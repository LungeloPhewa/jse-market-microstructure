"""
cost_of_trading.py -- Part III: Cost of trading.

Per-trade measures (D = +1 buy / -1 sell, p = trade price, m0 = mid
prevailing just BEFORE the trade, m1 = mid prevailing `impact_horizon` later):

    Effective spread   ES = 2 D (p - m0) / m0
    Quoted spread      QS = (ask0 - bid0) / m0
    Price impact       PI = 2 D (m1 - m0) / m0
    Realised spread    RS = 2 D (p - m1) / m0          so that ES = RS + PI

VPIN (Easley, Lopez de Prado & O'Hara, 2012), TRADE-CLASSIFIED variant:
trades are bucketed by volume and each bucket's buy/sell split comes from
the validated Lee-Ready `trade_sign`. (This is not "bulk volume
classification", which splits volume probabilistically from standardised
price changes.) VPIN is reported as the mean bucket imbalance over the day.

Expected shortfall, per trade, in the "half-spread" units of a single
execution measured against the pre-trade mid:

    direct cost    = QS / 2               (crossing half the posted spread)
    indirect cost  = D (m1 - m0) / m0     (= PI / 2, the post-trade mid move)
    total          = direct + indirect

The indirect (impact) curve is binned by trade size and its log-log slope is
compared with the square-root-law reference of 0.5.

Why this changed: the first version defined indirect = |ES| - QS/2. Since
ES is close to QS on average, that residual is close to QS/2 *by
construction*, so the direct and indirect curves tracked each other and the
"impact" slope mostly reflected how the spread varies with trade size. Price
impact is now measured from the subsequent mid-price move.
"""

import numpy as np
import pandas as pd
from scipy import stats

TZ_LOCAL = "Africa/Johannesburg"


# ---------------------------------------------------------------------------
# Spreads and impact
# ---------------------------------------------------------------------------

def _clean_quotes(q):
    return q[q["bid"].gt(0) & q["ask"].gt(q["bid"])]


def compute_spreads(trades, md, ric, impact_horizon="60s"):
    """
    Attach the pre-trade quote and the mid `impact_horizon` later to each
    signed trade, and compute ES, QS, PI and RS.

    The pre-trade quote is the last valid quote strictly BEFORE the trade
    (allow_exact_matches=False), so a book update stamped at the same instant
    as the print (often the trade's own footprint) is not used as "before".
    The post-trade mid is the last valid quote at or before trade time +
    horizon, and is only used if it falls on the same SAST trading day.
    """
    tx = trades[trades["ric"] == ric].sort_values("ts", kind="mergesort").copy()
    q = _clean_quotes(md[md["ric"] == ric][["ts", "bid", "ask", "mid"]]).sort_values("ts", kind="mergesort")

    before = pd.merge_asof(tx[["ts"]], q, on="ts", direction="backward",
                           allow_exact_matches=False)
    tx["mid_before"] = before["mid"].to_numpy()
    tx["bid_before"] = before["bid"].to_numpy()
    tx["ask_before"] = before["ask"].to_numpy()

    h = pd.Timedelta(impact_horizon)
    left = pd.DataFrame({"ts_after": tx["ts"].to_numpy() + h})
    left["ts_after"] = pd.to_datetime(left["ts_after"], utc=True).dt.as_unit(q["ts"].dt.unit)
    after = pd.merge_asof(left, q.rename(columns={"ts": "ts_after"})[["ts_after", "mid"]],
                          on="ts_after", direction="backward")
    same_day = (left["ts_after"].dt.tz_convert(TZ_LOCAL).dt.date.to_numpy()
                == tx["ts"].dt.tz_convert(TZ_LOCAL).dt.date.to_numpy())
    tx["mid_after"] = np.where(same_day, after["mid"].to_numpy(), np.nan)

    tx = tx.dropna(subset=["mid_before", "bid_before", "ask_before", "trade_sign"])
    D, p, m0 = tx["trade_sign"], tx["price"], tx["mid_before"]
    tx["effective_spread"] = 2 * D * (p - m0) / m0
    tx["quoted_spread"] = (tx["ask_before"] - tx["bid_before"]) / m0
    tx["price_impact"] = 2 * D * (tx["mid_after"] - m0) / m0
    tx["realised_spread"] = 2 * D * (p - tx["mid_after"]) / m0
    tx.attrs["impact_horizon"] = impact_horizon
    return tx


def spread_summary(tx):
    """Mean ES, QS, PI, RS in bps and the share of trades with ES < 0."""
    bps = lambda c: tx[c].mean() * 1e4
    return {
        "effective_bps": bps("effective_spread"),
        "quoted_bps": bps("quoted_spread"),
        "impact_bps": bps("price_impact"),
        "realised_bps": bps("realised_spread"),
        "es_over_qs": tx["effective_spread"].mean() / tx["quoted_spread"].mean(),
        "share_negative_es": (tx["effective_spread"] < 0).mean(),
        "n_trades": len(tx),
    }


def plot_spread_distributions(tx, ric, ax, clip_pct=(0.5, 99.5), verbose=True):
    """Overlaid histograms of ES and QS (bps) on COMMON bins over the central
    [clip_pct] range of the pooled data; the share of trades outside that
    range is stated in the legend. Returns spread_summary(tx)."""
    es, qs = tx["effective_spread"] * 1e4, tx["quoted_spread"] * 1e4
    pooled = pd.concat([es, qs]).dropna()
    lo, hi = np.percentile(pooled, clip_pct)
    bins = np.linspace(lo, hi, 61)
    out = ((es < lo) | (es > hi)).mean()
    ax.hist(es.clip(lo, hi), bins=bins, alpha=0.6, label="Effective spread")
    ax.hist(qs.clip(lo, hi), bins=bins, alpha=0.6, label="Quoted spread")
    summ = spread_summary(tx)
    ax.axvline(summ["effective_bps"], color="tab:blue", linestyle="--", linewidth=1)
    ax.axvline(summ["quoted_bps"], color="tab:orange", linestyle="--", linewidth=1)
    ax.legend(title=f"{out:.1%} of ES outside axis", fontsize=7, title_fontsize=7)
    ax.set_title(f"{ric}: effective vs quoted spread")
    ax.set_xlabel("bps (dashed: means)"); ax.set_ylabel("Trades")
    if verbose:
        print(f"[{ric}] mean effective spread: {summ['effective_bps']:.2f} bps, "
              f"mean quoted spread: {summ['quoted_bps']:.2f} bps")
    return summ


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

def compute_vpin(trades, ric, day, n_buckets=50, exclude_oversized=False):
    """
    Trade-classified VPIN for one (ric, day).

        VPIN = (1/n) * sum_i |V_buy_i - V_sell_i| / V,   V = day volume / n

    A trade larger than the room left in the current bucket is split across
    the boundary, so every bucket's imbalance stays in [0, 1]. Unsigned trades
    use bucket capacity but count as neither buy nor sell.

    exclude_oversized=True drops trades whose single print exceeds one
    bucket (V), recomputing V until none remain. Such prints (typically
    auction uncrosses or reported off-book blocks) fill several consecutive
    buckets with one sign and drive the bucket imbalance to exactly 1, so
    they say little about the persistence of order FLOW. The number and
    volume share of excluded trades are stored in buckets.attrs.
    """
    tx = trades[(trades["ric"] == ric) & (trades["date"] == day)].sort_values("ts", kind="mergesort")
    if tx.empty:
        print(f"[VPIN] no trades for {ric} on {day}")
        return None, None

    total_all = tx["volume"].sum()
    n_excl = 0
    if exclude_oversized:
        while True:
            V = tx["volume"].sum() / n_buckets
            big = tx["volume"] > V
            if not big.any():
                break
            n_excl += int(big.sum())
            tx = tx[~big]

    V = tx["volume"].sum() / n_buckets
    buy, sell = np.zeros(n_buckets), np.zeros(n_buckets)
    b, remaining = 0, V
    for vol, sign in zip(tx["volume"].to_numpy(), tx["trade_sign"].to_numpy()):
        v = vol
        while v > 1e-12 and b < n_buckets:
            take = min(v, remaining)
            if sign > 0:
                buy[b] += take
            elif sign < 0:
                sell[b] += take
            v -= take
            remaining -= take
            if remaining <= 1e-9:
                b += 1
                remaining = V
        if v > 1e-12:  # floating-point leftover at the end of the day
            if sign > 0:
                buy[-1] += v
            elif sign < 0:
                sell[-1] += v

    buckets = pd.DataFrame({"buy": buy, "sell": sell})
    buckets["imbalance"] = (buckets["buy"] - buckets["sell"]).abs() / V
    buckets.attrs.update(n_excluded=n_excl,
                         excluded_volume_share=1 - tx["volume"].sum() / total_all,
                         exclude_oversized=exclude_oversized)
    return buckets, float(buckets["imbalance"].mean())


def plot_vpin(buckets, vpin, ric, day, ax, label=None, color=None, verbose=True):
    """Plot bucket imbalances with the mean as a dashed line. Can be called
    twice on the same axes (e.g. all trades vs oversized prints excluded)."""
    if label is None:
        label = "all trades"
        if buckets.attrs.get("exclude_oversized"):
            label = (f"excl. {buckets.attrs['n_excluded']} oversized prints "
                     f"({buckets.attrs['excluded_volume_share']:.0%} of volume)")
    line, = ax.plot(buckets.index, buckets["imbalance"], marker="o", markersize=3,
                    linewidth=1, color=color, label=f"{label}: VPIN = {vpin:.3f}")
    ax.axhline(vpin, color=line.get_color(), linestyle="--", linewidth=1)
    ax.set_xlabel("Volume bucket"); ax.set_ylabel("|Buy − Sell| / V")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"{ric} {day}: order imbalance by volume bucket")
    ax.legend(fontsize=7)
    if verbose:
        print(f"[{ric}] VPIN on {day} ({len(buckets)} buckets, {label}): {vpin:.4f}")


# ---------------------------------------------------------------------------
# Expected shortfall and the square-root law
# ---------------------------------------------------------------------------

def compute_expected_shortfall(tx_with_spreads, n_q=20):
    """Bin signed trades by size (quantiles) and average, per bin:
    direct = QS/2, indirect = D (m1 - m0)/m0 (signed, so adverse moves are
    positive), total = direct + indirect. Also returns n per bin."""
    tx = tx_with_spreads.dropna(subset=["quoted_spread", "price_impact"]).copy()
    tx["direct_cost"] = tx["quoted_spread"] / 2
    tx["indirect_cost"] = tx["price_impact"] / 2
    tx["total_shortfall"] = tx["direct_cost"] + tx["indirect_cost"]
    grp = tx.groupby(pd.qcut(tx["volume"], n_q, duplicates="drop"), observed=True).agg(
        volume=("volume", "mean"), direct=("direct_cost", "mean"),
        indirect=("indirect_cost", "mean"), total=("total_shortfall", "mean"),
        n=("volume", "size"),
    ).dropna()
    grp.attrs["impact_horizon"] = tx_with_spreads.attrs.get("impact_horizon")
    return grp


def fit_impact_exponent(grp):
    """log-log OLS of indirect cost on mean bin volume, over bins with a
    positive (adverse) average impact. Returns dict(slope, intercept, r2,
    p_value, n_bins_used, n_bins)."""
    pos = grp[grp["indirect"] > 0]
    out = dict(slope=np.nan, intercept=np.nan, r2=np.nan, p_value=np.nan,
               n_bins_used=len(pos), n_bins=len(grp))
    if len(pos) > 2:
        fit = stats.linregress(np.log(pos["volume"]), np.log(pos["indirect"]))
        out.update(slope=fit.slope, intercept=fit.intercept, r2=fit.rvalue ** 2,
                   p_value=fit.pvalue)
    return out


def plot_expected_shortfall_and_sqrt_law(grp, ric, ax):
    """Direct, indirect (impact) and total cost by size bin on log-log axes,
    with the fitted impact slope and a 0.5 reference line through the same
    centre point. Returns the fitted slope (float); full fit in grp.attrs."""
    fit = fit_impact_exponent(grp)
    grp.attrs["fit"] = fit
    h = grp.attrs.get("impact_horizon") or "Δt"

    ax.loglog(grp["volume"], grp["direct"], marker="o", markersize=4,
              label="Direct: half quoted spread")
    pos = grp[grp["indirect"] > 0]
    ax.loglog(pos["volume"], pos["indirect"], marker="s", markersize=4,
              label=f"Indirect: mid move after {h}")
    tot = grp[grp["total"] > 0]
    ax.loglog(tot["volume"], tot["total"], marker="^", markersize=4, color="black",
              label="Total")
    if np.isfinite(fit["slope"]):
        lx = np.log(pos["volume"])
        xc = lx.mean()
        yc = fit["intercept"] + fit["slope"] * xc
        xs = np.linspace(lx.min(), lx.max(), 50)
        ax.loglog(np.exp(xs), np.exp(yc + fit["slope"] * (xs - xc)), color="tab:orange",
                  linestyle="-", alpha=0.5, label=f"Fitted impact slope = {fit['slope']:.2f}")
        ax.loglog(np.exp(xs), np.exp(yc + 0.5 * (xs - xc)), "k--", linewidth=1,
                  label="Square-root reference (slope 0.5)")
    ax.set_xlabel("Trade volume (shares, bin mean)")
    ax.set_ylabel("Cost (fraction of mid)")
    ax.set_title(f"{ric}: cost decomposition by trade size")
    ax.legend(fontsize=7)
    return fit["slope"]


# ---------------------------------------------------------------------------
# Case study: large market orders on one day, by time of day
# ---------------------------------------------------------------------------

CASE_WINDOWS = {"open": ("09:10", "10:10"), "midday": ("12:00", "13:00"),
                "close": ("15:50", "16:50")}


def _market_orders(trades, gap_ms=10.0):
    """Consecutive prints of the same ric and Lee-Ready sign, each within
    gap_ms of the previous one, form one market order (an aggressive order
    that walks the book prints once per level). Returns per order: ric,
    ts (first print), trade_sign, volume, vwap, n_prints."""
    t = trades.dropna(subset=["trade_sign"]).sort_values(["ric", "ts"], kind="mergesort").copy()
    tn = t["ts"].dt.tz_convert(None).astype("datetime64[ns]").to_numpy().view("int64")
    new = np.ones(len(t), dtype=bool)
    new[1:] = ((t["ric"].to_numpy()[1:] != t["ric"].to_numpy()[:-1])
               | (t["trade_sign"].to_numpy()[1:] != t["trade_sign"].to_numpy()[:-1])
               | (np.diff(tn) > gap_ms * 1e6))
    t["oid"] = np.cumsum(new)
    t["pv"] = t["price"] * t["volume"]
    g = t.groupby("oid")
    out = pd.DataFrame({"ric": g["ric"].first(), "ts": g["ts"].first(),
                        "trade_sign": g["trade_sign"].first(), "volume": g["volume"].sum(),
                        "pv": g["pv"].sum(), "n_prints": g.size()})
    out["vwap"] = out["pv"] / out["volume"]
    return out.drop(columns="pv").reset_index(drop=True)


def large_order_cost_sample(trades, md, day, rics, windows=None, n_per_window=3,
                            horizon="60s", gap_ms=10.0, price_unit=100.0):
    """For one SAST trading day: the n largest market orders in each window,
    with their trading costs.

    Per order (prices in ZAc; price_unit=100 converts to rand):
      quoted_bps    (ask - bid) / mid of the last quote before the first print
      effective_bps 2 D (vwap - mid) / mid
      impact_bps    2 D (mid_after - mid) / mid,   mid_after = mid `horizon` later
      realised_bps  2 D (vwap - mid_after) / mid  (effective = realised + impact)
      taker_cost_R    D (vwap - mid) x volume      what the taker paid vs the mid
      provider_gain_R D (vwap - mid_after) x volume what providers kept once the
                      price had moved (negative = providers lost)
    """
    windows = windows or CASE_WINDOWS
    tx = trades[(trades["date"] == day) & trades["ric"].isin(rics)]
    q = md[(md["date"] == day) & md["ric"].isin(rics)][["ric", "ts", "bid", "ask", "mid"]]
    q = _clean_quotes(q).sort_values("ts", kind="mergesort")
    mo = _market_orders(tx, gap_ms).sort_values("ts", kind="mergesort")
    tod = mo["ts"].dt.tz_convert(TZ_LOCAL).dt.time
    parts = []
    for ric in rics:
        for w, (a, b) in windows.items():
            a, b = pd.Timestamp(a).time(), pd.Timestamp(b).time()
            sel = mo[(mo["ric"] == ric) & (tod >= a) & (tod < b)].nlargest(n_per_window, "volume")
            parts.append(sel.assign(window=w))
    s = pd.concat(parts, ignore_index=True).sort_values("ts", kind="mergesort")
    pre = pd.merge_asof(s[["ts", "ric"]], q, on="ts", by="ric", direction="backward",
                        allow_exact_matches=False)
    after = pd.merge_asof(pd.DataFrame({"ts": s["ts"] + pd.Timedelta(horizon), "ric": s["ric"]}),
                          q[["ts", "ric", "mid"]], on="ts", by="ric", direction="backward")
    s["bid"], s["ask"], s["mid"] = pre["bid"].to_numpy(), pre["ask"].to_numpy(), pre["mid"].to_numpy()
    s["mid_after"] = after["mid"].to_numpy()
    D, m = s["trade_sign"], s["mid"]
    s["quoted_bps"] = 1e4 * (s["ask"] - s["bid"]) / m
    s["effective_bps"] = 1e4 * 2 * D * (s["vwap"] - m) / m
    s["impact_bps"] = 1e4 * 2 * D * (s["mid_after"] - m) / m
    s["realised_bps"] = 1e4 * 2 * D * (s["vwap"] - s["mid_after"]) / m
    s["notional_R"] = s["vwap"] * s["volume"] / price_unit
    s["taker_cost_R"] = D * (s["vwap"] - m) * s["volume"] / price_unit
    s["provider_gain_R"] = D * (s["vwap"] - s["mid_after"]) * s["volume"] / price_unit
    s["local_time"] = s["ts"].dt.tz_convert(TZ_LOCAL).dt.strftime("%H:%M:%S")
    s["side"] = np.where(D > 0, "buy", "sell")
    order = {w: i for i, w in enumerate(windows)}
    return (s.assign(_w=s["window"].map(order)).sort_values(["ric", "_w", "ts"])
             .drop(columns="_w").reset_index(drop=True))


def summarise_cost_sample(sample):
    """Totals by security and window: volume-weighted spreads (bps) and rand
    amounts for the taker and the providers."""
    def agg(g):
        w = g["notional_R"]
        return pd.Series({
            "orders": len(g), "notional_R": w.sum(),
            "quoted_bps": np.average(g["quoted_bps"], weights=w),
            "effective_bps": np.average(g["effective_bps"], weights=w),
            "impact_bps": np.average(g["impact_bps"], weights=w),
            "realised_bps": np.average(g["realised_bps"], weights=w),
            "taker_cost_R": g["taker_cost_R"].sum(),
            "provider_gain_R": g["provider_gain_R"].sum(),
        })
    order = {w: i for i, w in enumerate(CASE_WINDOWS)}
    out = sample.groupby(["ric", "window"]).apply(agg, include_groups=False).reset_index()
    return (out.assign(_w=out["window"].map(order)).sort_values(["ric", "_w"])
               .drop(columns="_w").reset_index(drop=True))


def vpin_shuffled_baseline(trades, ric, day, n_sims=200, seed=0, n_buckets=50,
                           exclude_oversized=True):
    """VPIN expected with NO order-flow persistence: the day's trades keep
    their times and sizes (and the day's buy share), but their signs are
    randomly permuted. A small number of large trades per bucket makes the
    bucket imbalance large even with random signs, so this is the level an
    observed VPIN should be compared with. Returns dict(mean, q025, q975)."""
    rng = np.random.default_rng(seed)
    m = (trades["ric"] == ric) & (trades["date"] == day)
    sub = trades.loc[m].copy()
    signs = sub["trade_sign"].to_numpy()
    vals = []
    for _ in range(n_sims):
        sub["trade_sign"] = rng.permutation(signs)
        vals.append(compute_vpin(sub, ric, day, n_buckets, exclude_oversized)[1])
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()), "q025": float(np.quantile(vals, 0.025)),
            "q975": float(np.quantile(vals, 0.975))}


def vpin_by_day(trades, ric, days, n_buckets=50, exclude_oversized=True):
    """Daily VPIN for every day in `days` (Series indexed by day)."""
    return pd.Series({d: compute_vpin(trades, ric, d, n_buckets, exclude_oversized)[1]
                      for d in days}, name=ric)
