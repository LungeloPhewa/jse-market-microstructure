"""Part I, section 2: intraday microstructure pictures and the stylised facts
of order flow and trade arrivals (the original ``part2_eda.py``).

The original module was not part of the Assignment 1 submission bundle; this
is a re-implementation from the report's function calls and descriptions,
following Jericevich, Chang & Gebbie (2020).

Contents
--------
* Intraday "bubble" charts of top-of-book updates and trades for a window
  of one day, and a ten-level depth snapshot (read from the globally sorted,
  merged event file, which still knows which side of the book changed).
* Order-flow autocorrelation: the ACF of Lee-Ready trade signs
  :math:`\\epsilon_i \\in \\{-1, +1\\}`,

  .. math:: \\hat\\rho(\\ell) = \\frac{\\sum_i (\\epsilon_i - \\bar\\epsilon)(\\epsilon_{i+\\ell} - \\bar\\epsilon)}
                                    {\\sum_i (\\epsilon_i - \\bar\\epsilon)^2},

  with the white-noise band :math:`\\pm 1.96/\\sqrt{n}`.
* Trade inter-arrival times :math:`\\tau_i = t_{i+1} - t_i` (within a day):
  histograms, a QQ-plot against the exponential law of a Poisson process,
  a power-law tail fitted by maximum likelihood with the
  Clauset-Shalizi-Newman (2009) choice of :math:`x_{\\min}`, and their ACF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

TZ_LOCAL = "Africa/Johannesburg"
LEVELS = range(1, 11)


# ----------------------------------------------------------------------------
# Intraday windows and depth snapshots (merged, globally sorted event file)
# ----------------------------------------------------------------------------

def _window_utc(date, start_local, end_local):
    lo = pd.Timestamp(f"{date} {start_local}", tz=TZ_LOCAL).tz_convert("UTC")
    hi = pd.Timestamp(f"{date} {end_local}", tz=TZ_LOCAL).tz_convert("UTC")
    return lo, hi


def load_intraday_window(merged_path, ric, date, start_local, end_local):
    """Events of one security between two SAST clock times on one day.

    Returns one row per event with ``ts`` (SAST), ``source`` (TX or MD), the
    raw top-of-book columns ``BID``/``ASK``/``BIDSIZE``/``ASKSIZE`` (NaN when
    that side did not change in a TX quote message) and the trade fields.
    """
    lo, hi = _window_utc(date, start_local, end_local)
    cols = ["ric", "ts_utc", "source", "BID", "ASK", "BIDSIZE", "ASKSIZE", "TRDPRC_1", "TRDVOL_1"]
    flt = (ds.field("ric") == ric) & (ds.field("ts_utc") >= lo.to_pydatetime()) & (ds.field("ts_utc") < hi.to_pydatetime())
    df = ds.dataset(str(merged_path)).to_table(columns=cols, filter=flt).to_pandas()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert(TZ_LOCAL)
    for c in ("BID", "ASK", "BIDSIZE", "ASKSIZE", "TRDPRC_1", "TRDVOL_1"):
        df[c] = df[c].where(df[c] > 0)
    return df.drop(columns="ts_utc").reset_index(drop=True)


def plot_intraday_microstructure(window, ric, title=None, ax=None, size_quantile=0.99):
    """Bubble chart of a window: bid (blue) and ask (red) quote updates with
    area proportional to the size at the touch, trades (black) with area
    proportional to volume, and the mid- and micro-price paths.

    Sizes are capped at their ``size_quantile`` so that a few very large
    orders resting ahead of the closing auction do not swamp the picture.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.8))
    q = window[window["source"] == "MD"].copy()
    if q.empty:                       # fall back to the TX feed's own quotes
        q = window[window["TRDPRC_1"].isna()].copy()
    q[["BID", "ASK", "BIDSIZE", "ASKSIZE"]] = q[["BID", "ASK", "BIDSIZE", "ASKSIZE"]].ffill()
    tr = window[window["TRDPRC_1"].notna()]
    cap = np.nanquantile(np.r_[q["BIDSIZE"].to_numpy(), q["ASKSIZE"].to_numpy(), tr["TRDVOL_1"].to_numpy()], size_quantile)

    def area(v):
        return 4 + 160 * np.clip(v, 0, cap) / cap

    changed_b = q["BID"].ne(q["BID"].shift()) | q["BIDSIZE"].ne(q["BIDSIZE"].shift())
    changed_a = q["ASK"].ne(q["ASK"].shift()) | q["ASKSIZE"].ne(q["ASKSIZE"].shift())
    qb, qa = q[changed_b], q[changed_a]
    ax.scatter(qb["ts"], qb["BID"], s=area(qb["BIDSIZE"]), color="tab:blue", alpha=0.35, linewidths=0, label="bid update")
    ax.scatter(qa["ts"], qa["ASK"], s=area(qa["ASKSIZE"]), color="tab:red", alpha=0.35, linewidths=0, label="ask update")
    mid = (q["BID"] + q["ASK"]) / 2
    micro = (q["BID"] * q["ASKSIZE"] + q["ASK"] * q["BIDSIZE"]) / (q["BIDSIZE"] + q["ASKSIZE"])
    ax.step(q["ts"], mid, where="post", color="0.3", linewidth=0.8, label="mid-price")
    ax.step(q["ts"], micro, where="post", color="tab:green", linewidth=0.8, label="micro-price")
    ax.scatter(tr["ts"], tr["TRDPRC_1"], s=area(tr["TRDVOL_1"]), facecolors="none", edgecolors="black",
               linewidths=0.8, label="trade")
    # Robust y-range: auction call periods can show indicative, crossed quotes
    # far from the market, which would otherwise flatten the whole hour.
    def _cont(d):                      # rows before the 16:50 closing-auction call, if enough
        m = (d["ts"].dt.hour * 60 + d["ts"].dt.minute) < 16 * 60 + 50
        return d[m] if m.sum() >= 10 else d
    qc, tc = _cont(q), _cont(tr)
    px = np.r_[qc["BID"].dropna().to_numpy(), qc["ASK"].dropna().to_numpy(), tc["TRDPRC_1"].to_numpy()]
    if len(px):
        lo, hi = np.nanquantile(px, [0.01, 0.99])
        pad = 0.15 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title or ric)
    ax.set_ylabel("Price (ZAc)")
    ax.set_xlabel("Time (SAST)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_LOCAL))
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="upper left", fontsize=6.5, ncol=3, markerscale=0.6, framealpha=0.8)
    return ax


def load_order_book_snapshot(merged_path, ric, timestamp):
    """The last ten-level depth snapshot at or before ``timestamp`` (SAST).

    Returns a long DataFrame with columns ``side``, ``level``, ``price``,
    ``size`` and the snapshot time in ``attrs["ts"]``.
    """
    t = pd.Timestamp(timestamp, tz=TZ_LOCAL).tz_convert("UTC")
    cols = ["ric", "ts_utc", "source"] + [f"L{k}-{s}" for k in LEVELS for s in ("BidPrice", "BidSize", "AskPrice", "AskSize")]
    flt = ((ds.field("ric") == ric) & (ds.field("source") == "MD")
           & (ds.field("ts_utc") <= t.to_pydatetime()) & (ds.field("ts_utc") > (t - pd.Timedelta("1h")).to_pydatetime()))
    df = ds.dataset(str(merged_path)).to_table(columns=cols, filter=flt).to_pandas()
    if df.empty:
        raise ValueError(f"No depth snapshot for {ric} in the hour before {timestamp}")
    row = df.sort_values("ts_utc").iloc[-1]
    rows = []
    for k in LEVELS:
        for side, s in (("bid", "Bid"), ("ask", "Ask")):
            p, v = row[f"L{k}-{s}Price"], row[f"L{k}-{s}Size"]
            if p > 0 and v > 0:
                rows.append({"side": side, "level": k, "price": p, "size": v})
    out = pd.DataFrame(rows)
    out.attrs["ts"] = pd.Timestamp(row["ts_utc"]).tz_convert(TZ_LOCAL)
    return out


def plot_order_book_depth(snapshot, ric, ax=None):
    """Bar chart of a depth snapshot: bars at each price, height = size."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for side, colour in (("bid", "tab:blue"), ("ask", "tab:red")):
        s = snapshot[snapshot["side"] == side]
        ax.bar(s["price"], s["size"], width=0.8 * max(np.diff(np.sort(snapshot["price"])).min(), 1),
               color=colour, alpha=0.75, label=f"{side}s")
    ts = snapshot.attrs.get("ts")
    ax.set_title(f"{ric}: ten-level depth at {ts:%Y-%m-%d %H:%M:%S} SAST" if ts is not None else ric)
    ax.set_xlabel("Price (ZAc)")
    ax.set_ylabel("Resting volume (shares)")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.legend()
    return ax


# ----------------------------------------------------------------------------
# Order-flow autocorrelation
# ----------------------------------------------------------------------------

def acf(x, max_lag):
    """Sample autocorrelation at lags 1..max_lag (FFT, O(n log n))."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 ** int(np.ceil(np.log2(2 * n))))
    r = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return r[1:] / r[0]


def compute_order_flow_acf(df, ric, max_lag=1000):
    """ACF of trade signs (continuous session, consecutive trades of one RIC)."""
    s = df.loc[(df["ric"] == ric) & (df["event_type"] == "trade"), "trade_sign"].dropna().to_numpy()
    r = acf(s, max_lag)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "acf": r, "bound": 1.96 / np.sqrt(len(s))})


def plot_order_flow_acf(acf_df, ric, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.8))
    ax.vlines(acf_df["lag"], 0, acf_df["acf"], color="tab:blue", linewidth=0.7)
    b = acf_df["bound"].iloc[0]
    ax.axhline(b, color="tab:red", linestyle="--", linewidth=0.8)
    ax.axhline(-b, color="tab:red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Lag (trades, log scale)")
    ax.set_ylabel("ACF of trade sign")
    ax.set_title(f"{ric}: order-flow autocorrelation (95% white-noise band dashed)")
    return ax


# ----------------------------------------------------------------------------
# Inter-arrival times
# ----------------------------------------------------------------------------

def compute_interarrival_times(df, ric):
    """Seconds between consecutive trades of one RIC within the same day."""
    t = df.loc[(df["ric"] == ric) & (df["event_type"] == "trade"), ["ts_utc", "date"]].sort_values("ts_utc")
    tau = t.groupby("date")["ts_utc"].diff().dt.total_seconds()
    return tau.dropna().to_numpy()


def plot_interarrival_histogram(tau, ric, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(tau, bins=200, color="tab:blue")
    axes[0].set_xlabel("Inter-arrival time (s)")
    axes[0].set_title(f"{ric}: inter-arrival times")
    pos = tau[tau > 0]
    axes[1].hist(np.log10(pos), bins=120, color="tab:orange")
    axes[1].set_xlabel(r"$\log_{10}$ inter-arrival time (s)")
    axes[1].set_title(f"{ric}: log scale ({(tau == 0).mean():.1%} zero durations omitted)")
    return axes


def plot_interarrival_qq_exponential(tau, ric, ax=None, n_points=2000):
    """QQ-plot of inter-arrival times against Exp(1/mean): points above the
    line in the upper tail mean longer quiet spells than a Poisson process."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x = np.sort(tau)
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    idx = np.unique(np.linspace(0, len(x) - 1, min(n_points, len(x))).astype(int))
    theo = -np.mean(x) * np.log1p(-p[idx])
    ax.scatter(theo, x[idx], s=6)
    lim = max(theo.max(), x[idx].max())
    ax.plot([0, lim], [0, lim], color="tab:red", linewidth=0.8)
    ax.set_xlabel("Exponential quantiles (s)")
    ax.set_ylabel("Empirical quantiles (s)")
    ax.set_title(f"{ric}: QQ-plot vs exponential")
    return ax


def power_law_mle(x, xmin):
    """Continuous power-law MLE (Clauset et al. 2009, eq. 3.1):
    :math:`\\hat\\alpha = 1 + n \\left[\\sum_{i} \\ln(x_i/x_{\\min})\\right]^{-1}`
    over the :math:`n` observations :math:`x_i \\ge x_{\\min}`."""
    t = x[x >= xmin]
    return 1.0 + len(t) / np.sum(np.log(t / xmin)), len(t)


def fit_power_law(x, xmin_method="clauset", n_candidates=200, min_tail=50):
    """Power-law tail fit. With ``xmin_method="clauset"`` the lower cut-off is
    the candidate that minimises the Kolmogorov-Smirnov distance between the
    empirical and fitted tail CDFs; ``"min"`` uses the smallest positive value."""
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    if xmin_method == "min":
        a, n = power_law_mle(x, x[0])
        return {"alpha": a, "xmin": x[0], "n_tail": n, "ks": np.nan}
    cands = np.unique(np.quantile(x, np.linspace(0, 0.99, n_candidates)))
    best = None
    for xm in cands:
        t = x[x >= xm]
        if len(t) < min_tail:
            break
        a = 1.0 + len(t) / np.sum(np.log(t / xm))
        emp = np.arange(len(t)) / len(t)
        fit = 1.0 - (t / xm) ** (1.0 - a)
        ks = np.max(np.abs(emp - fit))
        if best is None or ks < best["ks"]:
            best = {"alpha": a, "xmin": xm, "n_tail": len(t), "ks": ks}
    return best


def plot_interarrival_power_law_ccdf(tau, ric, xmin_method="clauset", ax=None):
    """Empirical CCDF of inter-arrival times (log-log) with the fitted tail."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    fit = fit_power_law(tau, xmin_method)
    x = np.sort(tau[tau > 0])
    ccdf = 1.0 - np.arange(len(x)) / len(x)
    ax.loglog(x, ccdf, ".", markersize=2, label="empirical")
    xt = np.logspace(np.log10(fit["xmin"]), np.log10(x.max()), 50)
    p_tail = fit["n_tail"] / len(x)
    ax.loglog(xt, p_tail * (xt / fit["xmin"]) ** (1 - fit["alpha"]), color="tab:red",
              label=fr"power law $\alpha$={fit['alpha']:.2f}, $x_{{min}}$={fit['xmin']:.0f}s")
    ax.set_xlabel("Inter-arrival time (s)")
    ax.set_ylabel(r"$P(\tau > x)$")
    ax.set_title(f"{ric}: tail of inter-arrival times")
    ax.legend()
    ax.figure.fit = fit
    return fit


def compute_interarrival_acf(df, ric, max_lag=100):
    tau = compute_interarrival_times(df, ric)
    r = acf(tau, max_lag)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "acf": r, "bound": 1.96 / np.sqrt(len(tau))})


def plot_interarrival_acf(acf_df, ric, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    ax.vlines(acf_df["lag"], 0, acf_df["acf"], color="tab:blue")
    b = acf_df["bound"].iloc[0]
    ax.axhline(b, color="tab:red", linestyle="--", linewidth=0.8)
    ax.axhline(-b, color="tab:red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (trades)")
    ax.set_ylabel("ACF")
    n_out = int((acf_df["acf"].abs() > b).sum())
    ax.set_title(f"{ric}: ACF of inter-arrival times ({n_out}/{len(acf_df)} lags outside the band)")
    return ax
