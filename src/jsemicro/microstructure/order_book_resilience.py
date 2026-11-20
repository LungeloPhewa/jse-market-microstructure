"""
order_book_resilience.py
------------------------

Order-book resilience around large market orders, after Large (2007), for
AGLJ.J / GRTJ.J.

Pipeline (run_resilience does all of it)
----------------------------------------
1. Candidates. For each security and SAST window, all signed trades at or
   above the window's `volume_quantile` volume (largest `top_n` kept).
2. Depth. L1-L10 sizes are streamed from the MD parquet for a window around
   every candidate: `pre_seconds` before (default 300 s) to the last horizon
   after. Within each window the sizes are forward-filled, because the
   displayed book at time t is the LAST state at or before t. The update
   rows need not repeat unchanged levels, and the book may simply not
   change for a while.
3. Qualification. A candidate is a usable liquidity shock only if
     a. there is depth data around it (the MD feed covers that moment);
     b. it is no larger than the displayed depth L1-L10 on the side it
        consumes. A bigger trade cannot have executed against the visible
        book: it is an off-book / reported / auction print;
     c. it is at least `min_l1_fraction` x the L1 depth on that side, so it
        actually depletes the touch (default 1.0 = clears the best level);
     d. the book updates within `react_seconds` after it (the trade left
        a footprint in the book).
   The largest qualifying candidate per (ric, window) becomes the shock.
4. Recovery. For each shock, the consumed side's L1, L2, L3 depth and their
   L1-L3 total are read at:
       before  last state strictly before the trade (baseline, 100%)
       0s      first state at/after the trade within `react_seconds`
       +h s    state prevailing at trade + h (as-of)
   and expressed as % of the baseline.

Why the previous version produced empty and misleading panels
-------------------------------------------------------------
* Only +/-10 s of depth was loaded, and the baseline was "the first update
  in that window". When the book had not updated in the 10 s before the
  trade there was no baseline at all (GRTJ.J midday), and when the feed had
  no updates within +/-10 s nothing was found (AGLJ.J pre-close).
* After the trade it used "the first update at or after t + h, within 10 s",
  rather than the book prevailing at t + h, so quiet periods dropped
  horizons.
* The largest trade in a window was taken without checking that it could
  have executed against the book. The 5.6M-share GRTJ.J and 0.7M-share
  AGLJ.J prints are block / off-book trades; the book did not react to
  them, so they say nothing about resilience.
* Rows were not forward-filled, so a level missing from an update row
  showed as an empty bar.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TZ_LOCAL = "Africa/Johannesburg"
DEFAULT_HORIZONS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0)   # seconds after the trade
BEFORE = -1.0                                    # horizon code for "just before"
DEFAULT_WINDOWS = {"open": ("09:10", "10:10"), "midday": ("12:00", "13:00"),
                   "close": ("15:50", "16:50")}            # SAST, one hour each


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _col(df, names, what):
    c = _find_column(df, names)
    if c is None:
        raise KeyError(f"Could not identify the {what} column (tried {names}).")
    return c


def _ric_col(df):  return _col(df, ["ric", "source_ric"], "RIC")
def _ts_col(df):   return _col(df, ["ts", "ts_utc", "timestamp", "datetime", "date_time"], "timestamp")
def _sign_col(df): return _col(df, ["trade_sign", "sign", "lee_ready_sign", "direction"], "trade sign")
def _vol_col(df):  return _col(df, ["volume", "TRDVOL_1", "trade_volume", "size"], "trade volume")


def infer_depth_columns(md: pd.DataFrame, explicit=None, max_level: int = 10
                        ) -> Dict[str, Dict[int, str]]:
    """{"bid": {1: col, 2: col, ...}, "ask": {...}} for every level present
    (up to max_level). Requires at least L1-L3 on both sides. Matching is
    side-specific, so ask can never resolve to a bid column."""
    if explicit is not None:
        return explicit
    out: Dict[str, Dict[int, str]] = {"bid": {}, "ask": {}}
    for side in ("bid", "ask"):
        pat = re.compile(
            rf"^(?:L(?P<a>\d+)[-_]?{side}_?size|{side}_?size_?(?P<b>\d+)|{side}(?P<c>\d+)_?size"
            rf"|{side}_?L(?P<d>\d+)_?size)$", re.I)
        for c in md.columns:
            m = pat.match(str(c))
            if m:
                lvl = int(next(g for g in m.groups() if g))
                if lvl <= max_level:
                    out[side].setdefault(lvl, c)
        out[side] = dict(sorted(out[side].items()))
        missing = [l for l in (1, 2, 3) if l not in out[side]]
        if missing:
            raise KeyError(f"No {side} size column for level(s) {missing}. "
                           f"Columns: {list(md.columns)}")
    return out


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _to_utc(values) -> pd.Series:
    s = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def _ns(series: pd.Series) -> np.ndarray:
    """Tz-aware timestamps -> int64 ns since epoch, whatever the stored unit."""
    return series.dt.tz_convert(None).astype("datetime64[ns]").to_numpy().view("int64")


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _tns(t: pd.Timestamp) -> int:
    return int(t.as_unit("ns").value)


# ---------------------------------------------------------------------------
# 1. Candidates
# ---------------------------------------------------------------------------

def build_market_orders(trades, gap_ms=10.0):
    """Group prints into market orders.

    An aggressive order that walks the book prints once per price level (and
    per resting order it fills), all within a few milliseconds. Consecutive
    prints of the same security and the same Lee-Ready sign, each no more
    than `gap_ms` after the previous one, are treated as one market order.

    Returns one row per order: ric, shock_ts (first print), last_ts,
    trade_sign, shock_volume (total), first_price, worst_price (highest for a
    buy, lowest for a sell), n_prints.
    """
    rc, tc, sc, vc = _ric_col(trades), _ts_col(trades), _sign_col(trades), _vol_col(trades)
    t = pd.DataFrame({
        "ric": trades[rc].to_numpy(),
        "ts": pd.to_datetime(_to_utc(trades[tc]).to_numpy(), utc=True),
        "sign": pd.to_numeric(trades[sc], errors="coerce").to_numpy(),
        "vol": pd.to_numeric(trades[vc], errors="coerce").to_numpy(),
        "price": pd.to_numeric(trades["price"], errors="coerce").to_numpy(),
    }).dropna(subset=["ts", "sign", "vol", "price"])
    t = t[t["sign"].isin([1, -1]) & (t["vol"] > 0)].sort_values(["ric", "ts"], kind="mergesort")
    tn = _ns(t["ts"])
    new = np.ones(len(t), dtype=bool)
    new[1:] = ((t["ric"].to_numpy()[1:] != t["ric"].to_numpy()[:-1])
               | (t["sign"].to_numpy()[1:] != t["sign"].to_numpy()[:-1])
               | (np.diff(tn) > gap_ms * 1e6))
    t["order_id"] = np.cumsum(new)
    g = t.groupby("order_id")
    out = pd.DataFrame({
        "ric": g["ric"].first(), "shock_ts": g["ts"].first(), "last_ts": g["ts"].last(),
        "trade_sign": g["sign"].first().astype(int), "shock_volume": g["vol"].sum(),
        "first_price": g["price"].first(), "n_prints": g.size(),
        "_pmax": g["price"].max(), "_pmin": g["price"].min(),
    }).reset_index(drop=True)
    out["worst_price"] = np.where(out["trade_sign"] == 1, out["_pmax"], out["_pmin"])
    return out.drop(columns=["_pmax", "_pmin"])


def select_shock_candidates(trades, rics=None, time_windows=None, volume_quantile=0.95,
                            top_n=60, gap_ms=10.0):
    """Market orders (see build_market_orders) at or above the window's
    `volume_quantile` of order size, largest first, at most top_n per ric
    and SAST window."""
    if time_windows is None:
        time_windows = DEFAULT_WINDOWS
    mo = build_market_orders(trades, gap_ms)
    mo["shock_local"] = mo["shock_ts"].dt.tz_convert(TZ_LOCAL)
    tod = mo["shock_local"].dt.time
    if rics is None:
        rics = sorted(mo["ric"].unique())
    parts = []
    for ric in rics:
        for name, (a, b) in time_windows.items():
            a, b = pd.Timestamp(a).time(), pd.Timestamp(b).time()
            inw = (mo["ric"] == ric) & (tod >= a) & (tod < b)
            w = mo[inw]
            if w.empty:
                continue
            thr = w["shock_volume"].quantile(volume_quantile)
            w = (w[w["shock_volume"] >= thr]
                 .sort_values("shock_volume", ascending=False).head(top_n).copy())
            w["window"] = name
            w["window_threshold"] = thr
            w["window_orders"] = int(inw.sum())
            parts.append(w)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Depth loading
# ---------------------------------------------------------------------------

def _size_cols_in_file(path):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    pat = re.compile(r"^L(\d+)-(Bid|Ask)(Size|Price)$")
    cols = [n for n in names if pat.match(n)]
    if not cols:  # fall back to anything that looks like a size column
        cols = [n for n in names if re.search(r"(bid|ask).*size|size.*(bid|ask)", n, re.I)]
    ts = next(n for n in ("ts_utc", "ts", "timestamp") if n in names)
    return ts, cols


def load_md_depth_for_shocks(path, shocks: pd.DataFrame, pre_seconds: float = 0.5,
                             post_seconds: float = 61.0, batch_size: int = 200_000,
                             horizons: Sequence[float] = DEFAULT_HORIZONS,
                             lookback_seconds: float = 600.0,
                             padding_seconds: Optional[float] = None) -> pd.DataFrame:
    """
    Stream the MD parquet and keep only the rows needed to know the book
    around each event in `shocks` (columns ric, shock_ts):

      * every row within +/- pre_seconds of each anchor time, where the
        anchors are the trade time and trade time + h for each horizon h;
      * for each anchor, the last row before (anchor - pre_seconds), looking
        back at most `lookback_seconds` -- the book state then in force.

    Each depth row is a full snapshot of the book, so this is enough to
    recover the book as-of any anchor, however long the book had been quiet,
    while holding a small fraction of the file in memory.
    (`post_seconds` and `padding_seconds` are accepted for compatibility.)
    """
    import pyarrow.parquet as pq

    if shocks is None or shocks.empty:
        return pd.DataFrame()
    ts_name, depth_cols = _size_cols_in_file(path)
    offsets = sorted({0.0, *[float(h) for h in horizons if h >= 0]})
    hw = int(pre_seconds * 1e9)
    lb = int(lookback_seconds * 1e9)

    anchors = {}
    for ric, g in shocks.groupby("ric"):
        t = np.array([_tns(_ts(x)) for x in g["shock_ts"]], dtype=np.int64)
        a = np.sort((t[:, None] + (np.array(offsets) * 1e9).astype(np.int64)[None, :]).ravel())
        anchors[ric] = a
    best_before = {ric: np.full(len(a), -1, dtype=np.int64) for ric, a in anchors.items()}
    best_rows = {ric: [None] * len(a) for ric, a in anchors.items()}

    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                    columns=["ric", ts_name, *depth_cols]):
        b = batch.to_pandas()
        b = b[b["ric"].isin(anchors.keys())]
        if b.empty:
            continue
        b[ts_name] = pd.to_datetime(_to_utc(b[ts_name]).to_numpy(), utc=True)
        b = b.dropna(subset=[ts_name])
        t = _ns(b[ts_name])
        keep = np.zeros(len(b), dtype=bool)
        ric_arr = b["ric"].to_numpy()
        for ric, a in anchors.items():
            m = np.flatnonzero(ric_arr == ric)
            if not len(m):
                continue
            tt = t[m]
            # rows within +/- hw of the nearest anchor
            k = np.searchsorted(a, tt)
            near = np.zeros(len(tt), dtype=bool)
            for kk in (k - 1, k):
                ok = (kk >= 0) & (kk < len(a))
                near[ok] |= np.abs(tt[ok] - a[kk[ok]]) <= hw
            keep[m[near]] = True
            # latest row before each anchor's window, within the look-back
            order = np.argsort(tt, kind="mergesort")
            ts_sorted = tt[order]
            pos = np.searchsorted(ts_sorted, a - hw, side="left") - 1
            for j in np.flatnonzero(pos >= 0):
                cand_t = ts_sorted[pos[j]]
                if cand_t >= a[j] - hw - lb and cand_t > best_before[ric][j]:
                    best_before[ric][j] = cand_t
                    best_rows[ric][j] = b.iloc[m[order[pos[j]]]]
        if keep.any():
            pieces.append(b.loc[keep])

    extra = [r for rows in best_rows.values() for r in rows if r is not None]
    if extra:
        pieces.append(pd.DataFrame(extra))
    if not pieces:
        return pd.DataFrame(columns=["ric", "ts", *depth_cols])
    df = pd.concat(pieces, ignore_index=True).rename(columns={ts_name: "ts"})
    # the look-back rows can duplicate rows already kept: drop exact copies only
    df = df.drop_duplicates(subset=["ric", "ts", *depth_cols], keep="first")
    num = [c for c in df.columns if c not in ("ric", "ts")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    sz = [c for c in num if c.endswith("Size")]
    df[sz] = df[sz].astype("float32")
    return df.sort_values(["ric", "ts"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Book state around one event
# ---------------------------------------------------------------------------

_BOOK_CACHE: dict = {}


def _ric_book(md, ric):
    """Per-ric depth rows sorted by time, converted once and cached:
    (times_ns, numeric DataFrame of all columns). Cache is keyed on the
    DataFrame's identity and length, so a new md_depth is picked up."""
    key = (id(md), len(md), ric)
    if key not in _BOOK_CACHE:
        if len(_BOOK_CACHE) > 16:
            _BOOK_CACHE.clear()
        rc, tc = _ric_col(md), _ts_col(md)
        sub = md.loc[md[rc].eq(ric)]
        if sub.empty:
            _BOOK_CACHE[key] = (None, None)
        else:
            t = _ns(pd.Series(pd.to_datetime(_to_utc(sub[tc]).to_numpy(), utc=True)))
            order = np.argsort(t, kind="mergesort")
            num = sub.drop(columns=[rc, tc]).iloc[order].apply(pd.to_numeric, errors="coerce")
            _BOOK_CACHE[key] = (t[order], num.reset_index(drop=True))
    return _BOOK_CACHE[key]


def _event_book(md, ric, shock_ts, side_cols, pre_seconds, post_seconds):
    """Rows of `ric` in the event window, sizes forward-filled, sorted.
    Returns (times_ns, sizes ndarray [n, levels]) or (None, None)."""
    t, num = _ric_book(md, ric)
    if t is None:
        return None, None
    s = _tns(shock_ts)
    a = np.searchsorted(t, s - int(pre_seconds * 1e9), side="left")
    b = np.searchsorted(t, s + int(post_seconds * 1e9), side="right")
    if b <= a:
        return None, None
    sizes = num[side_cols].iloc[a:b].ffill().to_numpy(dtype=float)
    return t[a:b], sizes


def _state(times, sizes, t_ns, mode):
    """mode 'before': last row strictly before t; 'asof': last row at/before t;
    'first_after': first row at/after t. Returns (row_index or None)."""
    if mode == "before":
        i = np.searchsorted(times, t_ns, side="left") - 1
    elif mode == "asof":
        i = np.searchsorted(times, t_ns, side="right") - 1
    else:
        i = np.searchsorted(times, t_ns, side="left")
    return int(i) if 0 <= i < len(times) else None


# ---------------------------------------------------------------------------
# 3. Qualification
# ---------------------------------------------------------------------------

def _l1_price_col(md, side):
    c = f"L1-{side.capitalize()}Price"
    return c if c in md.columns else None


def _align_shock(md, ric, st, sign, cols, pre_seconds, post_seconds, search_seconds=2.0):
    """Locate the trade's footprint in the depth feed.

    The trade and depth feeds are separate streams, so a trade's effect on
    the book can be stamped a few milliseconds before or after the trade
    print. The footprint is the depth update within +/- search_seconds of
    the trade, closest to it in time, at which the hit side's best level
    moves against the trade: for a buy the best ask price rises or its size
    falls; for a sell the best bid price falls or its size falls.

    Returns dict(times, sizes [n, levels on hit side], i_before, j_after,
    lag_s, found) or None if there is no depth around the trade.
    i_before = last row strictly before the footprint's timestamp,
    j_after  = last row carrying the footprint's timestamp.
    Without a footprint: i_before = last row before the trade, j_after =
    first update within search_seconds after it (or None).
    """
    side = "ask" if sign == 1 else "bid"
    hit_cols = list(cols[side].values())
    times, sizes = _event_book(md, ric, st, hit_cols, pre_seconds, post_seconds)
    if times is None:
        return None
    s = _tns(st)
    pc = _l1_price_col(md, side)
    if pc is not None:
        _, price = _event_book(md, ric, st, [pc], pre_seconds, post_seconds)
        price = price[:, 0]
    else:
        price = np.full(len(times), np.nan)
    l1 = sizes[:, 0]
    with np.errstate(invalid="ignore"):
        if sign == 1:
            adverse = (price[1:] > price[:-1]) | ((price[1:] == price[:-1]) & (l1[1:] < l1[:-1]))
        else:
            adverse = (price[1:] < price[:-1]) | ((price[1:] == price[:-1]) & (l1[1:] < l1[:-1]))
        if np.isnan(price).all():
            adverse = l1[1:] < l1[:-1]
    k_all = np.flatnonzero(adverse) + 1
    k_all = k_all[np.abs(times[k_all] - s) <= search_seconds * 1e9]
    if len(k_all):
        k = k_all[np.argmin(np.abs(times[k_all] - s))]
        tk = int(times[k])
        i = _state(times, sizes, tk, "before")
        j = _state(times, sizes, tk, "asof")
        return dict(times=times, sizes=sizes, i_before=i, j_after=j,
                    lag_s=(tk - s) / 1e9, found=True)
    i = _state(times, sizes, s, "before")
    j = _state(times, sizes, s, "first_after")
    if j is not None and times[j] > s + search_seconds * 1e9:
        j = None
    return dict(times=times, sizes=sizes, i_before=i, j_after=j, lag_s=np.nan, found=False)


def _price_cols(md, side, levels=range(1, 11)):
    S = side.capitalize()
    return [f"L{l}-{S}Price" for l in levels if f"L{l}-{S}Price" in md.columns]


def qualify_shocks(candidates, md_depth, depth_columns=None, min_l1_fraction=1.0,
                   pre_seconds=60.0, search_seconds=0.5, capacity_tolerance=0.10):
    """Decide which candidate market orders are genuine shocks to the
    DISPLAYED book, using the book just before the order's footprint:

      inside spread    the first print is better than the displayed best
                       price on the side hit (a buy below the best ask, a
                       sell above the best bid): it filled against hidden
                       liquidity rather than the displayed book;
      exceeds displayed depth
                       total volume is more than (1 + capacity_tolerance)
                       times the displayed quantity at the prices the order
                       traded through (up to its worst print): it filled
                       against hidden or iceberg volume, or was reported;
      smaller than L1  it did not clear the best level;
      no footprint     no matching change in the book within
                       +/- search_seconds of the first print.
    """
    cand = candidates.copy()
    if cand.empty:
        return cand
    cols = infer_depth_columns(md_depth, depth_columns)
    recs = []
    for _, c in cand.iterrows():
        sign = int(c["trade_sign"])
        side = "ask" if sign == 1 else "bid"
        st = _ts(c["shock_ts"])
        r = dict(pre_L1=np.nan, pre_L1_L3=np.nan, pre_L1_L10=np.nan, pre_best=np.nan,
                 displayed_capacity=np.nan, pre_staleness_s=np.nan, footprint_lag_s=np.nan,
                 has_depth=False, at_or_beyond_touch=False, within_displayed=False,
                 clears_l1=False, footprint=False)
        pc = _price_cols(md_depth, side)
        a = _align_shock(md_depth, c["ric"], st, sign, cols, pre_seconds,
                         search_seconds + 0.001, search_seconds=search_seconds)
        if a is not None and a["i_before"] is not None and pc:
            i = a["i_before"]
            sizes = np.nan_to_num(a["sizes"][i])
            _, prices = _event_book(md_depth, c["ric"], st, pc, pre_seconds,
                                    search_seconds + 0.001)
            prices = prices[i]
            n = min(len(prices), len(sizes))
            prices, sizes = prices[:n], sizes[:n]
            valid = np.isfinite(prices) & (prices > 0)
            if n and valid[0] and sizes[0] > 0:
                best = prices[0]
                worst = c.get("worst_price", c.get("price", np.nan))
                first = c.get("first_price", c.get("price", np.nan))
                eps = 1e-6 * abs(best)
                if sign == 1:
                    reach, at_touch = valid & (prices <= worst + eps), first >= best - eps
                else:
                    reach, at_touch = valid & (prices >= worst - eps), first <= best + eps
                cap = float(sizes[reach].sum())
                v = float(c["shock_volume"])
                r.update(has_depth=True, pre_L1=sizes[0], pre_L1_L3=sizes[:3].sum(),
                         pre_L1_L10=sizes.sum(), pre_best=best, displayed_capacity=cap,
                         pre_staleness_s=(_tns(st) - a["times"][i]) / 1e9,
                         footprint_lag_s=a["lag_s"], footprint=bool(a["found"]),
                         at_or_beyond_touch=bool(at_touch),
                         within_displayed=bool(v <= cap * (1 + capacity_tolerance)),
                         clears_l1=bool(v >= min_l1_fraction * sizes[0]))
        recs.append(r)
    cand = pd.concat([cand.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    checks = ["has_depth", "at_or_beyond_touch", "within_displayed", "clears_l1", "footprint"]
    cand["qualifies"] = cand[checks].all(axis=1)
    cand["reason"] = np.select(
        [~cand[c] for c in checks],
        ["no depth data", "inside spread (hidden liquidity)", "exceeds displayed depth",
         "smaller than L1", "no footprint in book"], default="ok")
    return cand


def rejection_summary(qualified):
    """Counts of candidates by outcome, per ric and window."""
    if qualified.empty:
        return pd.DataFrame()
    order = ["ok", "inside spread (hidden liquidity)", "exceeds displayed depth",
             "smaller than L1", "no footprint in book", "no depth data"]
    t = (qualified.groupby(["ric", "window", "reason"]).size()
         .unstack("reason", fill_value=0))
    return t[[c for c in order if c in t.columns]].reset_index()


def pick_shocks(qualified):
    """Largest qualifying candidate per (ric, window)."""
    q = qualified[qualified["qualifies"]]
    if q.empty:
        return q
    idx = q.groupby(["ric", "window"])["shock_volume"].idxmax()
    return q.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Recovery
# ---------------------------------------------------------------------------

def extract_depth_recovery(trades, md, shock, depth_columns=None,
                           horizons: Sequence[float] = DEFAULT_HORIZONS,
                           pre_seconds=60.0, react_seconds=0.5):
    """L1, L2, L3 and L1-L3 depth on the consumed side before and after one
    shock, as shares and as % of the pre-trade level. (`trades` is accepted
    for interface compatibility and not used.)"""
    cols = infer_depth_columns(md, depth_columns)
    sign = int(shock["trade_sign"])
    side = "ask" if sign == 1 else "bid"
    st = _ts(shock["shock_ts"])
    al = _align_shock(md, shock["ric"], st, sign, cols, pre_seconds, max(horizons) + 2.001,
                      search_seconds=react_seconds)
    if al is None or al["i_before"] is None:
        return pd.DataFrame()
    times, sizes = al["times"], al["sizes"][:, :3]
    opp = "bid" if side == "ask" else "ask"
    _, opp_sizes = _event_book(md, shock["ric"], st, [cols[opp][l] for l in (1, 2, 3)],
                               pre_seconds, max(horizons) + 2.001)

    # event time: the footprint if found, else the trade print
    s = int(times[al["j_after"]]) if al["found"] else _tns(st)
    rows = [(BEFORE, al["i_before"])]
    for h in horizons:
        if h == 0:
            j = al["j_after"] if al["j_after"] is not None else _state(times, sizes, s, "asof")
            if not al["found"] and j is not None:
                j = _state(times, sizes, int(times[j]), "asof")
        else:
            j = _state(times, sizes, s + int(h * 1e9), "asof")
        if j is not None:
            rows.append((float(h), j))

    recs = []
    for h, k in rows:
        d = sizes[k]
        rec = {
            "ric": shock["ric"], "window": shock["window"], "shock_ts": st,
            "shock_local": st.tz_convert(TZ_LOCAL), "trade_sign": sign,
            "shock_volume": float(shock["shock_volume"]), "side_examined": side,
            "horizon_seconds": h,
            "md_ts": pd.Timestamp(int(times[k]), unit="ns", tz="UTC"),
            "staleness_seconds": (s + (0 if h < 0 else int(h * 1e9)) - times[k]) / 1e9,
            "footprint_lag_s": al["lag_s"],
            "L1_depth": d[0], "L2_depth": d[1], "L3_depth": d[2],
            "L1_L3_depth": np.nansum(d),
            "direction": "buy" if sign == 1 else "sell",
            "opp_L1_L3_depth": np.nansum(opp_sizes[k]),
        }
        recs.append(rec)
    out = pd.DataFrame(recs)
    base = out[out["horizon_seconds"] == BEFORE]
    for c in ("L1", "L2", "L3", "L1_L3", "opp_L1_L3"):
        b = float(base.iloc[0][f"{c}_depth"]) if not base.empty else np.nan
        out[f"{c}_recovery_pct"] = 100 * out[f"{c}_depth"] / b if b and b > 0 else np.nan
    # each level as % of the pre-trade L1-L3 TOTAL: these stack to the
    # L1-L3 recovery and stay bounded when the touch is cleared (the new L1
    # is a different price level, so "% of old L1" can explode)
    tot = float(base.iloc[0]["L1_L3_depth"]) if not base.empty else np.nan
    for c in ("L1", "L2", "L3"):
        out[f"{c}_share_pct"] = 100 * out[f"{c}_depth"] / tot if tot and tot > 0 else np.nan
    return out.sort_values("horizon_seconds").reset_index(drop=True)


def run_resilience(trades, md_path, rics=None, time_windows=None, volume_quantile=0.95,
                   top_n=60, gap_ms=10.0, min_l1_fraction=1.0, capacity_tolerance=0.10,
                   search_seconds=0.5, pre_seconds=600.0,
                   horizons: Sequence[float] = DEFAULT_HORIZONS, batch_size=200_000):
    """Whole pipeline: market orders -> candidates -> displayed-book checks ->
    depth paths. Returns dict with candidates (all, with flags), rejections
    (counts), shocks (largest qualifying per ric/window, for illustration),
    recovery (their paths) and recovery_all (paths of every qualifying shock)."""
    if time_windows is None:
        time_windows = DEFAULT_WINDOWS
    cand = select_shock_candidates(trades, rics, time_windows, volume_quantile, top_n, gap_ms)
    md_depth = load_md_depth_for_shocks(md_path, cand, pre_seconds=search_seconds + 0.05,
                                        horizons=horizons, lookback_seconds=pre_seconds,
                                        batch_size=batch_size)
    if md_depth.empty:
        raise ValueError("No depth rows found around any candidate: check that the MD "
                         "file covers these dates and that ric values match.")
    cols = infer_depth_columns(md_depth)
    q = qualify_shocks(cand, md_depth, cols, min_l1_fraction, pre_seconds, search_seconds,
                       capacity_tolerance)
    shocks = pick_shocks(q)

    def _extract(frame):
        parts = []
        for i, s in frame.iterrows():
            r = extract_depth_recovery(trades, md_depth, s, cols, horizons, pre_seconds,
                                       search_seconds)
            if not r.empty:
                parts.append(r.assign(shock_id=i))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return {
        "candidates": q,
        "rejections": rejection_summary(q),
        "shocks": shocks,
        "recovery": _extract(shocks),
        "recovery_all": _extract(q[q["qualifies"]]),
        "md_depth": md_depth,
        "depth_columns": cols,
        "windows": dict(time_windows),
    }


# Backward-compatible names -------------------------------------------------

def select_shocks(trades, rics=None, time_windows=None, volume_quantile=0.95,
                  one_per_window=True, max_volume=None):
    """Old interface: largest trade per window WITHOUT the book checks. Kept
    for compatibility; use run_resilience instead."""
    c = select_shock_candidates(trades, rics, time_windows, volume_quantile, top_n=10**9)
    if max_volume is not None:
        c = c[c["shock_volume"] <= max_volume]
    if one_per_window and not c.empty:
        c = c.loc[c.groupby(["ric", "window"])["shock_volume"].idxmax()]
    return c.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting and summary
# ---------------------------------------------------------------------------

def _label(h):
    return "Before" if h < 0 else ("0s" if h == 0 else f"+{h:g}s")


def plot_depth_recovery(recovery, ric, window, ax=None, use_recovery_pct=True,
                        message=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    sub = recovery[(recovery["ric"] == ric) & (recovery["window"] == window)] \
        if not recovery.empty else recovery
    ax.set_title(f"{ric}, {window.replace('_', '-')}", fontsize=9, loc="left")
    if sub.empty:
        ax.text(0.5, 0.5, message or "No qualifying shock", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.35", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    levels = ["L1", "L2", "L3"]
    d = sub.set_index("horizon_seconds").sort_index()
    x = np.arange(len(d))
    suffix = "_share_pct" if use_recovery_pct else "_depth"
    bottom = np.zeros(len(d))
    for lv in levels:
        vals = np.nan_to_num(d[f"{lv}{suffix}"].to_numpy())
        ax.bar(x, vals, bottom=bottom, width=0.6, label=lv)
        bottom += vals
    ax.set_xticks(x, [_label(h) for h in d.index], fontsize=8)
    if use_recovery_pct:
        ax.axhline(100, linestyle="--", linewidth=0.8, color="0.4")
        ax.set_ylabel("% of pre-trade L1–L3 depth", fontsize=8)
    else:
        ax.set_ylabel("Shares", fontsize=8)
    r0 = sub.iloc[0]
    side = "sell into bid" if r0["trade_sign"] < 0 else "buy from ask"
    ax.set_title(f"{r0['shock_local']:%d %b %H:%M:%S}, {side}, {r0['shock_volume']:,.0f} sh",
                 loc="right", fontsize=7, color="0.3")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_all_cases(recovery, rics, windows=("midday", "pre_close"), use_recovery_pct=True,
                   rejections=None, figsize=None):
    """Grid: rows = rics, columns = windows. Uses constrained layout, which
    re-lays out at save time, so Quarto/knitr resizing the figure cannot
    make labels overlap or get clipped. Returns the Figure."""
    rics, windows = list(rics), list(windows)
    fig, axs = plt.subplots(len(rics), len(windows), squeeze=False, layout="constrained",
                            figsize=figsize or (4.8 * len(windows), 3.0 * len(rics)))
    for i, ric in enumerate(rics):
        for j, win in enumerate(windows):
            msg = None
            if rejections is not None and not rejections.empty:
                r = rejections[(rejections["ric"] == ric) & (rejections["window"] == win)]
                if not r.empty:
                    counts = r.drop(columns=["ric", "window"]).iloc[0]
                    counts = counts[counts > 0]
                    msg = "No qualifying shock\n" + "\n".join(
                        f"{n} × {k}" for k, n in counts.items() if k != "ok")
            plot_depth_recovery(recovery, ric, win, axs[i, j], use_recovery_pct, msg)
    h, l = next((a.get_legend_handles_labels() for a in axs.flat
                 if a.get_legend_handles_labels()[0]), ([], []))
    if h:
        fig.legend(h, l, loc="outside lower center", ncol=len(l), fontsize=8, frameon=False)
    return fig


def recovery_summary(recovery: pd.DataFrame, horizons=(0, 1, 10, 30, 60)) -> pd.DataFrame:
    """Per shock: L1-L3 depth on the consumed side as % of its pre-trade
    level at selected horizons, the trough over horizons >= 0, and T90 = the
    first sampled horizon AFTER the trough at which depth is back to >= 90%
    (NaN = not within the last horizon)."""
    if recovery.empty:
        return pd.DataFrame()
    rows = []
    for (ric, win), sub in recovery.groupby(["ric", "window"], sort=False):
        r0 = sub.iloc[0]
        row = {"ric": ric, "window": win, "volume": r0["shock_volume"],
               "side": r0["side_examined"],
               "pre L1–L3 (sh)": float(sub.loc[sub["horizon_seconds"] < 0, "L1_L3_depth"].iloc[0])
               if (sub["horizon_seconds"] < 0).any() else np.nan}
        for h in horizons:
            hit = sub[sub["horizon_seconds"] == h]
            row[f"{h}s (%)"] = float(hit.iloc[0]["L1_L3_recovery_pct"]) if len(hit) else np.nan
        post = sub[sub["horizon_seconds"] >= 0].sort_values("horizon_seconds")
        if len(post):
            k = post["L1_L3_recovery_pct"].to_numpy().argmin()
            row["trough (%)"] = float(post["L1_L3_recovery_pct"].iloc[k])
            after = post.iloc[k:]
            ok = after[after["L1_L3_recovery_pct"] >= 90]
            row["T90 (s)"] = (float(ok.iloc[0]["horizon_seconds"]) if len(ok) and row["trough (%)"] < 90
                              else (0.0 if row["trough (%)"] >= 90 else np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def average_recovery(recovery_all: pd.DataFrame) -> pd.DataFrame:
    """Across all qualifying shocks: median and inter-quartile range of
    L1-L3 depth (% of each shock's own pre-trade level) per horizon, by ric
    and window, plus the number of shocks."""
    if recovery_all.empty:
        return pd.DataFrame()
    r = recovery_all[recovery_all["horizon_seconds"] >= 0]
    g = r.groupby(["ric", "window", "horizon_seconds"])["L1_L3_recovery_pct"]
    out = g.agg(median="median", q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75), n="count").reset_index()
    return out


def share_recovered(recovery_all: pd.DataFrame, level=90.0, horizons=(1, 10, 60)) -> pd.DataFrame:
    """Share of qualifying shocks whose L1-L3 depth is back to >= `level`%
    of its pre-trade value at each horizon."""
    r = recovery_all[recovery_all["horizon_seconds"].isin(horizons)]
    t = (r.assign(ok=r["L1_L3_recovery_pct"] >= level)
         .groupby(["ric", "window", "horizon_seconds"])["ok"].mean().unstack("horizon_seconds"))
    t.columns = [f"≥{level:g}% at {h:g}s" for h in t.columns]
    n = recovery_all.groupby(["ric", "window"])["shock_id"].nunique().rename("shocks")
    return pd.concat([n, (100 * t).round(0)], axis=1).reset_index()


def plot_average_recovery(recovery_all, rics, windows=("midday", "pre_close"), figsize=None):
    """One panel per ric: median L1-L3 recovery path (% of pre-trade depth)
    with inter-quartile band, one line per window. Constrained layout."""
    avg = average_recovery(recovery_all)
    rics = list(rics)
    fig, axs = plt.subplots(1, len(rics), squeeze=False, sharey=True, layout="constrained",
                            figsize=figsize or (4.8 * len(rics), 3.4))
    for ax, ric in zip(axs[0], rics):
        for win in windows:
            a = avg[(avg["ric"] == ric) & (avg["window"] == win)].sort_values("horizon_seconds")
            if a.empty:
                continue
            x = np.arange(len(a))
            line, = ax.plot(x, a["median"], marker="o", markersize=3,
                            label=f"{win.replace('_', '-')} (n = {int(a['n'].max())})")
            ax.fill_between(x, a["q25"], a["q75"], color=line.get_color(), alpha=0.15, linewidth=0)
            ax.set_xticks(x, [_label(h) for h in a["horizon_seconds"]], fontsize=8)
        ax.axhline(100, linestyle="--", linewidth=0.8, color="0.4")
        ax.set_title(ric, fontsize=9, loc="left")
        ax.set_xlabel("Time after trade", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    axs[0, 0].set_ylabel("L1–L3 depth, % of pre-trade\n(median, IQR shaded)", fontsize=8)
    return fig


def signed_recovery_summary(recovery_all, horizons=(1, 10, 60), by_window=True):
    """Depth after the shock as % of pre-trade, split by trade direction,
    for the side the trade hit (ask after buys, bid after sells) and the
    opposite side. Reports median and mean (the ratios are right-skewed, so
    the mean is pulled up by a few shocks after which a large order arrived)."""
    r = recovery_all[recovery_all["horizon_seconds"].isin(horizons)]
    keys = ["ric", "window", "direction"] if by_window else ["ric", "direction"]
    rows = []
    for k, g in r.groupby(keys + ["horizon_seconds"]):
        k = dict(zip(keys + ["horizon_seconds"], k))
        rows.append({**k, "n": g["shock_id"].nunique(),
                     "hit median": g["L1_L3_recovery_pct"].median(),
                     "hit mean": g["L1_L3_recovery_pct"].mean(),
                     "opposite median": g["opp_L1_L3_recovery_pct"].median(),
                     "opposite mean": g["opp_L1_L3_recovery_pct"].mean()})
    return pd.DataFrame(rows)


def plot_signed_recovery(recovery_all, rics, figsize=None):
    """Rows = rics, columns = buyer- / seller-initiated shocks (windows
    pooled). In each panel: median L1-L3 depth on the side the trade hit
    (solid, IQR band) and on the opposite side (dashed), as % of pre-trade."""
    rics = list(rics)
    r = recovery_all[recovery_all["horizon_seconds"] >= 0]
    dirs = [("buy", "Buyer-initiated: ask side hit"), ("sell", "Seller-initiated: bid side hit")]
    fig, axs = plt.subplots(len(rics), 2, squeeze=False, sharey=True, layout="constrained",
                            figsize=figsize or (9, 3.0 * len(rics)))
    for i, ric in enumerate(rics):
        for j, (d, title) in enumerate(dirs):
            ax = axs[i, j]
            g = r[(r["ric"] == ric) & (r["direction"] == d)]
            if g.empty:
                ax.text(0.5, 0.5, "No shocks", ha="center", va="center", transform=ax.transAxes)
                continue
            hs = sorted(g["horizon_seconds"].unique())
            x = np.arange(len(hs))
            hit = g.groupby("horizon_seconds")["L1_L3_recovery_pct"]
            opp = g.groupby("horizon_seconds")["opp_L1_L3_recovery_pct"]
            n = g["shock_id"].nunique()
            ax.fill_between(x, hit.quantile(0.25).reindex(hs), hit.quantile(0.75).reindex(hs),
                            color="tab:red", alpha=0.12, linewidth=0)
            ax.plot(x, hit.median().reindex(hs), color="tab:red", marker="o", markersize=3,
                    label="side hit (median, IQR)")
            ax.plot(x, opp.median().reindex(hs), color="tab:blue", marker="s", markersize=3,
                    linestyle="--", label="opposite side (median)")
            ax.axhline(100, linestyle=":", linewidth=0.8, color="0.4")
            ax.set_xticks(x, [_label(h) for h in hs], fontsize=8)
            ax.set_title(f"{ric} – {title} (n = {n})", fontsize=9, loc="left")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel("L1–L3 depth, % of pre-trade", fontsize=8)
            if i == len(rics) - 1:
                ax.set_xlabel("Time after trade", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    return fig


def _order_windows(df, col="window"):
    order = {w: i for i, w in enumerate(DEFAULT_WINDOWS)}
    return df.assign(_w=df[col].map(order).fillna(99)).sort_values(
        [c for c in ["ric", "_w"] if c in df.columns or c == "_w"], kind="mergesort").drop(columns="_w")


def refill_table(recovery_all, horizons=(5, 10, 30, 60), min_depletion=10.0):
    """Per shock: depletion (100 - trough of L1-L3 depth over 0-1 s, % of
    pre-trade) and the share of that depletion refilled by each horizon,
    100 * (depth(h) - trough) / (100 - trough). Shocks that removed less
    than `min_depletion`% are excluded (nothing to refill)."""
    p = recovery_all.pivot_table(index=["ric", "window", "direction", "shock_id"],
                                 columns="horizon_seconds", values="L1_L3_recovery_pct")
    trough = p[[c for c in (0.0, 1.0) if c in p.columns]].min(axis=1)
    gap = 100 - trough
    keep = gap >= min_depletion
    out = pd.DataFrame({f"refill_{h:g}s": 100 * (p[float(h)] - trough) / gap
                        for h in horizons if float(h) in p.columns})
    out["depletion"] = gap
    return out[keep].reset_index()


def window_recovery_summary(recovery_all, horizons=(1, 10, 60)):
    """By security and window (buys and sells pooled, each measured on the
    side it hit). Columns: shocks; median L1-L3 depth on the side hit at
    each horizon (% of pre-trade); median opposite-side depth at the last
    horizon; median depletion; median share of the depletion refilled by the
    last horizon; share of shocks with at least half the depletion refilled
    by then."""
    if recovery_all.empty:
        return pd.DataFrame()
    H = max(horizons)
    r = recovery_all[recovery_all["horizon_seconds"].isin(horizons)]
    ref = refill_table(recovery_all, horizons=(H,))
    rows = []
    for (ric, win), g in r.groupby(["ric", "window"]):
        row = {"ric": ric, "window": win, "shocks": g["shock_id"].nunique()}
        for h in horizons:
            row[f"depth {h:g}s"] = g.loc[g["horizon_seconds"] == h, "L1_L3_recovery_pct"].median()
        row[f"opposite {H:g}s"] = g.loc[g["horizon_seconds"] == H, "opp_L1_L3_recovery_pct"].median()
        f = ref[(ref["ric"] == ric) & (ref["window"] == win)]
        row["depletion"] = f["depletion"].median()
        row[f"refilled {H:g}s"] = f[f"refill_{H:g}s"].median()
        row[f"half refilled {H:g}s (% shocks)"] = 100 * (f[f"refill_{H:g}s"] >= 50).mean()
        rows.append(row)
    return _order_windows(pd.DataFrame(rows))


def plot_recovery_by_window(recovery_all, rics, windows=None, figsize=None):
    """Rows = securities, columns = buyer- / seller-initiated shocks. One
    line per window: median L1-L3 depth on the side the trade hit, as % of
    its pre-trade level. Constrained layout."""
    windows = list(windows or DEFAULT_WINDOWS)
    rics = list(rics)
    r = recovery_all[recovery_all["horizon_seconds"] >= 0]
    dirs = [("buy", "buyer-initiated (ask side hit)"), ("sell", "seller-initiated (bid side hit)")]
    colors = dict(zip(windows, ["tab:orange", "tab:green", "tab:purple", "tab:gray"]))
    fig, axs = plt.subplots(len(rics), 2, squeeze=False, sharey=True, layout="constrained",
                            figsize=figsize or (9, 3.0 * len(rics)))
    for i, ric in enumerate(rics):
        for j, (d, title) in enumerate(dirs):
            ax = axs[i, j]
            for win in windows:
                g = r[(r["ric"] == ric) & (r["direction"] == d) & (r["window"] == win)]
                if g.empty:
                    continue
                med = g.groupby("horizon_seconds")["L1_L3_recovery_pct"].median().sort_index()
                x = np.arange(len(med))
                ax.plot(x, med.to_numpy(), marker="o", markersize=3, color=colors.get(win),
                        label=f"{win} (n = {g['shock_id'].nunique()})")
                ax.set_xticks(x, [_label(h) for h in med.index], fontsize=8)
            ax.axhline(100, linestyle=":", linewidth=0.8, color="0.4")
            ax.set_title(f"{ric}, {title}", fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="lower left")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel("Side hit: L1–L3 depth,\n% of pre-trade (median)", fontsize=8)
            if i == len(rics) - 1:
                ax.set_xlabel("Time after trade", fontsize=8)
    return fig


__all__ = [
    "build_market_orders", "refill_table", "window_recovery_summary", "plot_recovery_by_window",
    "signed_recovery_summary", "plot_signed_recovery",
    "average_recovery", "share_recovered", "plot_average_recovery",
    "run_resilience", "select_shock_candidates", "load_md_depth_for_shocks",
    "infer_depth_columns", "qualify_shocks", "rejection_summary", "pick_shocks",
    "extract_depth_recovery", "plot_depth_recovery", "plot_all_cases",
    "recovery_summary", "select_shocks",
]
