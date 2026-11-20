"""
data_io.py -- Shared data-loading utilities for Part II.

The merged TX+MD file of Part I loses
late-session events because of the different timestamp/SEQNUM handling
between feeds, so here the two feeds are loaded and kept SEPARATE throughout:

    load_tx()  -> trade-level data, from the trimmed Transactions feed
    load_md()  -> quote-level data, from the flattened Market Depth feed

Downstream modules (bar_construction, stylised_facts, price_impact,
intraday_seasonality, cost_of_trading) each take `trades` and/or `md`
DataFrames produced here as arguments. Where a computation genuinely needs
both (e.g. the mid-price just before/after a trade), it uses a `merge_asof`
on just the two columns needed -- never a row-level merge of the full
feeds -- so the 16:00-17:00 SAST data loss documented in Assignment 1 is
not reintroduced.
"""

import numpy as np
import pandas as pd

# TIMESTAMPS: ts_utc in both clean files is the true UTC instant ONLY if they
# were produced by the corrected flatten_taq.py / market_depth_prepare.py
# (Refinitiv's Date[G]/Time[G] are GMT). Files built by the earlier versions
# are 2 hours early; rebuild them rather than shifting here.

# Column names expected on the external trade_sign source (e.g. your
# Assignment-1 Stage-4 merged file). Adjust if these don't match -- use
# _peek_parquet_columns(path) below to check cheaply without loading data.
SIGN_COLS = dict(ric="ric", ts="ts_utc", sign="trade_sign")

SESSION_START = "09:10:00"   # exclude pre-open auction noise -- JSE LOCAL (SAST) time
SESSION_END = "16:50:00"     # exclude closing auction noise  -- JSE LOCAL (SAST) time
TZ_LOCAL = "Africa/Johannesburg"  # SAST = UTC+2, no daylight saving

# Column names in the trimmed Transactions feed (AGL_GRT_TX_trimmed.csv),
# per the Assignment-1 provenance report schema.
TX_COLS = dict(
    ric="ric", ts="ts_utc", trdprc="TRDPRC_1", trdvol="TRDVOL_1",
)

# Column names in the flattened Market Depth feed (AGL_GRT_MD_flat.parquet),
# top-of-book (L1) only. Adjust if your flatten step named these columns
# differently.
MD_COLS = dict(
    ric="ric", ts="ts_utc",
    bid="L1-BidPrice", bidsize="L1-BidSize",
    ask="L1-AskPrice", asksize="L1-AskSize",
)


def _restrict_session(df, time_col="time"):
    """Continuous-session hours (SAST) on weekdays only. Weekend rows (e.g.
    a reported trade stamped Saturday 28 Sep 2013) are not trading-day
    data."""
    start = pd.to_datetime(SESSION_START).time()
    end = pd.to_datetime(SESSION_END).time()
    weekday = pd.to_datetime(pd.Series(df["date"], index=df.index)).dt.dayofweek < 5
    return df[(df[time_col] >= start) & (df[time_col] <= end) & weekday]


def _attach_local_date_time(df, ts_col="ts"):
    """
    SESSION_START/END are JSE LOCAL (SAST) trading hours, but `ts` is UTC
    (the feed's `ts_utc` column). Comparing a UTC wall-clock time directly
    against SAST-intended session boundaries silently filters out most of
    the real trading session (SAST = UTC+2), so `date`/`time` used for
    session-hours filtering and day-grouping are derived from the
    SAST-converted timestamp -- `ts` itself is left in UTC throughout, so
    merge_asof joins between TX and MD stay on a consistent clock.
    """
    local = df[ts_col].dt.tz_convert(TZ_LOCAL)
    df["date"] = local.dt.date
    df["time"] = local.dt.time
    return df


def _peek_parquet_columns(path):
    """
    Return the column names of a parquet file WITHOUT loading any row data
    -- reads only the file's footer metadata, so this is safe to call even
    on a very large file that would otherwise cause an out-of-memory error
    if read in full (as stage4_final.parquet did above).
    """
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).schema_arrow.names


def _read_parquet_columns(path, columns, filters=None):
    """
    Read only the specified columns of a parquet file (optionally also
    passing row-group `filters`, e.g. [("ric", "in", rics)], which lets
    pyarrow skip whole row-groups whose statistics rule out any match,
    without first materialising them). Parquet is a columnar format, so
    restricting columns alone already only allocates memory for the
    requested columns -- dramatically cheaper than pd.read_parquet(path)
    followed by a column selection, which (as seen above) can OOM on a
    wide file even when only a couple of columns are actually needed.
    """
    try:
        return pd.read_parquet(str(path), columns=columns, filters=filters)
    except MemoryError as e:
        raise MemoryError(
            f"\n\nOut of memory reading {path} even restricted to columns "
            f"{columns}. This file may simply be too large for available "
            f"RAM on this machine. Options:\n"
            f"  1) Close other applications to free RAM and retry.\n"
            f"  2) Skip external_sign_path entirely (pass None) and let "
            f"load_tx() fall back to the built-in tick-test trade sign.\n"
            f"  3) Pre-extract just (ric, ts_utc, trade_sign) from the "
            f"Stage-4 file ONCE into a small separate parquet file (e.g. "
            f"via DuckDB or a one-off script with row-group chunked "
            f"reading), then point external_sign_path at that small file "
            f"instead of the full stage4_final.parquet."
        ) from e


def _require_columns(df, col_map, feed_name):
    """Check every raw column name referenced in col_map actually exists in
    df, and fail with a clear, actionable message (listing what IS present)
    rather than a bare KeyError three calls deep in pandas. col_map maps
    our internal name -> raw column name in the file."""
    missing = {k: v for k, v in col_map.items() if v not in df.columns}
    if missing:
        raise KeyError(
            f"\n\n[{feed_name}] Expected column(s) not found in the loaded file:\n"
            f"  {missing}\n"
            f"Columns actually present in this file are:\n"
            f"  {sorted(df.columns.tolist())}\n"
            f"Fix this by editing the {feed_name}_COLS dict near the top of "
            f"data_io.py so its values match the names above."
        )


def load_tx(path, rics, external_sign_path=None, md=None, sign_method=None,
            external_ts_shift_hours=0.0, external_tolerance="5ms"):
    """
    Load trade-level data from the trimmed Transactions feed.

    Parameters
    ----------
    path : str
        Path to AGL_GRT_TX_trimmed.csv
    rics : list[str]
        Securities to keep.
    external_sign_path : str, optional
        Path to a parquet/csv file carrying a previously-validated
        `trade_sign` column (e.g. your Assignment-1 Stage-4 Lee-Ready +
        Lim-Coggins classification), keyed by (ric, ts). If supplied, this
        sign is attached via merge_asof instead of the tick-test fallback
        below -- prefer this when available, since it is already validated.

    md : DataFrame from load_md(), optional
        Needed for sign_method="lee_ready".
    sign_method : {"lee_ready", "external", "tick"}, optional
        How trade_sign is set. Default: "external" if external_sign_path is
        given, else "lee_ready" if md is given, else "tick".
          lee_ready -- Lee & Ready (1991): quote rule against the mid of the
                       last L1 quote strictly before the trade, tick test for
                       trades at the mid.
          external  -- attach a previously computed sign from
                       external_sign_path, matched on (ric, ts) within
                       `external_tolerance`. Unmatched trades stay NaN (the
                       old version used an unbounded nearest match, which
                       would silently copy the sign of a trade hours away).
          tick      -- tick test only.
    external_ts_shift_hours : float
        Hours ADDED to the external file's timestamps before matching. The
        Assignment-1 files were built with the GMT bug fixed in
        flatten_taq.py / market_depth_prepare.py, so their ts_utc is 2 hours
        early: use 2.0 for those files.

    Returns
    -------
    pd.DataFrame with columns: ric, ts, date, price, volume, trade_sign
    (trades.attrs["sign_report"] describes how signs were assigned)
    """
    # usecols restricts which columns pandas even parses out of the CSV,
    # reducing peak memory for this read -- helpful since everything else
    # loaded earlier in the pipeline (this trades frame, in particular)
    # stays resident while load_md() runs next.
    needed_tx_cols = [TX_COLS["ric"], TX_COLS["ts"], TX_COLS["trdprc"], TX_COLS["trdvol"]]
    header_cols = pd.read_csv(str(path), nrows=0).columns.tolist()
    missing_tx = [c for c in needed_tx_cols if c not in header_cols]
    if missing_tx:
        raise KeyError(
            f"\n\n[TX] Expected column(s) {missing_tx} not found in {path}.\n"
            f"Columns actually present: {sorted(header_cols)}\n"
            f"Fix this by editing the TX_COLS dict near the top of data_io.py "
            f"so its values match the names above."
        )
    df = pd.read_csv(str(path), usecols=needed_tx_cols)
    _require_columns(df, TX_COLS, "TX")
    # format="ISO8601" handles mixed sub-second precision within the same
    # column (e.g. some rows "...05:08:33+00:00", others
    # "...05:08:33.123456+00:00") without pandas trying to force one
    # single strptime format across every row.
    df[TX_COLS["ts"]] = pd.to_datetime(df[TX_COLS["ts"]], format="ISO8601", utc=True)
    df = df[df[TX_COLS["ric"]].isin(rics)].copy()
    df = df.rename(columns={TX_COLS["ric"]: "ric", TX_COLS["ts"]: "ts"})

    trades = df[df[TX_COLS["trdprc"]].notna() & (df[TX_COLS["trdprc"]] > 0)].copy()
    trades["price"] = trades[TX_COLS["trdprc"]]
    trades["volume"] = trades[TX_COLS["trdvol"]]
    trades = trades.sort_values(["ric", "ts"])

    if sign_method is None:
        sign_method = ("external" if external_sign_path is not None
                       else "lee_ready" if md is not None else "tick")

    if sign_method == "external":
        if external_sign_path is None:
            raise ValueError("sign_method='external' needs external_sign_path")
        trades = _attach_external_sign(trades, external_sign_path,
                                       external_ts_shift_hours, external_tolerance)
    elif sign_method == "lee_ready":
        if md is None:
            raise ValueError("sign_method='lee_ready' needs md (from load_md)")
        trades = lee_ready_sign(trades, md)
    elif sign_method == "tick":
        trades["trade_sign"] = _tick_test(trades)
        trades.attrs["sign_report"] = {"method": "tick"}
    else:
        raise ValueError(f"Unknown sign_method {sign_method!r}")

    trades = _attach_local_date_time(trades)
    trades = _restrict_session(trades)
    return trades[["ric", "ts", "date", "price", "volume", "trade_sign"]].reset_index(drop=True)


def _tick_test(trades):
    """+1 uptick / -1 downtick vs the last different price of the same ric
    and SAST day; zero ticks inherit the previous non-zero tick."""
    day = trades["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    d = trades.groupby([trades["ric"], day])["price"].diff()
    tick = np.sign(d).replace(0, np.nan)
    return tick.groupby([trades["ric"], day]).ffill()


def lee_ready_sign(trades, md):
    """Lee & Ready (1991) trade classification.

    Quote rule: compare the trade price with the mid of the last valid L1
    quote STRICTLY before the trade (a quote stamped at the same instant is
    usually the trade's own footprint). Above the mid -> buyer-initiated
    (+1), below -> seller-initiated (-1). Trades exactly at the mid, or with
    no prior quote that day, fall back to the tick test.
    """
    out = trades.sort_values("ts", kind="mergesort").copy()
    q = md[["ric", "ts", "mid"]].sort_values("ts", kind="mergesort")
    snap = pd.merge_asof(out[["ts", "ric"]], q, on="ts", by="ric",
                         direction="backward", allow_exact_matches=False)
    # never use a quote from a previous day
    same_day = (snap["ts"].dt.tz_convert(TZ_LOCAL).dt.date.to_numpy()
                == out["ts"].dt.tz_convert(TZ_LOCAL).dt.date.to_numpy())
    mid = np.where(same_day, snap["mid"].to_numpy(), np.nan)
    quote = np.sign(out["price"].to_numpy() - mid)
    quote = np.where(quote == 0, np.nan, quote)
    tick = _tick_test(out).to_numpy()
    out["trade_sign"] = np.where(np.isnan(quote), tick, quote)
    out.attrs["sign_report"] = {
        "method": "lee_ready",
        "quote_rule_share": float(np.isfinite(quote).mean()),
        "tick_test_share": float((np.isnan(quote) & np.isfinite(tick)).mean()),
        "unsigned_share": float(np.isnan(out["trade_sign"]).mean()),
    }
    return out.sort_values(["ric", "ts"])


def _attach_external_sign(trades, external_sign_path, shift_hours=0.0, tolerance="5ms"):
    ext_path_str = str(external_sign_path)
    if ext_path_str.lower().endswith(".parquet"):
        needed = [SIGN_COLS["ric"], SIGN_COLS["ts"], SIGN_COLS["sign"]]
        available = _peek_parquet_columns(ext_path_str)
        missing = [c for c in needed if c not in available]
        if missing:
            raise KeyError(f"[external_sign_path] Expected column(s) {missing} not found. "
                           f"Columns present: {sorted(available)}")
        ext = _read_parquet_columns(ext_path_str, columns=needed)
    else:
        ext = pd.read_csv(ext_path_str, usecols=[SIGN_COLS["ric"], SIGN_COLS["ts"], SIGN_COLS["sign"]])
    ext = ext.rename(columns={SIGN_COLS["ric"]: "ric", SIGN_COLS["ts"]: "ts",
                              SIGN_COLS["sign"]: "trade_sign"})
    ext = ext.dropna(subset=["trade_sign"])
    ext["ts"] = pd.to_datetime(ext["ts"], utc=True, format="ISO8601") + pd.Timedelta(hours=shift_hours)
    ext["ts"] = ext["ts"].dt.as_unit("ns")
    ext = ext.drop_duplicates(["ric", "ts"]).sort_values("ts", kind="mergesort")

    t = trades.copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True).dt.as_unit("ns")
    t = pd.merge_asof(t.sort_values("ts", kind="mergesort"), ext[["ric", "ts", "trade_sign"]], on="ts", by="ric",
                      direction="nearest", tolerance=pd.Timedelta(tolerance))
    t.attrs["sign_report"] = {
        "method": "external", "shift_hours": shift_hours, "tolerance": tolerance,
        "matched_share": float(t["trade_sign"].notna().mean()),
    }
    return t.sort_values(["ric", "ts"])


def compare_signs(a, b):
    """Agreement between two signed trade frames (same trades, e.g. new
    Lee-Ready vs the Assignment-1 signs). Returns a small summary dict."""
    m = a[["ric", "ts", "trade_sign"]].merge(b[["ric", "ts", "trade_sign"]],
                                             on=["ric", "ts"], suffixes=("_a", "_b"))
    both = m.dropna()
    return {"n_common": len(m), "n_both_signed": len(both),
            "agreement": float((both["trade_sign_a"] == both["trade_sign_b"]).mean())
            if len(both) else float("nan")}


def load_md(path, rics):
    """
    Load quote-level data from the flattened Market Depth feed (top-of-book).

    Returns
    -------
    pd.DataFrame with columns:
        ric, ts, date, bid, ask, bidsize, asksize, mid, micro, spread, rel_spread

    micro-price = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    i.e. weighted toward the side with LESS resting size opposing it.
    """
    needed = [MD_COLS["ric"], MD_COLS["ts"], MD_COLS["bid"], MD_COLS["bidsize"],
              MD_COLS["ask"], MD_COLS["asksize"]]
    available = _peek_parquet_columns(path)
    missing = [c for c in needed if c not in available]
    if missing:
        raise KeyError(
            f"\n\n[MD] Expected column(s) {missing} not found in {path}.\n"
            f"Columns actually present ({len(available)} total): {sorted(available)}\n"
            f"Fix this by editing the MD_COLS dict near the top of data_io.py "
            f"so its values match the names above."
        )
    # Only pull the 6 columns actually needed (not e.g. a full L1-L10 depth
    # book), and let pyarrow skip row-groups that can't contain our rics
    # where its statistics allow it -- both reduce peak memory substantially
    # versus pd.read_parquet(path) followed by a column/row selection.
    df = _read_parquet_columns(path, columns=needed,
                                filters=[(MD_COLS["ric"], "in", rics)])
    df = df[df[MD_COLS["ric"]].isin(rics)].copy()
    df = df.rename(columns={
        MD_COLS["ric"]: "ric", MD_COLS["ts"]: "ts",
        MD_COLS["bid"]: "bid", MD_COLS["bidsize"]: "bidsize",
        MD_COLS["ask"]: "ask", MD_COLS["asksize"]: "asksize",
    })
    # MD timestamps are typically already a proper datetime dtype inside the
    # parquet file, but coerce defensively in case they were stored as
    # ISO8601 strings with mixed sub-second precision (see the TX loader
    # above for why format="ISO8601" is used rather than plain to_datetime).
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["micro"] = (df["bid"] * df["asksize"] + df["ask"] * df["bidsize"]) / (
        df["bidsize"] + df["asksize"]
    )
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid"]

    df = _attach_local_date_time(df)
    df = _restrict_session(df)
    return df[["ric", "ts", "date", "bid", "ask", "bidsize", "asksize",
               "mid", "micro", "spread", "rel_spread"]].reset_index(drop=True)
