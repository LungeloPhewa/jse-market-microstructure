"""
a3_data.py -- load the Step-0-cleaned data for Assignment 3: continuous-phase
order-book snapshots (L1-L10) and continuous on-book trades, Lee-Ready signed.

The ten-level book is large (7.8 million updates), so it is read one
security-day at a time.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import market_phases as mp

TZ_LOCAL = "Africa/Johannesburg"
LEVEL_COLS = [f"L{k}-{s}" for k in range(1, 11) for s in ("BidPrice", "BidSize", "AskPrice", "AskSize")]


def _read_md(md_path, columns, flt):
    tab = ds.dataset(str(md_path)).to_table(columns=["ric", "ts_utc"] + columns, filter=flt)
    md = tab.to_pandas().rename(columns={"ts_utc": "ts"})
    md["ts"] = pd.to_datetime(md["ts"], utc=True)
    for c in columns:
        md[c] = md[c].astype("int32")
    md["row"] = np.arange(len(md))
    return md


def load_l1(md_path, rics, markers):
    """Best bid and ask for every update, with its trading phase. Returns the
    continuous-phase quotes and a count of updates by phase."""
    md = _read_md(md_path, ["L1-BidPrice", "L1-AskPrice"], ds.field("ric").isin(rics))
    md = mp.classify_quotes(md, markers).sort_values(["ric", "ts", "row"], kind="mergesort")
    counts = md.groupby(["ric", "phase"]).size().rename("book updates").reset_index()
    q = md[md["phase"] == "continuous"].drop(columns=["phase", "row"]).reset_index(drop=True)
    q["date"] = q["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    return q, counts


def load_book_day(md_path, ric, day, markers):
    """Continuous-phase ten-level snapshots for one security and one day."""
    start = pd.Timestamp(day, tz=TZ_LOCAL).tz_convert("UTC")
    flt = ((ds.field("ric") == ric) & (ds.field("ts_utc") >= start.to_pydatetime())
           & (ds.field("ts_utc") < (start + pd.Timedelta("1D")).to_pydatetime()))
    md = _read_md(md_path, LEVEL_COLS, flt)
    md = mp.classify_quotes(md, markers[markers["ric"] == ric]).sort_values(["ts", "row"], kind="mergesort")
    return md[md["phase"] == "continuous"].drop(columns=["phase", "row"]).reset_index(drop=True)


def sign_trades(trades, l1):
    """Lee & Ready (1991): quote rule against the mid of the last valid L1
    quote strictly before the trade, tick test for trades at the mid."""
    q = l1[(l1["L1-BidPrice"] > 0) & (l1["L1-AskPrice"] > l1["L1-BidPrice"])]
    q = q.assign(mid=(q["L1-BidPrice"] + q["L1-AskPrice"]) / 2)[["ric", "ts", "mid", "date"]]
    out = trades.sort_values("ts", kind="mergesort").reset_index(drop=True)
    snap = pd.merge_asof(out[["ts", "ric"]], q.sort_values("ts", kind="mergesort"), on="ts", by="ric",
                         direction="backward", allow_exact_matches=False)
    mid = np.where(snap["date"].to_numpy() == out["date"].to_numpy(), snap["mid"].to_numpy(), np.nan)
    quote = np.sign(out["price"].to_numpy() - mid)
    quote[quote == 0] = np.nan
    out["quote_sign"] = quote
    out = out.sort_values(["ric", "ts"], kind="mergesort")
    dp = out.groupby(["ric", "date"])["price"].diff()
    tick = np.sign(dp.replace(0, np.nan)).groupby([out["ric"], out["date"]]).ffill()
    out["trade_sign"] = out["quote_sign"].fillna(tick)
    return out.drop(columns="quote_sign").reset_index(drop=True)


def load_clean(tx_path, md_path, rics):
    markers = mp.load_phase_markers(tx_path, rics)
    trades_all = mp.classify_trades(tx_path, markers, rics)
    l1, book_phases = load_l1(md_path, rics, markers)
    trades = sign_trades(trades_all[trades_all["trade_class"] == "continuous"], l1)
    return {"markers": markers, "trades_all": trades_all, "trades": trades,
            "l1": l1, "book_phases": book_phases}
