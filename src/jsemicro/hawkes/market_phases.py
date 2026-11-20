"""
market_phases.py -- Step 0: separate auction events, uncrossing prices and
off-book reports from continuous-trading events.

The Refinitiv (TRTH) transactions export carries the exchange's own
market-state markers in the PRC_QL2 / PRC_QL3 qualifier fields. Each marker
is a separate UPDATE message (no trade price) sent when a security changes
trading phase. Their numeric codes and meanings in this export:

    PRC_QL2  PRC_QL3   SAST           phase that starts
    102 REP   -        08:00          pre-open
    238 AUC   -        08:30          opening auction (call period)
    105 TRD   -        ~09:00:0x      continuous trading
    238 AUC   138 STL  12:00 (19 Sep) intraday auction (index-futures close-out)
    105 TRD   -        ~12:15         continuous trading resumes
    238 AUC   291 CLA  16:50          closing auction (call period)
    94  CLS   131 OSP  ~17:00:0x      post-close (off-book reporting)
    94  CLS   320 ETR  18:15          end of trade reporting
    94  CLS   -        18:30          closed

The same qualifiers also appear on VERIFY_SYNC messages, which are periodic
re-synchronisation snapshots, not phase changes; only UPDATE messages
without a trade are used as markers.

Each auction ends with an UNCROSS: a single price at which the accumulated
auction orders are matched, printed as the first trade(s) at the moment the
next phase starts. Uncross prints are auction volume, not continuous-trading
transactions.

OFF-BOOK trades (negotiated and reported to the exchange) leave the
cumulative on-book volume ACVOL_1 empty and increase OFFBK_VOL.

Functions
---------
load_phase_markers(tx_path)            -> marker timeline per security
classify_trades(tx_path, markers)      -> every trade with its phase and class
classify_quotes(md, markers)           -> phase of every order-book update
phase_summary(trades, quotes)          -> counts and volumes by class
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ_LOCAL = "Africa/Johannesburg"

QL2 = {102: "REP", 238: "AUC", 105: "TRD", 94: "CLS"}
QL3 = {291: "CLA", 138: "STL", 131: "OSP", 320: "ETR"}

AUCTIONS = ("opening auction", "intraday auction", "closing auction")


def _phase_name(q2, q3):
    if q2 == "REP":
        return "pre-open"
    if q2 == "AUC":
        return {"CLA": "closing auction", "STL": "intraday auction"}.get(q3, "opening auction")
    if q2 == "TRD":
        return "continuous"
    if q2 == "CLS":
        return "post-close" if q3 == "OSP" else "closed"
    return None


def _read_tx(tx_path, rics, columns):
    cols = ["ric", "ts_utc", "message_type"] + columns
    out = []
    for ch in pd.read_csv(tx_path, usecols=cols, chunksize=500_000):
        out.append(ch[ch["ric"].isin(rics)])
    df = pd.concat(out, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    return df.drop(columns="ts_utc")


def load_phase_markers(tx_path, rics):
    """Trading-phase timeline per security: one row per phase change."""
    df = _read_tx(tx_path, rics, ["PRC_QL2", "PRC_QL3", "TRDPRC_1"])
    m = df[(df["message_type"] == "UPDATE_PC") & df["PRC_QL2"].notna() & df["TRDPRC_1"].isna()].copy()
    m["q2"] = m["PRC_QL2"].astype(int).map(QL2)
    m["q3"] = m["PRC_QL3"].fillna(0).astype(int).map(QL3)
    m["phase"] = [_phase_name(a, b) for a, b in zip(m["q2"], m["q3"])]
    m = m.dropna(subset=["phase"]).sort_values(["ric", "ts"], kind="mergesort")
    # keep only genuine changes of phase
    m = m[m["phase"] != m.groupby("ric")["phase"].shift()]
    m["previous"] = m.groupby("ric")["phase"].shift()
    m["date"] = m["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    return m[["ric", "ts", "date", "phase", "previous"]].reset_index(drop=True)


def _attach_phase(df, markers):
    out = df.sort_values("ts", kind="mergesort")
    out = pd.merge_asof(out, markers[["ric", "ts", "phase"]].sort_values("ts"), on="ts", by="ric",
                        direction="backward")
    # a marker from a previous day never applies
    mk_day = pd.merge_asof(out[["ts", "ric"]], markers[["ric", "ts", "date"]].sort_values("ts"),
                           on="ts", by="ric", direction="backward")["date"]
    day = out["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    out["phase"] = np.where(mk_day.to_numpy() == day.to_numpy(), out["phase"], "no marker")
    return out


def classify_trades(tx_path, markers, rics, uncross_window="2s"):
    """Every trade print with its trading phase and class.

    trade_class is one of
      continuous      on-book trade during continuous trading
      off-book        negotiated trade reported to the exchange
      uncross: <auction>  the auction's uncrossing print(s)
      outside         any other phase (pre-open, closed, ...)
    """
    df = _read_tx(tx_path, rics, ["TRDPRC_1", "TRDVOL_1", "ACVOL_1", "OFFBK_VOL"])
    t = df[(df["message_type"] == "UPDATE_PC") & df["TRDPRC_1"].notna() & (df["TRDVOL_1"] > 0)].copy()
    t = t.rename(columns={"TRDPRC_1": "price", "TRDVOL_1": "volume"})
    t["offbook"] = t["ACVOL_1"].isna() & t["OFFBK_VOL"].notna()
    t = _attach_phase(t, markers)

    # Uncross: the first on-book print(s) at the end of each auction, i.e. at
    # the first timestamp within `uncross_window` of the marker that ends it.
    t["uncross"] = ""
    w = pd.Timedelta(uncross_window)
    ends = markers[markers["previous"].isin(AUCTIONS)]
    for _, e in ends.iterrows():
        near = t[(t["ric"] == e["ric"]) & ~t["offbook"] & (t["ts"] >= e["ts"] - w) & (t["ts"] <= e["ts"] + w)]
        if len(near):
            t.loc[near.index[near["ts"] == near["ts"].min()], "uncross"] = e["previous"]

    cls = np.where(t["offbook"], "off-book",
          np.where(t["uncross"] != "", "uncross: " + t["uncross"].str.replace(" auction", ""),
          np.where(t["phase"] == "continuous", "continuous", "outside")))
    t["trade_class"] = cls
    t["date"] = t["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    return t[["ric", "ts", "date", "price", "volume", "phase", "trade_class"]].reset_index(drop=True)


def classify_quotes(md, markers):
    """Phase of every order-book update (md needs columns ric, ts)."""
    return _attach_phase(md, markers)


def phase_summary(trades, quotes=None):
    """Trades, volume and (optionally) order-book updates by class."""
    tr = (trades.groupby(["ric", "trade_class"])
          .agg(trades=("volume", "size"), volume=("volume", "sum")).reset_index())
    tr["volume share"] = tr["volume"] / tr.groupby("ric")["volume"].transform("sum")
    if quotes is None:
        return tr
    q = quotes.groupby(["ric", "phase"]).size().rename("book updates").reset_index()
    return tr, q
