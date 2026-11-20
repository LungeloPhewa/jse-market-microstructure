"""
Derived Microstructure Variables
===================================

Implements the formulas from the assignment's "useful derived definitions"
sheet, applied to a cleaned, time-ordered event table (i.e. the output of
taq_stage2_ordering_and_cleaning.py, after merging TX and Market Depth).

FORMULAS IMPLEMENTED
-----------------------
- Mid-price:            m_ij = (1/2)(b_ij + a_ij)
  NOTE: the assignment sheet's OCR'd formula shows a MINUS sign
  (m_ij = (1/2)(b_ij - a_ij)), which would make mid-price negative
  whenever ask > bid -- i.e. essentially always. This is treated here as
  a transcription artifact from PDF extraction, and the standard,
  universally-used PLUS-sign definition is implemented instead. The
  feed's own MID_PRICE column is kept alongside for cross-checking.

- Micro-price:          s_ij = (v^a/(v^b+v^a)) * b_ij + (v^b/(v^b+v^a)) * a_ij
  A size-weighted mid-price: weights lean the "wrong way" round
  deliberately -- a larger volume waiting on the BID side pulls the
  micro-price CLOSER TO THE ASK (and vice versa), reflecting that heavier
  resting size on one side signals more pressure likely to move price
  toward the other side.

- Mid-price change:     Delta m_ij = m_{i+1,j} - m_{i,j}
  Computed as the change in mid-price from just before a transaction to
  just after -- i.e. between the two nearest surrounding quote-update
  events, per the assignment's description.

- Inter-arrival time:   tau_ij = t_{i+1,j} - t_{i,j}
  Time between consecutive events, per instrument.

- Normalised trade volume: w_ij = v_ij / [ sum_k(v_kj) * (sum_j(T_j)/N) ]
  Trade volume normalised by (a) the total volume traded on that same day
  and (b) the average number of trading events per day across the full
  sample -- see normalised_trade_volume() for the exact two-level
  aggregation this requires.

- VWAP:                 vwap_{t+1,t} = sum(v_i * p_i) / sum(v_i)
  Volume-weighted average price over a window of events.

Trade-sign (epsilon_ij) is NOT computed here -- it is the OUTPUT of the
Lee-Ready / Lim-Coggins classification step, which is a separate,
substantial piece of logic still to be implemented once the TX and Market
Depth tables are merged.
"""

from __future__ import annotations

import pandas as pd


# ----------------------------------------------------------------------------
# Mid-price and micro-price
# ----------------------------------------------------------------------------

def add_mid_price(df: pd.DataFrame, bid_col: str = "BID", ask_col: str = "ASK") -> pd.DataFrame:
    """Add m_ij = (1/2)(bid + ask), and compare against the feed's own MID_PRICE.

    WHY cross-check: if MID_PRICE (feed-provided) and our own computed
    mid-price diverge meaningfully, that's worth investigating BEFORE
    building further derived variables on top of either one -- e.g. it
    could reveal the feed's MID_PRICE uses a different price convention,
    a different set of quotes, or a timing lag relative to BID/ASK.
    """
    df = df.copy()
    df["mid_price_computed"] = 0.5 * (df[bid_col] + df[ask_col])
    if "MID_PRICE" in df.columns:
        df["mid_price_diff_from_feed"] = df["mid_price_computed"] - df["MID_PRICE"]
    return df


def add_micro_price(
    df: pd.DataFrame,
    bid_col: str = "BID", ask_col: str = "ASK",
    bid_size_col: str = "BIDSIZE", ask_size_col: str = "ASKSIZE",
) -> pd.DataFrame:
    """Add the size-weighted micro-price s_ij (equation 1 in the brief).

    WHY the weights are "crossed" (ask size weights the BID term and vice
    versa): a large resting size on the bid side signals many buyers
    queued at that price, which (all else equal) suggests price is more
    likely to move up toward the ask -- so heavier bid-side volume should
    pull the micro-price TOWARD the ask, not away from it. This is
    exactly what the crossed weighting achieves.
    """
    df = df.copy()
    va = df[ask_size_col]
    vb = df[bid_size_col]
    denom = vb + va
    df["micro_price"] = (va / denom) * df[bid_col] + (vb / denom) * df[ask_col]
    return df


# ----------------------------------------------------------------------------
# Mid-price change
# ----------------------------------------------------------------------------

def add_mid_price_change(
    df: pd.DataFrame,
    mid_price_col: str = "mid_price_computed",
    group_col: str = "ric",
) -> pd.DataFrame:
    """Add Delta m_ij = m_{i+1,j} - m_{i,j}, computed per instrument.

    WHY grouped by RIC: mid-price changes must be computed WITHIN each
    security's own event sequence -- computing a raw .diff() across the
    whole (multi-instrument) table would incorrectly compute a "change"
    between the last event of one RIC and the first event of the next,
    since AGLJ.J and GRTJ.J rows are interleaved by timestamp once
    merged and sorted.

    WHY this assumes the input is ALREADY in Stage 2's deterministic
    event-time order: if the events aren't correctly ordered first, this
    "next event" difference is meaningless -- this function does not
    re-sort, by design, so it can be safely reused on data whose order
    has already been established and verified upstream.
    """
    df = df.copy()
    df["mid_price_change"] = df.groupby(group_col)[mid_price_col].diff().shift(-1)
    # .diff() gives m_i - m_{i-1} at row i; shifting by -1 re-aligns it so
    # that row i holds m_{i+1} - m_i, matching the brief's Delta m_ij
    # definition (the change AFTER event i, not the change ending at i).
    return df


# ----------------------------------------------------------------------------
# Inter-arrival times
# ----------------------------------------------------------------------------

def add_inter_arrival_time(
    df: pd.DataFrame,
    ts_col: str = "ts_utc",
    group_col: str = "ric",
) -> pd.DataFrame:
    """Add tau_ij = t_{i+1,j} - t_{i,j}, per instrument.

    WHY per-instrument grouping (same rationale as mid-price change):
    without grouping, the gap between the last AGLJ.J event and the next
    GRTJ.J event in a merged, time-sorted table would be computed as an
    "inter-arrival time," which conflates two unrelated securities'
    event streams.
    """
    df = df.copy()
    next_ts = df.groupby(group_col)[ts_col].shift(-1)
    df["inter_arrival_time"] = next_ts - df[ts_col]
    return df


# ----------------------------------------------------------------------------
# Normalised trade volume
# ----------------------------------------------------------------------------

def normalised_trade_volume(
    df: pd.DataFrame,
    volume_col: str = "TRDVOL_1",
    ric_col: str = "ric",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Add w_ij per equation (2): trade volume normalised by that day's
    total volume, further scaled by the ratio of (average events per day
    across all days) to (that day's own event count).

    WHY this needs a `trade_date` column: the formula operates over TWO
    levels of aggregation -- within a single day (sum_k v_kj, the day's
    total volume) and across all days (mean T_j, the average number of
    events per day) -- so this function expects the caller to first
    derive a `trade_date` column (e.g. df["ts_local"].dt.date) marking
    which calendar day each event belongs to, since that's not something
    this function should silently assume a column name for without the
    caller's explicit input.

    WHY this is a SEPARATE function (not folded into add_* above): unlike
    mid-price/micro-price/mid-price change, this genuinely needs the full
    dataset's day-level statistics before it can compute a SINGLE row's
    value -- it cannot be computed one event (or even one day) at a time
    in isolation, so it's written to operate on the whole DataFrame at
    once rather than being combinable into a simple per-row/per-group
    transform.
    """
    df = df.copy()

    # Total volume traded on each (RIC, day) -- the sum_{k=1}^{T_j} v_kj term.
    daily_total_volume = df.groupby([ric_col, date_col])[volume_col].transform("sum")

    # Number of trading events on each (RIC, day) -- T_j.
    events_per_day = df.groupby([ric_col, date_col])[volume_col].transform("count")

    # Average number of events per day, across all days for that RIC -- the
    # [ sum_j(T_j) / N ] term. Computed once per RIC (not per row) via a
    # separate groupby, then broadcast back onto every row of that RIC.
    daily_event_counts = df.groupby([ric_col, date_col])[volume_col].count()
    avg_events_per_day = daily_event_counts.groupby(level=ric_col).transform("mean")
    avg_events_per_day_per_row = df[ric_col].map(avg_events_per_day.groupby(level=0).first())

    df["normalised_trade_volume"] = (
        df[volume_col] / daily_total_volume
    ) * (avg_events_per_day_per_row / events_per_day)

    return df


# ----------------------------------------------------------------------------
# VWAP
# ----------------------------------------------------------------------------

def vwap(prices: pd.Series, volumes: pd.Series) -> float:
    """Volume-weighted average price over the given window (equation 3).

    WHY a plain function rather than a DataFrame-adding one: VWAP is
    naturally computed over a chosen WINDOW of events (e.g. "the last
    hour", "this trading day") rather than being a per-row value the way
    mid-price or inter-arrival time are -- the caller selects the window
    (e.g. `df.loc[mask]`) and passes its price/volume columns in here,
    rather than this function making an assumption about window
    boundaries on the caller's behalf.
    """
    return (prices * volumes).sum() / volumes.sum()


def rolling_vwap(df: pd.DataFrame, price_col: str, volume_col: str, window: str, group_col: str = "ric") -> pd.Series:
    """Rolling VWAP over a time-based window (e.g. window="1H"), per instrument.

    WHY time-based rather than row-count-based windows: trading activity
    is uneven through the day (recall the earlier PCA-style
    "visualise an hour of trading" requirement) -- a fixed ROW-COUNT
    window would span very different amounts of real time depending on
    how active trading was, whereas a time-based window (e.g. exactly one
    hour) is directly comparable across different periods of the day.
    Requires a DatetimeIndex, so the caller should set the timestamp
    column as the index before calling this.
    """
    def _vwap_window(g: pd.DataFrame) -> pd.Series:
        numer = (g[price_col] * g[volume_col]).rolling(window).sum()
        denom = g[volume_col].rolling(window).sum()
        return numer / denom

    return df.groupby(group_col, group_keys=False).apply(_vwap_window)
