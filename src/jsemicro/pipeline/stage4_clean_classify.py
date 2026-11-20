"""
stage4_clean_classify.py
--------------------------
Stage 4 of the TAQ data-work pipeline: clean, classify, and build the
final time-ordered dataset.

Pipeline:
  1. CLEAN: drop rows with no usable timestamp (NaT ts_utc) -- confirmed
     via inspect_nat_rows.py to be genuinely timestampless (not a UTC
     conversion artifact: ts_local is also NaT for all of them). ~0.11%
     of rows. NOTE: message_type == 'UPDATE_PC' also appears on ~99.9%
     of ALL price-bearing rows (quotes and trades alike), so it does NOT
     indicate a correction/non-market message as initially suspected --
     it's most likely just a generic "price field updated" tag. It is
     NOT used as an exclusion criterion here.

  2. CLASSIFY: identify each row as a quote or trade event, forward-fill
     the prevailing best bid/ask per RIC, and classify each trade's
     initiator (buyer- vs seller-initiated) using:
       - Lee & Ready (2001) quote rule: trade above the prevailing
         midpoint -> buy; below -> sell.
       - Lim & Coggins (2005)-style tick-test tie-break for trades AT
         the midpoint: compare to the most recent DIFFERING trade price
         (not just the immediately preceding trade, which fails during
         runs of identical consecutive prices -- exactly the discreteness
         issue Lim & Coggins highlight for markets like the ASX/JSE where
         price ties are common) -- uptick -> buy, downtick -> sell.
     Also flags (but does NOT remove) large-volume outlier trades --
     see add_volume_outlier_flag(). Prices on these look economically
     sane relative to the prevailing quote (not garbage ticks), so they
     are most likely genuine large block/"special bargain" trades rather
     than data errors, and are kept with a flag rather than dropped.

  3. BUILD final columns: date, time, mid-price, mid-price change,
     volume, normalised volume, bid, bid volume, ask, ask volume,
     event type, transaction price, transaction volume, plus
     is_crossed_quote and is_volume_outlier flags.

Usage from Command Prompt OR Quarto:

    import stage4_clean_classify as s4
    df_final, stats = s4.run()

Or from Command Prompt directly:

    python stage4_clean_classify.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
from ..config import PATHS

INPUT_PATH = PATHS.merged_sorted
OUTPUT_PATH = PATHS.stage4

SORT_KEY = ["ric", "ts_utc", "SEQNUM", "event_id"]

# Columns actually needed for Stage 4 -- reading only these keeps memory
# well under what the full 59-column file would need.
NEEDED_COLUMNS = [
    "ric", "ts_utc", "SEQNUM", "event_id", "source", "type", "message_type",
    "BID", "ASK", "BIDSIZE", "ASKSIZE", "TRDPRC_1", "TRDVOL_1",
]
# --------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_and_clean(input_path: Path = INPUT_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Load the merged file (needed columns only) and drop rows with no
    usable timestamp. Returns the cleaned dataframe plus a stats dict
    for your write-up / record keeping.
    """
    logger.info("Loading %s (columns: %s)...", input_path.name, NEEDED_COLUMNS)
    df = pd.read_parquet(input_path, columns=NEEDED_COLUMNS)
    n_total = len(df)
    logger.info("Loaded %s rows.", f"{n_total:,}")

    nat_mask = df["ts_utc"].isna()
    n_nat = int(nat_mask.sum())

    stats = {
        "n_total_rows": n_total,
        "n_dropped_nat_ts": n_nat,
        "pct_dropped_nat_ts": round(100 * n_nat / n_total, 4) if n_total else 0.0,
        "n_dropped_by_source": df.loc[nat_mask, "source"].value_counts(dropna=False).to_dict() if n_nat else {},
        "n_dropped_by_message_type": df.loc[nat_mask, "message_type"].value_counts(dropna=False).to_dict() if n_nat else {},
    }
    logger.info(
        "Dropping %s rows (%.4f%%) with no usable timestamp (NaT ts_utc): %s",
        f"{n_nat:,}", stats["pct_dropped_nat_ts"], stats["n_dropped_by_source"],
    )

    df = df.loc[~nat_mask].copy()
    n_after = len(df)
    stats["n_rows_after_cleaning"] = n_after

    # Defensive re-sort: should already be globally sorted from Stage 3,
    # but guarantees correctness regardless of what's fed into this step.
    df = df.sort_values(SORT_KEY, kind="mergesort", na_position="last").reset_index(drop=True)

    logger.info("Rows after cleaning: %s", f"{n_after:,}")
    return df, stats


def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    SEQNUM/event_id were only needed for the defensive re-sort in
    load_and_clean; source/type/message_type were only needed to
    characterize the NaT-drop stats (already captured). None of them
    are used by any later pipeline step, so dropping them here frees
    real memory (two string columns + two more) for the rest of the
    9M+-row pipeline, rather than carrying dead weight all the way to
    the end.
    """
    cols_to_drop = [c for c in ["SEQNUM", "event_id", "source", "type", "message_type"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped no-longer-needed columns: %s", cols_to_drop)
    return df


def sanitize_zero_placeholders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Bloomberg appears to report exactly 0.0 for BID/ASK/TRDPRC_1 as a
    placeholder before a side of the book (or the first trade) has
    genuinely populated -- most visible in the first few minutes of a
    session, before both sides of the top-of-book have a real quote.
    Since 0.0 is not NaN, downstream .notna() checks and ffill would
    otherwise treat these as real prices: a bid of 23500 paired with an
    ask placeholder of 0.0 produces a nonsense mid-price of 11750, which
    then corrupts everything forward-filled after it (and produces -inf
    from log(0) in the mid-price-change calculation).

    A price of exactly 0 is not economically meaningful for an equity, so
    this treats BID <= 0, ASK <= 0, and TRDPRC_1 <= 0 as missing (NaN),
    and nulls the paired size field too (a size without a real price is
    meaningless) so bid/bid-size and ask/ask-size stay consistent pairs.
    """
    counts = {}

    for price_col, size_col in [("BID", "BIDSIZE"), ("ASK", "ASKSIZE"), ("TRDPRC_1", "TRDVOL_1")]:
        bad = df[price_col] <= 0
        n_bad = int(bad.sum())
        counts[price_col] = n_bad
        if n_bad:
            df.loc[bad, price_col] = np.nan
            df.loc[bad, size_col] = np.nan

    logger.info("Zero/negative price placeholders sanitized to NaN: %s", counts)
    return df, counts


def add_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each row as a 'quote' or 'trade' event. Uses the presence of
    a transaction price as the primary signal (more robust than trusting
    the 'source'/'type' labels alone), falling back to source/type.
    """
    is_trade = df["TRDPRC_1"].notna()
    is_quote = df["BID"].notna() | df["ASK"].notna()

    event_type = np.where(is_trade, "trade", np.where(is_quote, "quote", "other"))
    df["event_type"] = event_type

    n_other = int((df["event_type"] == "other").sum())
    if n_other:
        logger.warning(
            "%s rows (%.4f%%) are neither a trade nor a quote (no TRDPRC_1, "
            "no BID/ASK) -- inspect these separately; they're kept in the "
            "output but excluded from trade classification and quote fill.",
            f"{n_other:,}", 100 * n_other / len(df),
        )
    return df


def add_prevailing_quotes_and_mid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill the prevailing best bid/ask (and sizes) per RIC, so every
    row -- including trade rows, which carry NaN bid/ask themselves --
    has the most recent known quote. Then compute the mid-price from that
    prevailing quote: m_t = 0.5 * (bid_t + ask_t).

    Crossed quotes (bid > ask) are set to NaN mid-price rather than the
    row being dropped -- Lee & Ready's quote rule is mathematically
    undefined for a crossed market (there's no valid midpoint to compare
    a trade against). The raw bid_prevailing/ask_prevailing values are
    kept as-is for transparency/diagnostics; only mid_price (and anything
    derived from it downstream: mid_price_change, trade classification)
    becomes NaN for these rows. This preserves the full event stream
    (needed for inter-arrival times etc.) while correctly disqualifying
    periods with no valid quote from mid-price-dependent calculations.

    Crossed quotes often cluster in pre-market/opening-auction periods,
    where exchanges commonly broadcast crossed or locked indicative
    prices as part of normal call-auction mechanics -- a different
    regime from continuous double-auction trading, not necessarily a
    data error. Check the time-of-day distribution of crossed rows to
    confirm this for your specific dataset.
    """
    grp = df.groupby("ric", sort=False)

    df["bid_prevailing"] = grp["BID"].ffill()
    df["ask_prevailing"] = grp["ASK"].ffill()
    df["bid_size_prevailing"] = grp["BIDSIZE"].ffill()
    df["ask_size_prevailing"] = grp["ASKSIZE"].ffill()

    is_crossed = df["bid_prevailing"] > df["ask_prevailing"]
    n_crossed = int(is_crossed.sum())
    if n_crossed:
        logger.warning(
            "%s rows (%.4f%%) have a CROSSED prevailing quote (bid > ask). "
            "mid_price is set to NaN for these rows (raw bid/ask values are "
            "kept for diagnostics). Check the time-of-day distribution -- "
            "this often concentrates in pre-market/auction periods rather "
            "than continuous trading.",
            f"{n_crossed:,}", 100 * n_crossed / len(df),
        )

    df["is_crossed_quote"] = is_crossed
    df["mid_price"] = np.where(is_crossed, np.nan, 0.5 * (df["bid_prevailing"] + df["ask_prevailing"]))

    n_no_quote_yet = df["mid_price"].isna().sum() - n_crossed
    if n_no_quote_yet > 0:
        logger.info(
            "%s rows (%.4f%%) have no mid-price because no quote has been "
            "seen yet for that RIC at that point (e.g. a trade before the "
            "first quote of the day) -- these will have NaN mid-price/change.",
            f"{n_no_quote_yet:,}", 100 * n_no_quote_yet / len(df),
        )
    return df


def add_mid_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    General, row-to-row mid-price change (log return), computed at every
    event within each RIC: Delta m_t = ln(m_t) - ln(m_{t-1}).

    Note: this differs from the assignment slide's transaction-specific
    definition (Delta m_ij = ln(mid AFTER trade i) - ln(mid BEFORE trade
    i), defined only at trade events). Both are useful for different
    purposes -- this one gives a value at every row for the general
    time-ordered dataset requested in step 1.4; see
    add_transaction_mid_price_change() below for the trade-specific
    before/after version used in price-impact style analysis.
    """
    df["mid_price_change"] = df.groupby("ric", sort=False)["mid_price"].transform(
        lambda s: np.log(s).diff()
    )
    return df


def add_transaction_mid_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    The assignment slide's specific Delta m_ij: for each TRADE event i,
    the log change between the mid-price just AFTER the trade and just
    BEFORE it. NaN for non-trade rows.

    'Just before' = the prevailing mid-price at the trade's own row
    (mid_price column, which correctly reflects the last quote seen
    strictly before the trade, since trade rows carry no bid/ask of
    their own). 'Just after' = the prevailing mid-price at the NEXT row
    for that RIC (whatever it settles to next, whether that's the same
    trade's row if a quote updates immediately, or a later quote event).
    """
    next_mid = df.groupby("ric", sort=False)["mid_price"].shift(-1)
    df["transaction_mid_price_change"] = np.where(
        df["event_type"] == "trade",
        np.log(next_mid) - np.log(df["mid_price"]),
        np.nan,
    )
    return df


def _last_differing_trade_price(prices: np.ndarray) -> np.ndarray:
    """
    For a 1D array of trade prices (already in chronological order, one
    RIC only), return, for each element, the most recent DIFFERING price
    that came before it (NaN if none exists, e.g. the RIC's very first
    trade). Runs of identical consecutive prices all map to the same
    prior differing price -- this is the Lim & Coggins (2005) refinement
    to the classic tick test, which naively compares only to the single
    immediately-preceding trade and breaks down during price-tie runs.
    """
    n = len(prices)
    if n == 0:
        return np.full(0, np.nan)

    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = prices[1:] != prices[:-1]
    run_id = np.cumsum(changed) - 1

    run_start_idx = np.flatnonzero(changed)
    run_prices = prices[run_start_idx]

    prev_run_price = np.full(run_prices.shape[0], np.nan)
    prev_run_price[1:] = run_prices[:-1]

    return prev_run_price[run_id]


def classify_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lee & Ready (2001) quote rule, with a Lim & Coggins (2005)-style tick
    test as the tie-break when a trade prints exactly at the prevailing
    midpoint:

      trade_price > mid_price  -> +1 (buyer-initiated)
      trade_price < mid_price  -> -1 (seller-initiated)
      trade_price == mid_price -> tick test against the most recent
                                   DIFFERING trade price (not just the
                                   immediately preceding trade -- see
                                   _last_differing_trade_price):
          uptick   (curr > prior differing price) -> +1
          downtick (curr < prior differing price) -> -1
          still no differing price found (e.g. RIC's very first trade)
                                          -> NaN (unclassifiable)

    Adds a 'trade_sign' column: +1, -1, or NaN. NaN for all non-trade
    rows and for the small number of unclassifiable edge-case trades.
    """

    trade_mask = (df["event_type"] == "trade") & df["mid_price"].notna()
    n_trades = int(trade_mask.sum())
    logger.info("Classifying %s trade events...", f"{n_trades:,}")

    price = df["TRDPRC_1"]
    mid = df["mid_price"]

    sign = pd.Series(np.full(len(df), np.nan), index=df.index)
    above = trade_mask & (price > mid)
    below = trade_mask & (price < mid)
    tied = trade_mask & (price == mid)

    sign.loc[above] = 1.0
    sign.loc[below] = -1.0

    n_tied = int(tied.sum())
    logger.info(
        "%s trades (%.2f%% of trades) print exactly at the midpoint and "
        "need the tick-test tie-break.", f"{n_tied:,}",
        100 * n_tied / n_trades if n_trades else 0.0,
    )

    if n_tied:
        for ric, idx in df.loc[trade_mask].groupby("ric", sort=False).groups.items():
            idx = pd.Index(idx)
            prices_this_ric = price.loc[idx].to_numpy()
            prev_diff = _last_differing_trade_price(prices_this_ric)

            tied_local = tied.loc[idx].to_numpy()
            if not tied_local.any():
                continue

            local_sign = np.full(len(idx), np.nan)
            local_sign[prices_this_ric > prev_diff] = 1.0
            local_sign[prices_this_ric < prev_diff] = -1.0

            update_positions = idx[tied_local]
            update_values = local_sign[tied_local]
            sign.loc[update_positions] = update_values

    df["trade_sign"] = sign.to_numpy()

    n_classified = int(df.loc[trade_mask, "trade_sign"].notna().sum())
    n_unclassified = n_trades - n_classified
    if n_unclassified:
        logger.info(
            "%s trades (%.4f%%) remain unclassified (tied at the midpoint "
            "with no prior differing trade price to reference -- almost "
            "always just each RIC's very first trade).",
            f"{n_unclassified:,}", 100 * n_unclassified / n_trades if n_trades else 0,
        )
    return df


def add_normalised_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised trade volume, per the assignment slide's definition:

        omega_ij = v_ij / ( (1/N) * sum_{k=1}^{N} T_k )

    i.e. each trade's volume divided by the AVERAGE number of trade
    events per day across the full N-day sample (T_k = number of trade
    events on day k). NOTE: the slide's own subscript (sum of T_j, not
    T_k) looks like a typo -- this implements the sensible reading
    (average daily trade-event count across all days). Flag this
    assumption explicitly in your methodology write-up.
    """
    df["date"] = df["ts_utc"].dt.date

    trade_mask = df["event_type"] == "trade"
    daily_trade_counts = df.loc[trade_mask].groupby(["ric", "date"]).size()
    avg_daily_trades_per_ric = daily_trade_counts.groupby("ric").mean()

    logger.info("Average daily trade-event count per RIC (for volume normalisation):")
    for ric, avg_t in avg_daily_trades_per_ric.items():
        logger.info("  %s: %.1f trades/day", ric, avg_t)

    denom = df["ric"].map(avg_daily_trades_per_ric)
    df["normalised_volume"] = np.where(
        trade_mask, df["TRDVOL_1"] / denom, np.nan,
    )
    return df


def compact_same_timestamp_quotes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep only the last quote of each run of quotes sharing a timestamp.

    Standard TAQ cleaning practice: of several quotes with the same
    timestamp, only the most recent in sequence describes the market at
    that instant. Removing the others reduces noise from repeated or
    re-broadcast quotes that do not represent a new market state.

    "Most recent in sequence" is resolved by the dataframe's own row
    order: Stage 3's external merge sort already ordered same-timestamp
    ties correctly via SEQNUM/event_id, so simply keeping the LAST row
    within each (ric, ts_utc) group -- in existing row order -- is
    exactly the right notion of "most recent", without needing to carry
    SEQNUM/event_id back into memory.

    Memory-lean by construction: only a 2-column (ric, ts_utc) subset is
    ever copied for the grouping itself; the filter back onto the full
    dataframe is a single boolean mask, not a split-then-concat (an
    earlier version of this function did split-and-concat, and its
    extra transient full-width copies were traced to a MemoryError on
    the full 9.3M-row run).
    """
    is_quote = (df["event_type"] == "quote").to_numpy()
    n_before = int(is_quote.sum())

    key_cols = df.loc[is_quote, ["ric", "ts_utc"]]
    pos_from_end = key_cols.groupby(["ric", "ts_utc"], sort=False).cumcount(ascending=False)
    is_last = (pos_from_end.to_numpy() == 0)

    keep = np.ones(len(df), dtype=bool)
    keep[is_quote] = is_last
    df = df[keep].reset_index(drop=True)

    n_after = int((df["event_type"] == "quote").sum())
    stats = {
        "n_quote_rows_before": n_before,
        "n_quote_rows_after": n_after,
        "n_same_timestamp_quotes_compacted": n_before - n_after,
    }
    logger.info(
        "Same-timestamp quote compacting: %s -> %s quote rows (%s compacted away).",
        f"{n_before:,}", f"{n_after:,}", f"{n_before - n_after:,}",
    )
    return df, stats


def compact_same_timestamp_trades(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aggregate trade prints that share a timestamp into one trade.

    One aggressive order executed against several resting orders prints
    once per counterparty or price level, but it is economically a single
    trade. Prints sharing an exact (ric, ts_utc) timestamp are therefore
    combined into one trade with summed volume and a volume-weighted
    average price.

    Simplification: ideally only prints of the same order type would be
    combined, but no reliable order-type qualifier survives cleaning
    (message_type is a near-universal tag), so all prints sharing a
    timestamp are aggregated.

    Memory-lean by construction (see compact_same_timestamp_quotes'
    docstring for why this matters): only a small (ric, ts_utc,
    TRDPRC_1, TRDVOL_1) subset is copied for the aggregation itself; the
    aggregated VWAP/volume are written back into the ORIGINAL dataframe
    via targeted .loc assignment (not a fresh full-width copy), and the
    final filter is a single boolean mask, not a split-then-concat.
    """
    is_trade = (df["event_type"] == "trade").to_numpy()
    n_before = int(is_trade.sum())

    key_cols = df.loc[is_trade, ["ric", "ts_utc", "TRDPRC_1", "TRDVOL_1"]].copy()
    key_cols["_notional"] = key_cols["TRDPRC_1"] * key_cols["TRDVOL_1"]
    grp = key_cols.groupby(["ric", "ts_utc"], sort=False)
    agg_volume = grp["TRDVOL_1"].transform("sum").to_numpy()
    agg_notional = grp["_notional"].transform("sum").to_numpy()
    vwap = agg_notional / agg_volume

    pos_from_end = grp.cumcount(ascending=False).to_numpy()
    is_last = (pos_from_end == 0)

    trade_row_positions = np.flatnonzero(is_trade)
    kept_trade_positions = trade_row_positions[is_last]

    # Do the scatter-write in pure NumPy (fast), then replace each whole
    # column once (cheap, columnar) -- NOT `.iloc[array_of_positions, col]
    # = values` directly on the DataFrame, which is a known slow path in
    # pandas for scattered-row assignment into a wide, mixed-dtype frame
    # (this was the actual cause of the run appearing to hang/never
    # finish on the full 9.3M-row dataset).
    trdprc_full = df["TRDPRC_1"].to_numpy(copy=True)
    trdvol_full = df["TRDVOL_1"].to_numpy(copy=True)
    trdprc_full[kept_trade_positions] = vwap[is_last]
    trdvol_full[kept_trade_positions] = agg_volume[is_last]
    df["TRDPRC_1"] = trdprc_full
    df["TRDVOL_1"] = trdvol_full

    keep = np.ones(len(df), dtype=bool)
    keep[is_trade] = is_last
    df = df[keep].reset_index(drop=True)

    n_after = int((df["event_type"] == "trade").sum())
    stats = {
        "n_trade_rows_before": n_before,
        "n_trade_rows_after": n_after,
        "n_same_timestamp_trades_compacted": n_before - n_after,
    }
    logger.info(
        "Same-timestamp trade compacting: %s -> %s trade rows (%s compacted into VWAP'd prints).",
        f"{n_before:,}", f"{n_after:,}", f"{n_before - n_after:,}",
    )
    return df, stats


JSE_TZ = "Africa/Johannesburg"
CONTINUOUS_SESSION_START = "09:10:00"
CONTINUOUS_SESSION_END = "16:49:59"


def add_session_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each event with its trading-session phase.

    Auction-period order-book events and quote changes follow a different
    mechanism from continuous trading and are separated before any
    analysis. Continuous trading on the JSE runs from 09:00 to 16:50 SAST;
    the window used here is 09:10-16:49:59, which avoids the illiquid
    first minutes after the open and the run-up to the switch into the
    closing auction (Jericevich, Chang & Gebbie 2020).

    Adds a `session_phase` label rather than dropping rows outright:
    'pre_open_auction', 'continuous', or 'post_close'. This is
    deliberately non-destructive -- Part I's 1.2 daily auction price
    extraction specifically NEEDS the pre-open/post-close auction data,
    so those rows stay in the dataset; downstream analyses that want
    the "clean continuous trading period only" view (e.g. Part II's
    order-flow ACF, inter-arrival times, adverse selection) should
    filter to session_phase == 'continuous' explicitly.
    """
    df = df.copy()
    local_time = df["ts_utc"].dt.tz_convert(JSE_TZ).dt.time
    start_t = pd.Timestamp(CONTINUOUS_SESSION_START).time()
    end_t = pd.Timestamp(CONTINUOUS_SESSION_END).time()

    is_continuous = (local_time >= start_t) & (local_time <= end_t)
    is_pre_open = local_time < start_t

    df["session_phase"] = np.where(
        is_continuous, "continuous", np.where(is_pre_open, "pre_open_auction", "post_close")
    )

    counts = df["session_phase"].value_counts().to_dict()
    logger.info("Session phase counts (continuous window %s-%s SAST): %s",
                CONTINUOUS_SESSION_START, CONTINUOUS_SESSION_END, counts)
    return df


def add_volume_outlier_flag(df: pd.DataFrame, top_pct: float = 0.1) -> pd.DataFrame:
    """
    Flags (does NOT remove) trades whose volume sits in the extreme top
    tail (default top 0.1%) of that RIC's trade-volume distribution.

    Investigated during EDA: these large trades have transaction prices
    very close to the prevailing mid-price at the time (not wild
    mispricing), which is the signature of a genuine large negotiated
    block trade (e.g. a JSE "special bargain") rather than a bad tick or
    data error -- so they're kept, just flagged, letting downstream
    analysis decide per-purpose whether to include them (e.g. you might
    want to exclude them from "typical trade size" statistics but keep
    them for total volume/turnover).
    """
    trade_mask = df["event_type"] == "trade"

    df["is_volume_outlier"] = False
    if trade_mask.any():
        cutoff_per_ric = df.loc[trade_mask].groupby("ric")["TRDVOL_1"].quantile(1 - top_pct / 100)
        cutoff = df["ric"].map(cutoff_per_ric)
        is_outlier = trade_mask & (df["TRDVOL_1"] > cutoff)
        df["is_volume_outlier"] = is_outlier.fillna(False)

    n_outliers = int(df["is_volume_outlier"].sum())
    logger.info(
        "Flagged %s trades (%.4f%% of trades) as top-%.1f%% volume outliers per RIC "
        "(kept in output, not removed).",
        f"{n_outliers:,}", 100 * n_outliers / trade_mask.sum() if trade_mask.sum() else 0.0, top_pct,
    )
    return df


def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble exactly the columns requested in assignment step 1.4:
    date, time, mid-price, mid-price change, volume, normalised volume,
    bid, bid volume, ask, ask volume, event type, transaction price,
    transaction volume.

    'volume' is the event's own volume: TRDVOL_1 for trades. For quote
    events 'volume' is NaN (a quote update has no traded volume; use
    bid_volume/ask_volume for quote-side size instead).
    """
    out = pd.DataFrame({
        "ric": df["ric"],
        "date": df["ts_utc"].dt.date,
        "time": df["ts_utc"].dt.time,
        "ts_utc": df["ts_utc"],
        "mid_price": df["mid_price"],
        "mid_price_change": df["mid_price_change"],
        "transaction_mid_price_change": df["transaction_mid_price_change"],
        "volume": np.where(df["event_type"] == "trade", df["TRDVOL_1"], np.nan),
        "normalised_volume": df["normalised_volume"],
        "bid": df["bid_prevailing"],
        "bid_volume": df["bid_size_prevailing"],
        "ask": df["ask_prevailing"],
        "ask_volume": df["ask_size_prevailing"],
        "event_type": df["event_type"],
        "session_phase": df["session_phase"],
        "is_crossed_quote": df["is_crossed_quote"],
        "is_volume_outlier": df["is_volume_outlier"],
        "trade_sign": df["trade_sign"],
        "transaction_price": np.where(df["event_type"] == "trade", df["TRDPRC_1"], np.nan),
        "transaction_volume": np.where(df["event_type"] == "trade", df["TRDVOL_1"], np.nan),
    })
    return out


def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> tuple[pd.DataFrame, dict]:
    """Run the full Stage 4 pipeline end to end and save the result."""
    import time
    t_start = time.time()

    def _elapsed():
        return f"{time.time() - t_start:.1f}s elapsed"

    df, clean_stats = load_and_clean(input_path)
    df = drop_unneeded_columns(df)
    df, zero_placeholder_stats = sanitize_zero_placeholders(df)
    clean_stats["n_zero_placeholder_prices"] = zero_placeholder_stats

    df = add_event_type(df)

    # Compacting must happen BEFORE Lee-Ready classification: it changes
    # what counts as "one trade" (aggregating same-timestamp prints) and
    # reduces quote noise, both of which classify_trades()/add_prevailing_
    # quotes_and_mid() should see the CLEANED version of, not the raw one.
    logger.info("Starting same-timestamp quote compacting... (%s)", _elapsed())
    df, quote_compact_stats = compact_same_timestamp_quotes(df)
    clean_stats["quote_compacting"] = quote_compact_stats

    logger.info("Starting same-timestamp trade compacting... (%s)", _elapsed())
    df, trade_compact_stats = compact_same_timestamp_trades(df)
    clean_stats["trade_compacting"] = trade_compact_stats

    logger.info("Starting session-phase labeling... (%s)", _elapsed())
    df = add_session_phase(df)

    logger.info("Starting quote forward-fill / mid-price... (%s)", _elapsed())
    df = add_prevailing_quotes_and_mid(df)
    df = add_mid_price_change(df)
    df = add_transaction_mid_price_change(df)

    logger.info("Starting trade classification... (%s)", _elapsed())
    df = classify_trades(df)
    df = add_normalised_volume(df)
    df = add_volume_outlier_flag(df)

    logger.info("Starting final dataset assembly... (%s)", _elapsed())
    final = build_final_dataset(df)

    logger.info("Writing final dataset (%s rows) to %s... (%s)", f"{len(final):,}", output_path, _elapsed())
    final.to_parquet(output_path, index=False)
    logger.info("Done.")

    event_counts = final["event_type"].value_counts().to_dict()
    trade_sign_counts = final["trade_sign"].value_counts(dropna=False).to_dict()
    session_phase_counts = final["session_phase"].value_counts().to_dict()
    n_crossed = int(final["is_crossed_quote"].sum())
    n_volume_outliers = int(final["is_volume_outlier"].sum())

    stats = {
        **clean_stats,
        "n_final_rows": len(final),
        "event_type_counts": event_counts,
        "trade_sign_counts": trade_sign_counts,
        "session_phase_counts": session_phase_counts,
        "n_crossed_quote_rows": n_crossed,
        "n_volume_outlier_rows": n_volume_outliers,
        "output_path": str(output_path),
    }
    return final, stats


if __name__ == "__main__":
    final_df, stats = run()
    print()
    print("=" * 70)
    print("STAGE 4 SUMMARY")
    print("=" * 70)
    for k, v in stats.items():
        print(f"{k}: {v}")
