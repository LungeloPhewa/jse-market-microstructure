"""
stage5_eda.py
--------------
Assignment 1.5: EDA sanity checks on the Stage 4 cleaned & classified
dataset (AGL_GRT_stage4_final.parquet).

Each check is a standalone function so you can run them individually in
Quarto cells (charts render inline) or call run() for everything at once.

Usage in Quarto:

    import stage5_eda as s5
    df = s5.load()

    s5.check_row_counts(df)
    s5.check_spread(df)
    s5.check_trade_vs_quote_sanity(df)
    s5.check_trade_sign_balance(df)
    s5.check_normalised_volume(df)
    s5.check_intraday_intensity(df)
    s5.check_interarrival_times(df)

Or just: df = s5.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
from ..config import PATHS

INPUT_PATH = PATHS.stage4
# --------------------------------------------------------------------


def _as_date(d):
    """Return a plain date whether d is a datetime.date or a pandas Timestamp."""
    return d.date() if hasattr(d, "date") and callable(getattr(d, "date")) else d


def load(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df


def _section(title: str, verbose: bool = True) -> None:
    if not verbose:
        return
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------
# 1. Row counts / date coverage / gap check
# --------------------------------------------------------------------

def check_row_counts(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Overall counts, date range, and a per-RIC daily event-count series --
    a big unexplained gap or a day with near-zero events usually signals
    a missing data pull rather than a genuinely quiet trading day.

    verbose: set False to suppress the printed tables/warnings while
    still producing the plot -- useful for a presentation document where
    the narrative text already covers the findings and the full printed
    dump would only add length without adding information.
    """
    _section("1. ROW COUNTS & DATE COVERAGE", verbose)

    if verbose:
        print(f"Total rows: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"RICs: {sorted(df['ric'].unique())}")
        print()
        print("Rows per RIC:")
        print(df["ric"].value_counts().to_string())

    daily_counts = df.groupby(["date", "ric"]).size().unstack("ric").fillna(0)

    if verbose:
        print()
        print("Daily event count summary per RIC:")
        print(daily_counts.describe().to_string())

        for ric in daily_counts.columns:
            median = daily_counts[ric].median()
            low_days = daily_counts.index[daily_counts[ric] < 0.1 * median]
            if len(low_days):
                print()
                print(f"WARNING: {ric} has {len(low_days)} day(s) with < 10% of median daily "
                      f"event count (median={median:.0f}) -- check these for missing data:")
                for d in low_days:
                    print(f"    {_as_date(d)}: {int(daily_counts.loc[d, ric])} events")

    fig, ax = plt.subplots(figsize=(11, 4))
    daily_counts.plot(ax=ax)
    ax.set_title("Daily event count per RIC (gaps/dips may indicate missing data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Event count")
    plt.tight_layout()
    plt.show()

    return daily_counts


# --------------------------------------------------------------------
# 2. Spread sanity
# --------------------------------------------------------------------

def check_spread(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Bid-ask spread should be >= 0 for any row NOT flagged as a crossed
    quote. Also checks the intraday spread pattern -- spreads are
    typically wider at the open/close and tighter mid-session (a partial
    U-shape), which is a good general microstructure sanity check.

    verbose: see check_row_counts.
    """
    _section("2. SPREAD SANITY", verbose)

    # Only pull the columns actually needed, before filtering/copying --
    # the full dataset carries ~13 numeric columns; copying all of them
    # for the (large) quote-only subset needs one contiguous block sized
    # for all 13, which was traced to a genuine MemoryError on the full
    # ~5M-row quote subset partway through a long render (cumulative
    # memory pressure from everything already rendered by that point).
    # Selecting only the ~5 columns this function actually uses cuts
    # that block down to a fraction of the size.
    needed_cols = ["ric", "ts_utc", "event_type", "bid", "ask"]
    if "is_crossed_quote" in df.columns:
        needed_cols.append("is_crossed_quote")

    quotes = df.loc[df["event_type"] == "quote", needed_cols].copy()
    quotes["spread"] = quotes["ask"] - quotes["bid"]

    if "is_crossed_quote" in quotes.columns:
        clean_quotes = quotes.loc[~quotes["is_crossed_quote"]].copy()
        n_crossed = int(quotes["is_crossed_quote"].sum())
        if verbose:
            print(f"Crossed quotes (excluded from this check): {n_crossed:,} "
                  f"({100*n_crossed/len(quotes):.3f}% of quote rows). Note this is narrower "
                  f"than Stage 4's own is_crossed_quote total across ALL quote rows, since "
                  f"this check further requires both bid and ask to be non-null (needed for "
                  f"the spread calculation itself).")
    else:
        clean_quotes = quotes
        if verbose:
            print("No 'is_crossed_quote' column found -- rerun Stage 4 with the "
                  "latest script version to get this flag. Proceeding without "
                  "excluding crossed quotes; spread stats below may include them.")

    n_negative = int((clean_quotes["spread"] < 0).sum())
    if verbose:
        print(f"Negative spreads among NON-crossed-flagged quotes: {n_negative:,} "
              f"(should be 0 -- investigate if not)")

        print()
        print("Spread (ask - bid) summary per RIC:")
        print(clean_quotes.groupby("ric")["spread"].describe().to_string())

    # clean_quotes is already an independent copy from the .loc[...].copy()
    # above (or IS quotes itself, already copied) -- no need to copy again,
    # avoiding a second full-size allocation for the same data.
    clean_quotes["hour_sast"] = clean_quotes["ts_utc"].dt.tz_convert("Africa/Johannesburg").dt.hour
    hourly_spread = clean_quotes.groupby(["hour_sast", "ric"])["spread"].median().unstack("ric")

    fig, ax0 = plt.subplots(figsize=(7, 4))
    hourly_spread.plot(ax=ax0, marker="o")
    ax0.set_title("Median spread by hour of day (SAST)")
    ax0.set_xlabel("Hour (SAST)")
    ax0.set_ylabel("Median spread")
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    rics_sorted = sorted(clean_quotes["ric"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, ric, color in zip(axes2, rics_sorted, colors):
        vals = clean_quotes.loc[clean_quotes["ric"] == ric, "spread"]
        vals = vals[vals.between(vals.quantile(0.001), vals.quantile(0.999))]
        ax.hist(vals, bins=60, color=color)
        ax.set_title(ric)
        ax.set_xlabel("Spread")
    axes2[0].set_ylabel("Frequency")
    fig2.suptitle("Spread distribution (1st-99.9th pctile)")
    plt.tight_layout()
    plt.show()

    return clean_quotes


# --------------------------------------------------------------------
# 3. Trade price vs prevailing quote sanity
# --------------------------------------------------------------------

def check_trade_vs_quote_sanity(df: pd.DataFrame, pct_threshold: float = 0.05, verbose: bool = True) -> pd.DataFrame:
    """
    How far do trade prices sit from the prevailing mid-price? After
    cleaning, trades should generally be within (or very close to) the
    bid-ask band. Trades far outside that band (beyond pct_threshold,
    default 5%) are candidates for bad ticks / fat-finger errors worth a
    closer look, not necessarily something to drop automatically.

    verbose: see check_row_counts.
    """
    _section("3. TRADE PRICE vs PREVAILING QUOTE SANITY", verbose)

    trades = df[(df["event_type"] == "trade") & df["mid_price"].notna()].copy()
    trades["pct_from_mid"] = (trades["transaction_price"] - trades["mid_price"]) / trades["mid_price"]

    far_trades = trades[trades["pct_from_mid"].abs() > pct_threshold]

    if verbose:
        print(f"Trades with a valid prevailing mid-price: {len(trades):,}")
        print()
        print("Distance from mid-price (%) summary per RIC:")
        desc = trades.groupby("ric")["pct_from_mid"].describe()
        desc.loc[:, desc.columns != "count"] *= 100  # everything except count is a % value
        print(desc.to_string())

        print()
        print(f"Trades more than {pct_threshold:.0%} away from prevailing mid-price: "
              f"{len(far_trades):,} ({100*len(far_trades)/len(trades):.4f}%)")
        if len(far_trades):
            print(far_trades[["ric", "date", "time", "transaction_price", "mid_price", "pct_from_mid"]]
                  .sort_values("pct_from_mid", key=abs, ascending=False).head(15).to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    for ric in trades["ric"].unique():
        vals = trades.loc[trades["ric"] == ric, "pct_from_mid"] * 100
        vals = vals[vals.between(vals.quantile(0.005), vals.quantile(0.995))]
        ax.hist(vals, bins=80, alpha=0.5, label=ric)
    ax.set_title("Trade price distance from prevailing mid-price (%)")
    ax.set_xlabel("% from mid-price")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return far_trades


# --------------------------------------------------------------------
# 4. Trade sign balance
# --------------------------------------------------------------------

def check_trade_sign_balance(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Buy- vs sell-initiated trade counts overall and per RIC. A wildly
    skewed ratio (e.g. 90/10) across the WHOLE sample would be unusual
    and worth a second look at the classification logic; some imbalance
    is normal and expected, not itself a red flag.

    verbose: see check_row_counts.
    """
    _section("4. TRADE SIGN BALANCE (Lee-Ready / Lim-Coggins classification)", verbose)

    trades = df[df["event_type"] == "trade"].copy()
    by_ric = trades.groupby("ric")["trade_sign"].value_counts(dropna=False).unstack()

    if verbose:
        overall = trades["trade_sign"].value_counts(dropna=False)
        print("Overall trade_sign counts:")
        print(overall.to_string())

        print()
        print("By RIC:")
        print(by_ric.to_string())

        by_ric_pct = by_ric.div(by_ric.sum(axis=1), axis=0) * 100
        print()
        print("By RIC (%):")
        print(by_ric_pct.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [c for c in [1.0, -1.0] if c in by_ric.columns]
    by_ric[cols].plot(kind="bar", ax=ax)
    ax.set_title("Buy (+1) vs sell (-1) initiated trade counts by RIC")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    return by_ric


# --------------------------------------------------------------------
# 5. Normalised volume outliers
# --------------------------------------------------------------------

def check_normalised_volume(df: pd.DataFrame, top_pct: float = 0.1, verbose: bool = True) -> pd.DataFrame:
    """
    Distribution of normalised trade volume, flagging the extreme top
    tail (default top 0.1%) as candidate fat-finger / block trades worth
    a closer look, not necessarily errors to remove.

    The threshold is computed PER RIC (matching Stage 4's own
    is_volume_outlier flag), not globally across tickers -- a global
    threshold would unfairly flag whichever RIC happens to sit on a
    structurally higher normalised-volume scale (e.g. a less liquid
    ticker with fewer trades/day normalises to bigger numbers for the
    same raw volume), even when nothing is actually unusual for that
    RIC specifically.

    verbose: see check_row_counts.
    """
    _section("5. NORMALISED VOLUME OUTLIERS", verbose)

    trades = df[df["event_type"] == "trade"].copy()

    cutoff_per_ric = trades.groupby("ric")["normalised_volume"].transform(
        lambda s: s.quantile(1 - top_pct / 100)
    )
    outliers = trades[trades["normalised_volume"] > cutoff_per_ric]

    if verbose:
        print("Normalised volume summary per RIC:")
        print(trades.groupby("ric")["normalised_volume"].describe().to_string())

        print()
        print(f"Top {top_pct}% normalised-volume outliers (per RIC): {len(outliers):,} trades")
        print(outliers[["ric", "date", "time", "transaction_volume", "normalised_volume"]]
              .sort_values(["ric", "normalised_volume"], ascending=[True, False]).groupby("ric").head(8).to_string())

    rics = sorted(trades["ric"].unique())
    fig, axes = plt.subplots(1, len(rics), figsize=(6 * len(rics), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, ric, color in zip(axes, rics, colors):
        vals = trades.loc[trades["ric"] == ric, "normalised_volume"].dropna()
        vals = vals[vals <= vals.quantile(0.99)]
        ax.hist(vals, bins=60, color=color)
        ax.set_title(ric)
        ax.set_xlabel("Normalised volume")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Normalised trade volume distribution (up to 99th pctile)")
    plt.tight_layout()
    plt.show()

    return outliers


# --------------------------------------------------------------------
# 6. Intraday intensity (event counts by time of day)
# --------------------------------------------------------------------

def check_intraday_intensity(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Event count by hour of day (UTC). Continuous equity markets typically
    show a partial U-shape (busier at open/close, quieter mid-session);
    a distinct early-morning spike separate from that pattern is also
    useful to visually confirm the pre-open auction window identified
    during Stage 4 cleaning.

    verbose: see check_row_counts.
    """
    _section("6. INTRADAY EVENT INTENSITY", verbose)

    # Only ric/ts_utc are needed here -- copying the full wide dataframe
    # (all ~13-19 columns) just to add one derived column is the same
    # avoidable memory cost already found and fixed in check_spread.
    df = df[["ric", "ts_utc"]].copy()
    df["hour_sast"] = df["ts_utc"].dt.tz_convert("Africa/Johannesburg").dt.hour
    hourly = df.groupby(["hour_sast", "ric"]).size().unstack("ric").fillna(0)

    if verbose:
        print("Event counts by hour (SAST):")
        print(hourly.to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    hourly.plot(kind="bar", ax=ax)
    ax.set_title("Event count by hour of day (SAST)")
    ax.set_xlabel("Hour (SAST)")
    ax.set_ylabel("Event count")
    plt.tight_layout()
    plt.show()

    return hourly


# --------------------------------------------------------------------
# 7. Inter-arrival times
# --------------------------------------------------------------------

def check_interarrival_times(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    tau_ij = t_{i+1,j} - t_ij, the event inter-arrival time definition
    from the assignment slide. Computed per RIC (day boundaries are NOT
    excluded here -- consider filtering to within-day pairs only if you
    want a strictly intraday-only version for later analysis).

    verbose: see check_row_counts.
    """
    _section("7. INTER-ARRIVAL TIMES", verbose)

    # Same fix as check_spread/check_intraday_intensity: only ric/ts_utc
    # are needed here. sort_values() already returns a new object, so
    # the .copy() previously chained after it was also redundant on top
    # of being unnecessarily wide.
    df = df[["ric", "ts_utc"]].sort_values(["ric", "ts_utc"])
    df["tau_seconds"] = df.groupby("ric")["ts_utc"].diff().dt.total_seconds().shift(-1)

    if verbose:
        print("Inter-arrival time (seconds) summary per RIC:")
        print(df.groupby("ric")["tau_seconds"].describe().to_string())

    rics = sorted(df["ric"].unique())
    fig, axes = plt.subplots(1, len(rics), figsize=(6 * len(rics), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, ric, color in zip(axes, rics, colors):
        vals = df.loc[df["ric"] == ric, "tau_seconds"].dropna()
        vals = vals[(vals > 0) & (vals < vals.quantile(0.999))]
        ax.hist(np.log10(vals + 1e-6), bins=60, color=color)
        ax.set_title(ric)
        ax.set_xlabel("log10(tau + epsilon)")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Inter-arrival time distribution (log10 seconds)")
    plt.tight_layout()
    plt.show()

    return df[["ric", "ts_utc", "tau_seconds"]]


# --------------------------------------------------------------------
# 8. (Optional) reconciliation against the Part 1.2 daily OHLCV/auction pull
# --------------------------------------------------------------------

def reconcile_with_daily_auction(
    df: pd.DataFrame,
    daily_path: Optional[Path] = None,
    daily_ric_col: str = "ric",
    daily_date_col: str = "date",
    daily_open_col: str = "open",
    daily_close_col: str = "close",
    min_open_hour_utc: int = 7,
) -> Optional[pd.DataFrame]:
    """
    Compares the first and last MID-PRICE of each trading day (from the
    intraday TAQ data) against the OPEN/CLOSE auction prices from your
    Part 1.2 daily OHLCV pull. Large, systematic discrepancies would
    suggest a timestamp/timezone misalignment or an auction-price
    capture issue in one of the two pulls.

    min_open_hour_utc: the intraday "first mid-price" is restricted to
    quotes at or after this UTC hour before picking the day's first one.
    This matters because the JSE's pre-open indicative call auction can
    start hours before the actual 09:00 SAST open (observed as early as
    ~04:32 UTC during the crossed-quote investigation in Stage 4) --
    comparing the literal first tick of the whole day against the
    matched OPENING AUCTION price compares two different things by
    definition (an indicative, still-updating pre-open quote vs. the
    final uncrossed print), not a real discrepancy. Default is 5 (UTC),
    since build_daily_auction_prices.py's own output shows every
    opening auction print lands within the 05:00 UTC hour (05:00:0X-
    05:00:29) -- this restricts to "the auction hour or later" without
    excluding the auction moment itself. (With the corrected GMT handling in
    flatten_taq.py the opening uncross is at about 07:00 UTC, hence the
    default of 7; the 05:00 quoted above came from the two-hour shift.)
    The CLOSE side needs no such
    adjustment -- a closing auction is a clean, single settlement moment
    with no analogous pre-close indicative-price issue.

    You'll need to point daily_path at your actual Part 1.2 daily file
    and adjust the column name arguments to match its real schema --
    this function doesn't know that schema, since it wasn't part of the
    files inspected so far.
    """
    _section("8. RECONCILIATION vs DAILY OHLCV/AUCTION PULL")

    if daily_path is None:
        print("Skipping -- no daily_path provided. Call again with "
              "daily_path=<path to your Part 1.2 daily OHLCV file> and "
              "the matching column names to enable this check.")
        return None

    daily = pd.read_parquet(daily_path)
    daily[daily_date_col] = pd.to_datetime(daily[daily_date_col])

    df_for_open = df[df["ts_utc"].dt.hour >= min_open_hour_utc]

    intraday_daily = (
        df.sort_values(["ric", "ts_utc"])
        .groupby(["ric", "date"])["mid_price"]
        .agg(intraday_last="last")
        .reset_index()
    )
    intraday_open = (
        df_for_open.sort_values(["ric", "ts_utc"])
        .groupby(["ric", "date"])["mid_price"]
        .agg(intraday_first="first")
        .reset_index()
    )
    intraday_daily = intraday_daily.merge(intraday_open, on=["ric", "date"], how="left")
    intraday_daily["date"] = pd.to_datetime(intraday_daily["date"])

    merged = intraday_daily.merge(
        daily, left_on=["ric", "date"], right_on=[daily_ric_col, daily_date_col], how="inner",
    )
    merged["open_diff_pct"] = 100 * (merged["intraday_first"] - merged[daily_open_col]) / merged[daily_open_col]
    merged["close_diff_pct"] = 100 * (merged["intraday_last"] - merged[daily_close_col]) / merged[daily_close_col]

    print(f"Matched {len(merged)} (ric, date) pairs between intraday and daily data.")
    print(f"(Intraday 'first' restricted to ts_utc.hour >= {min_open_hour_utc} "
          f"to avoid comparing against pre-open indicative quotes.)")
    print()
    print("Open reconciliation (% diff) summary:")
    print(merged["open_diff_pct"].describe().to_string())
    print()
    print("Close reconciliation (% diff) summary:")
    print(merged["close_diff_pct"].describe().to_string())

    large_diff = merged[(merged["open_diff_pct"].abs() > 1) | (merged["close_diff_pct"].abs() > 1)]
    if len(large_diff):
        print()
        print(f"WARNING: {len(large_diff)} day(s) with >1% open/close discrepancy:")
        print(large_diff[["ric", "date", "open_diff_pct", "close_diff_pct"]].to_string())

    return merged


def run(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    df = load(input_path)
    check_row_counts(df)
    check_spread(df)
    check_trade_vs_quote_sanity(df)
    check_trade_sign_balance(df)
    check_normalised_volume(df)
    check_intraday_intensity(df)
    check_interarrival_times(df)
    print()
    print("=" * 78)
    print("Reconciliation against your Part 1.2 daily OHLCV pull was skipped --")
    print("call reconcile_with_daily_auction(df, daily_path=..., ...) separately")
    print("once you point it at your actual daily file and column names.")
    print("=" * 78)
    return df


if __name__ == "__main__":
    run()
