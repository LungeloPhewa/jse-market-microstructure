"""
md_premerge_eda.py
--------------------
Pre-merging EDA on the prepared Market Depth file (AGL_GRT_MD_flat.parquet)
-- i.e. before it gets combined with Transactions in merge_tx_md.py.

Computes and plots the L1 (top-of-book) mid-price time series for both
tickers, as a sanity check on the Market Depth data on its own terms
before it's merged with anything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_mid_price_series(
    md_path: str | Path,
    ric: str,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads L1 bid/ask for one ric from the prepared Market Depth file and
    computes mid-price. Only the columns actually needed are read (cheap
    with Parquet's columnar format, even though the full file carries
    ~40 depth-level columns across millions of rows).

    Zero-price placeholders (a price of exactly 0 meaning "no quote yet",
    not a genuine observation -- see stage4_clean_classify.py's
    sanitize_zero_placeholders(), the same convention applied here since
    market_depth_prepare.py's numeric coercion doesn't itself remove
    them, only ensures the column is numeric) are treated as missing
    before computing mid-price.

    resample: optional pandas offset alias (e.g. '1min', '5min') to
    downsample to a regular interval via last-observation-carried-forward
    within each bucket -- worth using for a full-month plot, since the
    raw tick-level series can run into millions of points and render as
    a dense, hard-to-read blur otherwise. Leave as None for the raw,
    full-resolution series (fine for a single day or a short window).
    """
    cols = ["ric", "ts_utc", "L1-BidPrice", "L1-AskPrice"]
    df = pd.read_parquet(md_path, columns=cols)
    df = df[df["ric"] == ric].sort_values("ts_utc").reset_index(drop=True)

    df["L1-BidPrice"] = df["L1-BidPrice"].replace(0, np.nan)
    df["L1-AskPrice"] = df["L1-AskPrice"].replace(0, np.nan)
    df["mid_price"] = (df["L1-BidPrice"] + df["L1-AskPrice"]) / 2

    df = df.dropna(subset=["mid_price"])

    if resample is not None:
        df = (
            df.set_index("ts_utc")["mid_price"]
            .resample(resample)
            .last()
            .ffill()
            .reset_index()
        )
    else:
        df = df[["ts_utc", "mid_price"]]

    return df


def plot_mid_price_timeseries(
    mid_price_by_ric: dict[str, pd.DataFrame],
    tz: str = "Africa/Johannesburg",
    title: str = "L1 mid-price (pre-merging, Market Depth only)",
) -> plt.Axes:
    """
    One subplot per ric, side by side, each with its own appropriately-
    scaled y-axis -- AGLJ.J and GRTJ.J differ in price by roughly 10x,
    so a single shared axis would make the smaller-scale series nearly
    invisible (the same issue already fixed for the normalised-volume
    and inter-arrival-time distribution charts elsewhere in this
    project).
    """
    rics = list(mid_price_by_ric.keys())
    fig, axes = plt.subplots(1, len(rics), figsize=(7 * len(rics), 4.5), sharex=False)
    axes = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, ric, color in zip(axes, rics, colors):
        df = mid_price_by_ric[ric]
        ts_local = df["ts_utc"].dt.tz_convert(tz)
        ax.plot(ts_local, df["mid_price"], color=color, linewidth=0.6)
        ax.set_title(ric)
        ax.set_xlabel(f"Time ({tz})")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b", tz=ts_local.dt.tz))
    axes[0].set_ylabel("Mid-price")
    fig.suptitle(title)
    plt.tight_layout()
    return axes


def load_liquidity_series(
    md_path: str | Path,
    ric: str,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads L1 bid/ask SIZE (volume resting at the best price) for one ric.

    A size value is only meaningful when its own side's price is a
    genuine quote, not the same "no quote yet" zero-price placeholder
    already established for mid-price (load_mid_price_series) -- a size
    of e.g. 500 paired with a price of 0 would be a placeholder artifact,
    not a real 500-share resting order, so it's nulled out here too.

    resample: see load_mid_price_series -- same rationale, use e.g.
    '5min' for a full-month plot to avoid an unreadably dense line.
    'last' (not mean/sum) is used per bucket, representing the
    prevailing liquidity snapshot at that point in time, consistent
    with how mid-price is resampled.
    """
    cols = ["ric", "ts_utc", "L1-BidPrice", "L1-AskPrice", "L1-BidSize", "L1-AskSize"]
    df = pd.read_parquet(md_path, columns=cols)
    df = df[df["ric"] == ric].sort_values("ts_utc").reset_index(drop=True)

    df["L1-BidSize"] = df["L1-BidSize"].where(df["L1-BidPrice"] != 0, np.nan)
    df["L1-AskSize"] = df["L1-AskSize"].where(df["L1-AskPrice"] != 0, np.nan)

    df = df.dropna(subset=["L1-BidSize", "L1-AskSize"], how="all")
    df = df[["ts_utc", "L1-BidSize", "L1-AskSize"]]

    if resample is not None:
        df = (
            df.set_index("ts_utc")
            .resample(resample)
            .last()
            .ffill()
            .reset_index()
        )

    return df


def plot_liquidity_timeseries(
    liquidity_by_ric: dict[str, pd.DataFrame],
    tz: str = "Africa/Johannesburg",
    title: str = "L1 liquidity (bid/ask size) over time, pre-merging",
) -> plt.Axes:
    """
    One subplot per ric, side by side -- same rationale as
    plot_mid_price_timeseries (AGLJ.J and GRTJ.J differ enough in
    typical resting size that a shared axis would misrepresent one of
    them). Blue = bid size, red = ask size, matching the bid/ask colour
    convention already used throughout this project's other charts
    (the intraday bubble charts and the order-book depth bars).
    """
    rics = list(liquidity_by_ric.keys())
    fig, axes = plt.subplots(1, len(rics), figsize=(7 * len(rics), 4.5), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, ric in zip(axes, rics):
        df = liquidity_by_ric[ric]
        ts_local = df["ts_utc"].dt.tz_convert(tz)
        ax.plot(ts_local, df["L1-BidSize"], color="blue", linewidth=0.6, label="Bid size")
        ax.plot(ts_local, df["L1-AskSize"], color="red", linewidth=0.6, label="Ask size")
        ax.set_title(ric)
        ax.set_xlabel(f"Time ({tz})")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b", tz=ts_local.dt.tz))
    axes[0].set_ylabel("Size (volume)")
    fig.suptitle(title)
    plt.tight_layout()
    return axes


# --------------------------------------------------------------------
# Daily-frequency linear filter / moving-average trend estimation
# --------------------------------------------------------------------

# Fitting a quadratic trend by least squares over a moving 5-point window
# and evaluating it at the window's midpoint gives this symmetric weight
# vector (the 5-point Savitzky-Golay smoother). Unlike a plain equal-
# weight moving average, this filter is exact (unbiased) for a
# quadratic trend, not just a constant or linear one.
QUADRATIC_FILTER_5 = [-3, 12, 17, 12, -3]
QUADRATIC_FILTER_DENOM_5 = 35


def resample_to_daily(series_df: pd.DataFrame, value_col: str, tz: str = "Africa/Johannesburg",
                       agg: str = "mean") -> pd.Series:
    """
    Collapses an already-loaded tick-level series (e.g. from
    load_mid_price_series or load_liquidity_series) down to one point
    per calendar day (local time) -- the basis for a daily-frequency
    linear filter, since linear filters are defined over equally-
    spaced observations and a "day" is the natural spacing for a
    month-long trend view (rather than smoothing at the tick or 5-minute
    level, which is what the existing pre-merging charts already show).

    agg: 'mean' (average price/size across the day) or 'last' (closing
    value of the day). 'mean' is generally more representative for a
    trend line; 'last' matches a more traditional "daily close" series
    if you'd rather have that.
    """
    s = series_df.set_index(series_df["ts_utc"].dt.tz_convert(tz))[value_col]
    daily = s.resample("1D").agg(agg)
    return daily.dropna()


def apply_linear_filter(daily_series: pd.Series, weights: list[float] | None = None,
                         window: int = 5) -> pd.Series:
    """
    Applies a linear filter (weighted moving average) to a daily series:
    y-tilde_t = sum_r w_r * y_{t+r}, weights summing to 1.

    weights: a FULL symmetric weight vector (e.g. QUADRATIC_FILTER_5),
    which gets rescaled to sum to 1 automatically. If None, uses a
    plain equal-weight moving average of the given `window` size (e.g.
    window=5 -> [1,1,1,1,1]/5) -- preserves a constant or linear trend
    exactly but NOT a quadratic one, unlike
    QUADRATIC_FILTER_5.

    Returns NaN at the first/last (len(weights)//2) points, where a
    symmetric window doesn't fit (the end-point problem of linear
    filters) rather than silently using a truncated/asymmetric filter there.
    """
    if weights is None:
        weights = [1.0] * window
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # normalise to sum to 1, as a linear filter requires

    m = len(weights) // 2
    values = daily_series.to_numpy()
    filtered = np.full(len(values), np.nan)
    for t in range(m, len(values) - m):
        filtered[t] = np.dot(weights, values[t - m: t + m + 1])

    return pd.Series(filtered, index=daily_series.index, name=f"{daily_series.name}_filtered")


def plot_daily_trend(daily_series: pd.Series, filtered_series: pd.Series, ric: str,
                      ylabel: str = "Value", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots the raw daily series alongside its linear-filter trend estimate.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(daily_series.index, daily_series.values, color="grey", alpha=0.6,
            marker="o", markersize=3, linewidth=1, label="Daily (raw)")
    ax.plot(filtered_series.index, filtered_series.values, color="black", linewidth=2,
            label="Linear filter (trend)")
    ax.set_title(f"{ric}: daily trend via linear filter")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax
