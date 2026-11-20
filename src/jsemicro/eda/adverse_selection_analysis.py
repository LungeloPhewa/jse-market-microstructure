"""
adverse_selection_analysis.py
--------------------------------
Buckets trades by normalised volume (log-spaced bins, matching the
reference paper's own binning convention -- Jericevich, Chang & Gebbie
2020, eq. 3.2) and tracks two adverse-selection signatures per bucket,
separately for buyer- and seller-initiated trades:

  1. VWAP vs. pre-trade mid-price ("slippage cost"): how far does the
     volume-weighted average execution price sit from fair value at the
     time of the trade? Growing slippage with size is the direct,
     literal signature of adverse selection -- larger trades face worse
     execution because liquidity providers protect themselves against
     the possibility the counterparty is informed.

  2. Signed price impact (the paper's Delta p*, eq. 3.1): does the
     mid-price keep drifting in the trade's own direction afterward?
     Growing impact with size is the "informed trading" signature --
     the market itself revises its view of fair value because of what
     the trade size reveals, not just the cost of crossing the spread.

Both are computed from columns already present in the Stage 4 final
dataset (AGL_GRT_stage4_final.parquet) -- no need to touch raw data.

SIGN CONVENTION (matches the reference paper's eqs. 3.9/3.11 exactly):
  cost_ij = trade_sign_ij * (transaction_price_ij - mid_price_ij)
  impact_ij = trade_sign_ij * transaction_mid_price_change_ij
Both are POSITIVE-ORIENTED as a "cost to the liquidity taker becoming
more informed" measure regardless of trade direction: for a buy
(trade_sign=+1), cost/impact is positive if price is above/rises above
the pre-trade mid; for a sell (trade_sign=-1), positive if price is
below/falls below the pre-trade mid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_signed_cost_measures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slippage_cost"] = df["trade_sign"] * (df["transaction_price"] - df["mid_price"])
    df["price_impact"] = df["trade_sign"] * df["transaction_mid_price_change"]
    return df


def compute_volume_buckets(
    df: pd.DataFrame,
    ric: str,
    n_bins: int = 20,
    vol_min: float = 1e-3,
    vol_max: float = 10.0,
) -> pd.DataFrame:
    """
    Buckets trades for one RIC into n_bins logarithmically-spaced
    normalised-volume bins between [vol_min, vol_max] (matching the
    reference paper's own [10^-3, 10^1] range), separately summarised
    for buyer- and seller-initiated trades.

    Returns one row per (side, bucket): average normalised volume,
    VWAP, average pre-trade mid-price, average slippage cost, average
    signed price impact, and trade count -- everything needed to plot
    either quantity against bucket size.
    """
    trades = df[
        (df["ric"] == ric) & (df["event_type"] == "trade")
        & df["normalised_volume"].notna() & df["trade_sign"].notna()
        & df["mid_price"].notna()
    ].copy()
    trades = add_signed_cost_measures(trades)

    bin_edges = np.logspace(np.log10(vol_min), np.log10(vol_max), n_bins + 1)
    trades["volume_bucket"] = pd.cut(trades["normalised_volume"], bins=bin_edges, include_lowest=True)

    rows = []
    for side_label, side_val in [("Buyer-initiated", 1.0), ("Seller-initiated", -1.0)]:
        subset = trades[trades["trade_sign"] == side_val]
        for bucket, grp in subset.groupby("volume_bucket", observed=True):
            if len(grp) == 0:
                continue
            vwap = np.average(grp["transaction_price"], weights=grp["transaction_volume"])
            rows.append({
                "ric": ric,
                "side": side_label,
                "bucket_mid": bucket.mid if hasattr(bucket, "mid") else np.nan,
                "avg_normalised_volume": grp["normalised_volume"].mean(),
                "vwap": vwap,
                "avg_pre_trade_mid": grp["mid_price"].mean(),
                "avg_slippage_cost": grp["slippage_cost"].mean(),
                "avg_price_impact": grp["price_impact"].mean(),
                "n_trades": len(grp),
            })

    result = pd.DataFrame(rows).sort_values(["side", "avg_normalised_volume"]).reset_index(drop=True)
    return result


def plot_vwap_vs_bucket(bucket_df: pd.DataFrame, ric: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Direct visual: VWAP vs. the average pre-trade mid-price, per bucket.
    Growing separation between the two lines as bucket size increases
    is the literal, visual signature of adverse selection.
    """
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = np.atleast_1d(ax)
    else:
        ax = np.atleast_1d(ax)

    for i, side in enumerate(["Buyer-initiated", "Seller-initiated"]):
        sub = bucket_df[bucket_df["side"] == side]
        a = ax[i] if len(ax) > 1 else ax[0]
        a.plot(sub["avg_normalised_volume"], sub["vwap"], "o-", color="black", label="VWAP")
        a.plot(sub["avg_normalised_volume"], sub["avg_pre_trade_mid"], "s--", color="grey", label="Avg pre-trade mid")
        a.set_xscale("log")
        a.set_xlabel("Normalised volume (bucket avg)")
        a.set_ylabel("Price")
        a.set_title(f"{ric}: {side}")
        a.legend()

    return ax


def plot_cost_vs_bucket(
    bucket_df: pd.DataFrame,
    ric: str,
    y_col: str = "avg_slippage_cost",
    log_y: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    y_col: 'avg_slippage_cost' (execution cost relative to pre-trade
    mid, in raw price units -- naturally O(1) for a typical spread) or
    'avg_price_impact' (the paper's Delta p*, a signed LOG RETURN --
    naturally O(1e-4)). A clear UPWARD trend as normalised volume
    increases, for either quantity, is the adverse-selection signature
    -- larger trades costing more / moving the price further in their
    own direction.

    log_y uses a symlog scale with linthresh chosen FROM THE DATA
    ITSELF (a small fraction of its own typical magnitude), rather than
    a fixed threshold -- slippage cost (price units) and price impact
    (log-return units) sit on completely different natural scales, and
    a single fixed threshold sized for one badly compresses the other's
    dynamic range into an unreadable band.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for side, marker, color in [("Buyer-initiated", "o", "tab:blue"), ("Seller-initiated", "s", "tab:red")]:
        sub = bucket_df[bucket_df["side"] == side]
        vals = sub[y_col]
        ax.plot(sub["avg_normalised_volume"], vals, marker=marker, linestyle="-",
                color=color, label=side, alpha=0.8)

    if log_y:
        all_vals = bucket_df[y_col].dropna()
        nonzero_abs = all_vals[all_vals != 0].abs()
        linthresh = max(nonzero_abs.quantile(0.1), 1e-12) if len(nonzero_abs) else 1e-4
        ax.set_yscale("symlog", linthresh=linthresh)

    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Normalised volume (bucket avg)")
    label = "Slippage cost" if y_col == "avg_slippage_cost" else "Signed price impact"
    ax.set_ylabel(label)
    ax.set_title(f"{ric}: {label} vs. trade size")
    ax.legend()
    return ax
