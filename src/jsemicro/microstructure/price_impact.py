"""
price_impact.py -- Part II.A: Price impact (Harvey, Hendricks, Gebbie &
Wilcox, 2017, Physica A, methodology).

Price-impact curve construction:

1. Trades are normalised by each security's average daily traded volume:
       omega = trade volume / average daily traded volume

2. Normalised trade volumes are grouped into 20 logarithmically-spaced
   bins over [1e-3, 10^0.5].

3. For each trade, the log mid-price change immediately following the
   trade is calculated.

4. The absolute price response is calculated:
       abs_delta_m = |log(mid_after) - log(mid_before)|

5. Within each volume bin, the following are calculated separately for
   buyer- and seller-initiated trades:
       - average normalised volume (omega_star)
       - average absolute mid-price change
       - number of trades

6. The resulting bin averages are plotted on log-log axes.

7. A power-law regression is fitted separately to buyer- and
   seller-initiated observations:

       I(omega*) = C * (omega*)^alpha

   which is equivalent to:

       log(I) = log(C) + alpha * log(omega*)

   The fitted slope alpha is the estimated price-impact exponent.
"""

import numpy as np
import pandas as pd
from scipy import stats


def attach_mid_before_after(trades, md, ric):
    """
    Attach the prevailing mid-price just before and just after each
    trade, via merge_asof.

    Two single-column lookups are used rather than a full row-level
    merge of the transaction and market-data feeds.
    """
    tx = trades[trades["ric"] == ric].copy()

    mid = (
        md[md["ric"] == ric][["ts", "mid"]]
        .sort_values("ts")
    )

    before = pd.merge_asof(
        tx[["ts"]],
        mid,
        on="ts",
        direction="backward"
    )

    after = pd.merge_asof(
        tx[["ts"]],
        mid,
        on="ts",
        direction="forward"
    )

    tx["mid_before"] = before["mid"].values
    tx["mid_after"] = after["mid"].values

    # Log mid-price change
    tx["delta_m"] = (
        np.log(tx["mid_after"])
        - np.log(tx["mid_before"])
    )

    # Absolute price response used for the Harvey et al.-style curve
    tx["abs_delta_m"] = tx["delta_m"].abs()

    return tx


def normalise_volume(tx):
    """
    omega = trade volume / that security's average daily trade volume.
    """
    avg_daily_vol = (
        tx.groupby("date")["volume"]
        .sum()
        .mean()
    )

    tx = tx.copy()

    tx["omega"] = (
        tx["volume"] / avg_daily_vol
    )

    return tx


def compute_price_impact_bins(
    trades,
    md,
    ric,
    n_bins=20,
    vol_lo=1e-3,
    vol_hi=10 ** 0.5
):
    """
    Calculate the 20 logarithmically-spaced price-impact bins.

    Returns
    -------
    dict
        {
            "Buyer-initiated": DataFrame,
            "Seller-initiated": DataFrame
        }

    Each DataFrame contains:

        bin
        bin_lower
        bin_upper
        omega_star
        abs_delta_m_star
        n_trades

    Empty bins are retained in the returned DataFrames so that the
    complete 20-bin structure can be inspected for reporting.
    """

    # ------------------------------------------------------------
    # 1. Attach mid-prices and calculate price response
    # ------------------------------------------------------------
    tx = attach_mid_before_after(trades, md, ric)

    # ------------------------------------------------------------
    # 2. Normalise trade volume
    # ------------------------------------------------------------
    tx = normalise_volume(tx)

    # ------------------------------------------------------------
    # 3. Remove invalid observations
    # ------------------------------------------------------------
    tx = tx.dropna(
        subset=[
            "delta_m",
            "abs_delta_m",
            "omega",
            "trade_sign"
        ]
    )

    tx = tx[tx["omega"] > 0]

    # ------------------------------------------------------------
    # 4. Construct logarithmically-spaced bins
    # ------------------------------------------------------------
    bins = np.logspace(
        np.log10(vol_lo),
        np.log10(vol_hi),
        n_bins + 1
    )

    tx["bin"] = pd.cut(
        tx["omega"],
        bins=bins,
        include_lowest=True
    )

    # ------------------------------------------------------------
    # 5. Calculate bin-level averages separately by trade side
    # ------------------------------------------------------------
    out = {}

    for sign, label in [
        (1, "Buyer-initiated"),
        (-1, "Seller-initiated")
    ]:

        sub = tx[
            tx["trade_sign"] == sign
        ].copy()

        # Group by volume bin
        grp = (
            sub.groupby(
                "bin",
                observed=False
            )
            .agg(
                omega_star=(
                    "omega",
                    "mean"
                ),
                abs_delta_m_star=(
                    "abs_delta_m",
                    "mean"
                ),
                n_trades=(
                    "abs_delta_m",
                    "size"
                )
            )
        )

        # --------------------------------------------------------
        # Extract numerical bin boundaries
        # --------------------------------------------------------
        grp["bin_lower"] = [
            interval.left if pd.notna(interval) else np.nan
            for interval in grp.index
        ]

        grp["bin_upper"] = [
            interval.right if pd.notna(interval) else np.nan
            for interval in grp.index
        ]

        # Keep bin as a readable column
        grp = grp.reset_index()

        # Put columns in report-friendly order
        grp = grp[
            [
                "bin",
                "bin_lower",
                "bin_upper",
                "omega_star",
                "abs_delta_m_star",
                "n_trades"
            ]
        ]

        out[label] = grp

    return out


def combine_price_impact_bins(binned):
    """
    Combine buyer- and seller-initiated bin results into one
    side-by-side report-friendly DataFrame.

    Each row represents one logarithmic volume bin.

    Returns
    -------
    pandas.DataFrame

    Columns:
        bin
        bin_lower
        bin_upper

        buyer_omega_star
        buyer_abs_delta_m_star
        buyer_n_trades

        seller_omega_star
        seller_abs_delta_m_star
        seller_n_trades
    """

    buyer = binned["Buyer-initiated"].copy()
    seller = binned["Seller-initiated"].copy()

    # ------------------------------------------------------------
    # Keep only the columns needed for the report
    # ------------------------------------------------------------

    buyer = buyer[
        [
            "bin",
            "bin_lower",
            "bin_upper",
            "omega_star",
            "abs_delta_m_star",
            "n_trades"
        ]
    ].rename(
        columns={
            "omega_star": "buyer_omega_star",
            "abs_delta_m_star": "buyer_abs_delta_m_star",
            "n_trades": "buyer_n_trades"
        }
    )

    seller = seller[
        [
            "bin",
            "bin_lower",
            "bin_upper",
            "omega_star",
            "abs_delta_m_star",
            "n_trades"
        ]
    ].rename(
        columns={
            "omega_star": "seller_omega_star",
            "abs_delta_m_star": "seller_abs_delta_m_star",
            "n_trades": "seller_n_trades"
        }
    )

    # ------------------------------------------------------------
    # Merge buyer and seller results by volume bin
    # ------------------------------------------------------------

    combined = pd.merge(
        buyer,
        seller,
        on=["bin", "bin_lower", "bin_upper"],
        how="outer"
    )

    # ------------------------------------------------------------
    # Sort by lower volume boundary
    # ------------------------------------------------------------

    combined = combined.sort_values(
        "bin_lower"
    ).reset_index(drop=True)

    return combined

def fit_power_law(grp):
    """
    Fit the power-law relationship:

        I(omega*) = C * (omega*)^alpha

    by performing linear regression in log-log space:

        log(I) = log(C) + alpha * log(omega*)

    Returns a dictionary containing:
        slope
        intercept
        r_squared
        p_value
        std_err
    """

    # Only use positive, non-missing observations
    valid = grp[
        (grp["omega_star"] > 0)
        & (grp["abs_delta_m_star"] > 0)
        & grp["omega_star"].notna()
        & grp["abs_delta_m_star"].notna()
    ].copy()

    if len(valid) < 3:
        return None

    logx = np.log(valid["omega_star"])
    logy = np.log(valid["abs_delta_m_star"])

    result = stats.linregress(
        logx,
        logy
    )

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "std_err": result.stderr
    }


def plot_price_impact_curve(binned, ric, ax):
    """
    Plot the price-impact curve.

    The observed bin averages are shown as points only.

    A fitted power-law regression line is shown separately for
    buyer- and seller-initiated trades.

    Returns
    -------
    dict
        Regression statistics for each trade direction.
    """

    colours = {
        "Buyer-initiated": "tab:blue",
        "Seller-initiated": "tab:red"
    }

    fit_results = {}

    for label, grp in binned.items():

        # --------------------------------------------------------
        # Only use valid observations for plotting/regression
        # --------------------------------------------------------
        valid = grp[
            (grp["omega_star"] > 0)
            & (grp["abs_delta_m_star"] > 0)
            & grp["omega_star"].notna()
            & grp["abs_delta_m_star"].notna()
        ].copy()

        if len(valid) < 3:
            continue

        # --------------------------------------------------------
        # Observed bin averages: POINTS ONLY
        # --------------------------------------------------------
        ax.scatter(
            valid["omega_star"],
            valid["abs_delta_m_star"],
            color=colours[label],
            marker="o",
            label=label
        )

        # --------------------------------------------------------
        # Power-law regression
        # --------------------------------------------------------
        fit = fit_power_law(valid)

        fit_results[label] = fit

        slope = fit["slope"]
        intercept = fit["intercept"]

        # --------------------------------------------------------
        # Generate smooth fitted power-law curve
        # --------------------------------------------------------
        x_fit = np.logspace(
            np.log10(valid["omega_star"].min()),
            np.log10(valid["omega_star"].max()),
            100
        )

        y_fit = np.exp(
            intercept
            + slope * np.log(x_fit)
        )

        # --------------------------------------------------------
        # Plot fitted regression line
        # --------------------------------------------------------
        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color=colours[label],
            label=(
                f"{label} fit "
                f"($\\alpha$ = {slope:.3f})"
            )
        )

    # ------------------------------------------------------------
    # Log-log axes
    # ------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Average normalised volume $\omega^*$"
    )

    ax.set_ylabel(
        r"Average $|\Delta m^*|$"
    )

    ax.set_title(
        f"{ric}: price impact curve "
        f"(Harvey et al. 2017 style)"
    )

    ax.legend()

    return fit_results
