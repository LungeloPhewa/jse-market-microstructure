"""
intraday_seasonality.py -- Part II.B: Order-book seasonality.

Builds three intraday curves on a common grid of local (SAST) time-of-day
buckets, each normalised within day and then averaged across days:

    volume      (TX feed)  total traded volume in the bucket
                           / that day's mean bucket volume
    volatility  (MD feed)  realised volatility of the mid-price in the bucket,
                           sqrt(sum of squared log mid changes)
                           / that day's mean bucket value
    spread      (MD feed)  mean relative quoted spread in the bucket
                           / that day's mean bucket value

A value of 1 therefore means "an average bucket for that day", and the
curves are comparable across names and across measures.

Changes from the first version
------------------------------
* The volume curve divided each TRADE's volume by the day's total and then
  averaged per bucket, which measures the size of a typical trade in that
  bucket, not how much volume trades there. A handful of very large prints
  then dominated the curve and every other bucket looked like zero. Volume
  is now summed per (day, bucket) first.
* Volatility and spread were averaged per quote UPDATE, so buckets with few
  updates (late session, thin book) were dominated by single large jumps.
  They are now computed per (day, bucket) first, and quotes with a zero or
  crossed touch are dropped.
* Plotting: the three curves were drawn on a shared categorical x-axis with
  different sets of buckets, so a bucket missing from the first curve was
  appended at the END of the axis by later curves (e.g. 14:50 after 16:10),
  making the lines zigzag backwards. All curves are now reindexed onto one
  sorted grid and drawn against numeric positions.
* Each bucket carries the number of days contributing to it. Buckets with
  coverage below `min_day_frac` of days (e.g. buckets only reached on a few
  late-closing days) are drawn hollow, so a thinly supported spike is
  visibly distinguished from a robust one.
"""

import numpy as np
import pandas as pd

# Must match data_io.TZ_LOCAL / bar_construction.TZ_LOCAL.
TZ_LOCAL = "Africa/Johannesburg"  # SAST = UTC+2, no daylight saving


def _tod(ts, bucket):
    """SAST time-of-day bucket label 'HH:MM' (sorts correctly as a string)."""
    return ts.dt.tz_convert(TZ_LOCAL).dt.floor(bucket).dt.strftime("%H:%M")


def _valid_quotes(md):
    """Drop zero, missing or crossed/locked touches."""
    ok = md["bid"].gt(0) & md["ask"].gt(md["bid"])
    if "mid" in md:
        ok &= np.isfinite(md["mid"]) & md["mid"].gt(0)
    return md[ok]


def _normalise_and_average(table):
    """table: rows = days, columns = buckets (NaN where a day has no value).
    Divide each day by its own mean across buckets, then average across days.
    Returns DataFrame indexed by bucket with columns mean, n_days."""
    norm = table.div(table.mean(axis=1), axis=0)
    out = pd.DataFrame({"mean": norm.mean(axis=0), "n_days": table.notna().sum(axis=0)})
    return out.sort_index()


def compute_intraday_volume_curve(trades, ric, bucket="10min"):
    tx = trades[trades["ric"] == ric]
    tx = tx.assign(tod=_tod(tx["ts"], bucket))
    vol = tx.groupby(["date", "tod"])["volume"].sum().unstack("tod")
    # A bucket inside the day's traded span with no trades is genuinely zero
    # volume; a bucket after the day's last trade is "no trading", also zero.
    curve = _normalise_and_average(vol.fillna(0.0))
    curve["n_days"] = vol.notna().sum(axis=0)          # days with any trades
    return curve


def compute_intraday_volatility_curve(md, ric, bucket="10min"):
    q = _valid_quotes(md[md["ric"] == ric]).sort_values("ts", kind="mergesort")
    q = q.assign(tod=_tod(q["ts"], bucket),
                 r=np.log(q["mid"]).groupby(q["date"]).diff())
    rv = (q.groupby(["date", "tod"])["r"]
          .apply(lambda r: np.sqrt(np.nansum(r.to_numpy() ** 2)))
          .unstack("tod"))
    return _normalise_and_average(rv)


def compute_intraday_spread_curve(md, ric, bucket="10min"):
    q = _valid_quotes(md[md["ric"] == ric])
    q = q.assign(tod=_tod(q["ts"], bucket))
    sp = q.groupby(["date", "tod"])["rel_spread"].mean().unstack("tod")
    return _normalise_and_average(sp)


def plot_seasonality_curves(trades, md, ric, axes, bucket="10min", min_day_frac=0.5,
                            label_every=3):
    """axes: a length-3 iterable of Axes (volume, volatility, spread).
    Returns (vol_curve, vola_curve, spread_curve), each a DataFrame with
    columns mean and n_days, indexed by 'HH:MM' bucket."""
    curves = [
        compute_intraday_volume_curve(trades, ric, bucket),
        compute_intraday_volatility_curve(md, ric, bucket),
        compute_intraday_spread_curve(md, ric, bucket),
    ]
    titles = ["Traded volume", "Mid-price realised volatility", "Relative quoted spread"]
    n_days_total = [trades.loc[trades["ric"] == ric, "date"].nunique(),
                    md.loc[md["ric"] == ric, "date"].nunique(),
                    md.loc[md["ric"] == ric, "date"].nunique()]

    grid = sorted(set().union(*[c.index for c in curves]))
    x = np.arange(len(grid))

    for ax, curve, title, n_tot in zip(axes, curves, titles, n_days_total):
        c = curve.reindex(grid)
        robust = (c["n_days"] >= min_day_frac * n_tot).to_numpy()
        y = c["mean"].to_numpy()
        ax.plot(x, y, color="tab:blue", linewidth=1)
        ax.plot(x[robust], y[robust], "o", color="tab:blue", markersize=3)
        if (~robust & np.isfinite(y)).any():
            ax.plot(x[~robust], y[~robust], "o", markersize=4, markerfacecolor="white",
                    markeredgecolor="tab:blue",
                    label=f"< {min_day_frac:.0%} of days")
            ax.legend(loc="upper left", fontsize=7)
        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
        ax.set_title(f"{ric}: {title} (1 = day's average bucket)")

    last = list(axes)[-1]
    last.set_xticks(x[::label_every])
    last.set_xticklabels(grid[::label_every], rotation=90)
    last.set_xlabel("Time of day (SAST, bucket start)")
    return tuple(curves)
