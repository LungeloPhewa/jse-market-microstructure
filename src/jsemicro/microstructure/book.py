"""Shared set-up for the Part II chapters of the book.

Each chapter runs in its own Jupyter kernel, so the data loading that the
original single Assignment 2 report did once is done here, once, and cached:
the loaded quotes and Lee-Ready-signed trades are pickled to
``<data root>/out/part2_data.pkl`` and reused by later chapters (and later
renders) as long as the two input files are unchanged.
"""
from __future__ import annotations

import pickle
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ..config import PATHS, RICS
from . import data_io

SAST = ZoneInfo("Africa/Johannesburg")

PLOT_STYLE = {
    # Matplotlib formats datetime axes in UTC unless told otherwise; make SAST the default.
    "timezone": "Africa/Johannesburg",
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 8,
}


def tidy_time_axis(ax, start=None, end=None, day=None):
    """SAST HH:MM tick labels, no y-offset, optional fixed window."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SAST))
    ax.set_xlabel("Time (SAST)")
    try:
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    except AttributeError:
        pass
    if start and end and day is not None:
        lo = pd.Timestamp(f"{day} {start}").tz_localize(SAST).to_pydatetime()
        hi = pd.Timestamp(f"{day} {end}").tz_localize(SAST).to_pydatetime()
        ax.set_xlim(lo, hi)


def show_table(df, index=False, floatfmt=".4g"):
    """Print a DataFrame as a Markdown table (use with ``#| output: asis``)."""
    out = df.copy()
    if index:
        out = out.reset_index()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            s = out[c]
            if s.dt.tz is not None:
                s = s.dt.tz_convert(SAST)
            out[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S")
    print(out.to_markdown(index=False, floatfmt=floatfmt))


def sast_minutes(ts):
    """Minutes after midnight, SAST, for a timestamp Series."""
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    t = ts.dt.tz_convert(SAST)
    return t.dt.hour * 60 + t.dt.minute + t.dt.second / 60


def hhmm(m):
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


def _stamp():
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in (PATHS.tx_trimmed, PATHS.md_flat))


def load_part2():
    """Quotes (L1, continuous session) and Lee-Ready-signed trades, plus the
    example day used by every single-day illustration.

    Returns a dict with keys ``md``, ``trades``, ``sign_report``,
    ``EXAMPLE_DAY``, ``EXAMPLE_DAY_STR``, ``EXAMPLE_DAY_TRADES`` and
    ``overlap_days``.
    """
    plt.rcParams.update(PLOT_STYLE)
    cache = PATHS.out_dir / "part2_data.pkl"
    if cache.exists():
        try:
            with open(cache, "rb") as fh:
                d = pickle.load(fh)
            if d.get("_stamp") == _stamp():
                return d
        except (EOFError, pickle.UnpicklingError):
            pass                                  # incomplete cache from an interrupted run: rebuild
    md = data_io.load_md(PATHS.md_flat, RICS)
    trades = data_io.load_tx(PATHS.tx_trimmed, RICS, md=md)      # Lee-Ready signs
    sign_report = trades.attrs["sign_report"]
    # EXAMPLE_DAY: a day present in BOTH feeds with the median AGLJ.J trade
    # count, derived from the data rather than hard-coded.
    overlap_days = sorted(set(trades["date"].unique()) & set(md["date"].unique()))
    daily_counts = (trades[trades["ric"] == "AGLJ.J"]
                    .groupby("date").size().reindex(overlap_days).fillna(0))
    example_day = daily_counts.sort_values(ascending=False).index[len(daily_counts) // 2]
    d = {"md": md, "trades": trades, "sign_report": sign_report,
         "EXAMPLE_DAY": example_day, "EXAMPLE_DAY_STR": str(example_day)[:10],
         "EXAMPLE_DAY_TRADES": int(daily_counts[example_day]), "overlap_days": overlap_days,
         "_stamp": _stamp()}
    PATHS.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    tmp.replace(cache)                             # atomic: never leave a half-written cache
    return d
