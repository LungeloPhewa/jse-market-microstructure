"""
a3_model.py -- turn identified events into point-process input and fit the
10-variate Hawkes model for one security-day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hawkes as hk

TZ_LOCAL = "Africa/Johannesburg"
M = 10


def continuous_window(markers, ric, day):
    """Start and end of continuous trading (first TRD marker, closing-auction
    marker) for one security-day; the intraday auction, if any, is returned
    as a gap to exclude."""
    mk = markers[(markers["ric"] == ric) & (markers["date"] == day)].sort_values("ts")
    start = mk.loc[mk["phase"] == "continuous", "ts"].min()
    end = mk.loc[mk["phase"] == "closing auction", "ts"].min()
    gaps = []
    for _, r in mk[mk["phase"] == "intraday auction"].iterrows():
        resume = mk[(mk["ts"] > r["ts"]) & (mk["phase"] == "continuous")]["ts"].min()
        gaps.append((r["ts"], resume))
    return start, end, gaps


def point_process(events, start, end, gaps=(), seed=0):
    """Event times in seconds from `start`, made strictly increasing.

    The feeds are stamped to the millisecond and several events often share
    one millisecond (a book update can change several price levels). A
    Hawkes process has no simultaneous events, so each time gets a uniform
    random offset within its millisecond (seeded, reproducible). Time spent
    in an intraday auction is removed from the clock."""
    ev = events[(events["ts"] >= start) & (events["ts"] < end)].copy()
    t = (ev["ts"] - start).dt.total_seconds().to_numpy()
    shift = np.zeros_like(t)
    for g0, g1 in gaps:
        a, b = (g0 - start).total_seconds(), (g1 - start).total_seconds()
        t = np.where((t >= a) & (t < b), np.nan, t)
        shift += np.where(t >= b, b - a, 0.0)
    t = t - shift
    keep = ~np.isnan(t)
    rng = np.random.default_rng(seed)
    t = np.floor(t[keep] * 1000) / 1000 + rng.uniform(0, 1e-3, keep.sum())
    types = ev["event"].to_numpy()[keep] - 1
    o = np.argsort(t, kind="mergesort")
    T = (end - start).total_seconds() - sum((b - a).total_seconds() for a, b in gaps)
    return t[o], types[o].astype(np.int64), T


def fit_day(events, markers, ric, day, seed=0):
    start, end, gaps = continuous_window(markers, ric, day)
    t, k, T = point_process(events[(events["ric"] == ric) & (events["date"] == day)], start, end, gaps, seed)
    fit = hk.fit_hawkes(t, k, T, M)
    _, tests = hk.residuals(t, k, fit, M)
    fit.update(ric=ric, day=day, n_events=len(t), tests=pd.DataFrame(tests),
               poisson_loglik=hk.poisson_loglik(k, T, M), times=t, types=k)
    return fit
