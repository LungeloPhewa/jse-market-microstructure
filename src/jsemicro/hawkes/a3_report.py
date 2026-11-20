"""
a3_report.py -- helpers that turn the cached pipeline outputs (out/) into the
tables and summary objects used in the Assignment 3 report.
"""
from __future__ import annotations

import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .order_events import EVENT_NAMES


def load_fits(out_dir, ric):
    fits = {}
    for f in sorted(glob.glob(str(Path(out_dir) / f"fit_{ric}_*.pkl"))):
        r = pickle.load(open(f, "rb"))
        fits[r["day"]] = r
    return fits


def median_fit(fits):
    """Element-wise median of mu, branching ratio and half-life across days,
    and mean daily event counts: a day-robust summary for bubble plots."""
    days = list(fits)
    G = np.stack([fits[d]["branching"] for d in days])
    H = np.stack([fits[d]["half_life"] for d in days])
    C = np.stack([fits[d]["counts"] for d in days])
    mu = np.stack([fits[d]["mu"] for d in days])
    return {"branching": np.median(G, axis=0), "half_life": np.median(H, axis=0),
            "counts": C.mean(axis=0), "mu": np.median(mu, axis=0), "days": len(days),
            "G_all": G, "H_all": H}


def event_counts(events, day=None):
    ev = events if day is None else events[events["date"] == day]
    c = ev.groupby("event").size().reindex(range(1, 11), fill_value=0)
    return c


def parameter_table(fit):
    """mu, own-type branching ratio, total branching ratio received and
    exerted, for each event type."""
    G = fit["branching"]
    return pd.DataFrame({
        "Event": [f"{k}: {EVENT_NAMES[k]}" for k in range(1, 11)],
        "Count": fit["counts"],
        "Baseline mu (per s)": fit["mu"],
        "Self-excitation": np.diag(G),
        "Triggered by others (row sum)": G.sum(axis=1) - np.diag(G),
        "Triggers in others (column sum)": G.sum(axis=0) - np.diag(G),
    })


def top_effects(fit, k=10, min_count=0):
    """The k largest branching ratios with their half-lives."""
    G, H, n = fit["branching"], fit["half_life"], fit["counts"]
    rows = [{"source": s + 1, "target": t + 1, "Gamma": G[t, s], "half_life": H[t, s]}
            for t in range(10) for s in range(10) if n[s] >= min_count]
    out = pd.DataFrame(rows).sort_values("Gamma", ascending=False).head(k)
    out["effect"] = [f"{EVENT_NAMES[s]} → {EVENT_NAMES[t]}" for s, t in zip(out["source"], out["target"])]
    return out.reset_index(drop=True)


def spectral_radius(fit):
    return float(np.max(np.abs(np.linalg.eigvals(fit["branching"]))))


def stability(fits, pairs):
    """Branching ratio and half-life of chosen (source, target) effects on
    every day."""
    rows = []
    for d, f in fits.items():
        for s, t in pairs:
            rows.append({"date": d, "effect": f"{s}→{t}", "Gamma": f["branching"][t - 1, s - 1],
                         "half_life": f["half_life"][t - 1, s - 1]})
    return pd.DataFrame(rows)
