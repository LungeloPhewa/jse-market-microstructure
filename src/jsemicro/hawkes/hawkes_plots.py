"""
hawkes_plots.py -- visualise a fitted 10-variate Hawkes model in the style of
Jericevich, Chang & Gebbie (2021), Figures C.7-C.8.

Bubble plot for a target event type m: one marker per source type n, at
    x = half-life of the effect n -> m, ln(2) / beta^{mn}   (seconds)
    y = branching ratio Gamma^{mn} = alpha^{mn} / beta^{mn}
        (expected number of type-m events triggered by one type-n event)
with marker area proportional to the number of type-n events. The x-axis is
logarithmic because fitted half-lives run from one millisecond to minutes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .order_events import EVENT_NAMES

# Colours grouped by order type (MOs red, aggressive LOs blue, passive LOs
# green, aggressive cancels purple, passive cancels orange/brown), with the
# bid (buy) side darker; marker shape repeats the side (circle bid/buy,
# square ask/sell) so the plots do not rely on colour alone.
COLOURS = {1: "#d62728", 2: "#ff9896", 3: "#1f4e9c", 4: "#6fa8dc", 5: "#2a7d2e", 6: "#93c47d",
           7: "#6a3d9a", 8: "#c2a5cf", 9: "#a0522d", 10: "#f4a460"}
MARKERS = {k: ("o" if k % 2 == 1 else "s") for k in range(1, 11)}
RESOLUTION = 1e-3      # timestamps are to the millisecond


def bubble(ax, fit, target, size_scale=1500.0, min_branching=0.0, label_top=3):
    """Bubble plot of the excitation of event type `target` (1..10) by every
    source type."""
    m = target - 1
    hl = fit["half_life"][m]
    br = fit["branching"][m]
    counts = fit["counts"].astype(float)
    area = size_scale * counts / counts.max() + 15
    ax.axvspan(RESOLUTION * 0.3, RESOLUTION * 1.5, color="0.92", zorder=0)
    for n in range(10):
        if br[n] < min_branching:
            continue
        ax.scatter(hl[n], br[n], s=area[n], color=COLOURS[n + 1], marker=MARKERS[n + 1],
                   edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3, label=EVENT_NAMES[n + 1])
    for n in np.argsort(br)[::-1][:label_top]:
        ax.annotate(str(n + 1), (hl[n], br[n]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(RESOLUTION * 0.3, 2e3)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Excitation of {target}: {EVENT_NAMES[target]}", fontsize=9)
    ax.set_xlabel("Half-life of effect (s, log scale)", fontsize=8)
    ax.set_ylabel("Branching ratio\n(events triggered per source event)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def bubble_grid(fit, targets, ncols=2, title=None, size_scale=1500.0):
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, targets):
        bubble(ax, fit, k, size_scale=size_scale)
    for ax in axes[len(targets):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker=MARKERS[k], color="w", markerfacecolor=COLOURS[k],
                          markeredgecolor="black", markersize=8, label=f"{k}: {EVENT_NAMES[k]}")
               for k in range(1, 11)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig


def branching_heatmap(ax, fit, title=None, vmax=None):
    G = fit["branching"]
    vmax = vmax or np.quantile(G, 0.98)
    im = ax.imshow(G, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if G[i, j] > 0.6 * vmax else "black")
    ax.set_xticks(range(10), [str(k) for k in range(1, 11)], fontsize=7)
    ax.set_yticks(range(10), [f"{k} {EVENT_NAMES[k]}" for k in range(1, 11)], fontsize=7)
    ax.set_xlabel("Source event n (excites)", fontsize=8)
    ax.set_ylabel("Target event m (excited)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def residual_qq(ax, res, title=None):
    for m, e in res.items():
        if len(e) < 5:
            continue
        q = np.sort(e)
        p = (np.arange(1, len(q) + 1) - 0.5) / len(q)
        ax.plot(stats.expon.ppf(p), q, ".", markersize=2, color=COLOURS[m + 1], label=str(m + 1))
    lim = 10
    ax.plot([0, lim], [0, lim], "k-", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Exponential(1) quantiles", fontsize=8)
    ax.set_ylabel("Generalised residual quantiles", fontsize=8)
    ax.legend(fontsize=6, ncol=2, markerscale=3, title="event", title_fontsize=6)
    if title:
        ax.set_title(title, fontsize=9)
