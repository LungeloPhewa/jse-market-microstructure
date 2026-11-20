"""
stylised_facts.py -- Part I.B: time-series stylised facts.

For a given 1-minute bar-price series (VWAP or micro-price), computes log
returns and produces: the sample distribution, a QQ-plot against the
Normal, the empirical CCDF of both tails above/below the 95th/5th
percentile, and the ACF.

Each `compute_*` / `plot_*` pair can be used independently; `run_all` is a
convenience wrapper that produces all four figures for one series in one
call (mirrors the Assignment-1 module convention of pairing a `compute_`
step with its matching `plot_` step). `summarise_returns` returns the
numbers quoted in the text.

Changes from the first version
------------------------------
* Returns are computed WITHIN each trading day. The first version took
  np.log(series).diff() over the whole sample, so each day's first return
  ran from the previous day's last minute (16:49) to the next day's first
  (09:10). Those 20 overnight returns were 8 of the 10 largest AGLJ.J VWAP
  returns, drove its excess kurtosis from about 48 to over 300, and halved
  the lag-1 autocorrelation. Within a day, a return runs from one non-empty
  1-minute bar to the next.
* Histogram: x-axis limited to the 0.1-99.9th percentiles (the few returns
  outside are counted in the panel) and a Normal density with the same mean
  and standard deviation overlaid.
* Tails: CCDF is P(|r| >= x), so the largest point is plotted rather than
  falling at zero; a Hill estimate of the tail exponent alpha is drawn as a
  reference line of slope -alpha. Short panel titles (they overlapped).
* ACF: starts at lag 1 (lag 0 is always 1 and squashed the plot), with the
  +/-1.96/sqrt(n) band and a y-range fitted to the data.
"""

import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from scipy import stats


def compute_log_returns(series):
    """1-minute log returns of a bar-price series (VWAP or micro-price),
    within each trading day only. The series index is SAST local time."""
    logp = np.log(series.astype(float))
    return logp.groupby(series.index.date).diff().dropna()


def hill_alpha(x, tail_frac=0.05):
    """Hill estimator of the power-law tail exponent from the largest
    `tail_frac` of the positive values of x."""
    x = np.sort(np.asarray(x)[np.asarray(x) > 0])[::-1]
    k = max(int(len(x) * tail_frac), 10)
    if len(x) <= k:
        return np.nan
    return float(k / np.sum(np.log(x[:k] / x[k])))


def acf(ret, max_lag=40):
    """Sample autocorrelation at lags 1..max_lag."""
    r = np.asarray(ret, dtype=float) - np.mean(ret)
    denom = np.sum(r ** 2)
    return np.array([np.sum(r[:-k] * r[k:]) / denom for k in range(1, max_lag + 1)])


def summarise_returns(ret):
    """Numbers quoted in the report for one return series."""
    a = acf(ret, 2)
    return {"n": int(len(ret)),
            "sd_bps": float(ret.std() * 1e4),
            "zero_share": float((ret == 0).mean()),
            "skew": float(stats.skew(ret)),
            "exkurt": float(stats.kurtosis(ret)),
            "jb_p": float(stats.jarque_bera(ret).pvalue),
            "alpha_right": hill_alpha(ret.to_numpy()),
            "alpha_left": hill_alpha(-ret.to_numpy()),
            "acf1": float(a[0]), "acf2": float(a[1]),
            "band": float(1.96 / np.sqrt(len(ret))),
            "max_abs_bps": float(ret.abs().max() * 1e4)}


def plot_distribution(ret, ric, label, ax):
    lo, hi = ret.quantile([0.001, 0.999])
    inside = ret[(ret >= lo) & (ret <= hi)]
    ax.hist(inside, bins=60, density=True, alpha=0.75, color="tab:blue", label="sample")
    x = np.linspace(lo, hi, 400)
    ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), color="tab:red", linewidth=1.2,
            label="Normal, same mean and s.d.")
    ax.set_xlim(lo, hi)
    n_out = len(ret) - len(inside)
    ax.text(0.98, 0.95, f"{n_out} returns outside\n0.1–99.9th pct not shown",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.3")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"{ric}: 1-min {label} return distribution")
    ax.set_xlabel("log return"); ax.set_ylabel("density")


def plot_qq(ret, ric, label, ax):
    stats.probplot(ret, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(3)
    ax.set_title(f"{ric}: 1-min {label} QQ-plot vs Normal")
    ax.set_ylabel("sample quantiles (log return)")


def plot_tails(ret, ric, label, axes):
    """axes: a length-2 iterable of Axes (right tail, left tail)."""
    q95, q05 = ret.quantile(0.95), ret.quantile(0.05)
    right_tail = ret[ret > q95].sort_values().to_numpy()
    left_tail = (-ret[ret < q05]).sort_values().to_numpy()  # mirrored, so both read as right-tail CCDFs
    alphas = [hill_alpha(ret.to_numpy()), hill_alpha(-ret.to_numpy())]

    for ax, tail, name, alpha in zip(axes, [right_tail, left_tail],
                                     ["Right tail (>95th pct)", "Left tail (<5th pct)"], alphas):
        if len(tail) > 2:
            n = len(tail)
            ccdf = (n - np.arange(n)) / n                    # P(|r| >= x) within the tail
            ax.loglog(tail, ccdf, marker=".", linestyle="none", markersize=3)
            if np.isfinite(alpha):
                x = np.array([tail[0], tail[-1]])
                ax.loglog(x, (x / tail[0]) ** (-alpha), color="tab:red", linewidth=1,
                          label=f"slope −α, α = {alpha:.1f}")
                ax.legend(loc="lower left", fontsize=7)
        # Tails often span less than one decade, which leaves a log axis with
        # no labelled tick; label 1-2-5 steps instead.
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.set_title(f"{label}: {name}", fontsize=9)
        ax.set_xlabel("|log return|"); ax.set_ylabel("CCDF (log scale)")


def plot_acf_custom(ret, ric, label, ax, max_lag=40):
    a = acf(ret, max_lag)
    band = 1.96 / np.sqrt(len(ret))
    lags = np.arange(1, max_lag + 1)
    ax.vlines(lags, 0, a, color="tab:blue", linewidth=1)
    ax.plot(lags, a, "o", color="tab:blue", markersize=3)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.fill_between([0.5, max_lag + 0.5], -band, band, color="tab:blue", alpha=0.15,
                    label=f"±1.96/√n = ±{band:.3f}")
    lim = max(0.1, 1.2 * np.abs(a).max())
    ax.set_ylim(-lim, lim); ax.set_xlim(0.5, max_lag + 0.5)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{ric}: ACF of 1-min {label} returns (lag-1 = {a[0]:.3f})")
    ax.set_xlabel("lag (bars)"); ax.set_ylabel("autocorrelation")


def run_all(series, ric, label, fig_dist, fig_qq, fig_tails, fig_acf):
    """Convenience wrapper producing all four stylised-fact figures for one
    series. Each `fig_*` argument is a (fig, ax) or (fig, axes) tuple
    already created by the caller (so the calling .qmd chunk controls
    figure sizing/layout)."""
    ret = compute_log_returns(series)
    if len(ret) < 20:
        print(f"[stylised_facts] too few points for {label}/{ric}, skipping")
        return ret

    plot_distribution(ret, ric, label, fig_dist[1])
    plot_qq(ret, ric, label, fig_qq[1])
    plot_tails(ret, ric, label, fig_tails[1])
    plot_acf_custom(ret, ric, label, fig_acf[1])
    return ret
