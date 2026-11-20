"""Simulation of a multivariate Hawkes process with exponential kernels.

The order flow of the synthetic market is driven by the same model that Part
III estimates: an M-variate Hawkes process with intensity

.. math::

    \\lambda^m(t) = s(t)\\,\\mu^m + \\sum_{n=1}^{M} \\sum_{t^n_k < t}
        \\alpha^{mn} e^{-\\beta^{mn}(t - t^n_k)},

where :math:`s(t)` is a deterministic intraday seasonality factor (U-shape)
and :math:`\\Gamma^{mn} = \\alpha^{mn}/\\beta^{mn}` is the branching ratio.
Events are drawn by Ogata's (1981) thinning algorithm: between events every
excitation term decays, so the intensity at the current time (with the
seasonal factor at its maximum) bounds the intensity until the next event.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _thin(mu, alpha, beta, T, s_grid, dt_grid, seed):
    """Ogata thinning. Returns event times and types (0-based) on [0, T)."""
    np.random.seed(seed)
    M = mu.shape[0]
    R = np.zeros((M, M))               # R[m, n]: current excitation of m from source n
    s_max = s_grid.max()
    mu_tot = mu.sum()
    cap = 4_000_000
    times = np.empty(cap)
    types = np.empty(cap, dtype=np.int64)
    n = 0
    t = 0.0
    while True:
        lam_bar = s_max * mu_tot + R.sum()
        w = -np.log(np.random.random()) / lam_bar
        decay = np.exp(-beta * w)
        R *= decay
        t += w
        if t >= T or n >= cap:
            break
        s = s_grid[min(int(t / dt_grid), s_grid.shape[0] - 1)]
        lam = s * mu + R.sum(axis=1)
        u = np.random.random() * lam_bar
        if u > lam.sum():
            continue                   # rejected candidate
        # choose the type in proportion to its intensity
        acc = 0.0
        k = M - 1
        for m in range(M):
            acc += lam[m]
            if u <= acc:
                k = m
                break
        times[n] = t
        types[n] = k
        n += 1
        R[:, k] += alpha[:, k]
    return times[:n], types[:n]


def seasonality(T: float, n: int = 2000, open_boost: float = 1.5, close_boost: float = 1.0,
                open_scale: float = 2400.0, close_scale: float = 1800.0) -> tuple[np.ndarray, float]:
    """U-shaped intraday activity factor on a grid over [0, T] (seconds).

    .. math:: s(t) = 1 + a_o e^{-t/\\tau_o} + a_c e^{-(T - t)/\\tau_c}

    normalised to mean one, so ``mu`` keeps its meaning as the average
    baseline rate.
    """
    t = np.linspace(0.0, T, n)
    s = 1.0 + open_boost * np.exp(-t / open_scale) + close_boost * np.exp(-(T - t) / close_scale)
    s /= s.mean()
    return s, T / (n - 1)


def simulate(mu, alpha, beta, T, seed=0, seasonal=True):
    """Simulate one session of length ``T`` seconds.

    Parameters
    ----------
    mu : (M,) baseline intensities (events per second, before seasonality)
    alpha, beta : (M, M) excitation jumps and decay rates; ``alpha[m, n]`` is
        the jump in the intensity of type m caused by an event of type n.
    T : session length in seconds.
    seed : integer seed (the simulation is fully deterministic given it).

    Returns
    -------
    times : (N,) increasing event times in seconds
    types : (N,) event types, 0-based
    """
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float)
    if seasonal:
        s, dt = seasonality(T)
    else:
        s, dt = np.ones(2), T
    rho = np.max(np.abs(np.linalg.eigvals(alpha / beta)))
    if rho >= 1:
        raise ValueError(f"Branching matrix has spectral radius {rho:.3f} >= 1 (explosive process)")
    return _thin(mu, alpha, beta, float(T), s, float(dt), int(seed))
