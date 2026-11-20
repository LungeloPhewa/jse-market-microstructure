"""
hawkes.py -- maximum-likelihood calibration of an M-variate Hawkes process
with exponential kernels (Jericevich, Chang & Gebbie 2021, eq. 1-2 and
Appendix B):

    lambda^m(t) = mu^m + sum_n sum_{t_k^n < t} alpha^{mn} exp(-beta^{mn} (t - t_k^n))

alpha^{mn} / beta^{mn} is the branching ratio Gamma^{mn}: the expected number
of type-m events triggered by one type-n event. ln(2) / beta^{mn} is the
half-life of that effect.

The log-likelihood separates by target dimension m, so each m is fitted on
its own (1 + 2M parameters). The intensity is evaluated with the Markov
recursion of Appendix B (eq. B.4), one pass over the merged event stream,
with exact analytic gradients. Optimisation is L-BFGS-B on the logarithms of
the parameters.

Validation follows section 3.4: generalised residuals (the compensator
between consecutive type-m events) are Exp(1) and independent under a
correct model; they are tested with Kolmogorov-Smirnov and Ljung-Box.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import optimize, stats


@njit(cache=True)
def _loglik_target(times, types, m, T, mu, alpha, beta):
    """Log-likelihood of target dimension m and its gradient with respect to
    (mu, alpha[0..M-1], beta[0..M-1])."""
    M = alpha.shape[0]
    R = np.zeros(M)          # sum_k exp(-beta (t - t_k))         for sources n
    S = np.zeros(M)          # sum_k (t - t_k) exp(-beta (t - t_k))
    g_mu = 0.0
    g_a = np.zeros(M)
    g_b = np.zeros(M)
    ll = 0.0
    t_prev = 0.0
    for i in range(times.shape[0]):
        t = times[i]
        dt = t - t_prev
        if dt > 0.0:
            for n in range(M):
                e = np.exp(-beta[n] * dt)
                S[n] = e * (S[n] + dt * R[n])
                R[n] = e * R[n]
        t_prev = t
        if types[i] == m:
            lam = mu
            for n in range(M):
                lam += alpha[n] * R[n]
            ll += np.log(lam)
            inv = 1.0 / lam
            g_mu += inv
            for n in range(M):
                g_a[n] += R[n] * inv
                g_b[n] -= alpha[n] * S[n] * inv
        R[types[i]] += 1.0
    # compensator: mu T + sum_n alpha/beta sum_k (1 - exp(-beta (T - t_k)))
    ll -= mu * T
    g_mu -= T
    A = np.zeros(M)          # sum_k (1 - exp(-beta (T - t_k)))
    B = np.zeros(M)          # sum_k (T - t_k) exp(-beta (T - t_k))
    for i in range(times.shape[0]):
        n = types[i]
        u = T - times[i]
        e = np.exp(-beta[n] * u)
        A[n] += 1.0 - e
        B[n] += u * e
    for n in range(M):
        ll -= alpha[n] / beta[n] * A[n]
        g_a[n] -= A[n] / beta[n]
        g_b[n] -= -alpha[n] / beta[n] ** 2 * A[n] + alpha[n] / beta[n] * B[n]
    return ll, g_mu, g_a, g_b


@njit(cache=True)
def _compensator_at_events(times, types, m, mu, alpha, beta):
    """Lambda^m(t) evaluated at every type-m event time."""
    M = alpha.shape[0]
    R = np.zeros(M)
    N = np.zeros(M)
    out = []
    t_prev = 0.0
    for i in range(times.shape[0]):
        t = times[i]
        dt = t - t_prev
        if dt > 0.0:
            for n in range(M):
                R[n] *= np.exp(-beta[n] * dt)
        t_prev = t
        if types[i] == m:
            lam_int = mu * t
            for n in range(M):
                lam_int += alpha[n] / beta[n] * (N[n] - R[n])
            out.append(lam_int)
        R[types[i]] += 1.0
        N[types[i]] += 1.0
    return np.array(out)


def fit_target(times, types, m, T, M, starts=(0.1, 1.0, 10.0), beta_bounds=(1e-3, 1e3)):
    """Fit (mu^m, alpha^{m.}, beta^{m.}) by maximum likelihood. Several
    starting decay rates are tried and the best optimum kept."""
    n_m = (types == m).sum()
    best = None
    lb, ub = np.log(beta_bounds[0]), np.log(beta_bounds[1])
    # upper bounds keep exp() finite; they are far above any fitted value
    bounds = [(np.log(1e-10), np.log(1e4))] + [(np.log(1e-10), np.log(1e5))] * M + [(lb, ub)] * M

    def f(x):
        mu, a, b = np.exp(x[0]), np.exp(x[1:M + 1]), np.exp(x[M + 1:])
        ll, gmu, ga, gb = _loglik_target(times, types, m, T, mu, a, b)
        if not np.isfinite(ll):
            return 1e300, np.zeros_like(x)
        grad = np.concatenate([[gmu * mu], ga * a, gb * b])
        return -ll, -grad

    # Starting values: half of the target's events baseline, half triggered,
    # spread evenly over the sources. Scaling the starting branching ratios
    # by the event counts matters when the counts differ by orders of
    # magnitude (AGLJ.J: ~600 market orders against ~100,000 passive orders
    # a day); equal starting values push every alpha of a rare target to its
    # lower bound, where the log-parameter gradient vanishes.
    n = np.maximum(np.bincount(types, minlength=M), 1).astype(float)
    G0 = np.minimum(0.5 * max(n_m, 1) / (M * n), 0.5)
    for b0 in starts:
        x0 = np.concatenate([[np.log(max(n_m, 1) / T * 0.5)], np.log(G0 * b0), np.full(M, np.log(b0))])
        r = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000, "maxfun": 5000})
        if best is None or r.fun < best.fun:
            best = r
    # if the best run stopped early (line search could not improve further,
    # or the iteration limit), restart it from where it stopped
    for _ in range(3):
        if best.success:
            break
        r = optimize.minimize(f, best.x, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000, "maxfun": 5000})
        if r.fun <= best.fun:
            best = r
    x = best.x
    return {"mu": float(np.exp(x[0])), "alpha": np.exp(x[1:M + 1]), "beta": np.exp(x[M + 1:]),
            "loglik": float(-best.fun), "converged": bool(best.success), "message": str(best.message)}


def fit_hawkes(times, types, T, M, **kw):
    """Fit all M targets. times in seconds from the start of the window
    (strictly increasing), types in 0..M-1."""
    times = np.asarray(times, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    fits = [fit_target(times, types, m, T, M, **kw) for m in range(M)]
    mu = np.array([f["mu"] for f in fits])
    alpha = np.vstack([f["alpha"] for f in fits])          # alpha[m, n]: n -> m
    beta = np.vstack([f["beta"] for f in fits])
    return {"mu": mu, "alpha": alpha, "beta": beta, "branching": alpha / beta,
            "half_life": np.log(2) / beta, "loglik": sum(f["loglik"] for f in fits),
            "converged": [f["converged"] for f in fits], "counts": np.bincount(types, minlength=M),
            "T": T}


def poisson_loglik(types, T, M):
    n = np.bincount(types, minlength=M).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(n * np.log(n / T)) - n.sum())


def ljung_box(x, lags=10):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    denom = np.sum(x ** 2)
    rho = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(Q), float(stats.chi2.sf(Q, lags))


def residuals(times, types, fit, M):
    """Generalised residuals per dimension (eq. 5) and their KS and Ljung-Box
    p-values."""
    times = np.asarray(times, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    res, tests = {}, []
    for m in range(M):
        lam = _compensator_at_events(times, types, m, fit["mu"][m], fit["alpha"][m], fit["beta"][m])
        e = np.diff(lam)
        res[m] = e
        ks = stats.kstest(e, "expon").pvalue if len(e) > 5 else np.nan
        lb = ljung_box(e)[1] if len(e) > 20 else np.nan
        tests.append({"event": m + 1, "n": len(e), "mean": float(e.mean()) if len(e) else np.nan,
                      "KS p": float(ks), "Ljung-Box p": lb})
    return res, tests
