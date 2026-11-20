"""Statistical building blocks against known answers."""
import numpy as np
import pandas as pd

from jsemicro.eda import order_flow as of
from jsemicro.hawkes import hawkes as hk
from jsemicro.hawkes.order_events import _epoch_ns, market_orders
from jsemicro.synthetic import hawkes_sim


def test_acf_of_white_noise_is_small_and_ar1_is_phi():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    assert np.abs(of.acf(x, 20)).max() < 0.02
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = 0.6 * y[i - 1] + x[i]
    assert abs(of.acf(y, 1)[0] - 0.6) < 0.02


def test_power_law_mle_recovers_alpha():
    rng = np.random.default_rng(2)
    alpha, xmin = 2.7, 5.0
    x = xmin * (1 - rng.random(20_000)) ** (-1 / (alpha - 1))     # Pareto(alpha) by inversion
    a, n = of.power_law_mle(x, xmin)
    assert abs(a - alpha) < 0.05 and n == len(x)


def test_power_law_xmin_selection_ignores_a_non_power_law_body():
    rng = np.random.default_rng(3)
    body = rng.exponential(1.0, 20_000)
    tail = 10 * (1 - rng.random(5_000)) ** (-1 / 1.5)              # alpha = 2.5 above 10
    fit = of.fit_power_law(np.r_[body[body < 10], tail])
    assert abs(fit["alpha"] - 2.5) < 0.2 and fit["xmin"] > 5


def test_epoch_ns_is_independent_of_resolution():
    s = pd.Series(pd.to_datetime(["2013-09-03 07:00:00.001", "2013-09-03 07:00:00.011"], utc=True))
    for unit in ("ns", "us", "ms"):
        ns = _epoch_ns(s.dt.as_unit(unit))
        assert np.diff(ns)[0] == 10_000_000                         # 10 ms in nanoseconds


def test_market_orders_merge_same_sign_prints_within_10ms():
    t0 = pd.Timestamp("2013-09-03 07:00:00", tz="UTC")
    ts = [t0, t0 + pd.Timedelta("5ms"), t0 + pd.Timedelta("50ms"), t0 + pd.Timedelta("52ms")]
    trades = pd.DataFrame({"ric": "AGLJ.J", "ts": pd.Series(ts).dt.as_unit("us"),
                           "trade_sign": [1, 1, 1, -1], "volume": [100, 50, 10, 20]})
    mo = market_orders(trades)
    assert list(mo["volume"]) == [150, 10, 20]                       # 10 ms window, sign change splits
    assert list(mo["event"]) == [1, 1, 2]


def test_hawkes_mle_recovers_simulated_parameters():
    mu = np.array([0.3, 0.2])
    beta = np.array([[2.0, 1.0], [5.0, 1.0]])
    gamma = np.array([[0.4, 0.0], [0.3, 0.2]])
    times, types = hawkes_sim.simulate(mu, gamma * beta, beta, 20_000.0, seed=7, seasonal=False)
    fit = hk.fit_hawkes(times, types, 20_000.0, 2)
    assert np.allclose(fit["branching"], gamma, atol=0.06)
    assert np.allclose(fit["mu"], mu, rtol=0.15)


def test_hawkes_residuals_are_unit_exponential_under_the_true_model():
    mu, beta = np.array([0.5]), np.array([[1.5]])
    alpha = 0.5 * beta
    t, k = hawkes_sim.simulate(mu, alpha, beta, 20_000.0, seed=3, seasonal=False)
    fit = hk.fit_hawkes(t, k, 20_000.0, 1)
    _, tests = hk.residuals(t, k, fit, 1)
    assert abs(tests[0]["mean"] - 1) < 0.03
    assert tests[0]["KS p"] > 0.01


def test_simulator_rejects_explosive_branching():
    import pytest
    with pytest.raises(ValueError):
        hawkes_sim.simulate([0.1], [[1.2]], [[1.0]], 100.0)
