import numpy as np
import pandas as pd
import pytest

from investor_toolkit import backtest as bt
from investor_toolkit.diversification import portfolio_returns


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    index = pd.date_range("2000-01-31", periods=120, freq="ME")
    data = {
        "eq": rng.normal(0.007, 0.045, 120),
        "bd": rng.normal(0.002, 0.012, 120),
        "cash": np.full(120, 0.001),
    }
    return pd.DataFrame(data, index=index)


@pytest.mark.validation
def test_buy_and_hold_matches_analytic_wealth(returns):
    weights = {"eq": 0.6, "bd": 0.4}
    result = bt.run_backtest(
        returns, bt.FixedWeights(weights), rebalance=bt.Rebalance(every=None), cost_bps=0
    )
    expected = sum(w * (1 + returns[a]).prod() for a, w in weights.items())
    assert result.wealth.iloc[-1] == pytest.approx(expected)


@pytest.mark.validation
def test_monthly_rebalancing_matches_constant_mix_returns(returns):
    weights = {"eq": 0.6, "bd": 0.4}
    result = bt.run_backtest(returns, bt.FixedWeights(weights), cost_bps=0)
    np.testing.assert_allclose(result.returns, portfolio_returns(returns, weights))
    np.testing.assert_allclose(result.weights["eq"], 0.6)


@pytest.mark.validation
def test_transaction_cost_of_initial_purchase():
    index = pd.date_range("2000-01-31", periods=2, freq="ME")
    flat = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]}, index=index)
    result = bt.run_backtest(flat, bt.FixedWeights({"a": 0.5, "b": 0.5}), cost_bps=50)
    # Buying 1.0 of assets costs 0.5 % once; nothing is traded afterwards.
    assert result.wealth.iloc[-1] == pytest.approx(1 - 0.005)
    assert result.costs.iloc[1] == pytest.approx(0.0)


@pytest.mark.validation
def test_contributions_with_zero_returns_accumulate_exactly():
    index = pd.date_range("2000-01-31", periods=24, freq="ME")
    zero = pd.DataFrame({"a": 0.0, "b": 0.0}, index=index)
    result = bt.run_backtest(
        zero, bt.FixedWeights({"a": 0.7, "b": 0.3}), initial=0.0, contributions=100.0, cost_bps=0
    )
    assert result.wealth.iloc[-1] == pytest.approx(2400.0)
    np.testing.assert_allclose(result.returns, 0.0)
    assert result.irr() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.validation
def test_time_weighted_return_ignores_contributions_and_irr_matches_constant_rate():
    index = pd.date_range("2000-01-31", periods=36, freq="ME")
    constant = pd.DataFrame({"a": 0.01}, index=index)
    strategy = bt.FixedWeights({"a": 1.0})
    lump = bt.run_backtest(constant, strategy, cost_bps=0)
    saver = bt.run_backtest(constant, strategy, cost_bps=0, contributions=50.0)
    np.testing.assert_allclose(lump.returns, saver.returns)
    assert saver.irr() == pytest.approx(1.01**12 - 1, rel=1e-8)


def test_no_look_ahead(returns):
    strategy = bt.TrendFollowing({"eq": 0.6, "bd": 0.4}, cash="cash")
    base = bt.run_backtest(returns, strategy)
    shocked = returns.copy()
    shocked.iloc[80:, 0] = -0.3
    altered = bt.run_backtest(shocked, strategy)
    # Decisions up to and including period 80 only use data before period 80.
    pos = returns.index.get_loc(returns.index[80]) - strategy.lookback
    pd.testing.assert_frame_equal(base.weights.iloc[: pos + 1], altered.weights.iloc[: pos + 1])
    assert not base.weights.iloc[pos + 2 :].equals(altered.weights.iloc[pos + 2 :])


def test_trend_following_moves_to_cash_in_downtrend():
    index = pd.date_range("2000-01-31", periods=24, freq="ME")
    falling = pd.DataFrame({"eq": -0.02, "cash": 0.001}, index=index)
    result = bt.run_backtest(falling, bt.TrendFollowing({"eq": 1.0}, cash="cash"))
    np.testing.assert_allclose(result.weights["cash"], 1.0)


def test_band_rebalancing_trades_less_than_monthly(returns):
    strategy = bt.FixedWeights({"eq": 0.6, "bd": 0.4})
    monthly = bt.run_backtest(returns, strategy)
    banded = bt.run_backtest(returns, strategy, rebalance=bt.Rebalance(every=None, band=0.05))
    assert banded.turnover.sum() < monthly.turnover.sum()
    assert np.all(np.abs(banded.weights["eq"] - 0.6) <= 0.05 + 0.03)


def test_dynamic_strategies_produce_valid_weights(returns):
    assets = ["eq", "bd"]
    for strategy in [
        bt.InverseVolatility(assets),
        bt.RiskParity(assets),
        bt.MinimumVariance(assets),
    ]:
        result = bt.run_backtest(returns, strategy)
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
        assert (result.weights >= -1e-12).all().all()
        assert len(result.returns) == len(returns) - strategy.lookback


def test_compare_returns_one_row_per_strategy(returns):
    results = [
        bt.run_backtest(returns, bt.FixedWeights({"eq": 1.0}), name="equity"),
        bt.run_backtest(returns, bt.FixedWeights({"bd": 1.0}), name="bonds"),
    ]
    table = bt.compare(results)
    assert list(table.index) == ["equity", "bonds"]


@pytest.mark.validation
def test_sharpe_ratio_standard_error_matches_simulation():
    rng = np.random.default_rng(1)
    n, sr = 120, 0.2
    estimates = [
        (lambda x: x.mean() / x.std(ddof=1))(rng.normal(sr, 1.0, n)) for _ in range(20_000)
    ]
    assert np.std(estimates) == pytest.approx(bt.sharpe_ratio_std(sr, n), rel=0.03)


@pytest.mark.validation
def test_expected_max_sharpe_of_noise_strategies():
    rng = np.random.default_rng(2)
    n, n_trials, reps = 120, 50, 400
    maxima = []
    for _ in range(reps):
        x = rng.normal(0, 1, (n_trials, n))
        maxima.append(np.max(x.mean(axis=1) / x.std(axis=1, ddof=1)))
    expected = bt.expected_max_sharpe(n_trials, bt.sharpe_ratio_std(0.0, n))
    assert np.mean(maxima) == pytest.approx(expected, rel=0.05)


@pytest.mark.validation
def test_probabilistic_sharpe_ratio_is_calibrated_under_null():
    rng = np.random.default_rng(3)
    psr = [bt.probabilistic_sharpe_ratio(rng.normal(0, 1, 240)) for _ in range(2000)]
    # Under a true Sharpe ratio of zero the PSR is uniformly distributed.
    assert np.mean(np.array(psr) > 0.95) == pytest.approx(0.05, abs=0.015)


def test_deflated_sharpe_is_lower_than_probabilistic_sharpe():
    r = np.random.default_rng(4).normal(0.01, 0.04, 240)
    psr = bt.probabilistic_sharpe_ratio(r)
    dsr = bt.deflated_sharpe_ratio(r, n_trials=20, trial_sr_std=0.05)
    assert dsr < psr


def test_bootstrap_difference_of_identical_series_is_zero(returns):
    out = bt.bootstrap_difference(returns["eq"], returns["eq"], n_boot=200)
    assert out["difference"] == 0.0
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(0.0)


@pytest.mark.validation
def test_lump_sum_vs_dca_closed_form():
    index = pd.date_range("2000-01-31", periods=30, freq="ME")
    r, months, horizon = 0.01, 6, 12
    table = bt.lump_sum_vs_dca(
        pd.Series(r, index=index), pd.Series(0.0, index=index), months, horizon
    )
    dca = sum((1 / months) * (1 + r) ** (horizon - k) for k in range(months))
    assert table["dca"].iloc[0] == pytest.approx(dca)
    assert table["lump_sum"].iloc[0] == pytest.approx((1 + r) ** horizon)
    assert table["lump_sum_wins"].all()
    assert len(table) == 30 - horizon + 1
