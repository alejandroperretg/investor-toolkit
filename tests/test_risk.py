import numpy as np
import pandas as pd
import pytest
from scipy import stats

from investor_toolkit import risk


@pytest.fixture
def monthly():
    index = pd.date_range("2000-01-31", periods=8, freq="ME")
    # Wealth: 1.1, 0.88, 0.968, 1.1132, 1.2245, 1.1021, 1.1021, 1.2123
    return pd.Series([0.10, -0.20, 0.10, 0.15, 0.10, -0.10, 0.0, 0.10], index=index)


def test_drawdown_and_episodes(monthly):
    dd = risk.drawdown(monthly)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == pytest.approx(-0.20)
    assert risk.max_drawdown(monthly) == pytest.approx(0.20)
    episodes = risk.drawdown_episodes(monthly, top=None)
    assert len(episodes) == 2
    first = episodes.iloc[0]
    assert first["depth"] == pytest.approx(0.20)
    assert first["peak"] == pd.Timestamp("2000-01-31")
    assert first["trough"] == pd.Timestamp("2000-02-29")
    assert first["recovery"] == pd.Timestamp("2000-04-30")
    assert first["to_trough"] == 1
    assert first["under_water"] == 3
    second = episodes.iloc[1]
    assert second["depth"] == pytest.approx(0.10)
    assert pd.isna(second["recovery"])  # 1.2123 at the end is still below 1.2245


def test_drawdown_counts_initial_loss():
    returns = pd.Series([-0.1, 0.05], index=pd.date_range("2000-01-31", periods=2, freq="ME"))
    assert risk.max_drawdown(returns) == pytest.approx(0.1)
    episode = risk.drawdown_episodes(returns).iloc[0]
    assert pd.isna(episode["peak"])
    assert pd.isna(episode["recovery"])


def test_ratios_on_simple_series():
    r = np.array([0.02, -0.01, 0.03, 0.00])
    expected_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    assert risk.sharpe_ratio(r, 12) == pytest.approx(expected_sharpe)
    expected_dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(12)
    assert risk.downside_deviation(r, 12) == pytest.approx(expected_dd)
    assert risk.sortino_ratio(r, 12) == pytest.approx(r.mean() * 12 / expected_dd)


@pytest.mark.validation
def test_var_methods_agree_on_large_normal_sample():
    mu, sigma, alpha = 0.005, 0.04, 0.05
    # n = 50,000: the standard error of the empirical quantile is about 0.6 % of VaR.
    sample = np.random.default_rng(0).normal(mu, sigma, 50_000)
    exact_var = -(mu + sigma * stats.norm.ppf(alpha))
    for method in risk.VAR_METHODS:
        assert risk.value_at_risk(sample, alpha, method) == pytest.approx(exact_var, rel=0.02)
    exact_es = risk.gaussian_expected_shortfall(mu, sigma, alpha)
    assert risk.expected_shortfall(sample, alpha, "historical") == pytest.approx(exact_es, rel=0.02)


@pytest.mark.validation
def test_student_t_expected_shortfall_formula_matches_monte_carlo():
    loc, scale, df, alpha = 0.0, 0.03, 4.0, 0.01
    sample = stats.t.rvs(df, loc=loc, scale=scale, size=2_000_000, random_state=1)
    analytic = risk.student_t_expected_shortfall(loc, scale, df, alpha)
    assert risk.expected_shortfall(sample, alpha) == pytest.approx(analytic, rel=0.02)


@pytest.mark.validation
def test_cornish_fisher_reduces_to_gaussian_for_symmetric_light_tails():
    sample = np.random.default_rng(2).normal(0, 1, 500_000)
    gauss = risk.value_at_risk(sample, 0.01, "gaussian")
    assert risk.value_at_risk(sample, 0.01, "cornish_fisher") == pytest.approx(gauss, rel=0.01)


@pytest.mark.validation
def test_kupiec_test_has_correct_size_for_correct_model():
    rng = np.random.default_rng(3)
    alpha = 0.05
    rejections = [risk.kupiec_test(rng.random(1000) < alpha, alpha)[1] < 0.05 for _ in range(2000)]
    assert 0.03 < np.mean(rejections) < 0.07


def test_kupiec_and_christoffersen_detect_bad_models():
    rng = np.random.default_rng(4)
    too_many = rng.random(1000) < 0.10
    assert risk.kupiec_test(too_many, 0.05)[1] < 1e-6
    clustered = np.repeat(rng.random(100) < 0.05, 10)
    assert risk.christoffersen_test(clustered)[1] < 1e-6
    independent = rng.random(5000) < 0.05
    assert risk.christoffersen_test(independent)[1] > 0.001


def test_backtest_var_uses_only_past_data():
    returns = pd.Series(np.random.default_rng(5).normal(0, 0.01, 300))
    result = risk.backtest_var(returns, window=100)
    assert len(result) == 200
    shocked = returns.copy()
    shocked.iloc[250:] = -0.5
    shocked_result = risk.backtest_var(shocked, window=100)
    # The forecast for period 250 cannot know about the shock at 250.
    assert shocked_result["var"].iloc[150] == result["var"].iloc[150]
    assert shocked_result["exception"].iloc[150]


@pytest.mark.validation
def test_expected_max_drawdown_of_brownian_motion():
    """Simulated max drawdown converges to sqrt(pi/2) * sigma * sqrt(T).

    Discrete monitoring misses part of each peak and trough. The leading-order
    correction for each extremum is beta * sigma * sqrt(dt), with
    beta = -zeta(1/2) / sqrt(2 pi) ~= 0.5826 (Broadie, Glasserman and Kou, 1997).
    """
    rng = np.random.default_rng(6)
    sigma, horizon, steps, paths = 1.0, 1.0, 2000, 4000
    dt = horizon / steps
    log_wealth = np.cumsum(rng.normal(0, sigma * np.sqrt(dt), (paths, steps)), axis=1)
    log_wealth = np.hstack([np.zeros((paths, 1)), log_wealth])
    mdd = np.max(np.maximum.accumulate(log_wealth, axis=1) - log_wealth, axis=1)
    corrected = mdd.mean() + 2 * 0.5826 * sigma * np.sqrt(dt)
    standard_error = mdd.std(ddof=1) / np.sqrt(paths)
    exact = risk.expected_max_drawdown_brownian(sigma, horizon)
    assert abs(corrected - exact) < 3 * standard_error
    assert mdd.mean() < exact  # discrete monitoring underestimates drawdowns


def test_risk_summary_has_one_row_per_asset(monthly):
    table = risk.risk_summary(pd.DataFrame({"a": monthly, "b": monthly * 0.5}), 12)
    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "max_drawdown"] == pytest.approx(0.20)
