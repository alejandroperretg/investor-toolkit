import numpy as np
import pandas as pd
import pytest
from scipy import stats

from investor_toolkit import returns as rt


@pytest.fixture
def prices():
    index = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, index.size))), index=index)


def test_log_and_simple_returns_are_consistent(prices):
    simple = rt.simple_returns(prices)
    log = rt.log_returns(prices)
    np.testing.assert_allclose(rt.to_log(simple), log)
    np.testing.assert_allclose(rt.to_simple(log), simple)
    # Log returns add up over time; simple returns compound.
    total = prices.iloc[-1] / prices.iloc[0]
    assert np.exp(log.sum()) == pytest.approx(total)
    assert rt.wealth_index(simple).iloc[-1] == pytest.approx(total)


def test_annualized_return_of_constant_growth():
    assert rt.annualized_return(np.full(36, 0.01), 12) == pytest.approx(1.01**12 - 1)


def test_infer_periods_per_year():
    assert rt.infer_periods_per_year(pd.bdate_range("2020-01-01", periods=100)) == 252
    assert rt.infer_periods_per_year(pd.date_range("2020-01-31", periods=24, freq="ME")) == 12


def test_monthly_returns_drop_partial_final_month(prices):
    partial = prices.loc[:"2020-12-15"]
    assert rt.monthly_returns(partial).index[-1] == pd.Timestamp("2020-11-30")
    assert rt.monthly_returns(prices).index[-1] == pd.Timestamp("2020-12-31")
    assert len(rt.monthly_returns(partial, drop_partial=False)) == 11


@pytest.mark.validation
def test_student_t_fit_recovers_parameters():
    sample = stats.t.rvs(df=4, loc=0.001, scale=0.02, size=50_000, random_state=1)
    fit = rt.fit_student_t(sample)
    assert fit.params["df"] == pytest.approx(4, rel=0.1)
    assert fit.params["scale"] == pytest.approx(0.02, rel=0.03)
    table = rt.compare_fits(sample)
    assert table.loc["student_t", "aic"] < table.loc["normal", "aic"]


@pytest.mark.validation
def test_normal_sample_has_normal_moments_and_tails():
    sample = np.random.default_rng(2).normal(0, 1, 200_000)
    m = rt.moments(sample)
    assert abs(m["skew"]) < 0.03
    assert abs(m["excess_kurtosis"]) < 0.06
    tails = rt.tail_frequencies(sample, thresholds=(2.0, 3.0))
    np.testing.assert_allclose(tails["ratio"], 1.0, atol=0.08)
    assert rt.jarque_bera(sample)[1] > 0.001


@pytest.mark.validation
@pytest.mark.parametrize("nu", [3.0, 5.0])
def test_hill_estimator_recovers_student_t_tail_index(nu):
    sample = stats.t.rvs(df=nu, size=400_000, random_state=3)
    assert rt.hill_tail_index(sample, tail_fraction=0.005, side="both") == pytest.approx(
        nu, rel=0.12
    )


@pytest.mark.validation
def test_variance_ratio_matches_ar1_theory():
    # For AR(1) with coefficient phi, VR(2) = 1 + phi.
    rng = np.random.default_rng(4)
    phi, n = 0.3, 200_000
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    result = rt.variance_ratio_test(x, q=2)
    assert result.variance_ratio == pytest.approx(1 + phi, abs=0.01)
    assert result.p_value < 1e-6


@pytest.mark.validation
def test_variance_ratio_test_has_correct_size_under_iid():
    rng = np.random.default_rng(5)
    rejections = [
        rt.variance_ratio_test(rng.standard_t(5, size=1000), q=4).p_value < 0.05 for _ in range(400)
    ]
    assert 0.025 < np.mean(rejections) < 0.08


@pytest.mark.validation
def test_iid_volatility_scales_with_square_root_of_horizon():
    x = np.random.default_rng(6).normal(0, 0.04, 100_000)
    table = rt.horizon_volatility(x, horizons=(1, 4, 12))
    np.testing.assert_allclose(table["ratio"], 1.0, atol=0.02)


def test_qq_points_are_sorted_and_matched():
    theoretical, sample = rt.qq_points(np.array([3.0, -1.0, 0.5, 2.0]))
    assert np.all(np.diff(sample) >= 0)
    assert np.all(np.diff(theoretical) > 0)
    assert theoretical.size == sample.size
