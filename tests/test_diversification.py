import numpy as np
import pandas as pd
import pytest
from scipy import stats

from investor_toolkit import diversification as dv


@pytest.fixture
def three_assets():
    vols = np.array([0.15, 0.05, 0.20])
    corr = np.array([[1.0, -0.2, 0.3], [-0.2, 1.0, 0.1], [0.3, 0.1, 1.0]])
    cov = np.outer(vols, vols) * corr
    mu = np.array([0.07, 0.02, 0.06])
    return mu, cov


def test_portfolio_returns_accepts_mapping_and_array():
    returns = pd.DataFrame({"a": [0.1, -0.1], "b": [0.0, 0.2]})
    expected = [0.05, 0.05]
    np.testing.assert_allclose(dv.portfolio_returns(returns, {"a": 0.5, "b": 0.5}), expected)
    np.testing.assert_allclose(dv.portfolio_returns(returns, [0.5, 0.5]), expected)


def test_risk_contributions_sum_to_one(three_assets):
    _, cov = three_assets
    rc = dv.risk_contributions([0.5, 0.3, 0.2], cov)
    assert rc.sum() == pytest.approx(1.0)


@pytest.mark.validation
def test_equal_weight_variance_matches_simulation():
    sigma, rho, n = 0.2, 0.3, 10
    cov = sigma**2 * (rho * np.ones((n, n)) + (1 - rho) * np.eye(n))
    sample = np.random.default_rng(0).multivariate_normal(np.zeros(n), cov, 200_000)
    simulated = sample.mean(axis=1).var()
    assert simulated == pytest.approx(dv.equal_weight_variance(sigma, rho, n), rel=0.01)
    # The diversification floor: rho * sigma^2.
    assert dv.equal_weight_variance(sigma, rho, 1e9) == pytest.approx(rho * sigma**2)


@pytest.mark.validation
def test_ledoit_wolf_matches_scikit_learn():
    from sklearn.covariance import ledoit_wolf as sk_ledoit_wolf

    x = np.random.default_rng(1).normal(size=(60, 15))
    cov, delta = dv.ledoit_wolf(x)
    sk_cov, sk_delta = sk_ledoit_wolf(x)
    assert delta == pytest.approx(sk_delta, rel=1e-10)
    np.testing.assert_allclose(cov, sk_cov, rtol=1e-10)


@pytest.mark.validation
def test_ledoit_wolf_beats_sample_covariance_with_few_observations():
    rng = np.random.default_rng(2)
    p, n = 20, 40
    true_cov = np.diag(np.linspace(0.5, 2.0, p))
    errors_lw, errors_sample = [], []
    for _ in range(50):
        x = rng.multivariate_normal(np.zeros(p), true_cov, n)
        errors_lw.append(np.sum((dv.ledoit_wolf(x)[0] - true_cov) ** 2))
        errors_sample.append(np.sum((np.cov(x, rowvar=False, bias=True) - true_cov) ** 2))
    assert np.mean(errors_lw) < 0.6 * np.mean(errors_sample)


@pytest.mark.validation
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.8])
def test_conditional_correlation_benchmark_matches_simulation(rho):
    cov = [[1.0, rho], [rho, 1.0]]
    x = np.random.default_rng(3).multivariate_normal([0, 0], cov, 1_000_000)
    down = x[:, 0] < np.quantile(x[:, 0], 0.2)
    simulated = np.corrcoef(x[down].T)[0, 1]
    assert simulated == pytest.approx(dv.conditional_correlation_normal(rho, 0.2), abs=0.01)
    # Conditioning on down markets lowers measured correlation under normality.
    assert dv.conditional_correlation_normal(rho, 0.2) <= rho


@pytest.mark.validation
def test_truncated_normal_variance_matches_scipy():
    z = stats.norm.ppf(0.1)
    expected = stats.truncnorm(-np.inf, z).var()
    assert dv.truncated_normal_variance(z) == pytest.approx(expected)


@pytest.mark.validation
def test_unconstrained_min_variance_matches_closed_form(three_assets):
    _, cov = three_assets
    w = dv.min_variance_weights(cov, long_only=False)
    inv = np.linalg.inv(cov)
    expected = inv @ np.ones(3) / (np.ones(3) @ inv @ np.ones(3))
    np.testing.assert_allclose(w, expected)


@pytest.mark.validation
def test_long_only_optimizer_matches_analytic_frontier_when_constraints_inactive():
    vols = np.array([0.15, 0.10, 0.12])
    corr = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.1], [0.3, 0.1, 1.0]])
    cov = np.outer(vols, vols) * corr
    mu = np.array([0.07, 0.03, 0.05])
    w_min = dv.min_variance_weights(cov, long_only=False)
    assert np.all(w_min > 0)  # constraints inactive at the minimum-variance point
    np.testing.assert_allclose(dv.min_variance_weights(cov, long_only=True), w_min, atol=1e-6)
    frontier = dv.efficient_frontier(mu, cov, n_points=10, long_only=True)
    inside = frontier[(frontier[["w0", "w1", "w2"]] > 1e-4).all(axis=1)]
    assert len(inside) >= 3
    analytic = np.sqrt(dv.frontier_variance(mu, cov, inside["mean"]))
    np.testing.assert_allclose(inside["volatility"], analytic, rtol=1e-5)


@pytest.mark.validation
def test_two_asset_frontier_closed_form():
    s1, s2, rho = 0.2, 0.1, 0.25
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    mu = np.array([0.08, 0.03])
    frontier = dv.efficient_frontier(mu, cov, n_points=5, long_only=False)
    for _, row in frontier.iterrows():
        w = (row["mean"] - mu[1]) / (mu[0] - mu[1])
        var = w**2 * s1**2 + (1 - w) ** 2 * s2**2 + 2 * w * (1 - w) * rho * s1 * s2
        assert row["volatility"] == pytest.approx(np.sqrt(var))


@pytest.mark.validation
def test_unconstrained_tangency_maximises_sharpe(three_assets):
    mu, cov = three_assets
    w = dv.max_sharpe_weights(mu, cov, risk_free=0.01, long_only=False)

    def sharpe(weights):
        return (weights @ mu - 0.01) / np.sqrt(weights @ cov @ weights)

    rng = np.random.default_rng(4)
    for _ in range(200):
        other = w + rng.normal(0, 0.05, 3)
        other /= other.sum()
        assert sharpe(other) <= sharpe(w) + 1e-12


@pytest.mark.validation
def test_risk_parity_equalises_risk_contributions(three_assets):
    _, cov = three_assets
    w = dv.risk_parity_weights(cov)
    np.testing.assert_allclose(dv.risk_contributions(w, cov), 1 / 3, atol=1e-9)
    budgets = [0.5, 0.25, 0.25]
    w_b = dv.risk_parity_weights(cov, budgets)
    np.testing.assert_allclose(dv.risk_contributions(w_b, cov), budgets, atol=1e-9)


@pytest.mark.validation
def test_risk_parity_with_uncorrelated_assets_is_inverse_volatility():
    cov = np.diag([0.2, 0.1, 0.05]) ** 2
    np.testing.assert_allclose(dv.risk_parity_weights(cov), dv.inverse_volatility_weights(cov))


def test_diversification_ratio_and_effective_number():
    cov = np.diag([0.04, 0.04])
    assert dv.diversification_ratio([0.5, 0.5], cov) == pytest.approx(np.sqrt(2))
    assert dv.effective_number([0.25] * 4) == pytest.approx(4)


def test_downside_correlations_table():
    rng = np.random.default_rng(5)
    x = rng.multivariate_normal([0, 0, 0], [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]], 5000)
    table = dv.downside_correlations(pd.DataFrame(x, columns=["mkt", "a", "b"]), "mkt")
    assert list(table.index) == ["a", "b"]
    assert abs(table.loc["a", "excess"]) < 0.06
