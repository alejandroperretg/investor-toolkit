import numpy as np
import pandas as pd
import pytest

from investor_toolkit import factors as fm


@pytest.fixture
def synthetic():
    """Returns generated from known factor exposures with AR(1), heteroskedastic noise."""
    rng = np.random.default_rng(0)
    n = 600
    index = pd.date_range("1975-01-31", periods=n, freq="ME")
    f = pd.DataFrame(rng.normal(0.005, 0.04, (n, 3)), index=index, columns=["Mkt-RF", "SMB", "HML"])
    f["RF"] = 0.002
    true = {"alpha": 0.001, "Mkt-RF": 1.1, "SMB": 0.3, "HML": -0.4}
    noise = np.empty(n)
    noise[0] = 0.0
    shocks = rng.normal(0, 0.01, n) * (1 + 2 * np.abs(f["Mkt-RF"].to_numpy()) / 0.04)
    for t in range(1, n):
        noise[t] = 0.4 * noise[t - 1] + shocks[t]
    excess = true["alpha"] + f[["Mkt-RF", "SMB", "HML"]].to_numpy() @ [1.1, 0.3, -0.4] + noise
    returns = pd.Series(excess, index=index) + f["RF"]
    return returns, f, true


@pytest.mark.validation
def test_factor_regression_recovers_planted_betas(synthetic):
    returns, factors, true = synthetic
    result = fm.factor_regression(returns, factors)
    for name, value in true.items():
        assert abs(result.params[name] - value) < 3 * result.std_errors[name] + 1e-3
    # Signal variance 0.04^2 * (1.1^2 + 0.3^2 + 0.4^2) = 0.00234. Noise variance:
    # 0.01^2 * E[(1 + 2|Z|)^2] / (1 - 0.4^2) = 0.01^2 * (5 + 8 / sqrt(2 pi)) / 0.84 = 0.00098.
    assert result.r_squared == pytest.approx(0.00234 / (0.00234 + 0.00098), abs=0.05)


@pytest.mark.validation
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HAC"])
def test_standard_errors_match_statsmodels(synthetic, cov_type):
    import statsmodels.api as sm

    returns, factors, _ = synthetic
    excess = returns - factors["RF"]
    x = factors[["Mkt-RF", "SMB", "HML"]]
    ours = fm.ols(excess, x, cov_type=cov_type, max_lags=5)
    kwargs = {"maxlags": 5, "use_correction": False} if cov_type == "HAC" else {}
    reference = sm.OLS(excess, sm.add_constant(x)).fit(cov_type=cov_type, cov_kwds=kwargs or None)
    np.testing.assert_allclose(ours.params.to_numpy(), reference.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(ours.std_errors.to_numpy(), reference.bse.to_numpy(), rtol=1e-8)
    assert ours.r_squared == pytest.approx(reference.rsquared)


@pytest.mark.validation
def test_hac_confidence_intervals_have_nominal_coverage():
    """With autocorrelated regressor and errors, classical intervals are too narrow;
    Newey-West restores roughly 95 % coverage."""
    rng = np.random.default_rng(1)
    n, phi, beta, reps = 400, 0.7, 0.5, 400
    hits = {"nonrobust": 0, "HAC": 0}

    def ar1(size):
        e = rng.normal(size=size)
        out = np.empty(size)
        out[0] = e[0]
        for t in range(1, size):
            out[t] = phi * out[t - 1] + e[t]
        return out

    for _ in range(reps):
        x = ar1(n)
        y = beta * x + ar1(n)
        frame = pd.DataFrame({"x": x})
        for cov_type in hits:
            r = fm.ols(pd.Series(y), frame, cov_type=cov_type, max_lags=12)
            hits[cov_type] += abs(r.params["x"] - beta) < 1.96 * r.std_errors["x"]
    assert hits["nonrobust"] / reps < 0.85
    assert 0.88 < hits["HAC"] / reps < 0.98


@pytest.mark.validation
def test_attribution_adds_up_to_mean_excess_return(synthetic):
    returns, factors, _ = synthetic
    result = fm.factor_regression(returns, factors)
    attribution = fm.return_attribution(result, factors)
    mean_excess = (returns - factors["RF"]).mean()
    assert attribution.sum() == pytest.approx(mean_excess, rel=1e-12)


def test_rolling_betas_match_full_regression_on_last_window(synthetic):
    returns, factors, _ = synthetic
    rolling = fm.rolling_betas(returns, factors, window=120)
    last = fm.factor_regression(returns.iloc[-120:], factors.iloc[-120:])
    np.testing.assert_allclose(rolling.iloc[-1].to_numpy(), last.params.to_numpy())
    assert len(rolling) == len(returns) - 120 + 1


def test_eur_to_usd_conversion():
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    fx = pd.Series([1.10, 1.21, 1.21], index=index)
    r = pd.Series([0.0, 0.0, 0.05], index=index)
    usd = fm.eur_to_usd_returns(r, fx_usd_per_eur=fx)
    assert np.isnan(usd.iloc[0])
    assert usd.iloc[1] == pytest.approx(0.10)
    assert usd.iloc[2] == pytest.approx(0.05)
