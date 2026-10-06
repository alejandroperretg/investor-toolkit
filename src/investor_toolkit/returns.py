"""Return conventions and return-distribution analysis.

Question answered: *what do returns actually look like?*

Conventions
-----------
* Simple return: ``R_t = P_t / P_{t-1} - 1``. Simple returns aggregate across
  assets (a portfolio return is the weighted sum of asset returns).
* Log return: ``r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)``. Log returns aggregate
  across time (the multi-period log return is the sum of one-period log returns).
* ``periods_per_year`` is 252 for daily trading data and 12 for monthly data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = np.ndarray | pd.Series

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """One-period simple returns; the first row is dropped."""
    return (prices / prices.shift(1) - 1.0).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """One-period log returns; the first row is dropped."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def to_log(simple: ArrayLike | pd.DataFrame) -> ArrayLike | pd.DataFrame:
    """Convert simple returns to log returns."""
    return np.log1p(simple)


def to_simple(log: ArrayLike | pd.DataFrame) -> ArrayLike | pd.DataFrame:
    """Convert log returns to simple returns."""
    return np.expm1(log)


def resample_prices(prices: pd.DataFrame | pd.Series, freq: str = "ME") -> pd.DataFrame:
    """Last available price in each period (``"ME"`` month end, ``"YE"`` year end)."""
    return prices.resample(freq).last()


def monthly_returns(
    prices: pd.DataFrame | pd.Series, drop_partial: bool = True
) -> pd.DataFrame | pd.Series:
    """Simple month-end to month-end returns from daily prices.

    With ``drop_partial=True`` the final month is dropped if the data end before
    that month's last business day.
    """
    returns = simple_returns(resample_prices(prices, "ME"))
    last_date = prices.index[-1]
    if drop_partial and last_date != last_date + pd.offsets.BMonthEnd(0):
        returns = returns.iloc[:-1]
    return returns


def wealth_index(returns: pd.DataFrame | pd.Series, start: float = 1.0):
    """Cumulative value of ``start`` invested, compounding simple returns."""
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the sampling frequency (252, 52, 12, 4 or 1) from a date index."""
    if len(index) < 3:
        raise ValueError("Need at least three dates to infer frequency")
    median_days = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    candidates = {252: 1.45, 52: 7.0, 12: 30.4, 4: 91.3, 1: 365.25}
    return min(candidates, key=lambda k: abs(np.log(median_days / candidates[k])))


# ---------------------------------------------------------------------------
# Annualised statistics
# ---------------------------------------------------------------------------


def annualized_return(returns: ArrayLike, periods_per_year: int) -> float:
    """Geometric average return per year (CAGR) from simple returns."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    growth = np.prod(1.0 + r)
    return float(growth ** (periods_per_year / r.size) - 1.0)


def annualized_volatility(returns: ArrayLike, periods_per_year: int) -> float:
    """Sample standard deviation scaled by ``sqrt(periods_per_year)``."""
    r = np.asarray(returns, dtype=float)
    return float(np.nanstd(r, ddof=1) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------


def moments(returns: ArrayLike) -> pd.Series:
    """Sample mean, standard deviation, skewness and excess kurtosis.

    Skewness and kurtosis use the bias-corrected estimators. A normal
    distribution has skewness 0 and excess kurtosis 0.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return pd.Series(
        {
            "n": r.size,
            "mean": r.mean(),
            "std": r.std(ddof=1),
            "skew": stats.skew(r, bias=False),
            "excess_kurtosis": stats.kurtosis(r, fisher=True, bias=False),
        }
    )


def jarque_bera(returns: ArrayLike) -> tuple[float, float]:
    """Jarque-Bera normality test. Returns ``(statistic, p_value)``."""
    r = np.asarray(returns, dtype=float)
    result = stats.jarque_bera(r[~np.isnan(r)])
    return float(result.statistic), float(result.pvalue)


@dataclass(frozen=True)
class FitResult:
    """A fitted distribution with goodness-of-fit diagnostics."""

    name: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float

    def frozen(self):
        """The fitted ``scipy.stats`` frozen distribution."""
        if self.name == "normal":
            return stats.norm(loc=self.params["loc"], scale=self.params["scale"])
        return stats.t(df=self.params["df"], loc=self.params["loc"], scale=self.params["scale"])


def _fit_result(name: str, dist, params: dict[str, float], r: np.ndarray) -> FitResult:
    frozen = dist(**params)
    loglik = float(np.sum(frozen.logpdf(r)))
    k = len(params)
    return FitResult(
        name=name,
        params=params,
        log_likelihood=loglik,
        aic=2 * k - 2 * loglik,
        bic=k * np.log(r.size) - 2 * loglik,
        ks_statistic=float(stats.kstest(r, frozen.cdf).statistic),
    )


def fit_normal(returns: ArrayLike) -> FitResult:
    """Maximum-likelihood normal fit."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return _fit_result("normal", stats.norm, {"loc": r.mean(), "scale": r.std(ddof=0)}, r)


def fit_student_t(returns: ArrayLike) -> FitResult:
    """Maximum-likelihood Student-t fit (degrees of freedom, location, scale)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    df, loc, scale = stats.t.fit(r)
    return _fit_result("student_t", stats.t, {"df": df, "loc": loc, "scale": scale}, r)


def compare_fits(returns: ArrayLike) -> pd.DataFrame:
    """Normal vs Student-t fit comparison (lower AIC/BIC is better)."""
    fits = [fit_normal(returns), fit_student_t(returns)]
    return pd.DataFrame(
        {
            f.name: {
                **f.params,
                "log_likelihood": f.log_likelihood,
                "aic": f.aic,
                "bic": f.bic,
                "ks_statistic": f.ks_statistic,
            }
            for f in fits
        }
    ).T


def tail_frequencies(returns: ArrayLike, thresholds=(2.0, 3.0, 4.0, 5.0)) -> pd.DataFrame:
    """Observed vs normal-implied counts of moves beyond k standard deviations."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    z = (r - r.mean()) / r.std(ddof=1)
    rows = {}
    for k in thresholds:
        expected = r.size * 2.0 * stats.norm.sf(k)
        observed = int(np.sum(np.abs(z) > k))
        rows[f"{k:g} sigma"] = {
            "observed": observed,
            "expected_normal": expected,
            "ratio": observed / expected if expected > 0 else np.nan,
        }
    return pd.DataFrame(rows).T


def hill_tail_index(returns: ArrayLike, tail_fraction: float = 0.05, side: str = "left") -> float:
    """Hill estimator of the tail index alpha.

    For a power-law tail ``P(|X| > x) ~ x^(-alpha)``; a Student-t with ``nu``
    degrees of freedom has ``alpha = nu``. Smaller alpha means fatter tails;
    a normal distribution has no finite tail index (alpha estimates grow with
    sample size).

    Parameters
    ----------
    tail_fraction
        Fraction of observations treated as the tail (``k = tail_fraction * n``).
    side
        ``"left"`` (losses), ``"right"`` (gains) or ``"both"`` (absolute values).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    x = {"left": -r, "right": r, "both": np.abs(r)}[side]
    x = np.sort(x[x > 0])[::-1]
    k = max(int(tail_fraction * r.size), 2)
    if k >= x.size:
        raise ValueError("Not enough positive tail observations")
    logs = np.log(x[:k]) - np.log(x[k])
    return float(1.0 / logs.mean())


def qq_points(returns: ArrayLike, dist=None) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical and sample quantiles for a Q-Q plot.

    ``dist`` is a frozen ``scipy.stats`` distribution; the default is a normal
    with the sample mean and standard deviation.
    """
    r = np.sort(np.asarray(returns, dtype=float))
    r = r[~np.isnan(r)]
    if dist is None:
        dist = stats.norm(loc=r.mean(), scale=r.std(ddof=1))
    probabilities = (np.arange(1, r.size + 1) - 0.5) / r.size
    return dist.ppf(probabilities), r


# ---------------------------------------------------------------------------
# Time aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceRatioResult:
    """Lo-MacKinlay variance-ratio test result for one horizon ``q``."""

    q: int
    variance_ratio: float
    z_statistic: float
    p_value: float


def variance_ratio_test(log_returns: ArrayLike, q: int) -> VarianceRatioResult:
    """Lo-MacKinlay (1988) variance-ratio test with heteroskedasticity-robust z.

    If log returns are serially uncorrelated, the variance of ``q``-period returns
    is ``q`` times the one-period variance and ``VR(q) = 1``. ``VR > 1`` indicates
    positive autocorrelation (trending), ``VR < 1`` mean reversion. For an AR(1)
    process, ``VR(2) = 1 + rho_1``.

    Uses overlapping ``q``-period sums with the unbiased variance estimators of
    Lo and MacKinlay (1988).
    """
    x = np.asarray(log_returns, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if q < 2 or q >= n:
        raise ValueError("Require 2 <= q < number of observations")
    mu = x.mean()
    dev = x - mu
    var_1 = np.sum(dev**2) / (n - 1)
    sums = np.convolve(x, np.ones(q), mode="valid")
    # Per-period variance from q-period sums; m contains the factor q.
    m = q * (n - q + 1) * (1.0 - q / n)
    var_q = np.sum((sums - q * mu) ** 2) / m
    vr = var_q / var_1

    dev2 = dev**2
    denominator = np.sum(dev2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta_j = np.sum(dev2[j:] * dev2[:-j]) / denominator * n
        theta += (2.0 * (q - j) / q) ** 2 * delta_j
    z = (vr - 1.0) / np.sqrt(theta / n)
    return VarianceRatioResult(q, float(vr), float(z), float(2.0 * stats.norm.sf(abs(z))))


def horizon_volatility(log_returns: ArrayLike, horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    """Observed volatility of overlapping ``h``-period log returns vs ``sqrt(h)`` scaling."""
    x = np.asarray(log_returns, dtype=float)
    x = x[~np.isnan(x)]
    sigma_1 = x.std(ddof=1)
    rows = {}
    for h in horizons:
        sums = np.convolve(x, np.ones(h), mode="valid")
        observed = sums.std(ddof=1)
        rows[h] = {
            "observed": observed,
            "sqrt_h_scaling": sigma_1 * np.sqrt(h),
            "ratio": observed / (sigma_1 * np.sqrt(h)),
        }
    return pd.DataFrame(rows).T.rename_axis("horizon")
