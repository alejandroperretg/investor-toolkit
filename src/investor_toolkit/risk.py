"""Risk measures.

Question answered: *how bad can it get, and for how long?*

Sign convention: Value at Risk (VaR) and Expected Shortfall (ES) are reported as
positive numbers representing losses, e.g. ``VaR = 0.05`` means a 5 % loss.
``alpha`` is the tail probability (``alpha = 0.05`` gives the 95 % VaR).
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from investor_toolkit.returns import annualized_return, annualized_volatility, fit_student_t

ArrayLike = np.ndarray | pd.Series

# ---------------------------------------------------------------------------
# Volatility-type measures
# ---------------------------------------------------------------------------


def downside_deviation(returns: ArrayLike, periods_per_year: int, target: float = 0.0) -> float:
    """Annualised root-mean-square of returns below ``target`` (per period)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Value at Risk and Expected Shortfall
# ---------------------------------------------------------------------------

VAR_METHODS = ("historical", "gaussian", "cornish_fisher", "student_t")


def _clean(returns: ArrayLike) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def value_at_risk(returns: ArrayLike, alpha: float = 0.05, method: str = "historical") -> float:
    """One-period Value at Risk as a positive loss.

    Methods
    -------
    historical
        Empirical ``alpha`` quantile of returns.
    gaussian
        ``-(mu + sigma * z_alpha)``.
    cornish_fisher
        Gaussian quantile corrected for sample skewness and excess kurtosis.
    student_t
        Quantile of a maximum-likelihood location-scale Student-t fit.
    """
    r = _clean(returns)
    if method == "historical":
        return float(-np.quantile(r, alpha))
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(alpha)
    if method == "gaussian":
        return float(-(mu + sigma * z))
    if method == "cornish_fisher":
        s = stats.skew(r, bias=False)
        k = stats.kurtosis(r, fisher=True, bias=False)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(-(mu + sigma * z_cf))
    if method == "student_t":
        fit = fit_student_t(r).params
        return float(-(fit["loc"] + fit["scale"] * stats.t.ppf(alpha, fit["df"])))
    raise ValueError(f"Unknown method {method!r}; choose from {VAR_METHODS}")


def gaussian_expected_shortfall(mu: float, sigma: float, alpha: float) -> float:
    """Analytical ES of a normal distribution: ``-mu + sigma * phi(z_alpha) / alpha``."""
    return float(-mu + sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)


def student_t_expected_shortfall(loc: float, scale: float, df: float, alpha: float) -> float:
    """Analytical ES of a location-scale Student-t distribution (``df > 1``).

    ``ES = -loc + scale * g(q) * (df + q^2) / ((df - 1) * alpha)``, where
    ``q = t^-1(1 - alpha)`` and ``g`` is the standard t density
    (McNeil, Frey and Embrechts, *Quantitative Risk Management*, 2015).
    """
    if df <= 1:
        raise ValueError("Expected shortfall is infinite for df <= 1")
    q = stats.t.ppf(1 - alpha, df)
    return float(-loc + scale * stats.t.pdf(q, df) * (df + q**2) / ((df - 1) * alpha))


def expected_shortfall(
    returns: ArrayLike, alpha: float = 0.05, method: str = "historical"
) -> float:
    """Expected Shortfall (CVaR): the average loss in the worst ``alpha`` of cases.

    Methods: ``historical`` (mean of returns at or below the VaR quantile),
    ``gaussian`` and ``student_t`` (analytical ES of the fitted distribution).
    """
    r = _clean(returns)
    if method == "historical":
        threshold = np.quantile(r, alpha)
        return float(-r[r <= threshold].mean())
    if method == "gaussian":
        return gaussian_expected_shortfall(r.mean(), r.std(ddof=1), alpha)
    if method == "student_t":
        fit = fit_student_t(r).params
        return student_t_expected_shortfall(fit["loc"], fit["scale"], fit["df"], alpha)
    raise ValueError(f"Unknown method {method!r}")


def var_table(returns: ArrayLike, alphas=(0.05, 0.01)) -> pd.DataFrame:
    """VaR and ES for several methods and tail probabilities."""
    rows = {}
    for alpha in alphas:
        for method in VAR_METHODS:
            rows[(f"{1 - alpha:.0%}", method)] = {
                "VaR": value_at_risk(returns, alpha, method),
                "ES": (
                    expected_shortfall(returns, alpha, method)
                    if method != "cornish_fisher"
                    else np.nan
                ),
            }
    return pd.DataFrame(rows).T.rename_axis(["confidence", "method"])


# ---------------------------------------------------------------------------
# VaR backtesting
# ---------------------------------------------------------------------------


def kupiec_test(exceptions: ArrayLike, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test.

    Tests whether the observed exception rate equals ``alpha``. Returns the
    likelihood-ratio statistic (chi-squared, 1 dof) and its p-value.
    """
    x = np.asarray(exceptions, dtype=bool)
    n, k = x.size, int(x.sum())
    p_hat = k / n

    def loglik(p: float) -> float:
        return sum(c * math.log(q) for c, q in [(k, p), (n - k, 1 - p)] if c > 0)

    lr = -2.0 * (loglik(alpha) - loglik(p_hat))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_test(exceptions: ArrayLike) -> tuple[float, float]:
    """Christoffersen (1998) independence test for VaR exceptions.

    Tests whether an exception today makes an exception tomorrow more likely
    (clustering). Returns the LR statistic (chi-squared, 1 dof) and p-value.
    """
    x = np.asarray(exceptions, dtype=int)
    prev, curr = x[:-1], x[1:]
    n = {(i, j): int(np.sum((prev == i) & (curr == j))) for i in (0, 1) for j in (0, 1)}

    def ll(count: int, p: float) -> float:
        return count * math.log(p) if count > 0 else 0.0

    pi01 = n[0, 1] / max(n[0, 0] + n[0, 1], 1)
    pi11 = n[1, 1] / max(n[1, 0] + n[1, 1], 1)
    pi = (n[0, 1] + n[1, 1]) / max(sum(n.values()), 1)
    unrestricted = (
        ll(n[0, 0], 1 - pi01) + ll(n[0, 1], pi01) + ll(n[1, 0], 1 - pi11) + ll(n[1, 1], pi11)
    )
    restricted = ll(n[0, 0] + n[1, 0], 1 - pi) + ll(n[0, 1] + n[1, 1], pi)
    lr = -2.0 * (restricted - unrestricted)
    return float(lr), float(stats.chi2.sf(lr, 1))


def backtest_var(
    returns: pd.Series, window: int, alpha: float = 0.05, method: str = "historical"
) -> pd.DataFrame:
    """Rolling out-of-sample VaR forecasts and exceptions.

    The VaR for period ``t`` is estimated from the ``window`` returns before ``t``
    only. Returns a frame with columns ``return``, ``var`` and ``exception``.
    """
    r = returns.dropna()
    values = r.to_numpy()
    forecasts = np.full(values.size, np.nan)
    for t in range(window, values.size):
        forecasts[t] = value_at_risk(values[t - window : t], alpha, method)
    out = pd.DataFrame({"return": values, "var": forecasts}, index=r.index).iloc[window:]
    out["exception"] = out["return"] < -out["var"]
    return out


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak of the wealth index (zero or negative)."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series | ArrayLike) -> float:
    """Largest peak-to-trough loss, as a positive fraction."""
    return float(-drawdown(pd.Series(np.asarray(returns, dtype=float))).min())


def drawdown_episodes(returns: pd.Series, top: int | None = 5) -> pd.DataFrame:
    """Drawdown episodes ranked by depth.

    Each episode runs from a peak to the first date that regains it. Columns:
    ``peak`` (last date at the high), ``trough``, ``recovery`` (NaT if still under
    water), ``depth`` (positive fraction), ``to_trough`` and ``under_water``
    (number of periods from peak to trough and from peak to recovery or the end
    of the sample).
    """
    dd = drawdown(returns)
    underwater = dd < 0
    episodes = []
    index = dd.index
    # Each run of consecutive under-water periods is one episode.
    run_id = (underwater != underwater.shift(fill_value=False)).cumsum()
    for _, run in dd[underwater].groupby(run_id[underwater]):
        start_pos = index.get_loc(run.index[0])
        end_pos = index.get_loc(run.index[-1])
        recovered = end_pos + 1 < len(index)
        trough = run.idxmin()
        episodes.append(
            {
                "peak": index[start_pos - 1] if start_pos > 0 else pd.NaT,
                "trough": trough,
                "recovery": index[end_pos + 1] if recovered else pd.NaT,
                "depth": float(-run.min()),
                "to_trough": index.get_loc(trough) - start_pos + 1,
                "under_water": end_pos - start_pos + 1 + int(recovered),
            }
        )
    table = pd.DataFrame(
        episodes, columns=["peak", "trough", "recovery", "depth", "to_trough", "under_water"]
    )
    table = table.sort_values("depth", ascending=False).reset_index(drop=True)
    return table if top is None else table.head(top)


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown (Martin, 1987): penalises both depth and duration."""
    return float(np.sqrt(np.mean(drawdown(returns) ** 2)))


def expected_max_drawdown_brownian(sigma: float, horizon: float) -> float:
    """Expected maximum drawdown of driftless Brownian motion.

    ``E[MDD] = sqrt(pi / 2) * sigma * sqrt(T)`` (Magdon-Ismail et al., 2004).
    Applied to log wealth, ``sigma`` is the volatility of log returns per unit
    time and the result is the expected maximum drawdown in log terms.
    """
    return float(np.sqrt(np.pi / 2.0) * sigma * np.sqrt(horizon))


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------


def _excess(returns: ArrayLike, risk_free: ArrayLike | float) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    rf = np.broadcast_to(np.asarray(risk_free, dtype=float), r.shape)
    excess = r - rf
    return excess[~np.isnan(excess)]


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: int, risk_free: ArrayLike | float = 0.0
) -> float:
    """Annualised Sharpe ratio: mean excess return over its standard deviation."""
    x = _excess(returns, risk_free)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike, periods_per_year: int, risk_free: ArrayLike | float = 0.0
) -> float:
    """Annualised mean excess return over the downside deviation of excess returns."""
    x = _excess(returns, risk_free)
    return float(x.mean() * periods_per_year / downside_deviation(x, periods_per_year))


def calmar_ratio(returns: ArrayLike, periods_per_year: int) -> float:
    """Annualised geometric return over the maximum drawdown."""
    return annualized_return(returns, periods_per_year) / max_drawdown(returns)


def risk_summary(
    returns: pd.DataFrame,
    periods_per_year: int,
    risk_free: pd.Series | float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Key return and risk statistics, one row per column of ``returns``."""
    rows = {}
    for column in returns.columns:
        r = returns[column].dropna()
        rf = risk_free.reindex(r.index) if isinstance(risk_free, pd.Series) else risk_free
        rows[column] = {
            "CAGR": annualized_return(r, periods_per_year),
            "volatility": annualized_volatility(r, periods_per_year),
            "sharpe": sharpe_ratio(r, periods_per_year, rf),
            "sortino": sortino_ratio(r, periods_per_year, rf),
            "max_drawdown": max_drawdown(r),
            "calmar": calmar_ratio(r, periods_per_year),
            "ulcer_index": ulcer_index(r),
            f"VaR_{1 - alpha:.0%}": value_at_risk(r, alpha),
            f"ES_{1 - alpha:.0%}": expected_shortfall(r, alpha),
            "worst_period": float(r.min()),
        }
    return pd.DataFrame(rows).T
