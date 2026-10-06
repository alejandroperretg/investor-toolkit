"""Factor models.

Question answered: *what actually drives an asset's returns?*

A linear factor model explains an asset's excess return with the returns of a
few long-short factor portfolios:

``r_t - rf_t = alpha + sum_k beta_k f_{k,t} + e_t``

* ``Mkt-RF``: market minus risk-free rate (equity market exposure).
* ``SMB``: small minus big (size).
* ``HML``: high minus low book-to-market (value).
* ``RMW``: robust minus weak profitability.
* ``CMA``: conservative minus aggressive investment.
* ``WML``: winners minus losers (momentum).

The betas measure exposures; alpha is the average return the factors do not
explain. Factor data come from the Kenneth R. French Data Library and are in
USD, so EUR-denominated returns are converted to USD before regressing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from investor_toolkit.data import load_fred, load_french

FIVE_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionResult:
    """Ordinary least squares fit with a chosen covariance estimator."""

    params: pd.Series
    std_errors: pd.Series
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted: pd.Series
    n_obs: int
    cov_type: str

    @property
    def t_stats(self) -> pd.Series:
        return self.params / self.std_errors

    @property
    def p_values(self) -> pd.Series:
        """Two-sided p-values from the normal distribution."""
        return pd.Series(2 * stats.norm.sf(np.abs(self.t_stats)), index=self.params.index)

    def summary(self) -> pd.DataFrame:
        """Coefficients, standard errors, t-statistics and p-values."""
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "t": self.t_stats,
                "p_value": self.p_values,
            }
        )


def newey_west_lags(n_obs: int) -> int:
    """Rule-of-thumb lag length ``floor(4 (n / 100)^(2/9))`` (Newey and West, 1994)."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def ols(
    y: pd.Series,
    x: pd.DataFrame,
    add_constant: bool = True,
    cov_type: str = "HAC",
    max_lags: int | None = None,
) -> RegressionResult:
    """Least-squares regression of ``y`` on ``x``.

    Parameters
    ----------
    cov_type
        ``"nonrobust"`` (classical, homoskedastic iid errors), ``"HC0"`` (White's
        heteroskedasticity-robust) or ``"HAC"`` (Newey-West heteroskedasticity-
        and autocorrelation-consistent, Bartlett kernel).
    max_lags
        HAC lag length; defaults to :func:`newey_west_lags`.
    """
    data = pd.concat([y.rename("_y"), x], axis=1).dropna()
    target = data["_y"].to_numpy()
    design = data.drop(columns="_y")
    if add_constant:
        design.insert(0, "alpha", 1.0)
    names = list(design.columns)
    xm = design.to_numpy(dtype=float)
    n, k = xm.shape

    beta, *_ = np.linalg.lstsq(xm, target, rcond=None)
    fitted = xm @ beta
    resid = target - fitted
    xtx_inv = np.linalg.inv(xm.T @ xm)

    if cov_type == "nonrobust":
        cov = xtx_inv * (resid @ resid) / (n - k)
    elif cov_type in ("HC0", "HAC"):
        scores = xm * resid[:, None]
        meat = scores.T @ scores
        if cov_type == "HAC":
            lags = newey_west_lags(n) if max_lags is None else max_lags
            for lag in range(1, lags + 1):
                weight = 1.0 - lag / (lags + 1.0)
                gamma = scores[lag:].T @ scores[:-lag]
                meat += weight * (gamma + gamma.T)
        cov = xtx_inv @ meat @ xtx_inv
    else:
        raise ValueError(f"Unknown cov_type {cov_type!r}")

    tss = np.sum((target - target.mean()) ** 2) if add_constant else np.sum(target**2)
    r2 = 1.0 - (resid @ resid) / tss
    return RegressionResult(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(np.sqrt(np.diag(cov)), index=names),
        r_squared=float(r2),
        adj_r_squared=float(1.0 - (1.0 - r2) * (n - 1) / (n - k)),
        residuals=pd.Series(resid, index=data.index),
        fitted=pd.Series(fitted, index=data.index),
        n_obs=n,
        cov_type=cov_type,
    )


# ---------------------------------------------------------------------------
# Factor models
# ---------------------------------------------------------------------------


def load_factors(
    region: str = "Europe",
    momentum: bool = True,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Monthly Fama-French five factors (plus momentum ``WML``) and ``RF``, in USD.

    ``region`` is a Kenneth French region prefix, e.g. ``"Europe"``,
    ``"Developed"`` or ``"North_America"``.
    """
    factors = load_french(f"{region}_5_Factors", refresh=refresh, cache_dir=cache_dir)
    if momentum:
        mom = load_french(f"{region}_Mom_Factor", refresh=refresh, cache_dir=cache_dir)
        factors = factors.join(mom.iloc[:, 0].rename("WML"), how="inner")
    return factors


def eur_to_usd_returns(
    returns: pd.DataFrame | pd.Series,
    *,
    fx_usd_per_eur: pd.Series | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
):
    """Convert monthly EUR returns to USD returns: ``(1 + r) * X_t / X_{t-1} - 1``.

    ``X`` is the month-end USD-per-EUR rate (FRED ``DEXUSEU`` by default).
    """
    if fx_usd_per_eur is None:
        fx_usd_per_eur = load_fred("DEXUSEU", refresh=refresh, cache_dir=cache_dir)
    fx = fx_usd_per_eur.resample("ME").last()
    growth = (fx / fx.shift(1)).reindex(returns.index)
    if isinstance(returns, pd.DataFrame):
        return (1.0 + returns).mul(growth, axis=0) - 1.0
    return (1.0 + returns) * growth - 1.0


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor_names: list[str] | None = None,
    cov_type: str = "HAC",
) -> RegressionResult:
    """Regress excess returns (``returns - RF``) on the chosen factors.

    ``returns`` and ``factors`` must be in the same currency and periodicity.
    """
    names = factor_names or [c for c in factors.columns if c != "RF"]
    excess = returns - factors["RF"].reindex(returns.index)
    return ols(excess, factors[names], cov_type=cov_type)


def return_attribution(result: RegressionResult, factors: pd.DataFrame) -> pd.Series:
    """Decompose the mean excess return into alpha and factor contributions.

    Contribution of factor ``k`` is ``beta_k * mean(f_k)`` over the regression
    sample. With an intercept, alpha plus all contributions equals the mean
    excess return exactly (the residuals average to zero).
    """
    sample = factors.loc[result.residuals.index]
    contributions = {"alpha": result.params["alpha"]}
    for name in result.params.index.drop("alpha"):
        contributions[name] = result.params[name] * sample[name].mean()
    return pd.Series(contributions)


def rolling_betas(
    returns: pd.Series, factors: pd.DataFrame, window: int, factor_names: list[str] | None = None
) -> pd.DataFrame:
    """Factor exposures estimated over a rolling window (point estimates only)."""
    names = factor_names or [c for c in factors.columns if c != "RF"]
    excess = (returns - factors["RF"].reindex(returns.index)).dropna()
    x = factors.loc[excess.index, names]
    rows = {}
    for end in range(window, len(excess) + 1):
        y_w = excess.iloc[end - window : end].to_numpy()
        x_w = np.column_stack([np.ones(window), x.iloc[end - window : end].to_numpy()])
        coef, *_ = np.linalg.lstsq(x_w, y_w, rcond=None)
        rows[excess.index[end - 1]] = coef
    return pd.DataFrame.from_dict(rows, orient="index", columns=["alpha", *names])
