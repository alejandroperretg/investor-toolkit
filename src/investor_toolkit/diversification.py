"""Correlation, covariance estimation and portfolio construction.

Question answered: *does combining assets reduce risk, and by how much?*

All functions work on per-period (not annualised) means and covariances unless
stated otherwise. Weight vectors sum to one.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Portfolio arithmetic
# ---------------------------------------------------------------------------


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Returns of a portfolio rebalanced to ``weights`` every period.

    ``weights`` may be a mapping or Series keyed by column, or an array in
    column order.
    """
    w = (
        pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
        if isinstance(weights, dict | pd.Series)
        else pd.Series(np.asarray(weights, dtype=float), index=returns.columns)
    )
    return (returns * w).sum(axis=1, min_count=1).rename("portfolio")


def portfolio_volatility(weights, cov) -> float:
    """``sqrt(w' Sigma w)``."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ np.asarray(cov) @ w))


def risk_contributions(weights, cov) -> np.ndarray:
    """Fractional risk contributions ``w_i (Sigma w)_i / (w' Sigma w)``; they sum to one."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov)
    marginal = sigma @ w
    return w * marginal / (w @ marginal)


def diversification_ratio(weights, cov) -> float:
    """Weighted average asset volatility over portfolio volatility (>= 1)."""
    w = np.asarray(weights, dtype=float)
    vols = np.sqrt(np.diag(np.asarray(cov)))
    return float(w @ vols / portfolio_volatility(w, cov))


def effective_number(shares) -> float:
    """Inverse Herfindahl index ``1 / sum(s_i^2)`` of weights or risk shares."""
    s = np.asarray(shares, dtype=float)
    return float(1.0 / np.sum(s**2))


def equal_weight_variance(sigma: float, rho: float, n) -> np.ndarray:
    """Variance of an equal-weight portfolio of ``n`` assets with common volatility
    ``sigma`` and pairwise correlation ``rho``: ``sigma^2 (rho + (1 - rho) / n)``.

    As ``n`` grows, the variance falls to the floor ``rho * sigma^2``: only the
    uncorrelated part of risk can be diversified away.
    """
    n = np.asarray(n, dtype=float)
    return sigma**2 * (rho + (1.0 - rho) / n)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def ledoit_wolf(returns: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the covariance matrix towards a scaled identity.

    ``Sigma* = delta * m * I + (1 - delta) * S`` where ``S`` is the sample covariance
    (normalised by ``n``), ``m = tr(S) / p``, and the intensity ``delta`` in [0, 1]
    minimises the expected Frobenius loss. Shrinkage matters when the number of
    observations is not much larger than the number of assets.

    Returns
    -------
    (covariance, delta)
    """
    x = np.asarray(returns, dtype=float)
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    m = np.trace(s) / p
    d2 = np.sum((s - m * np.eye(p)) ** 2) / p
    # Average squared distance of the rank-one terms x_k x_k' from S.
    b2_bar = np.sum(np.einsum("ki,kj->k", x**2, x**2) - 2 * np.einsum("ki,ij,kj->k", x, s, x)) / p
    b2_bar = (b2_bar + n * np.sum(s**2) / p) / n**2
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * m * np.eye(p) + (1.0 - delta) * s, float(delta)


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------


def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation over ``window`` periods."""
    return a.rolling(window).corr(b)


def truncated_normal_variance(z: float) -> float:
    """Variance of a standard normal conditioned on ``X < z``."""
    lam = stats.norm.pdf(z) / stats.norm.cdf(z)
    return float(1.0 - z * lam - lam**2)


def conditional_correlation_normal(rho: float, quantile: float) -> float:
    """Correlation of a bivariate normal pair, conditioned on the first variable
    lying below its ``quantile``.

    ``rho_A = rho / sqrt(rho^2 + (1 - rho^2) / v_A)``, where ``v_A`` is the variance
    of the conditioning variable inside the region (Boyer, Gibson and Loretan,
    1997). Because ``v_A < 1``, conditioning on bad markets *lowers* measured
    correlation even when the true correlation is constant. Observed
    down-market correlations above this benchmark indicate genuine asymmetry.
    """
    v = truncated_normal_variance(stats.norm.ppf(quantile))
    return float(rho / np.sqrt(rho**2 + (1.0 - rho**2) / v))


@dataclass(frozen=True)
class DownsideCorrelation:
    """Correlation in the market's worst periods vs a constant-correlation benchmark."""

    asset: str
    full_sample: float
    down_market: float
    normal_benchmark: float
    n_down: int

    @property
    def excess(self) -> float:
        """Down-market correlation above what conditioning alone would produce."""
        return self.down_market - self.normal_benchmark


def downside_correlations(
    returns: pd.DataFrame, market: str, quantile: float = 0.2
) -> pd.DataFrame:
    """Correlation of each asset with ``market`` in the market's worst periods.

    Compares the observed down-market correlation with the bivariate-normal
    benchmark from :func:`conditional_correlation_normal`.
    """
    data = returns.dropna()
    down = data[market] <= data[market].quantile(quantile)
    rows = []
    for asset in data.columns.drop(market):
        rho = data[market].corr(data[asset])
        rows.append(
            DownsideCorrelation(
                asset=asset,
                full_sample=float(rho),
                down_market=float(data.loc[down, market].corr(data.loc[down, asset])),
                normal_benchmark=conditional_correlation_normal(rho, quantile),
                n_down=int(down.sum()),
            )
        )
    table = pd.DataFrame([r.__dict__ for r in rows]).set_index("asset")
    table["excess"] = table["down_market"] - table["normal_benchmark"]
    return table


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def _solve_long_only(objective, n: int, constraints=(), x0=None) -> np.ndarray:
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    result = optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *constraints],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def min_variance_weights(cov, long_only: bool = True) -> np.ndarray:
    """Global minimum-variance portfolio.

    Without constraints the solution is ``Sigma^-1 1 / (1' Sigma^-1 1)``.
    """
    sigma = np.asarray(cov, dtype=float)
    n = sigma.shape[0]
    if not long_only:
        x = np.linalg.solve(sigma, np.ones(n))
        return x / x.sum()
    return _solve_long_only(lambda w: w @ sigma @ w, n)


def max_sharpe_weights(mu, cov, risk_free: float = 0.0, long_only: bool = True) -> np.ndarray:
    """Tangency (maximum Sharpe ratio) portfolio.

    Without constraints the solution is ``Sigma^-1 (mu - rf) / 1' Sigma^-1 (mu - rf)``
    (requires at least one asset with positive expected excess return).
    """
    sigma = np.asarray(cov, dtype=float)
    excess = np.asarray(mu, dtype=float) - risk_free
    if not long_only:
        x = np.linalg.solve(sigma, excess)
        return x / x.sum()
    return _solve_long_only(lambda w: -(w @ excess) / np.sqrt(w @ sigma @ w), sigma.shape[0])


def frontier_variance(mu, cov, target) -> np.ndarray:
    """Analytical minimum variance for a target mean (Merton, 1972), unconstrained.

    ``sigma^2(m) = (A m^2 - 2 B m + C) / D`` with ``A = 1' S^-1 1``, ``B = 1' S^-1 mu``,
    ``C = mu' S^-1 mu`` and ``D = A C - B^2``.
    """
    sigma = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ones = np.ones_like(mu)
    inv_ones, inv_mu = np.linalg.solve(sigma, ones), np.linalg.solve(sigma, mu)
    a, b, c = ones @ inv_ones, ones @ inv_mu, mu @ inv_mu
    m = np.asarray(target, dtype=float)
    return (a * m**2 - 2 * b * m + c) / (a * c - b**2)


def efficient_frontier(
    mu, cov, n_points: int = 40, long_only: bool = True, asset_names=None
) -> pd.DataFrame:
    """Minimum-variance portfolios for a grid of target means.

    The grid runs from the minimum-variance portfolio's mean to the highest
    asset mean (long-only) or twice that range (unconstrained). Returns one row
    per point with ``mean``, ``volatility`` and the weights.
    """
    sigma = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    names = list(asset_names) if asset_names is not None else [f"w{i}" for i in range(n)]
    w_min = min_variance_weights(sigma, long_only)
    lo, hi = float(w_min @ mu), float(mu.max())
    if not long_only:
        hi = lo + 2 * (hi - lo)
    rows = []
    previous = w_min
    for target in np.linspace(lo, hi, n_points):
        if long_only:
            w = _solve_long_only(
                lambda w: w @ sigma @ w,
                n,
                constraints=[{"type": "eq", "fun": lambda w, t=target: w @ mu - t}],
                x0=previous,
            )
            previous = w
        else:
            inv_ones, inv_mu = np.linalg.solve(sigma, np.ones(n)), np.linalg.solve(sigma, mu)
            a, b, c = inv_ones.sum(), inv_mu.sum(), mu @ inv_mu
            d = a * c - b**2
            w = ((c - b * target) * inv_ones + (a * target - b) * inv_mu) / d
        rows.append(
            {
                "mean": float(w @ mu),
                "volatility": portfolio_volatility(w, sigma),
                **dict(zip(names, w, strict=True)),
            }
        )
    return pd.DataFrame(rows)


def inverse_volatility_weights(cov) -> np.ndarray:
    """Weights proportional to ``1 / sigma_i`` (ignores correlations)."""
    inv_vol = 1.0 / np.sqrt(np.diag(np.asarray(cov, dtype=float)))
    return inv_vol / inv_vol.sum()


def risk_parity_weights(
    cov, budgets=None, tol: float = 1e-12, max_iter: int = 10_000
) -> np.ndarray:
    """Risk-budgeting portfolio: each asset contributes its budget share of risk.

    With equal budgets this is the equal-risk-contribution (ERC) portfolio. Solved
    by cyclical coordinate descent on Spinu's (2013) convex formulation
    ``min 0.5 y' Sigma y - sum b_i ln y_i``; each coordinate update is the positive
    root of ``Sigma_ii y_i^2 + c_i y_i - b_i = 0`` with ``c_i = sum_{j != i}
    Sigma_ij y_j`` (Griveau-Billion, Richard and Roncalli, 2013).
    """
    sigma = np.asarray(cov, dtype=float)
    n = sigma.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    b = b / b.sum()
    y = 1.0 / np.sqrt(np.diag(sigma))
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            c = sigma[i] @ y - sigma[i, i] * y[i]
            y[i] = (-c + np.sqrt(c**2 + 4 * sigma[i, i] * b[i])) / (2 * sigma[i, i])
        if np.max(np.abs(y - y_old) / y) < tol:
            break
    return y / y.sum()
