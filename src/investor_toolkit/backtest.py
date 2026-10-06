"""Historical simulation of investment strategies.

Question answered: *how would a strategy have performed, and how much of that
is luck?*

Timing convention (no look-ahead): the target weights for period ``t`` are
computed from returns strictly before ``t``. Trades happen at the start of the
period, then the holdings earn the period-``t`` returns.

Two return measures are reported:

* **Time-weighted return** (``BacktestResult.returns``): the per-period growth of
  invested capital, unaffected by the size and timing of contributions. It
  measures the strategy.
* **Money-weighted return** (``BacktestResult.irr``): the internal rate of
  return of the actual cash flows. It measures the saver's experience.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from investor_toolkit import diversification as dv
from investor_toolkit import risk
from investor_toolkit.resampling import bootstrap_statistic, percentile_interval
from investor_toolkit.returns import annualized_return, annualized_volatility

EULER_GAMMA = 0.5772156649015329

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class Strategy:
    """Base class. Subclasses implement :meth:`target_weights`.

    ``lookback`` is the number of past periods the strategy needs before it can
    trade.
    """

    name: str = "strategy"
    lookback: int = 0

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


@dataclass
class FixedWeights(Strategy):
    """Constant target weights (a constant-mix or, without rebalancing, buy-and-hold)."""

    weights: dict[str, float]
    name: str = "fixed"
    lookback: int = 0

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        return pd.Series(self.weights, dtype=float)


@dataclass
class InverseVolatility(Strategy):
    """Weights proportional to ``1 / volatility`` over the lookback window."""

    assets: list[str]
    lookback: int = 36
    name: str = "inverse_volatility"

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        window = history[self.assets].iloc[-self.lookback :]
        return pd.Series(dv.inverse_volatility_weights(window.cov().to_numpy()), index=self.assets)


@dataclass
class RiskParity(Strategy):
    """Equal risk contributions from a Ledoit-Wolf covariance estimate."""

    assets: list[str]
    lookback: int = 36
    name: str = "risk_parity"

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        window = history[self.assets].iloc[-self.lookback :]
        cov, _ = dv.ledoit_wolf(window.to_numpy())
        return pd.Series(dv.risk_parity_weights(cov), index=self.assets)


@dataclass
class MinimumVariance(Strategy):
    """Long-only minimum-variance weights from a Ledoit-Wolf covariance estimate."""

    assets: list[str]
    lookback: int = 36
    name: str = "minimum_variance"

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        window = history[self.assets].iloc[-self.lookback :]
        cov, _ = dv.ledoit_wolf(window.to_numpy())
        return pd.Series(dv.min_variance_weights(cov), index=self.assets)


@dataclass
class TrendFollowing(Strategy):
    """Moving-average timing rule (Faber, 2007).

    Each risky asset keeps its base weight while its total-return index is above
    its ``lookback``-period simple moving average; otherwise that weight moves to
    the ``cash`` asset.
    """

    weights: dict[str, float]
    cash: str = "cash"
    lookback: int = 10
    name: str = "trend_following"

    def target_weights(self, history: pd.DataFrame) -> pd.Series:
        target = pd.Series(self.weights, dtype=float)
        target[self.cash] = target.get(self.cash, 0.0)
        level = (1.0 + history.iloc[-self.lookback :]).cumprod()
        for asset in self.weights:
            if asset == self.cash:
                continue
            if level[asset].iloc[-1] < level[asset].mean():
                target[self.cash] += target[asset]
                target[asset] = 0.0
        return target


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rebalance:
    """When to trade back to target weights.

    ``every``: rebalance every ``every`` periods (1 monthly, 3 quarterly, 12 annual
    with monthly data; ``None`` never). ``band``: also rebalance whenever any weight
    deviates from target by more than ``band`` (absolute). Contributions are
    always invested at the target weights.
    """

    every: int | None = 1
    band: float | None = None


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`."""

    name: str
    wealth: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    contributions: pd.Series
    initial: float
    periods_per_year: int = 12
    extra: dict = field(default_factory=dict)

    def irr(self) -> float:
        """Annualised money-weighted return (internal rate of return)."""
        flows = -self.contributions.to_numpy(dtype=float).copy()
        flows[0] -= self.initial
        flows = np.append(flows, self.wealth.iloc[-1])
        return internal_rate_of_return(flows, self.periods_per_year)

    def summary(self, risk_free: pd.Series | float = 0.0) -> pd.Series:
        """Key performance and risk statistics of the time-weighted returns."""
        r = self.returns
        ppy = self.periods_per_year
        rf = risk_free.reindex(r.index) if isinstance(risk_free, pd.Series) else risk_free
        return pd.Series(
            {
                "CAGR": annualized_return(r, ppy),
                "volatility": annualized_volatility(r, ppy),
                "sharpe": risk.sharpe_ratio(r, ppy, rf),
                "max_drawdown": risk.max_drawdown(r),
                "calmar": risk.calmar_ratio(r, ppy),
                "turnover_per_year": float(self.turnover.mean() * ppy),
                "cost_drag_per_year": float(
                    (self.costs / (self.wealth / (1.0 + self.returns))).mean() * ppy
                ),
                "final_value": float(self.wealth.iloc[-1]),
            },
            name=self.name,
        )


def internal_rate_of_return(flows: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualised IRR of equally spaced cash flows (negative = money invested).

    The per-period rate is searched in [-30 %, +100 %]; the discount factors are
    computed in log space so long series do not overflow.
    """
    flows = np.asarray(flows, dtype=float)
    t = np.arange(flows.size)

    def npv(rate: float) -> float:
        return float(np.sum(flows * np.exp(-t * np.log1p(rate))))

    lo, hi = -0.3, 1.0
    if np.sign(npv(lo)) == np.sign(npv(hi)):
        raise ValueError("IRR not bracketed in [-30 %, +100 %] per period")
    per_period = optimize.brentq(npv, lo, hi, xtol=1e-14)
    return float((1.0 + per_period) ** periods_per_year - 1.0)


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    rebalance: Rebalance | None = None,
    cost_bps: float = 10.0,
    initial: float = 1.0,
    contributions: float | pd.Series = 0.0,
    start: int | pd.Timestamp | None = None,
    periods_per_year: int = 12,
    name: str | None = None,
) -> BacktestResult:
    """Simulate ``strategy`` on ``returns``.

    Parameters
    ----------
    returns
        Simple per-period returns, one column per asset. Assets the strategy
        holds must have no missing values in the simulated window.
    cost_bps
        Proportional transaction cost in basis points of traded value.
    initial, contributions
        Starting capital and the amount added at the start of every period
        (scalar or Series aligned to ``returns``).
    start
        First simulated period (position or date). Defaults to the strategy's
        lookback, so every target is based on at least ``lookback`` periods.
    """
    rebalance = Rebalance() if rebalance is None else rebalance
    columns = list(returns.columns)
    r = returns.to_numpy(dtype=float)
    n_periods = len(returns)
    if start is None:
        start_pos = strategy.lookback
    elif isinstance(start, int):
        start_pos = start
    else:
        start_pos = returns.index.get_indexer([pd.Timestamp(start)], method="bfill")[0]
    if start_pos < strategy.lookback:
        raise ValueError("Start leaves less history than the strategy's lookback")
    if isinstance(contributions, pd.Series):
        flows = contributions.reindex(returns.index).fillna(0.0).to_numpy(dtype=float)
    else:
        flows = np.full(n_periods, float(contributions))

    cost_rate = cost_bps / 1e4
    holdings = np.zeros(len(columns))
    cash_in = float(initial)
    sim_index = returns.index[start_pos:]
    values, rets, turnover, costs, weights, contribs = [], [], [], [], [], []

    for step, t in enumerate(range(start_pos, n_periods)):
        history = returns.iloc[:t]
        target = strategy.target_weights(history).reindex(columns).fillna(0.0).to_numpy(dtype=float)
        contribution = flows[t] + (cash_in if step == 0 else 0.0)
        value_before = holdings.sum()
        total = value_before + contribution

        current_w = holdings / value_before if value_before > 0 else target
        due = step == 0
        if rebalance.every is not None and step % rebalance.every == 0:
            due = True
        if rebalance.band is not None and np.max(np.abs(current_w - target)) > rebalance.band:
            due = True

        desired = target * total if due else holdings + contribution * target
        trade = desired - holdings
        cost = cost_rate * np.abs(trade).sum()
        holdings = desired * (1.0 - cost / total) if total > 0 else desired

        held = holdings > 0
        if np.any(np.isnan(r[t, held])):
            raise ValueError(f"Missing return for a held asset at {returns.index[t]}")
        invested = holdings.sum()
        weights.append(holdings / invested if invested > 0 else holdings)
        holdings = holdings * (1.0 + np.nan_to_num(r[t]))
        value_after = holdings.sum()

        values.append(value_after)
        rets.append(value_after / total - 1.0 if total > 0 else 0.0)
        turnover.append(np.abs(trade).sum() / total if total > 0 else 0.0)
        costs.append(cost)
        contribs.append(flows[t])

    label = name or strategy.name
    return BacktestResult(
        name=label,
        wealth=pd.Series(values, index=sim_index, name=label),
        returns=pd.Series(rets, index=sim_index, name=label),
        weights=pd.DataFrame(weights, index=sim_index, columns=columns),
        turnover=pd.Series(turnover, index=sim_index),
        costs=pd.Series(costs, index=sim_index),
        contributions=pd.Series(contribs, index=sim_index),
        initial=float(initial),
        periods_per_year=periods_per_year,
    )


def compare(results: list[BacktestResult], risk_free: pd.Series | float = 0.0) -> pd.DataFrame:
    """Side-by-side summary statistics of several backtests."""
    return pd.DataFrame([res.summary(risk_free) for res in results])


# ---------------------------------------------------------------------------
# Statistical significance and overfitting
# ---------------------------------------------------------------------------


def sharpe_ratio_std(sr: float, n: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Standard error of an estimated per-period Sharpe ratio.

    ``sqrt((1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (n - 1))`` (Mertens, 2002;
    Bailey and Lopez de Prado, 2012). For normal returns this is Lo's (2002)
    ``sqrt((1 + SR^2 / 2) / (n - 1))``. ``kurtosis`` is the raw (non-excess) value.
    """
    return float(np.sqrt((1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n - 1)))


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark_sr``.

    Accounts for sample length, skewness and kurtosis (Bailey and Lopez de Prado,
    2012). ``benchmark_sr`` is per period (not annualised).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sr = r.mean() / r.std(ddof=1)
    se = sharpe_ratio_std(sr, r.size, stats.skew(r), stats.kurtosis(r, fisher=False))
    return float(stats.norm.cdf((sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratio estimates whose true value is zero.

    ``E[max] ~ sd * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))`` with
    the Euler-Mascheroni constant ``g`` (Bailey and Lopez de Prado, 2014). This is
    the Sharpe ratio the best of ``N`` worthless strategies reaches by luck.
    """
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    return float(
        sr_std
        * (
            (1 - g) * stats.norm.ppf(1 - 1 / n_trials)
            + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(returns, n_trials: int, trial_sr_std: float) -> float:
    """Probabilistic Sharpe ratio against the best-of-``n_trials`` luck benchmark.

    ``trial_sr_std`` is the standard deviation of the per-period Sharpe ratios of
    all strategies that were tried. Values near 1 mean the result is unlikely to
    be a selection artefact.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, trial_sr_std))


def bootstrap_difference(
    a: pd.Series,
    b: pd.Series,
    metric: Callable[[np.ndarray], float] | str = "sharpe",
    periods_per_year: int = 12,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = 0,
    level: float = 0.95,
) -> pd.Series:
    """Paired stationary-bootstrap confidence interval for ``metric(a) - metric(b)``.

    Months are resampled jointly for both strategies so their correlation is kept.
    ``metric`` is ``"sharpe"``, ``"cagr"`` or a function of a return array.
    """
    funcs = {
        "sharpe": lambda x: x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year),
        "cagr": lambda x: np.prod(1 + x) ** (periods_per_year / x.size) - 1,
    }
    f = funcs[metric] if isinstance(metric, str) else metric
    paired = pd.concat([a, b], axis=1, join="inner").dropna().to_numpy()
    observed = f(paired[:, 0]) - f(paired[:, 1])
    boot = bootstrap_statistic(
        paired, lambda x: f(x[:, 0]) - f(x[:, 1]), n_boot=n_boot, mean_block=mean_block, seed=seed
    )
    lo, hi = percentile_interval(boot, level)
    return pd.Series(
        {
            "difference": observed,
            "ci_low": lo,
            "ci_high": hi,
            "share_positive": float(np.mean(boot > 0)),
        }
    )


# ---------------------------------------------------------------------------
# Lump sum vs dollar-cost averaging
# ---------------------------------------------------------------------------


def lump_sum_vs_dca(
    risky: pd.Series, cash: pd.Series, months: int = 12, horizon: int | None = None
) -> pd.DataFrame:
    """Compare investing a sum at once with spreading it over ``months``.

    For every start date, 1 unit is either invested immediately in ``risky``
    (lump sum) or held in ``cash`` and moved into ``risky`` in ``months`` equal
    instalments at the start of each month (DCA). Both are valued after
    ``horizon`` months (default ``months``).

    Returns
    -------
    pandas.DataFrame
        Indexed by start date with columns ``lump_sum``, ``dca`` and
        ``lump_sum_wins``.
    """
    horizon = months if horizon is None else horizon
    if horizon < months:
        raise ValueError("horizon must be at least the DCA period")
    data = pd.concat([risky.rename("risky"), cash.rename("cash")], axis=1, join="inner").dropna()
    growth_r = (1.0 + data["risky"]).to_numpy()
    growth_c = (1.0 + data["cash"]).to_numpy()
    rows = {}
    for s in range(len(data) - horizon + 1):
        gr, gc = growth_r[s : s + horizon], growth_c[s : s + horizon]
        lump = np.prod(gr)
        invested, waiting = 0.0, 1.0
        for k in range(horizon):
            if k < months:
                instalment = 1.0 / months if k < months - 1 else waiting
                instalment = min(instalment, waiting)
                waiting -= instalment
                invested += instalment
            invested *= gr[k]
            waiting *= gc[k]
        rows[data.index[s]] = {"lump_sum": lump, "dca": invested + waiting}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["lump_sum_wins"] = table["lump_sum"] > table["dca"]
    return table
