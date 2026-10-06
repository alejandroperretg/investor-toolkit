"""Monte Carlo projection of a savings and withdrawal plan.

Question answered: *what range of outcomes can a savings plan expect, and how
much can be withdrawn safely?*

The simulation runs month by month over many independent scenarios ("paths"):

1. A **return model** generates monthly portfolio returns and inflation.
2. The **plan** adds contributions during the accumulation phase and takes
   inflation-indexed withdrawals during the withdrawal phase.
3. Optional **fund costs** reduce returns, and an optional **German tax** model
   charges the annual Vorabpauschale and taxes realised gains (FIFO).

All amounts are nominal EUR unless the name says ``real`` (deflated to money
of the start date). The plan starts in January.

Return models
-------------
``GBM``
    Independent lognormal monthly returns (geometric Brownian motion). Has
    closed-form results used to validate the simulator.
``StudentT``
    Same mean and variance of log returns as GBM, but fat-tailed shocks.
``Bootstrap``
    Stationary block bootstrap of historical monthly returns *and* inflation,
    resampled jointly so their co-movement, autocorrelation and volatility
    clustering are kept.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from investor_toolkit.resampling import stationary_bootstrap_indices
from investor_toolkit.tax import (
    GermanTax,
    purchase_month_factor,
    sell_fifo,
    vorabpauschale_per_unit,
)

# ---------------------------------------------------------------------------
# Return models
# ---------------------------------------------------------------------------


def _monthly_inflation(annual: float) -> float:
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


@dataclass(frozen=True)
class GBM:
    """Lognormal monthly returns.

    Parameters
    ----------
    expected_return
        Annual expected simple return ``E[1 + R_year] - 1``.
    volatility
        Annual volatility of log returns.
    inflation
        Constant annual inflation rate.
    antithetic
        Pair every shock ``z`` with ``-z`` (variance reduction).

    Monthly log returns are ``N(a, b^2)`` with ``b = volatility / sqrt(12)`` and
    ``a = ln(1 + expected_return) / 12 - b^2 / 2``.
    """

    expected_return: float = 0.07
    volatility: float = 0.15
    inflation: float = 0.02
    antithetic: bool = False

    @property
    def log_mean(self) -> float:
        return np.log1p(self.expected_return) / 12.0 - self.log_std**2 / 2.0

    @property
    def log_std(self) -> float:
        return self.volatility / np.sqrt(12.0)

    @classmethod
    def from_returns(cls, monthly_returns, inflation: float = 0.02, **kwargs) -> GBM:
        """Calibrate to the mean and standard deviation of historical log returns."""
        log_r = np.log1p(np.asarray(monthly_returns, dtype=float))
        a, b = log_r.mean(), log_r.std(ddof=1)
        return cls(np.exp(12 * a + 6 * b**2) - 1.0, b * np.sqrt(12.0), inflation, **kwargs)

    def _shocks(self, n_paths: int, n_months: int, rng: np.random.Generator) -> np.ndarray:
        if not self.antithetic:
            return rng.standard_normal((n_paths, n_months))
        half = rng.standard_normal(((n_paths + 1) // 2, n_months))
        return np.vstack([half, -half])[:n_paths]

    def sample(self, n_paths: int, n_months: int, rng: np.random.Generator):
        """Monthly simple returns and inflation, each of shape ``(n_paths, n_months)``."""
        z = self._shocks(n_paths, n_months, rng)
        returns = np.expm1(self.log_mean + self.log_std * z)
        inflation = np.full((n_paths, n_months), _monthly_inflation(self.inflation))
        return returns, inflation


@dataclass(frozen=True)
class StudentT(GBM):
    """Fat-tailed monthly log returns with the same mean and variance as :class:`GBM`.

    Shocks are Student-t with ``df`` degrees of freedom, rescaled to unit
    variance (requires ``df > 2``). With the same mean and variance of log
    returns, the median outcome matches GBM but extreme months are more frequent.
    """

    df: float = 4.0

    def _shocks(self, n_paths: int, n_months: int, rng: np.random.Generator) -> np.ndarray:
        if self.df <= 2:
            raise ValueError("df must exceed 2 for finite variance")
        scale = np.sqrt((self.df - 2.0) / self.df)
        return rng.standard_t(self.df, (n_paths, n_months)) * scale


@dataclass(frozen=True)
class Bootstrap:
    """Stationary block bootstrap of historical monthly returns and inflation.

    Parameters
    ----------
    returns
        Historical monthly simple returns of the portfolio.
    inflation
        Historical monthly inflation aligned with ``returns``. If ``None``, a
        constant ``constant_inflation`` (annual) is used.
    mean_block
        Mean block length in months.
    return_adjustment
        Constant added to every monthly return, e.g. to lower historical returns
        to a more conservative forward-looking assumption.
    """

    returns: pd.Series
    inflation: pd.Series | None = None
    mean_block: float = 12.0
    return_adjustment: float = 0.0
    constant_inflation: float = 0.02

    def sample(self, n_paths: int, n_months: int, rng: np.random.Generator):
        if self.inflation is not None:
            joint = pd.concat([self.returns, self.inflation], axis=1, join="inner").dropna()
            r_hist, i_hist = joint.iloc[:, 0].to_numpy(), joint.iloc[:, 1].to_numpy()
        else:
            r_hist = self.returns.dropna().to_numpy()
            i_hist = np.full(r_hist.size, _monthly_inflation(self.constant_inflation))
        idx = stationary_bootstrap_indices(r_hist.size, n_months, n_paths, self.mean_block, rng)
        return r_hist[idx] + self.return_adjustment, i_hist[idx]


# ---------------------------------------------------------------------------
# Plan and simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavingsPlan:
    """A savings plan followed by an optional withdrawal phase.

    Parameters
    ----------
    initial
        Lump sum invested at the start.
    monthly_contribution
        Contribution at the start of every month of the accumulation phase, in
        money of the start date.
    contributions_indexed
        Grow contributions with inflation (constant real saving rate).
    accumulation_years, withdrawal_years
        Length of each phase.
    monthly_withdrawal
        Inflation-indexed withdrawal in money of the start date. Ignored if
        ``withdrawal_rate`` is set.
    withdrawal_rate
        Initial annual withdrawal as a fraction of wealth at the start of the
        withdrawal phase (e.g. 0.04), then indexed to inflation.
    annual_fee
        Fund costs (TER), deducted monthly from returns.
    tax
        German tax model, or ``None`` for a tax-free account.
    """

    initial: float = 0.0
    monthly_contribution: float = 500.0
    contributions_indexed: bool = True
    accumulation_years: int = 30
    withdrawal_years: int = 0
    monthly_withdrawal: float = 0.0
    withdrawal_rate: float | None = None
    annual_fee: float = 0.0
    tax: GermanTax | None = None

    @property
    def accumulation_months(self) -> int:
        return 12 * self.accumulation_years

    @property
    def total_months(self) -> int:
        return 12 * (self.accumulation_years + self.withdrawal_years)


@dataclass
class SimulationResult:
    """Simulated paths. Arrays have one row per path."""

    plan: SavingsPlan
    wealth: np.ndarray
    price_level: np.ndarray
    contributions: np.ndarray
    withdrawals: np.ndarray
    taxes: np.ndarray
    after_tax_value: np.ndarray
    depleted: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def real_wealth(self) -> np.ndarray:
        """Wealth in money of the start date."""
        return self.wealth / self.price_level

    def terminal(self, real: bool = True, after_tax: bool = False) -> np.ndarray:
        """Final value per path (after liquidation tax if ``after_tax``)."""
        value = self.after_tax_value if after_tax else self.wealth[:, -1]
        return value / self.price_level[:, -1] if real else value

    def wealth_at_retirement(self, real: bool = True) -> np.ndarray:
        """Wealth at the end of the accumulation phase."""
        m = self.plan.accumulation_months
        return self.real_wealth[:, m] if real else self.wealth[:, m]

    def percentiles(self, q=(5, 25, 50, 75, 95), real: bool = True) -> pd.DataFrame:
        """Wealth percentiles at each year end (rows: year, columns: percentile)."""
        data = self.real_wealth if real else self.wealth
        years = np.arange(0, data.shape[1], 12)
        table = np.percentile(data[:, years], q, axis=0).T
        return pd.DataFrame(table, index=pd.Index(years // 12, name="year"), columns=list(q))

    def success_rate(self) -> float:
        """Share of paths that never run out of money."""
        return float(1.0 - self.depleted.mean())

    def summary(self) -> pd.Series:
        """Headline statistics in money of the start date."""
        retire = self.wealth_at_retirement()
        out = {
            "invested_real": float(
                np.mean(np.sum(self.contributions / self.price_level[:, :-1], axis=1))
            ),
            "retirement_median": float(np.median(retire)),
            "retirement_p5": float(np.percentile(retire, 5)),
            "retirement_p95": float(np.percentile(retire, 95)),
            "terminal_median": float(np.median(self.terminal())),
            "terminal_after_tax_median": float(np.median(self.terminal(after_tax=True))),
            "taxes_paid_real_mean": float(
                np.mean(np.sum(self.taxes / self.price_level[:, 1:], axis=1))
            ),
            "success_rate": self.success_rate(),
        }
        return pd.Series(out)


def simulate(
    model, plan: SavingsPlan, n_paths: int = 10_000, seed: int | None = 0
) -> SimulationResult:
    """Run the Monte Carlo simulation of ``plan`` under return ``model``."""
    rng = np.random.default_rng(seed)
    n_months = plan.total_months
    acc_months = plan.accumulation_months
    returns, inflation = model.sample(n_paths, n_months, rng)
    returns = (1.0 + returns) * (1.0 - plan.annual_fee / 12.0) - 1.0

    price = np.ones((n_paths, n_months + 1))
    price[:, 1:] = np.cumprod(1.0 + returns, axis=1)
    cpi = np.ones((n_paths, n_months + 1))
    cpi[:, 1:] = np.cumprod(1.0 + inflation, axis=1)

    n_years = n_months // 12
    units = np.zeros((n_paths, n_years))
    cost = np.zeros((n_paths, n_years))
    prepaid = np.zeros((n_paths, n_years))
    vp_weight = np.zeros(n_paths)  # units bought this year x purchase-month factor

    wealth = np.zeros((n_paths, n_months + 1))
    contributions = np.zeros((n_paths, n_months))
    withdrawals = np.zeros((n_paths, n_months))
    taxes = np.zeros((n_paths, n_months))
    depleted = np.zeros(n_paths, dtype=bool)

    tax = plan.tax
    realized = np.zeros(n_paths)  # taxable realised gains of the current year
    carry = np.zeros(n_paths)  # loss carryforward (<= 0)
    # Last year's Vorabpauschale (after partial exemption). It is deemed received on the first
    # working day of the following year, so it is taxed with, and uses the allowance of, that year.
    pending_vp = np.zeros(n_paths)
    real_withdrawal = np.full(n_paths, plan.monthly_withdrawal)

    for t in range(n_months):
        year, month = divmod(t, 12)
        p = price[:, t]
        amount = np.zeros(n_paths)
        if t < acc_months:
            amount = plan.monthly_contribution * (cpi[:, t] if plan.contributions_indexed else 1.0)
        if t == 0:
            amount = amount + plan.initial
        if np.any(amount > 0):
            new_units = amount / p
            units[:, year] += new_units
            cost[:, year] += amount
            vp_weight += new_units * purchase_month_factor(month)
            contributions[:, t] = amount
        if t >= acc_months:
            if t == acc_months and plan.withdrawal_rate is not None:
                real_withdrawal = plan.withdrawal_rate / 12.0 * units.sum(axis=1) * p / cpi[:, t]
            proceeds, gain = sell_fifo(units, cost, prepaid, real_withdrawal * cpi[:, t], p)
            withdrawals[:, t] = proceeds
            depleted |= proceeds < real_withdrawal * cpi[:, t] * (1.0 - 1e-9)
            if tax is not None:
                realized += gain * (1.0 - tax.partial_exemption)

        wealth[:, t + 1] = units.sum(axis=1) * price[:, t + 1]

        if tax is not None and month == 11:
            start, end = price[:, t - 11], price[:, t + 1]
            vp_unit = vorabpauschale_per_unit(start, end, tax.base_rate)
            # The current year's lot holds this year's purchases (reduced weight).
            new_this_year = units[:, year]
            full_year_units = units.sum(axis=1) - new_this_year
            vp_total = vp_unit * (full_year_units + np.minimum(vp_weight, new_this_year))
            # Attribute the taxed Vorabpauschale to lots in proportion to their weight.
            lot_weight = units.copy()
            lot_weight[:, year] = np.minimum(vp_weight, new_this_year)
            share = np.divide(
                lot_weight,
                lot_weight.sum(axis=1, keepdims=True),
                out=np.zeros_like(lot_weight),
                where=lot_weight.sum(axis=1, keepdims=True) > 0,
            )
            prepaid += share * vp_total[:, None]
            taxable = pending_vp + realized
            pending_vp = vp_total * (1.0 - tax.partial_exemption)
            due, carry = tax.tax_due(taxable, carry)
            taxes[:, t] = due
            realized = np.zeros(n_paths)
            if tax.paid_from_portfolio:
                _, gain = sell_fifo(units, cost, prepaid, due, end)
                realized += gain * (1.0 - tax.partial_exemption)
                wealth[:, t + 1] = units.sum(axis=1) * end
            vp_weight = np.zeros(n_paths)

    final_price = price[:, -1]
    final_value = units.sum(axis=1) * final_price
    if tax is not None:
        _, gain = sell_fifo(units.copy(), cost.copy(), prepaid.copy(), final_value, final_price)
        # The sale happens in the year after the last year end, together with the
        # Vorabpauschale deemed received at its start.
        taxable = gain * (1.0 - tax.partial_exemption) + realized + pending_vp
        liquidation_tax, _ = tax.tax_due(taxable, carry)
        after_tax = final_value - liquidation_tax
    else:
        after_tax = final_value

    return SimulationResult(
        plan=plan,
        wealth=wealth,
        price_level=cpi,
        contributions=contributions,
        withdrawals=withdrawals,
        taxes=taxes,
        after_tax_value=after_tax,
        depleted=depleted,
        extra={"fund_price": price},
    )


# ---------------------------------------------------------------------------
# Analytical results (GBM) and helpers
# ---------------------------------------------------------------------------


def gbm_wealth_moments(model: GBM, monthly_contribution: float, months: int, initial: float = 0.0):
    """Exact mean and variance of terminal wealth under GBM with fixed contributions.

    With contributions ``c_k`` at the start of month ``k`` and iid monthly growth
    factors ``G`` (``E[G] = g``, ``E[G^2] = h``), ``W = sum_k c_k prod_{j>=k} G_j``,
    so ``E[W] = sum_k c_k g^(n-k)`` and
    ``E[W^2] = sum_{k,l} c_k c_l g^|k-l| h^(n - max(k,l))``.
    Contributions are not inflation-indexed and there are no costs or taxes.
    """
    a, b = model.log_mean, model.log_std
    g = np.exp(a + b**2 / 2.0)
    h = np.exp(2.0 * a + 2.0 * b**2)
    c = np.full(months, float(monthly_contribution))
    c[0] += initial
    k = np.arange(months)
    mean = float(np.sum(c * g ** (months - k)))
    kk, ll = np.meshgrid(k, k, indexing="ij")
    second = np.sum(np.outer(c, c) * g ** np.abs(kk - ll) * h ** (months - np.maximum(kk, ll)))
    return mean, float(second - mean**2)


def safe_withdrawal_rate(
    model,
    plan: SavingsPlan,
    target_success: float = 0.95,
    n_paths: int = 2000,
    seed: int = 0,
    bounds: tuple[float, float] = (0.001, 0.15),
) -> float:
    """Highest initial withdrawal rate whose success rate is at least ``target_success``.

    Uses the same random scenarios for every candidate rate (common random
    numbers), which makes the success rate monotone in the rate and the root
    search stable.
    """

    def excess(rate: float) -> float:
        trial = SavingsPlan(**{**plan.__dict__, "withdrawal_rate": rate})
        return simulate(model, trial, n_paths=n_paths, seed=seed).success_rate() - target_success

    lo, hi = bounds
    if excess(lo) < 0:
        return float("nan")
    if excess(hi) >= 0:
        return hi
    return float(optimize.brentq(lambda r: excess(r) + 1e-9, lo, hi, xtol=1e-4))
