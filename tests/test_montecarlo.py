from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from investor_toolkit import montecarlo as mc
from investor_toolkit.tax import GermanTax


@dataclass(frozen=True)
class Constant:
    """Deterministic model: the same monthly return and inflation on every path."""

    monthly_return: float = 0.0
    monthly_inflation: float = 0.0

    def sample(self, n_paths, n_months, rng):
        return (
            np.full((n_paths, n_months), self.monthly_return),
            np.full((n_paths, n_months), self.monthly_inflation),
        )


NO_ALLOWANCE = GermanTax(allowance=0.0, paid_from_portfolio=False)


# ---------------------------------------------------------------------------
# Deterministic accounting
# ---------------------------------------------------------------------------


def test_zero_returns_accumulate_contributions():
    plan = mc.SavingsPlan(monthly_contribution=100.0, accumulation_years=2)
    result = mc.simulate(Constant(), plan, n_paths=3)
    np.testing.assert_allclose(result.wealth[:, -1], 2400.0)
    np.testing.assert_allclose(result.contributions.sum(axis=1), 2400.0)


def test_indexed_contributions_keep_real_value_constant():
    plan = mc.SavingsPlan(monthly_contribution=100.0, accumulation_years=3)
    result = mc.simulate(Constant(monthly_inflation=0.003), plan, n_paths=2)
    real = result.contributions / result.price_level[:, :-1]
    np.testing.assert_allclose(real, 100.0)


def test_withdrawals_deplete_exactly_when_money_runs_out():
    """A retiree with 12,000 and zero returns can withdraw 1,000 for 12 months, not 1,100."""
    base = {"initial": 12_000.0, "monthly_contribution": 0.0, "accumulation_years": 0}
    enough = mc.simulate(
        Constant(), mc.SavingsPlan(**base, withdrawal_years=1, monthly_withdrawal=1000.0), n_paths=2
    )
    assert enough.success_rate() == 1.0
    np.testing.assert_allclose(enough.withdrawals.sum(axis=1), 12_000.0)
    assert enough.wealth[0, -1] == pytest.approx(0.0, abs=1e-6)
    short = mc.simulate(
        Constant(), mc.SavingsPlan(**base, withdrawal_years=1, monthly_withdrawal=1100.0), n_paths=2
    )
    assert short.success_rate() == 0.0
    np.testing.assert_allclose(short.withdrawals.sum(axis=1), 12_000.0)


@pytest.mark.validation
def test_safe_withdrawal_rate_with_zero_returns_is_one_over_years():
    plan = mc.SavingsPlan(
        initial=100_000.0, monthly_contribution=0.0, accumulation_years=1, withdrawal_years=25
    )
    rate = mc.safe_withdrawal_rate(Constant(), plan, target_success=0.95, n_paths=4)
    assert rate == pytest.approx(1 / 25, abs=2e-4)


# ---------------------------------------------------------------------------
# German tax
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_vorabpauschale_and_liquidation_tax_by_hand():
    monthly = 1.06 ** (1 / 12) - 1
    plan = mc.SavingsPlan(
        initial=10_000.0, monthly_contribution=0.0, accumulation_years=1, tax=NO_ALLOWANCE
    )
    result = mc.simulate(Constant(monthly), plan, n_paths=1)
    vorabpauschale = 10_000 * 0.7 * 0.025  # 175, below the 600 gain
    expected_vp_tax = 0.26375 * 0.7 * vorabpauschale
    assert result.taxes.sum() == pytest.approx(expected_vp_tax)
    gain_at_sale = 10_600 - 10_000 - vorabpauschale
    expected_after_tax = 10_600 - 0.26375 * 0.7 * gain_at_sale
    assert result.after_tax_value[0] == pytest.approx(expected_after_tax)


@pytest.mark.validation
def test_vorabpauschale_reduced_for_purchases_during_the_year():
    monthly = 0.01
    plan = mc.SavingsPlan(
        monthly_contribution=1000.0,
        contributions_indexed=False,
        accumulation_years=1,
        tax=NO_ALLOWANCE,
    )
    result = mc.simulate(Constant(monthly), plan, n_paths=1)
    price = (1 + monthly) ** np.arange(13)
    units = 1000.0 / price[:12]
    factors = (12 - np.arange(12)) / 12
    vp_unit = min(0.7 * 0.025 * price[0], price[12] - price[0])
    expected = 0.26375 * 0.7 * vp_unit * np.sum(units * factors)
    assert result.taxes.sum() == pytest.approx(expected)


@pytest.mark.validation
def test_lifetime_tax_equals_flat_tax_on_total_gain():
    """Vorabpauschalen are prepayments: with taxes paid externally, no allowance and
    rising prices, lifetime tax = rate x (1 - partial exemption) x total gain."""
    plan = mc.SavingsPlan(
        initial=5_000.0, monthly_contribution=300.0, accumulation_years=15, tax=NO_ALLOWANCE
    )
    result = mc.simulate(Constant(0.005, 0.002), plan, n_paths=1)
    final = result.wealth[0, -1]
    gain = final - result.contributions.sum()
    lifetime_tax = result.taxes.sum() + (final - result.after_tax_value[0])
    assert lifetime_tax == pytest.approx(0.26375 * 0.7 * gain, rel=1e-10)


def test_allowance_covers_small_vorabpauschale():
    plan = mc.SavingsPlan(
        initial=10_000.0, monthly_contribution=0.0, accumulation_years=3, tax=GermanTax()
    )
    result = mc.simulate(Constant(0.005), plan, n_paths=1)
    assert result.taxes.sum() == 0.0


def test_tax_paid_from_portfolio_reduces_wealth():
    common = {"initial": 50_000.0, "monthly_contribution": 0.0, "accumulation_years": 10}
    external = mc.simulate(Constant(0.006), mc.SavingsPlan(**common, tax=NO_ALLOWANCE), n_paths=1)
    internal_tax = GermanTax(allowance=0.0, paid_from_portfolio=True)
    internal = mc.simulate(Constant(0.006), mc.SavingsPlan(**common, tax=internal_tax), n_paths=1)
    assert internal.wealth[0, -1] < external.wealth[0, -1]
    assert internal.taxes.sum() > 0


# ---------------------------------------------------------------------------
# Return models against theory
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_lump_sum_terminal_wealth_is_lognormal():
    model = mc.GBM(expected_return=0.07, volatility=0.15)
    plan = mc.SavingsPlan(initial=1.0, monthly_contribution=0.0, accumulation_years=10)
    result = mc.simulate(model, plan, n_paths=20_000, seed=1)
    n = 120
    log_w = np.log(result.wealth[:, -1])
    dist = stats.norm(n * model.log_mean, np.sqrt(n) * model.log_std)
    assert stats.kstest(log_w, dist.cdf).pvalue > 0.01
    assert result.wealth[:, -1].mean() == pytest.approx(1.07**10, rel=0.01)


@pytest.mark.validation
def test_savings_plan_moments_match_closed_form():
    model = mc.GBM(expected_return=0.06, volatility=0.18)
    plan = mc.SavingsPlan(
        initial=1000.0,
        monthly_contribution=100.0,
        contributions_indexed=False,
        accumulation_years=20,
    )
    result = mc.simulate(model, plan, n_paths=40_000, seed=2)
    mean, var = mc.gbm_wealth_moments(model, 100.0, 240, initial=1000.0)
    terminal = result.wealth[:, -1]
    standard_error = np.sqrt(var / terminal.size)
    assert abs(terminal.mean() - mean) < 3 * standard_error
    assert terminal.var() == pytest.approx(var, rel=0.06)


@pytest.mark.validation
def test_monte_carlo_error_decreases_as_inverse_square_root_of_paths():
    model = mc.GBM(expected_return=0.07, volatility=0.2)
    plan = mc.SavingsPlan(initial=1.0, monthly_contribution=0.0, accumulation_years=5)
    exact = 1.07**5
    sizes = np.array([200, 800, 3200, 12800])
    rms = []
    for n in sizes:
        errors = [
            mc.simulate(model, plan, int(n), seed=s).wealth[:, -1].mean() - exact for s in range(40)
        ]
        rms.append(np.sqrt(np.mean(np.square(errors))))
    slope = np.polyfit(np.log(sizes), np.log(rms), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.1)


@pytest.mark.validation
def test_antithetic_variates_reduce_variance():
    plan = mc.SavingsPlan(initial=1.0, monthly_contribution=0.0, accumulation_years=5)
    plain = mc.GBM(0.07, 0.2)
    anti = mc.GBM(0.07, 0.2, antithetic=True)
    est_plain = [mc.simulate(plain, plan, 500, seed=s).wealth[:, -1].mean() for s in range(100)]
    est_anti = [mc.simulate(anti, plan, 500, seed=s).wealth[:, -1].mean() for s in range(100)]
    assert np.var(est_plain) / np.var(est_anti) > 2.0


@pytest.mark.validation
def test_student_t_matches_gbm_log_moments_with_fatter_tails():
    rng = np.random.default_rng(3)
    t_model = mc.StudentT(expected_return=0.07, volatility=0.15, df=4)
    returns, _ = t_model.sample(2000, 120, rng)
    log_r = np.log1p(returns).ravel()
    assert log_r.mean() == pytest.approx(t_model.log_mean, abs=3e-4)
    assert log_r.std() == pytest.approx(t_model.log_std, rel=0.02)
    assert stats.kurtosis(log_r) > 1.0


@pytest.mark.validation
def test_bootstrap_preserves_joint_observations_and_mean():
    rng = np.random.default_rng(4)
    index = pd.date_range("2000-01-31", periods=240, freq="ME")
    r = pd.Series(rng.normal(0.006, 0.04, 240), index=index)
    infl = pd.Series(rng.normal(0.002, 0.003, 240), index=index)
    model = mc.Bootstrap(r, infl, mean_block=12)
    sr, si = model.sample(3000, 120, np.random.default_rng(5))
    pairs = set(zip(r.round(12), infl.round(12), strict=True))
    sampled = set(zip(sr[:50].ravel().round(12), si[:50].ravel().round(12), strict=True))
    assert sampled <= pairs
    assert sr.mean() == pytest.approx(r.mean(), abs=5e-4)


def test_gbm_calibration_round_trip():
    model = mc.GBM(0.08, 0.16)
    returns, _ = model.sample(1, 200_000, np.random.default_rng(6))
    fitted = mc.GBM.from_returns(returns.ravel())
    assert fitted.expected_return == pytest.approx(0.08, abs=0.003)
    assert fitted.volatility == pytest.approx(0.16, rel=0.01)


def test_summary_and_percentiles():
    plan = mc.SavingsPlan(
        monthly_contribution=500.0,
        accumulation_years=5,
        withdrawal_years=5,
        withdrawal_rate=0.04,
        tax=GermanTax(),
    )
    result = mc.simulate(mc.GBM(), plan, n_paths=500)
    table = result.percentiles()
    assert list(table.index) == list(range(11))
    assert (table[5] <= table[50]).all() and (table[50] <= table[95]).all()
    summary = result.summary()
    assert summary["invested_real"] == pytest.approx(500 * 60)
    assert 0.0 <= summary["success_rate"] <= 1.0
