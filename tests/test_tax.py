import numpy as np
import pytest

from investor_toolkit import tax


def test_vorabpauschale_is_capped_by_actual_gain():
    start = np.array([100.0, 100.0, 100.0])
    end = np.array([110.0, 100.5, 90.0])
    np.testing.assert_allclose(tax.vorabpauschale_per_unit(start, end, 0.025), [1.75, 0.5, 0.0])


def test_purchase_month_factor():
    assert tax.purchase_month_factor(0) == 1.0
    assert tax.purchase_month_factor(6) == 0.5
    assert tax.purchase_month_factor(11) == pytest.approx(1 / 12)


def test_sell_fifo_sells_oldest_lots_first():
    units = np.array([[10.0, 10.0]])
    cost = np.array([[100.0, 150.0]])
    prepaid = np.array([[5.0, 0.0]])
    proceeds, gain = tax.sell_fifo(units, cost, prepaid, np.array([300.0]), np.array([20.0]))
    assert proceeds[0] == pytest.approx(300.0)
    # Cost of units sold: 100 (lot 0) + 75 (half of lot 1); prepaid Vorabpauschale 5.
    assert gain[0] == pytest.approx(300.0 - 175.0 - 5.0)
    np.testing.assert_allclose(units, [[0.0, 5.0]])
    np.testing.assert_allclose(cost, [[0.0, 75.0]])
    np.testing.assert_allclose(prepaid, [[0.0, 0.0]])


def test_sell_fifo_caps_at_holding_value():
    units, cost, prepaid = np.array([[2.0]]), np.array([[10.0]]), np.array([[0.0]])
    proceeds, _ = tax.sell_fifo(units, cost, prepaid, np.array([1e9]), np.array([10.0]))
    assert proceeds[0] == pytest.approx(20.0)
    assert units[0, 0] == 0.0


def test_tax_due_applies_carryforward_then_allowance():
    rules = tax.GermanTax(allowance=1000.0)
    due, carry = rules.tax_due(np.array([3000.0, 500.0, -200.0]), np.array([-500.0, 0.0, 0.0]))
    np.testing.assert_allclose(due, [0.26375 * 1500.0, 0.0, 0.0])
    np.testing.assert_allclose(carry, [0.0, 0.0, -200.0])
