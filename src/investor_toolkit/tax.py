"""German taxation of investment funds held by a private investor.

Implements the rules of the Investment Tax Act (InvStG, since 2018) that matter
for an accumulating ETF held in a taxable account:

* **Flat tax** (Abgeltungsteuer): 25 % plus 5.5 % solidarity surcharge on the
  tax, i.e. 26.375 %. Church tax is not modelled.
* **Partial exemption** (Teilfreistellung): 30 % of fund income is tax-free for
  equity funds (at least 51 % equities), 15 % for mixed funds (at least 25 %),
  0 % otherwise.
* **Saver's allowance** (Sparerpauschbetrag): EUR 1,000 per year for a single
  person.
* **Advance lump sum** (Vorabpauschale): accumulating funds are taxed each year
  on ``70 % x base rate x price at the start of the year``, capped at the
  year's actual price increase. Units bought during the year count 1/12 less
  for each full month before purchase. Vorabpauschalen already taxed are
  deducted from the gain when units are sold.
* **FIFO**: units are deemed sold in the order they were bought.

The base rate (Basiszins) is published annually by the Federal Ministry of
Finance; the default here is an assumed long-run value.

This module is a simplified model for planning purposes, not tax advice.
"""

from dataclasses import dataclass

import numpy as np

EQUITY_FUND = 0.30
MIXED_FUND = 0.15
OTHER_FUND = 0.0


@dataclass(frozen=True)
class GermanTax:
    """Tax parameters. ``partial_exemption`` is 0.30, 0.15 or 0.0 by fund type."""

    rate: float = 0.26375
    partial_exemption: float = EQUITY_FUND
    allowance: float = 1000.0
    base_rate: float = 0.025
    paid_from_portfolio: bool = True

    def tax_due(self, taxable_income: np.ndarray, loss_carryforward: np.ndarray):
        """Tax on one year's taxable income (after partial exemption).

        Losses carried forward offset income first, then the allowance applies.
        Returns ``(tax, new_loss_carryforward)``; the carryforward is <= 0.
        """
        net = taxable_income + loss_carryforward
        new_carry = np.minimum(net, 0.0)
        tax = self.rate * np.maximum(net - self.allowance, 0.0)
        return tax, new_carry


def vorabpauschale_per_unit(
    price_start: np.ndarray, price_end: np.ndarray, base_rate: float
) -> np.ndarray:
    """Vorabpauschale per fund unit held for the whole year (before partial exemption).

    ``min(0.7 * base_rate * P_start, max(P_end - P_start, 0))``.
    """
    base_income = 0.7 * base_rate * price_start
    return np.minimum(base_income, np.maximum(price_end - price_start, 0.0))


def purchase_month_factor(month_index: int) -> float:
    """Fraction of the Vorabpauschale for units bought in month ``month_index``
    (0 = January, 11 = December): reduced by 1/12 per full month before purchase."""
    return (12 - month_index) / 12.0


def sell_fifo(
    units: np.ndarray,
    cost: np.ndarray,
    prepaid: np.ndarray,
    amount: np.ndarray,
    price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sell fund units worth ``amount`` in first-in-first-out order, in place.

    Parameters
    ----------
    units, cost, prepaid
        Arrays of shape ``(paths, lots)`` with the units held, their total
        acquisition cost and the Vorabpauschalen already taxed on them. Lots are
        ordered oldest first. They are updated in place.
    amount
        Currency amount to raise per path; capped at the holding's value.
    price
        Current unit price per path.

    Returns
    -------
    (proceeds, gain)
        Proceeds and taxable gain before partial exemption
        (``proceeds - cost - prepaid Vorabpauschalen`` of the units sold).
    """
    held = units.sum(axis=1)
    to_sell = np.minimum(amount / price, held)
    cumulative = np.cumsum(units, axis=1)
    sold = np.clip(np.minimum(cumulative, to_sell[:, None]) - (cumulative - units), 0.0, None)
    fraction = np.divide(sold, units, out=np.zeros_like(units), where=units > 0)
    cost_sold = cost * fraction
    prepaid_sold = prepaid * fraction
    units -= sold
    cost -= cost_sold
    prepaid -= prepaid_sold
    proceeds = sold.sum(axis=1) * price
    gain = proceeds - cost_sold.sum(axis=1) - prepaid_sold.sum(axis=1)
    return proceeds, gain
