import numpy as np
import pandas as pd
import pytest

from investor_toolkit import proxies


@pytest.mark.validation
def test_annuity_factor_matches_explicit_sum_and_zero_limit():
    y, n = 0.03, 10
    explicit = sum((1 + y) ** -k for k in range(1, n + 1))
    assert proxies.annuity_factor(y, n) == pytest.approx(explicit)
    assert proxies.annuity_factor(0.0, 7.5) == pytest.approx(7.5)
    assert proxies.annuity_factor(1e-12, 7.5) == pytest.approx(7.5)


@pytest.mark.validation
def test_par_bond_price_matches_discounted_cash_flows():
    # With dt = 1 year the bond has exactly 9 annual coupons left.
    c, y = 0.04, 0.05
    cash_flows = sum(c / (1 + y) ** k for k in range(1, 10)) + 1 / (1 + y) ** 9
    expected = cash_flows - 1 + c
    assert proxies.par_bond_return(c, y, maturity=10, dt=1.0) == pytest.approx(expected)


@pytest.mark.validation
def test_unchanged_yield_earns_exactly_the_coupon():
    for y in [-0.005, 0.0, 0.02, 0.08]:
        assert proxies.par_bond_return(y, y) == pytest.approx(y / 12, abs=1e-14)


@pytest.mark.validation
def test_small_yield_change_follows_modified_duration():
    y0, dy = 0.03, 1e-5
    price_change = proxies.par_bond_return(y0, y0 + dy) - y0 / 12
    duration = proxies.annuity_factor(y0, 10 - 1 / 12)
    assert -price_change / dy == pytest.approx(duration, rel=1e-3)


@pytest.mark.network
@pytest.mark.validation
def test_proxies_track_investable_etfs():
    from investor_toolkit.data import clean_prices, load_prices
    from investor_toolkit.returns import monthly_returns

    long = proxies.long_history()
    etf = monthly_returns(clean_prices(load_prices(["EUNL.DE", "X03G.DE"])))
    both = pd.concat([long, etf], axis=1, join="inner").dropna()
    assert len(both) > 120
    assert both["world_equity"].corr(both["EUNL.DE"]) > 0.95
    assert both["bund_10y"].corr(both["X03G.DE"]) > 0.9
    tracking_error = (both["world_equity"] - both["EUNL.DE"]).std() * np.sqrt(12)
    assert tracking_error < 0.04
