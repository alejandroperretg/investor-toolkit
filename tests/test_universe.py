import pytest

from investor_toolkit.universe import ASSETS, LONG_HISTORY_PORTFOLIOS, MODEL_PORTFOLIOS, Portfolio


def test_model_portfolios_use_known_assets():
    for portfolio in MODEL_PORTFOLIOS.values():
        assert set(portfolio.tickers) <= set(ASSETS)


def test_long_history_portfolios_use_proxy_columns():
    for portfolio in LONG_HISTORY_PORTFOLIOS.values():
        assert set(portfolio.tickers) <= {"world_equity", "bund_10y", "cash"}


def test_portfolio_rejects_invalid_weights():
    with pytest.raises(ValueError):
        Portfolio("bad", "", {"A": 0.5, "B": 0.4})
    with pytest.raises(ValueError):
        Portfolio("short", "", {"A": 1.5, "B": -0.5})


def test_weight_vector_fills_missing_assets_with_zero():
    weights = MODEL_PORTFOLIOS["balanced"].weight_vector(["XGLE.DE", "4GLD.DE", "EUNL.DE"])
    assert weights.to_dict() == {"XGLE.DE": 0.4, "4GLD.DE": 0.0, "EUNL.DE": 0.6}
