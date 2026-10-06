"""Investable assets and model portfolios.

All instruments are traded on Xetra in EUR. Prices from Yahoo Finance are net of
fund costs (the ETF's NAV already reflects its total expense ratio).

The *concentrated* portfolio holds five large German companies from different
sectors. It illustrates single-stock risk; any fixed basket of named stocks is
subject to selection bias and should not be read as a forecast.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from investor_toolkit.data import clean_prices, load_prices
from investor_toolkit.returns import monthly_returns


@dataclass(frozen=True)
class Asset:
    """An investable instrument."""

    ticker: str
    name: str
    asset_class: str


ASSETS: dict[str, Asset] = {
    a.ticker: a
    for a in [
        Asset("EUNL.DE", "iShares Core MSCI World UCITS ETF (Acc)", "equity"),
        Asset("XGLE.DE", "Xtrackers II Eurozone Government Bond UCITS ETF 1C", "government_bond"),
        Asset(
            "IBCI.DE", "iShares Euro Inflation Linked Govt Bond UCITS ETF (Acc)", "inflation_linked"
        ),
        Asset("4GLD.DE", "Xetra-Gold", "gold"),
        Asset("SAP.DE", "SAP SE", "single_stock"),
        Asset("SIE.DE", "Siemens AG", "single_stock"),
        Asset("ALV.DE", "Allianz SE", "single_stock"),
        Asset("BAYN.DE", "Bayer AG", "single_stock"),
        Asset("DBK.DE", "Deutsche Bank AG", "single_stock"),
    ]
}


@dataclass(frozen=True)
class Portfolio:
    """A named set of target weights that sum to one."""

    name: str
    description: str
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights of {self.name!r} sum to {total}, not 1")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError(f"Portfolio {self.name!r} has negative weights")

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)

    def weight_vector(self, columns) -> pd.Series:
        """Weights aligned to ``columns`` (zero for assets not held)."""
        return pd.Series({c: self.weights.get(c, 0.0) for c in columns}, dtype=float)


_STOCKS = ["SAP.DE", "SIE.DE", "ALV.DE", "BAYN.DE", "DBK.DE"]

MODEL_PORTFOLIOS: dict[str, Portfolio] = {
    p.name: p
    for p in [
        Portfolio(
            "defensive",
            "25% global equity, 75% euro-area government bonds",
            {"EUNL.DE": 0.25, "XGLE.DE": 0.75},
        ),
        Portfolio(
            "balanced",
            "60% global equity, 40% euro-area government bonds",
            {"EUNL.DE": 0.60, "XGLE.DE": 0.40},
        ),
        Portfolio("growth", "100% global equity", {"EUNL.DE": 1.0}),
        Portfolio(
            "diversified",
            "40% equity, 30% nominal bonds, 15% inflation-linked bonds, 15% gold",
            {"EUNL.DE": 0.40, "XGLE.DE": 0.30, "IBCI.DE": 0.15, "4GLD.DE": 0.15},
        ),
        Portfolio(
            "concentrated",
            "Equal weights in five large German stocks",
            {t: 0.2 for t in _STOCKS},
        ),
    ]
}

LONG_HISTORY_PORTFOLIOS: dict[str, Portfolio] = {
    p.name: p
    for p in [
        Portfolio(
            "defensive", "25% world equity, 75% 10y Bund", {"world_equity": 0.25, "bund_10y": 0.75}
        ),
        Portfolio(
            "balanced", "60% world equity, 40% 10y Bund", {"world_equity": 0.60, "bund_10y": 0.40}
        ),
        Portfolio("growth", "100% world equity", {"world_equity": 1.0}),
    ]
}


def load_asset_prices(
    tickers=None, *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Daily cleaned EUR prices for ``tickers`` (default: the whole universe)."""
    tickers = list(ASSETS) if tickers is None else list(tickers)
    return clean_prices(load_prices(tickers, refresh=refresh, cache_dir=cache_dir))


def load_asset_returns(
    tickers=None, *, common_start: bool = True, refresh: bool = False, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Monthly EUR returns for ``tickers``.

    With ``common_start=True`` the sample starts at the first month in which all
    assets have a return, so every column covers the same period.
    """
    returns = monthly_returns(load_asset_prices(tickers, refresh=refresh, cache_dir=cache_dir))
    return returns.dropna() if common_start else returns
