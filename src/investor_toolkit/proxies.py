"""Long-history monthly return series in EUR, built from official sources.

UCITS ETFs listed in Europe mostly start trading between 2008 and 2014, which
misses the 2000-2002 equity bear market and most of the 2008 crisis. The series
below extend the history back to the introduction of the euro (January 1999):

* ``world_equity``: developed-market equities, from the Fama-French *Developed*
  market factor (``Mkt-RF + RF``, USD, value-weighted, dividends reinvested),
  converted to EUR with the FRED ``DEXUSEU`` exchange rate.
* ``bund_10y``: a synthetic constant-maturity 10-year German government bond,
  priced from month-end values of the Bundesbank daily 10-year yield.
* ``cash``: German 3-month interbank rate (FRED ``IR3TIB01DEM156N``).
* ``inflation``: euro-area HICP monthly change (FRED ``CP0000EZ19M086NEST``).

Limitations: the Bundesbank series is a zero-coupon (spot) yield and is used as an
approximation of the 10-year par yield. The cash rate is a monthly average.
Index proxies carry no fund costs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from investor_toolkit.data import load_bundesbank, load_fred, load_french

LONG_HISTORY_START = "1999-01-31"


def _month_end(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index + pd.offsets.MonthEnd(0), name="date")
    return out


def annuity_factor(y: np.ndarray, maturity: np.ndarray | float) -> np.ndarray:
    """Present value of 1 per year for ``maturity`` years at annual yield ``y``.

    ``a(y, T) = (1 - (1 + y)^-T) / y``, with the limit ``a(0, T) = T``.
    """
    y = np.asarray(y, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    small = np.abs(y) < 1e-10
    safe_y = np.where(small, 1.0, y)
    factor = -np.expm1(-maturity * np.log1p(safe_y)) / safe_y
    return np.where(small, maturity, factor)


def par_bond_return(
    yield_prev: np.ndarray, yield_now: np.ndarray, maturity: float = 10.0, dt: float = 1.0 / 12
) -> np.ndarray:
    """One-period total return of a constant-maturity par bond.

    At the start of the period a bond with ``maturity`` years and annual coupon
    equal to the prevailing yield is bought at par (price 1). At the end of the
    period it has ``maturity - dt`` years left and is valued at the new yield; the
    coupon accrued over ``dt`` is added. The bond is then rolled into a new par
    bond. Yields are annual decimals.

    ``R = c * a(y_1, M - dt) + (1 + y_1)^-(M - dt) - 1 + c * dt``, with ``c = y_0``.

    For small yield changes, ``R ~ y_0 * dt - D_mod * (y_1 - y_0)``, where
    ``D_mod = a(y_0, M)`` is the modified duration of an annual-pay par bond.
    """
    c = np.asarray(yield_prev, dtype=float)
    y = np.asarray(yield_now, dtype=float)
    remaining = maturity - dt
    price = c * annuity_factor(y, remaining) + np.exp(-remaining * np.log1p(y))
    return price - 1.0 + c * dt


def world_equity_eur(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Monthly EUR total return of developed-market equities (Fama-French)."""
    factors = load_french("Developed_5_Factors", refresh=refresh, cache_dir=cache_dir)
    usd_return = factors["Mkt-RF"] + factors["RF"]
    fx = load_fred("DEXUSEU", refresh=refresh, cache_dir=cache_dir).resample("ME").last()
    eur_return = (1.0 + usd_return) * (fx.shift(1) / fx).reindex(usd_return.index) - 1.0
    return eur_return.dropna().rename("world_equity")


def bund_10y(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Monthly total return of a synthetic constant-maturity 10-year Bund."""
    daily = load_bundesbank(refresh=refresh, cache_dir=cache_dir) / 100
    yields = daily.resample("ME").last()
    returns = par_bond_return(yields.shift(1).to_numpy(), yields.to_numpy(), maturity=10.0)
    return pd.Series(returns, index=yields.index, name="bund_10y").dropna()


def eur_cash(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Monthly return of EUR cash: last month's 3-month rate divided by 12."""
    rate = _month_end(load_fred("IR3TIB01DEM156N", refresh=refresh, cache_dir=cache_dir)) / 100
    return (rate.shift(1) / 12.0).dropna().rename("cash")


def euro_inflation(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Monthly euro-area HICP inflation (simple change of the price index)."""
    hicp = _month_end(load_fred("CP0000EZ19M086NEST", refresh=refresh, cache_dir=cache_dir))
    return (hicp / hicp.shift(1) - 1.0).dropna().rename("inflation")


def long_history(
    start: str = LONG_HISTORY_START, *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Aligned monthly EUR returns: world equity, 10y Bund, cash and inflation."""
    kwargs = {"refresh": refresh, "cache_dir": cache_dir}
    frame = pd.concat(
        [
            world_equity_eur(**kwargs),
            bund_10y(**kwargs),
            eur_cash(**kwargs),
            euro_inflation(**kwargs),
        ],
        axis=1,
        join="inner",
    )
    return frame.loc[start:]
