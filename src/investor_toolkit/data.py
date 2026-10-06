"""Data access with a local Parquet cache.

Three public sources are supported:

* **Yahoo Finance** (via ``yfinance``): daily prices adjusted for splits and
  distributions, i.e. total-return price series in the listing currency.
* **FRED** (Federal Reserve Bank of St. Louis): exchange rates, interest rates
  and inflation indices.
* **Deutsche Bundesbank**: daily yields of German federal securities.
* **Kenneth R. French Data Library**: monthly factor returns.

Every download is cached as a Parquet file so that analyses are reproducible and
work offline. Pass ``refresh=True`` to re-download.
"""

import io
import os
import re
import ssl
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

BUNDESBANK_URL = "https://api.statistiken.bundesbank.de/rest/download/{key}?format=csv&lang=en"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset}_CSV.zip"
_USER_AGENT = {"User-Agent": "Mozilla/5.0 (investor-toolkit)"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Return the cache directory.

    Resolution order: the ``INVESTOR_TOOLKIT_CACHE`` environment variable, then a
    ``data/`` folder in the nearest parent directory containing ``pyproject.toml``,
    then ``~/.cache/investor_toolkit``.
    """
    env = os.environ.get("INVESTOR_TOOLKIT_CACHE")
    if env:
        return Path(env)
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "data"
    return Path.home() / ".cache" / "investor_toolkit"


def _cache_path(cache_dir: Path | None, source: str, key: str) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    return (cache_dir or default_cache_dir()) / source / f"{safe_key}.parquet"


def _cached(path: Path, fetch: Callable[[], pd.DataFrame], refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers=_USER_AGENT)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------


def _fetch_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    history = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if history.empty:
        raise ValueError(f"No price data returned for ticker {ticker!r}")
    close = history["Close"].astype(float)
    close.index = pd.DatetimeIndex(close.index.tz_localize(None).normalize(), name="date")
    close = close[~close.index.duplicated(keep="last")]
    return close.to_frame(ticker)


def load_prices(
    tickers: str | Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load daily total-return prices from Yahoo Finance.

    Prices are closing prices adjusted for splits and distributions, so their
    percentage changes are total returns. They are quoted in each instrument's
    listing currency.

    Parameters
    ----------
    tickers
        One ticker or an iterable of tickers (Yahoo symbols, e.g. ``"EUNL.DE"``).
    start, end
        Optional inclusive date bounds applied after loading the full history.
    refresh
        Re-download instead of using the cache.
    cache_dir
        Cache location; see :func:`default_cache_dir`.

    Returns
    -------
    pandas.DataFrame
        Prices indexed by date, one column per ticker, outer-joined on dates.
    """
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    frames = [
        _cached(_cache_path(cache_dir, "yahoo", t), lambda t=t: _fetch_yahoo(t), refresh)
        for t in tickers
    ]
    prices = pd.concat(frames, axis=1, join="outer", sort=True)
    return prices.loc[start:end]


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def parse_fred_csv(content: bytes | str) -> pd.Series:
    """Parse a FRED ``fredgraph.csv`` download into a float Series."""
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    df = pd.read_csv(io.StringIO(content))
    date_col, value_col = df.columns[0], df.columns[1]
    series = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"),
        name=str(value_col),
    )
    return series.dropna()


def load_fred(series_id: str, *, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Load a FRED series (e.g. ``"DEXUSEU"``) as a float Series indexed by date."""

    def fetch() -> pd.DataFrame:
        return parse_fred_csv(_http_get(FRED_URL.format(series_id=series_id))).to_frame()

    df = _cached(_cache_path(cache_dir, "fred", series_id), fetch, refresh)
    return df.iloc[:, 0].rename(series_id)


# ---------------------------------------------------------------------------
# Deutsche Bundesbank
# ---------------------------------------------------------------------------

BUND_10Y_DAILY = "BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"
"""Daily 10-year spot rate of German federal securities (Svensson method), percent."""


def parse_bundesbank_csv(content: bytes | str) -> pd.Series:
    """Parse a Bundesbank time-series CSV download into a float Series.

    Metadata rows are skipped, as are rows whose value is ``"."`` (no value).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    dates, values = [], []
    for line in content.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", parts[0]):
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
        dates.append(parts[0])
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.Series(values, index=index, dtype=float)


def load_bundesbank(
    key: str = BUND_10Y_DAILY, *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.Series:
    """Load a Bundesbank time series (default: daily 10-year Bund spot rate, percent)."""

    def fetch() -> pd.DataFrame:
        return parse_bundesbank_csv(_http_get(BUNDESBANK_URL.format(key=key))).to_frame("value")

    df = _cached(_cache_path(cache_dir, "bundesbank", key), fetch, refresh)
    return df.iloc[:, 0].rename(key.split("/")[-1])


# ---------------------------------------------------------------------------
# Kenneth R. French Data Library
# ---------------------------------------------------------------------------


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first (monthly) table of a Kenneth French CSV file.

    Values are converted from percent to decimal returns, the missing-value marker
    ``-99.99`` becomes NaN, and the index is the calendar month end.
    """
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    columns = [c.strip() for c in lines[header_idx].split(",")[1:]]
    dates, rows = [], []
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if not re.fullmatch(r"\d{6}", parts[0]):
            break
        dates.append(parts[0])
        rows.append([float(p) for p in parts[1 : len(columns) + 1]])
    values = np.asarray(rows, dtype=float)
    values[np.isclose(values, -99.99)] = np.nan
    index = pd.DatetimeIndex(
        pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0), name="date"
    )
    return pd.DataFrame(values / 100.0, index=index, columns=columns)


def load_french(
    dataset: str, *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Load monthly factor returns from the Kenneth R. French Data Library.

    Parameters
    ----------
    dataset
        File stem without ``_CSV.zip``, e.g. ``"Europe_5_Factors"``,
        ``"Developed_5_Factors"``, ``"Europe_Mom_Factor"`` or
        ``"F-F_Research_Data_5_Factors_2x3"``.

    Returns
    -------
    pandas.DataFrame
        Decimal monthly returns (USD for international datasets), month-end index.
    """

    def fetch() -> pd.DataFrame:
        archive = zipfile.ZipFile(io.BytesIO(_http_get(FRENCH_URL.format(dataset=dataset))))
        text = archive.read(archive.namelist()[0]).decode("latin-1")
        return parse_french_csv(text)

    return _cached(_cache_path(cache_dir, "french", dataset), fetch, refresh)


# ---------------------------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------------------------


def load_fx_per_eur(
    currency: str, *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.Series:
    """Daily exchange rate in units of ``currency`` per 1 EUR.

    USD uses the FRED series ``DEXUSEU`` (from 1999); other currencies use Yahoo
    Finance ``EUR<CCY>=X``.
    """
    currency = currency.upper()
    if currency == "EUR":
        raise ValueError("No exchange rate needed for EUR")
    if currency == "USD":
        return load_fred("DEXUSEU", refresh=refresh, cache_dir=cache_dir).rename("USD")
    symbol = f"EUR{currency}=X"
    return load_prices(symbol, refresh=refresh, cache_dir=cache_dir)[symbol].rename(currency)


def convert_to_eur(
    prices: pd.DataFrame,
    currencies: Mapping[str, str],
    fx_per_eur: Mapping[str, pd.Series],
    max_fill_days: int = 5,
) -> pd.DataFrame:
    """Convert prices quoted in foreign currencies into EUR.

    ``price_eur = price_local / fx``, where ``fx`` is local currency per EUR. The
    exchange rate is forward-filled onto the price dates for at most
    ``max_fill_days`` observations to bridge holidays.
    """
    out = prices.copy()
    for column in prices.columns:
        currency = currencies[column].upper()
        if currency == "EUR":
            continue
        fx = fx_per_eur[currency].reindex(prices.index.union(fx_per_eur[currency].index))
        fx = fx.ffill(limit=max_fill_days).reindex(prices.index)
        out[column] = prices[column] / fx
    return out


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def quality_report(prices: pd.DataFrame, jump_threshold: float = 0.15) -> pd.DataFrame:
    """Summarise data quality per price series.

    Columns: first/last valid date, number of observations, share of missing
    values inside the series' own date range, largest absolute one-period return,
    number of returns larger than ``jump_threshold`` in absolute value, and the
    longest run of unchanged consecutive prices (stale quotes).
    """
    records = {}
    for column in prices.columns:
        series = prices[column]
        first, last = series.first_valid_index(), series.last_valid_index()
        live = series.loc[first:last]
        valid = live.dropna()
        returns = valid.pct_change().dropna()
        records[column] = {
            "first": first,
            "last": last,
            "n_obs": int(valid.size),
            "missing_pct": 100.0 * float(live.isna().mean()),
            "max_abs_return": float(returns.abs().max()) if returns.size else np.nan,
            "n_jumps": int((returns.abs() > jump_threshold).sum()),
            "longest_stale_run": _longest_run(np.diff(valid.to_numpy()) == 0.0),
        }
    return pd.DataFrame.from_dict(records, orient="index")


def clean_prices(prices: pd.DataFrame, max_fill: int = 5) -> pd.DataFrame:
    """Forward-fill short gaps inside each series' live range.

    Gaps longer than ``max_fill`` observations and values before a series' first
    or after its last valid observation are left as NaN.
    """
    filled = prices.ffill(limit=max_fill)
    for column in prices.columns:
        last = prices[column].last_valid_index()
        if last is not None:
            filled.loc[filled.index > last, column] = np.nan
    return filled
