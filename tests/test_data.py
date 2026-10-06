import numpy as np
import pandas as pd
import pytest

from investor_toolkit import data

FRENCH_SAMPLE = """This file was created using the 202608 Bloomberg database.

Missing data are indicated by -99.99.

,Mkt-RF,SMB,RF
199007    ,0.77    ,0.60   ,0.68
199008    ,-10.00  ,-99.99 ,0.66

 Annual Factors: January-December
,Mkt-RF,SMB,RF
 1991,  20.00,  1.00,  5.00
"""

FRED_SAMPLE = "observation_date,DEXUSEU\n1999-01-04,1.1812\n1999-01-05,.\n1999-01-06,1.1760\n"

BUNDESBANK_SAMPLE = (
    '﻿"",BBSIS.D.SERIES,BBSIS.D.SERIES_FLAGS\n'
    '"",Term structure / residual maturity of 10.0 years / daily data,\n'
    "Decimals,2,\n"
    "1997-08-01,.,No value available\n"
    "1997-08-04,5.73,\n"
    "1997-08-05,5.70,\n"
)


def test_parse_french_reads_only_monthly_table_in_decimal():
    df = data.parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "RF"]
    assert list(df.index) == [pd.Timestamp("1990-07-31"), pd.Timestamp("1990-08-31")]
    assert df.loc["1990-07-31", "Mkt-RF"] == pytest.approx(0.0077)
    assert df.loc["1990-08-31", "Mkt-RF"] == pytest.approx(-0.10)
    assert np.isnan(df.loc["1990-08-31", "SMB"])


def test_parse_fred_drops_missing_values():
    series = data.parse_fred_csv(FRED_SAMPLE)
    assert series.name == "DEXUSEU"
    assert len(series) == 2
    assert series.iloc[-1] == pytest.approx(1.1760)


def test_parse_bundesbank_skips_metadata_and_missing():
    series = data.parse_bundesbank_csv(BUNDESBANK_SAMPLE.encode("utf-8"))
    assert list(series.index) == [pd.Timestamp("1997-08-04"), pd.Timestamp("1997-08-05")]
    assert series.iloc[0] == pytest.approx(5.73)


def test_cache_fetches_once(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"x": [1.0, 2.0]})

    path = tmp_path / "src" / "key.parquet"
    first = data._cached(path, fetch, refresh=False)
    second = data._cached(path, fetch, refresh=False)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1
    data._cached(path, fetch, refresh=True)
    assert len(calls) == 2


def test_cache_dir_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("INVESTOR_TOOLKIT_CACHE", str(tmp_path))
    assert data.default_cache_dir() == tmp_path


def test_convert_to_eur_divides_by_rate_and_bridges_holidays():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"US": [110.0, 121.0, 99.0], "EU": [1.0, 2.0, 3.0]}, index=dates)
    fx = pd.Series([1.10, 1.21], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    out = data.convert_to_eur(prices, {"US": "USD", "EU": "EUR"}, {"USD": fx})
    np.testing.assert_allclose(out["US"], [100.0, 100.0, 99.0 / 1.21])
    pd.testing.assert_series_equal(out["EU"], prices["EU"])


def test_quality_report_flags_stale_prices_and_jumps():
    index = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.1, 1.5, np.nan, 1.5]}, index=index)
    report = data.quality_report(prices, jump_threshold=0.2)
    assert report.loc["A", "longest_stale_run"] == 3
    assert report.loc["A", "n_jumps"] == 1
    assert report.loc["A", "missing_pct"] == pytest.approx(100 / 8)


def test_clean_prices_fills_short_gaps_only():
    index = pd.bdate_range("2020-01-01", periods=9)
    prices = pd.DataFrame(
        {"A": [1.0, np.nan, 2.0, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan]}, index=index
    )
    out = data.clean_prices(prices, max_fill=2)
    assert out["A"].iloc[1] == 1.0
    assert out["A"].iloc[3:5].tolist() == [2.0, 2.0]
    assert np.isnan(out["A"].iloc[5])
    assert out["A"].iloc[7:].isna().all()


@pytest.mark.network
def test_download_sources(tmp_path):
    prices = data.load_prices("EUNL.DE", start="2020-01-01", cache_dir=tmp_path)
    assert prices["EUNL.DE"].notna().sum() > 1000
    assert data.load_fred("DEXUSEU", cache_dir=tmp_path).index[0].year == 1999
    assert "Mkt-RF" in data.load_french("Europe_5_Factors", cache_dir=tmp_path).columns
    assert data.load_bundesbank(cache_dir=tmp_path).index[0].year == 1997
