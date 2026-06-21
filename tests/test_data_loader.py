"""Tests for the data-loader normalisation across source schemas."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.data_loader import _normalise


def test_normalise_nse_shape():
    """niftyindices.com rows: upper-case OHLC, 'HistoricalDate', no volume."""
    records = [
        {"HistoricalDate": "03 Nov 1995", "OPEN": "1000.0", "HIGH": "1010.5",
         "LOW": "995.2", "CLOSE": "1007.3", "INDEX_NAME": "NIFTY 50"},
        {"HistoricalDate": "06 Nov 1995", "OPEN": "1007.3", "HIGH": "1020.0",
         "LOW": "1002.0", "CLOSE": "1015.8", "INDEX_NAME": "NIFTY 50"},
    ]
    df = _normalise(pd.DataFrame(records))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["volume"] == 0).all()          # index data carries no volume
    assert str(df.index[0].date()) == "1995-11-03"
    assert df["close"].iloc[0] == pytest.approx(1007.3)


def test_normalise_csv_shape():
    """A plain user CSV with standard columns still works."""
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Open": [100, 101], "High": [102, 103], "Low": [99, 100],
            "Close": [101, 102], "Volume": [1000, 1100],
        }
    )
    out = _normalise(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["volume"].iloc[1] == 1100


def test_normalise_requires_ohlc():
    df = pd.DataFrame({"Date": ["2020-01-01"], "Close": [100]})
    with pytest.raises(ValueError):
        _normalise(df)
