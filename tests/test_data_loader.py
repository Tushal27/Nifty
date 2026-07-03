"""Tests for the data-loader normalisation across source schemas."""

import os
import sys

import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty import data_loader
from nifty.data_loader import _EgressBlocked, _is_egress_block, _nse_post, _normalise


class _FakeResp:
    def __init__(self, status_code=200, text="", json_obj=None):
        self.status_code = status_code
        self.text = text
        self._json = json_obj
        self.url = "https://example/test"

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}", response=self)


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


def test_egress_block_detection():
    blocked = _FakeResp(403, "Host not in allowlist: niftyindices.com.")
    site_403 = _FakeResp(403, "Access denied by application")
    ok = _FakeResp(200, "{}")
    assert _is_egress_block(blocked) is True
    assert _is_egress_block(site_403) is False
    assert _is_egress_block(ok) is False


def test_nse_post_raises_egress_immediately():
    """An allowlist block must not be retried — it fails fast."""
    calls = {"n": 0}

    def post(url, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResp(403, "Host not in allowlist: niftyindices.com.")

    session = types.SimpleNamespace(post=post)
    with pytest.raises(_EgressBlocked):
        _nse_post(session, "u", {}, timeout=5, retries=3)
    assert calls["n"] == 1  # no retries on a hard egress block


def test_nse_post_retries_then_succeeds(monkeypatch):
    """Transient errors are retried with backoff, then succeed."""
    monkeypatch.setattr(data_loader.time, "sleep", lambda *_: None)
    import requests

    calls = {"n": 0}

    def post(url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _FakeResp(200, "{}", json_obj={"d": "[]"})

    session = types.SimpleNamespace(post=post)
    resp = _nse_post(session, "u", {}, timeout=5, retries=3)
    assert calls["n"] == 3
    assert resp.status_code == 200
