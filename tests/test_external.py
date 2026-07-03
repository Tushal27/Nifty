"""Tests for the exogenous (external) feature loader and its no-look-ahead lag."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.config import Config
from nifty.features import _load_external_one, add_external_features


@pytest.fixture
def base_index():
    return pd.bdate_range("2020-01-01", periods=60, name="date")


def test_external_is_lagged(tmp_path, base_index):
    """Level feature at row t must equal the source value at row t-lag."""
    vals = np.arange(100.0, 100.0 + len(base_index))
    csv = tmp_path / "vix.csv"
    pd.DataFrame({"Date": base_index, "close": vals}).to_csv(csv, index=False)

    out = _load_external_one(str(csv), "vix", lag=1, base_index=base_index)
    # row 1 should carry the source value from row 0 (lagged by 1 day)
    assert out["vix_close"].iloc[1] == pytest.approx(vals[0])
    assert out["vix_close"].iloc[5] == pytest.approx(vals[4])
    assert np.isnan(out["vix_close"].iloc[0])  # nothing before the first day
    # positive series -> pct_change feature exists
    assert "vix_close_chg" in out.columns


def test_external_negative_series_uses_diff(tmp_path, base_index):
    """Flows can go negative, so change must be diff (not pct_change)."""
    vals = np.linspace(-500, 500, len(base_index))
    csv = tmp_path / "flow.csv"
    pd.DataFrame({"Date": base_index, "fii_net": vals}).to_csv(csv, index=False)

    out = _load_external_one(str(csv), "flow", lag=1, base_index=base_index)
    expected_diff = vals[4] - vals[3]  # diff at row 5, then lagged by 1 -> row 5
    assert out["flow_fii_net_chg"].iloc[5] == pytest.approx(expected_diff)


def test_add_external_skips_missing_file(base_index, capsys):
    df = pd.DataFrame({"close": np.arange(len(base_index))}, index=base_index)
    cfg = Config(raw={"external": {
        "enabled": True, "lag_days": 1,
        "sources": [{"path": "does/not/exist.csv", "prefix": "nope"}],
    }})
    out = add_external_features(df, cfg)
    assert list(out.columns) == ["close"]  # unchanged
    assert "skipping nope" in capsys.readouterr().out


def test_add_external_disabled_is_noop(base_index):
    df = pd.DataFrame({"close": np.arange(len(base_index))}, index=base_index)
    cfg = Config(raw={"external": {"enabled": False}})
    assert add_external_features(df, cfg).equals(df)
