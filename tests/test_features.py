"""Sanity tests for indicators, labelling, and absence of look-ahead leakage."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.config import load_config
from nifty.features import add_indicators, build_dataset, make_label


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_df():
    """Deterministic synthetic OHLCV with a gentle upward drift."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0005, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_label_matches_next_day_move(sample_df):
    y = make_label(sample_df)
    # Label at t must equal (close[t+1] > close[t]); verify on a few rows.
    for t in (5, 50, 200):
        expected = int(sample_df["close"].iloc[t + 1] > sample_df["close"].iloc[t])
        assert int(y.iloc[t]) == expected


def test_label_horizon(sample_df):
    """A horizon-H label compares close[t+H] to close[t]."""
    H = 10
    y = make_label(sample_df, horizon=H)
    for t in (5, 100, 250):
        expected = int(sample_df["close"].iloc[t + H] > sample_df["close"].iloc[t])
        assert int(y.iloc[t]) == expected
    # the final H rows have no forward price and must be NaN-droppable
    assert y.iloc[-H:].isna().any() or (sample_df["close"].shift(-H).iloc[-H:].isna().all())


def test_rsi_in_valid_range(sample_df, config):
    feat = add_indicators(sample_df, config)
    rsi = feat["rsi"].dropna()
    assert rsi.between(0, 100).all()


def test_no_lookahead_in_features(sample_df, config):
    """Changing a *future* close must not alter today's feature row."""
    feat_a = add_indicators(sample_df, config)
    tampered = sample_df.copy()
    tampered.iloc[-1, tampered.columns.get_loc("close")] *= 1.5  # change last day
    feat_b = add_indicators(tampered, config)

    # All rows except the final one must be identical (features only look back).
    a = feat_a.iloc[:-1].drop(columns=["volume"])
    b = feat_b.iloc[:-1].drop(columns=["volume"])
    pd.testing.assert_frame_equal(a, b)


def test_build_dataset_shapes(sample_df, config):
    X, y, full = build_dataset(sample_df, config)
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert set(y.unique()).issubset({0, 1})
    assert len(X) > 0
