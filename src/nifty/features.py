"""Technical-indicator feature engineering and the next-day direction label.

Every indicator is computed using **only past data** (trailing windows / shifts),
so the feature matrix at row *t* never contains information from day *t+1*. The
label is the only thing that looks one day ahead, via ``Close.shift(-1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


# --- individual indicators --------------------------------------------------
def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Append a rich set of trailing technical indicators to ``df``."""
    cfg = config.features
    out = df.copy()
    close = out["close"]

    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close).diff()

    for w in cfg["sma_windows"]:
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"close_to_sma_{w}"] = close / out[f"sma_{w}"] - 1.0
    for w in cfg["ema_windows"]:
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        out[f"close_to_ema_{w}"] = close / out[f"ema_{w}"] - 1.0

    out["rsi"] = _rsi(close, cfg["rsi_period"])

    ema_fast = close.ewm(span=cfg["macd_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=cfg["macd_slow"], adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=cfg["macd_signal"], adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = close.rolling(cfg["bb_period"]).mean()
    bb_std = close.rolling(cfg["bb_period"]).std()
    upper = bb_mid + cfg["bb_std"] * bb_std
    lower = bb_mid - cfg["bb_std"] * bb_std
    out["bb_pct"] = (close - lower) / (upper - lower)

    out["atr"] = _atr(out, cfg["atr_period"])
    out["atr_pct"] = out["atr"] / close

    out["volatility"] = out["log_return_1d"].rolling(cfg["vol_window"]).std()

    for w in cfg["momentum_windows"]:
        out[f"momentum_{w}"] = close.pct_change(w)

    out["volume_change"] = out["volume"].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(
        cfg["vol_window"]
    ).mean()

    return out


def make_label(df: pd.DataFrame) -> pd.Series:
    """1 if the next day's close is higher than today's, else 0."""
    return (df["close"].shift(-1) > df["close"]).astype(int)


# Columns that are raw prices / intermediate values, not predictive features.
_NON_FEATURE = set(["open", "high", "low", "close", "volume", "target"])


def feature_columns(df: pd.DataFrame, config: Config) -> list[str]:
    """The model input columns: everything engineered, minus raw OHLCV/SMAs."""
    drop_prefixes = ("sma_", "ema_")  # keep the *_to_ ratios, drop absolute levels
    cols = []
    for c in df.columns:
        if c in _NON_FEATURE:
            continue
        if c.startswith(drop_prefixes) and not c.startswith("close_to_"):
            continue
        cols.append(c)
    return cols


def build_dataset(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(X, y, full)`` ready for modelling.

    ``full`` retains the price columns aligned to ``X`` so the backtester can map
    predictions back to realised next-day returns. The final row (no next-day
    label) and warm-up rows with NaN indicators are dropped from ``X``/``y`` but
    the most recent row is kept in ``full`` for live-signal generation.
    """
    feat = add_indicators(df, config)
    feat["target"] = make_label(feat)

    cols = feature_columns(feat, config)

    # Rows usable for *training*: features present AND a known next-day label.
    trainable = feat.dropna(subset=cols + ["target"])
    X = trainable[cols].copy()
    y = trainable["target"].astype(int).copy()

    return X, y, feat
