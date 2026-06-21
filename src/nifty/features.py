"""Technical-indicator feature engineering and the next-day direction label.

Every indicator is computed using **only past data** (trailing windows / shifts),
so the feature matrix at row *t* never contains information from day *t+1*. The
label is the only thing that looks one day ahead, via ``Close.shift(-1)``.
"""

from __future__ import annotations

import os

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

    # Volume features must stay robust for index series that carry no volume
    # (all zeros): a naive pct_change/ratio yields 0/0 -> NaN on every row, which
    # would drop the entire dataset. Collapse those degenerate cases to neutral
    # constants instead (0 change, ratio 1).
    out["volume_change"] = (
        out["volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    roll_vol = out["volume"].rolling(cfg["vol_window"]).mean()
    out["volume_ratio"] = (out["volume"] / roll_vol).replace(
        [np.inf, -np.inf], np.nan
    )
    out.loc[roll_vol == 0, "volume_ratio"] = 1.0  # divide-by-zero -> neutral

    return out


def _load_external_one(
    path: str, prefix: str, lag: int, base_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Read one exogenous CSV and align it to the index, lagged to avoid leakage.

    The CSV needs a Date column plus one or more numeric value columns. For each
    value column we emit a level feature and a change feature (pct_change for
    strictly-positive series like prices/VIX, plain diff for series that can go
    negative like net flows). Everything is forward-filled onto the trading
    calendar, then shifted by ``lag`` days so the model only ever sees PAST data.
    """
    ext = pd.read_csv(path)
    ext.columns = [str(c).strip().lower() for c in ext.columns]
    date_col = next(
        (c for c in ("date", "datetime", "timestamp") if c in ext.columns), None
    )
    if date_col is None:
        raise ValueError(f"{path}: no Date/Datetime column found.")
    ext[date_col] = pd.to_datetime(ext[date_col])
    ext = ext.set_index(date_col).sort_index()
    ext = ext[~ext.index.duplicated(keep="last")]

    value_cols = [c for c in ext.columns if c != date_col]
    # Align external dates onto the Nifty trading calendar via forward fill.
    aligned = (
        ext.reindex(base_index.union(ext.index)).sort_index().ffill().reindex(base_index)
    )

    out = pd.DataFrame(index=base_index)
    for c in value_cols:
        s = pd.to_numeric(aligned[c], errors="coerce")
        out[f"{prefix}_{c}"] = s
        positive = s.dropna()
        if len(positive) and (positive > 0).all():
            out[f"{prefix}_{c}_chg"] = s.pct_change()
        else:  # flows etc. can be negative -> pct_change is meaningless
            out[f"{prefix}_{c}_chg"] = s.diff()

    return out.shift(lag)  # critical: only past exogenous values are visible


def add_external_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Join optional exogenous (non-price) features defined in config.external.

    Files that don't exist yet are skipped with a notice, so the pipeline still
    runs before the CSVs are supplied.
    """
    ext_cfg = config.raw.get("external") or {}
    if not ext_cfg.get("enabled"):
        return df

    lag = int(ext_cfg.get("lag_days", 1))
    out = df
    for src in ext_cfg.get("sources", []):
        path = src["path"]
        prefix = src.get("prefix") or os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            print(f"      [external] skipping {prefix}: file not found ({path})")
            continue
        block = _load_external_one(path, prefix, lag, out.index)
        out = out.join(block)
        print(f"      [external] added {prefix}: {list(block.columns)}")
    return out


def make_label(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1 if the close ``horizon`` days ahead is higher than today's, else 0.

    ``horizon=1`` is next-day direction (the hardest, near-random case). Larger
    horizons (e.g. 5-20) capture multi-day *trend*, where the market's drift
    dominates daily noise, so the label is far more learnable.

    The trailing ``horizon`` rows have no forward price and return NaN (so
    ``build_dataset`` drops them) rather than a fabricated 0.
    """
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype("float")
    label[future.isna()] = np.nan
    return label


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
    feat = add_external_features(feat, config)
    feat["target"] = make_label(feat, horizon=config.horizon)

    cols = feature_columns(feat, config)

    # Rows usable for *training*: features present AND a known forward label. The
    # trailing ``horizon`` rows have no label yet and drop out here; the most
    # recent row stays in ``full`` for live-signal generation.
    trainable = feat.dropna(subset=cols + ["target"])
    X = trainable[cols].copy()
    y = trainable["target"].astype(int).copy()

    return X, y, feat
