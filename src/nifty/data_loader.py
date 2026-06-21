"""Load Nifty 50 OHLCV data from a user CSV or via the yfinance API.

The loader normalises everything to a clean, date-indexed DataFrame with the
columns ``open, high, low, close, volume`` so the rest of the pipeline does not
care where the data came from.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .config import Config

OHLCV = ["open", "high", "low", "close", "volume"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, coerce a date index, keep OHLCV, sort and dedupe."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Find a date column if it is not already the index.
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in ("date", "datetime", "timestamp") if c in df.columns), None
        )
        if date_col is None:
            raise ValueError(
                "Could not find a Date column. Expected one of: Date/Datetime/Timestamp."
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # 'adj close' is fine to drop for an index; keep the core OHLCV.
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    df = df[OHLCV]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    # Volume can legitimately be 0/NaN for an index; only require a valid close.
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _fetch_yfinance(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker!r}. The environment's network "
            "policy may block the request — set data.csv_path in config.yaml to "
            "load your own file instead."
        )
    # yfinance may return a MultiIndex (column, ticker); flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    return _normalise(raw)


def load_data(config: Config, refresh: bool = False) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame, using cache/CSV/API in that order.

    Parameters
    ----------
    config:
        Parsed configuration.
    refresh:
        If True, ignore any cached parquet and re-fetch from source.
    """
    data_cfg = config.data
    cache_path = data_cfg.get("cache_path")
    csv_path = data_cfg.get("csv_path")

    if cache_path and os.path.exists(cache_path) and not refresh:
        df = pd.read_parquet(cache_path)
        df = _normalise(df.reset_index())
        _report_span(df, source=f"cache ({cache_path})")
        return df

    if csv_path:
        df = _normalise(pd.read_csv(csv_path))
        source = f"CSV ({csv_path})"
    else:
        df = _fetch_yfinance(
            data_cfg["ticker"], data_cfg.get("start"), data_cfg.get("end")
        )
        source = f"yfinance ({data_cfg['ticker']})"

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path)

    _report_span(df, source=source)
    return df


def _report_span(df: pd.DataFrame, source: str) -> None:
    print(
        f"Loaded {len(df):,} rows from {source} | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
