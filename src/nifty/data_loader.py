"""Load Nifty 50 OHLCV data from a user CSV, the NSE API, or the yfinance API.

The loader normalises everything to a clean, date-indexed DataFrame with the
columns ``open, high, low, close, volume`` so the rest of the pipeline does not
care where the data came from.

Three sources, selected via ``data.source`` in config.yaml:
  * ``yfinance`` — Yahoo Finance ``^NSEI`` (history only reaches ~2007)
  * ``nse``      — niftyindices.com historical API (reaches index inception ~1996)
  * a CSV via ``data.csv_path`` always takes precedence over either API.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import pandas as pd

from .config import Config

OHLCV = ["open", "high", "low", "close", "volume"]
OHLC = ["open", "high", "low", "close"]

NIFTY_INDICES_HOME = "https://niftyindices.com"
NIFTY_INDICES_HISTORICAL = (
    "https://niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, coerce a date index, keep OHLCV, sort and dedupe."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Find a date column if it is not already the index.
    if not isinstance(df.index, pd.DatetimeIndex):
        date_candidates = ("date", "datetime", "timestamp", "historicaldate")
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            raise ValueError(
                "Could not find a Date column. Expected one of: "
                "Date/Datetime/Timestamp/HistoricalDate."
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Index series (e.g. NSE) carry no volume — synthesise a 0 column.
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # 'adj close' is fine to drop for an index; keep the core OHLCV.
    missing = [c for c in OHLC if c not in df.columns]
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


def _nse_session():
    """A requests session primed with the headers/cookies niftyindices expects."""
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://niftyindices.com/reports/historical-data",
            "Origin": NIFTY_INDICES_HOME,
        }
    )
    try:  # bootstrap cookies; failure here is non-fatal, the POST may still work
        session.get(NIFTY_INDICES_HOME, timeout=20)
    except Exception:  # noqa: BLE001 - network errors handled at call site
        pass
    return session


def _fetch_nse(index_name: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Pull full-history index data from niftyindices.com (back to ~1996).

    The endpoint only serves a limited window per request, so we page through
    the range one year at a time and concatenate.
    """
    import requests

    start_ts = pd.Timestamp(start or "1996-01-01")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    session = _nse_session()
    frames: list[pd.DataFrame] = []
    cursor = start_ts
    try:
        while cursor <= end_ts:
            chunk_end = min(
                cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end_ts
            )
            payload = {
                "cinfo": json.dumps(
                    {
                        "name": index_name,
                        "startDate": cursor.strftime("%d-%b-%Y"),
                        "endDate": chunk_end.strftime("%d-%b-%Y"),
                        "indexName": index_name,
                    }
                )
            }
            resp = session.post(NIFTY_INDICES_HISTORICAL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("d")
            records = json.loads(data) if isinstance(data, str) else data
            if records:
                frames.append(pd.DataFrame(records))
            cursor = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.4)  # be polite to the endpoint
    except requests.RequestException as exc:
        raise RuntimeError(
            "NSE (niftyindices.com) request failed — the environment's network "
            f"policy may block the host. Underlying error: {exc}. Set "
            "data.csv_path in config.yaml to load your own file instead."
        ) from exc

    if not frames:
        raise RuntimeError(
            f"NSE returned no data for index {index_name!r} in the requested range."
        )

    raw = pd.concat(frames, ignore_index=True)
    # niftyindices uses upper-case OHLC and 'HistoricalDate'; _normalise lowercases
    # and recognises those column names directly.
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

    source_kind = (data_cfg.get("source") or "yfinance").lower()
    if csv_path:  # an explicit CSV always wins over either API
        df = _normalise(pd.read_csv(csv_path))
        source = f"CSV ({csv_path})"
    elif source_kind == "nse":
        index_name = data_cfg.get("nse_index_name", "NIFTY 50")
        df = _fetch_nse(index_name, data_cfg.get("start"), data_cfg.get("end"))
        source = f"NSE niftyindices.com ({index_name})"
    elif source_kind == "yfinance":
        df = _fetch_yfinance(
            data_cfg["ticker"], data_cfg.get("start"), data_cfg.get("end")
        )
        source = f"yfinance ({data_cfg['ticker']})"
    else:
        raise ValueError(
            f"Unknown data.source {source_kind!r}. Use 'yfinance', 'nse', or set "
            "data.csv_path."
        )

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
