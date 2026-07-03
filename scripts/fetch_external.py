#!/usr/bin/env python
"""Fetch exogenous (non-price) daily series and write them as CSVs.

Pulls India VIX, the S&P 500, and USD-INR from Yahoo Finance and writes one
``Date,close`` CSV per series into ``external_data/`` (the directory the pipeline
reads via config.external). FII/DII flows are not on Yahoo, so this script only
writes a template for that one — fill it from the NSE / Moneycontrol daily report.

Run this in an environment that can reach Yahoo (the allowlisted/new session or
your local machine), then commit the CSVs to the branch:

    python scripts/fetch_external.py
    git add external_data/*.csv && git commit -m "Add external feature CSVs" && git push

Usage:
    python scripts/fetch_external.py [--start 1990-01-01] [--out external_data]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

# ticker -> output filename (Yahoo Finance symbols)
SERIES = {
    "^INDIAVIX": "india_vix.csv",   # India VIX (fear gauge), ~2008+
    "^GSPC": "sp500.csv",           # S&P 500 (overnight US lead)
    "INR=X": "usdinr.csv",          # USD-INR exchange rate
}


def _fetch_close(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw[["Close"]].reset_index()
    out.columns = ["Date", "close"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch exogenous feature CSVs")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--out", default="external_data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for ticker, fname in SERIES.items():
        dest = os.path.join(args.out, fname)
        try:
            df = _fetch_close(ticker, args.start)
            df.to_csv(dest, index=False)
            print(f"OK   {ticker:>10s} -> {dest}  "
                  f"({len(df)} rows, {df['Date'].min()} → {df['Date'].max()})")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {ticker:>10s}: {exc}")

    # FII/DII has no free API — leave a template for manual fill.
    fii_path = os.path.join(args.out, "fii_dii.csv")
    if not os.path.exists(fii_path):
        pd.DataFrame(
            {"Date": ["2024-01-01"], "fii_net": [0.0], "dii_net": [0.0]}
        ).to_csv(fii_path + ".template", index=False)
        print(
            f"NOTE fii_dii: no Yahoo source. Fill {fii_path} with columns "
            "Date,fii_net,dii_net (₹ cr) from the NSE/Moneycontrol daily report. "
            f"A header template was written to {fii_path}.template"
        )

    print("\nDone. Commit the CSVs in external_data/ and push to the branch.")


if __name__ == "__main__":
    main()
