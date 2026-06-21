#!/usr/bin/env python
"""Export the full Nifty 50 OHLCV history to a CSV at the repo root.

Pulls the configured data source (set ``data.source: "nse"`` in config.yaml to
reach index inception ~1996) via :func:`nifty.data_loader.load_data`, writes the
cleaned OHLCV to ``nifty50_full_history.csv`` with a ``Date`` column, and prints
the row count and min/max date so you can confirm the span.

Run this from a network that can reach niftyindices.com (e.g. a residential /
Indian IP). Datacenter / cloud egress IPs are blocked at the Akamai edge.

Usage:
    python scripts/export_full_history.py [--config config.yaml]
                                          [--out nifty50_full_history.csv]
                                          [--refresh]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import load_data  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full Nifty 50 history to CSV")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "nifty50_full_history.csv"),
        help="output CSV path (default: repo-root/nifty50_full_history.csv)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="allow loading from the cached parquet instead of a live pull",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Default to a live pull so the export reflects source data, not a stale cache.
    df = load_data(config, refresh=not args.use_cache)

    # _normalise gives a date-indexed frame named 'date'; expose it as a Date column.
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(args.out)

    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Date span: {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
