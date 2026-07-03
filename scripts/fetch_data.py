#!/usr/bin/env python
"""Download / refresh the Nifty 50 dataset into the local cache.

Usage:
    python scripts/fetch_data.py [--config config.yaml] [--refresh]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import load_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Nifty 50 data")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cache and re-download"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_data(config, refresh=args.refresh)
    print(df.tail())


if __name__ == "__main__":
    main()
