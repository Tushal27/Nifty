#!/usr/bin/env python
"""End-to-end pipeline: load -> features -> compare models -> backtest -> report.

Usage:
    python scripts/run_pipeline.py [--config config.yaml] [--refresh]
                                   [--threshold 0.5] [--ticker ^NSEI]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import load_data  # noqa: E402
from nifty.features import build_dataset, feature_columns  # noqa: E402
from nifty.models import build_models  # noqa: E402
from nifty.evaluate import evaluate_models, select_best  # noqa: E402
from nifty.backtest import (  # noqa: E402
    walk_forward_predict,
    run_backtest,
    latest_signal,
)
from nifty import plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nifty 50 pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="prediction horizon in days (1=next-day, 5-20=trend)")
    parser.add_argument("--mode", choices=["close_to_close", "open_to_close"],
                        default=None, help="target type (default from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.threshold is not None:
        config.backtest["threshold"] = args.threshold
    if args.ticker is not None:
        config.data["ticker"] = args.ticker
    if args.horizon is not None:
        config.raw.setdefault("target", {})["horizon"] = args.horizon
    if args.mode is not None:
        config.raw.setdefault("target", {})["mode"] = args.mode

    np.random.seed(config.random_seed)

    out_dir = config.output["dir"]
    charts_dir = config.output["charts_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/6] Loading data ...")
    df = load_data(config, refresh=args.refresh)

    print("[2/6] Engineering features ...")
    X, y, full = build_dataset(df, config)
    cols = feature_columns(full, config)
    horizon = config.horizon
    if config.target_mode == "open_to_close":
        target_desc = "same-day open→close (intraday)"
    else:
        target_desc = "next-day" if horizon == 1 else f"{horizon}-day-ahead trend"
    print(f"      target = {target_desc} direction | "
          f"{X.shape[0]:,} samples x {X.shape[1]} features | "
          f"up rate = {y.mean():.3f}")

    print("[3/6] Walk-forward model comparison ...")
    factories = build_models(config)
    table = evaluate_models(factories, X, y, config)
    table.to_csv(config.output["metrics_csv"])
    best = select_best(table, config)
    print(f"      Best model by {config.evaluate['select_metric']}: {best}")

    print("[4/6] Out-of-sample backtest ...")
    proba = walk_forward_predict(
        factories[best], X, y, config.evaluate["n_splits"],
        gap=max(horizon - 1, 0),
    )
    bt, metrics = run_backtest(proba, full, config)

    print("[5/6] Generating live signal & charts ...")
    final_model = factories[best]().fit(X.values, y.values)
    signal = latest_signal(final_model, X, full, config)
    with open(config.output["signal_json"], "w", encoding="utf-8") as fh:
        json.dump(signal, fh, indent=2)

    plots.plot_metrics_bar(table, charts_dir)
    plots.plot_equity_curve(bt, charts_dir, best)
    plots.plot_signals_on_price(bt, full, charts_dir)
    plots.plot_feature_importance(final_model, cols, charts_dir)

    print("[6/6] Summary")
    print("-" * 60)
    print(table.round(4).to_string())
    # Majority-class ("always predict the more common direction") baseline. Any
    # model that does not beat this is adding zero real skill — on an imbalanced
    # multi-day target a high accuracy can be pure base-rate drift.
    baseline = float(max(y.mean(), 1 - y.mean()))
    best_acc = float(table["accuracy"].max())
    print(f"Always-'{'up' if y.mean() >= 0.5 else 'down'}' baseline accuracy: "
          f"{baseline:.4f}  |  best model: {best_acc:.4f}  |  "
          f"skill (best - baseline): {best_acc - baseline:+.4f}")
    if best_acc <= baseline:
        print("  ⚠ No model beats the naive baseline — the accuracy is drift, "
              "not predictive skill.")
    print("-" * 60)
    s, b = metrics["strategy"], metrics["buy_and_hold"]
    print(f"Backtest {metrics['period_start']} → {metrics['period_end']} "
          f"({metrics['n_days']} days, {metrics['n_trades']} trades)")
    print(f"  {'':14s} {'Strategy':>12s} {'Buy&Hold':>12s}")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate"):
        print(f"  {k:14s} {s[k]:12.4f} {b[k]:12.4f}")
    print("-" * 60)
    print(f"Latest signal: {signal['signal']} "
          f"(P(up)={signal['probability_up']}) as of {signal['as_of_date']}")
    print(f"Artifacts written to: {out_dir}/")

    with open(os.path.join(out_dir, "backtest_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
