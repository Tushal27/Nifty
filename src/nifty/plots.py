"""Chart generation for the backtest and model diagnostics."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless / container-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_equity_curve(bt: pd.DataFrame, out_dir: str, best_name: str) -> str:
    _ensure(out_dir)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bt.index, bt["strategy_equity"], label=f"Strategy ({best_name})")
    ax.plot(bt.index, bt["buyhold_equity"], label="Buy & Hold", alpha=0.8)
    ax.set_title("Out-of-sample equity curve (growth of 1 unit)")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "equity_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_signals_on_price(
    bt: pd.DataFrame, full: pd.DataFrame, out_dir: str
) -> str:
    _ensure(out_dir)
    price = full["close"].reindex(bt.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(price.index, price.values, color="black", lw=0.8, label="Close")

    entries = bt["position"].diff().fillna(bt["position"])
    longs = bt.index[entries > 0]
    exits = bt.index[entries < 0]
    ax.scatter(longs, price.reindex(longs), marker="^", color="green", s=24, label="Long")
    ax.scatter(exits, price.reindex(exits), marker="v", color="red", s=24, label="Exit/Short")

    ax.set_title("Trade signals on price")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "signals_on_price.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, out_dir: str) -> str | None:
    est = getattr(model, "estimator", None)
    importances = getattr(est, "feature_importances_", None)
    if importances is None:
        return None
    _ensure(out_dir)
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
    )
    ax.set_title("Top feature importances")
    path = os.path.join(out_dir, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metrics_bar(table: pd.DataFrame, out_dir: str) -> str:
    _ensure(out_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    table[["accuracy", "roc_auc", "f1"]].plot.bar(ax=ax)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="random (0.5)")
    ax.set_title("Model comparison (walk-forward CV)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    path = os.path.join(out_dir, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
