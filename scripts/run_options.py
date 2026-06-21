#!/usr/bin/env python
"""Backtest Nifty options premium-selling strategies and emit a live recommendation.

Loads the Nifty spot (via the data loader / CSV) and India VIX, prices options with
Black-Scholes off VIX, backtests every structure (short straddle / strangle / iron
condor), prints a comparison vs buy-and-hold, saves charts, and writes today's
strategy recommendation from the variance-risk-premium signal.

Usage:
    python scripts/run_options.py [--config config.yaml] [--structure short_strangle]
                                  [--cycle-days 5] [--stop-loss-mult 2.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import load_data  # noqa: E402
from nifty.options.backtest import (  # noqa: E402
    STRUCTURES, OptionsConfig, recommend, run_options_backtest,
)
from nifty.options.sizing import (  # noqa: E402
    SizingConfig, apply_sizing, equity_stats,
)


def _load_vix(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in ("date", "datetime") if c in df.columns)
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)["close"].sort_index()
    return (s / 100.0).rename("vix")  # VIX level -> decimal vol


def main() -> None:
    p = argparse.ArgumentParser(description="Nifty options premium-selling backtest")
    p.add_argument("--config", default=None)
    p.add_argument("--structure", choices=STRUCTURES, default=None)
    p.add_argument("--cycle-days", type=int, default=None)
    p.add_argument("--stop-loss-mult", type=float, default=None)
    args = p.parse_args()

    config = load_config(args.config)
    opt_raw = dict(config.raw.get("options", {}))
    if args.structure:
        opt_raw["structure"] = args.structure
    if args.cycle_days:
        opt_raw["cycle_days"] = args.cycle_days
    if args.stop_loss_mult is not None:
        opt_raw["stop_loss_mult"] = args.stop_loss_mult
    cfg = OptionsConfig.from_dict(opt_raw)

    charts_dir = config.output["charts_dir"]
    out_dir = config.output["dir"]
    os.makedirs(charts_dir, exist_ok=True)

    print("\n[1/4] Loading spot + VIX ...")
    df = load_data(config)
    spot = df["close"]
    vix_path = opt_raw.get("vix_csv", "external_data/india_vix.csv")
    if not os.path.exists(vix_path):
        sys.exit(f"VIX file not found: {vix_path}. Fetch it with fetch_external.py.")
    vix = _load_vix(vix_path)
    print(f"      spot {spot.index.min().date()}→{spot.index.max().date()} | "
          f"VIX {vix.index.min().date()}→{vix.index.max().date()}")

    print("[2/4] Backtesting every structure ...")
    rows = {}
    curves = {}
    for struct in STRUCTURES:
        c = OptionsConfig.from_dict({**opt_raw, "structure": struct})
        bt, m = run_options_backtest(spot, vix, c)
        rows[struct] = m
        curves[struct] = bt["equity"]
        print(f"  {struct:>15s} | CAGR={m['cagr']:+.3f} Sharpe={m['sharpe']:+.2f} "
              f"maxDD={m['max_drawdown']:.3f} win={m['win_rate']:.3f} "
              f"worst_cycle={m['worst_cycle']:+.3f}")
    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(out_dir, "options_metrics.csv"))

    print("[3/4] Charts + risk-managed sizing ...")
    _plot_equity(curves, spot, charts_dir)
    bt_best, _ = run_options_backtest(spot, vix, cfg)
    _plot_cycle_hist(bt_best, cfg.structure, charts_dir)

    # Risk-managed account: size the chosen structure and enforce breakers.
    sizing_cfg = SizingConfig.from_dict({
        **dict(opt_raw.get("sizing", {})),
        "cycle_days": cfg.cycle_days, "trading_days": cfg.trading_days,
    })
    per_cycle = bt_best["ret"]
    sized = apply_sizing(per_cycle, sizing_cfg)
    flat = apply_sizing(per_cycle, SizingConfig.from_dict({
        "method": "flat", "capital": sizing_cfg.capital,
        "cycle_days": cfg.cycle_days, "trading_days": cfg.trading_days,
        "max_leverage": 1.0, "margin_pct": 1.0, "monthly_stop": 1.0,
        "max_drawdown_stop": 1.0,
    }))  # 1x-notional, no sizing/breakers — the naive baseline
    risk_table = pd.DataFrame([
        equity_stats(flat, sizing_cfg, "1x notional (no risk mgmt)"),
        equity_stats(sized, sizing_cfg, f"sized: {sizing_cfg.method} + breakers"),
    ]).set_index("label")
    risk_table.to_csv(os.path.join(out_dir, "options_sizing.csv"))
    _plot_sized_equity(flat, sized, sizing_cfg, charts_dir)

    print("[4/4] Recommendation ...")
    rec = recommend(spot, vix, int(opt_raw.get("realized_window", 20)), cfg)
    with open(os.path.join(out_dir, "options_signal.json"), "w") as fh:
        json.dump(rec, fh, indent=2)

    print("-" * 64)
    print("Per-notional edge by structure:")
    print(table[["cagr", "sharpe", "max_drawdown", "win_rate", "worst_cycle",
                 "total_return", "index_buy_hold_return"]].round(3).to_string())
    print("-" * 64)
    print(f"Risk-managed account ({cfg.structure}, start ₹{sizing_cfg.capital:,.0f}):")
    print(risk_table[["final_multiple", "cagr", "sharpe", "max_drawdown",
                      "worst_month", "avg_leverage", "cycles_halted"]].round(3).to_string())
    print("-" * 64)
    print(f"VRP signal as of {rec['as_of_date']}: implied(VIX)={rec['implied_vol_vix']} "
          f"vs realized={rec['realized_vol']}  ->  VRP={rec['variance_risk_premium']:+.4f}")
    print(f"Stance: {rec['stance']}")
    print(f"Suggested {rec['suggested_structure']} (~{rec['cycle_days']}d, "
          f"±{rec['expected_move_pts']:.0f} pts expected move), "
          f"net credit ≈ {rec['net_credit_pts']:.0f} pts:")
    for leg in rec["suggested_legs"]:
        print(f"    {leg['side']:>5s} {leg['kind']:>4s} @ {leg['strike']:.0f}")
    print("-" * 64)
    print("⚠ Educational only: BS priced off VIX (no skew); short premium has "
          "large tail risk. Not financial advice.")
    print(f"Artifacts: {out_dir}/options_metrics.csv, options_signal.json, charts/")


def _plot_equity(curves, spot, charts_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name)
    px = spot.reindex(list(curves.values())[0].index).ffill()
    ax.plot(px.index, (px / px.iloc[0]).values, label="index buy & hold",
            color="black", alpha=0.5, ls="--")
    ax.set_title("Options premium-selling: equity (return on notional) vs index")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "options_equity.png"), dpi=120)
    plt.close(fig)


def _plot_sized_equity(flat, sized, cfg, charts_dir):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(flat.index, flat["equity"], label="1x notional (no risk mgmt)",
            color="firebrick", alpha=0.8)
    ax.plot(sized.index, sized["equity"], label=f"sized: {cfg.method} + breakers",
            color="seagreen")
    ax.set_yscale("log")
    ax.set_title("Risk-managed account equity (log scale)")
    ax.set_ylabel("Account value (₹)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2.fill_between(sized.index, sized["drawdown"] * 100, 0, color="seagreen",
                     alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "options_sized_equity.png"), dpi=120)
    plt.close(fig)


def _plot_cycle_hist(bt, structure, charts_dir):
    traded = bt[bt["traded"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(traded["ret"] * 100, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(0, color="grey", lw=1)
    ax.set_title(f"Per-cycle P&L distribution — {structure} "
                 f"(note the fat LEFT tail = crash risk)")
    ax.set_xlabel("cycle return (% of notional)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "options_cycle_pnl.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
