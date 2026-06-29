#!/usr/bin/env python
"""Operate the options premium-selling strategy: tickets + paper trading.

Subcommands:
    ticket        Print today's sized trade ticket (what to place with a broker).
    paper-open    Log today's ticket into the paper-trading ledger.
    paper-mark    Settle expired positions + mark open ones to market; show status.
    paper-status  Show account equity and open positions.
    paper-report  Realized-P&L stats and an equity-curve chart.

Workflow: run `ticket` each cycle → if it says SELL, place it (or `paper-open`)
→ `paper-mark` daily → positions auto-settle at expiry. Validate on PAPER before
risking real money. Not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from nifty.config import load_config  # noqa: E402
from nifty.data_loader import load_data  # noqa: E402
from nifty.options.backtest import OptionsConfig, run_options_backtest  # noqa: E402
from nifty.options.paper import PaperLedger  # noqa: E402
from nifty.options.sizing import SizingConfig  # noqa: E402
from nifty.options.ticket import build_ticket, format_ticket  # noqa: E402


def _load_vix(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in ("date", "datetime") if c in df.columns)
    df[date_col] = pd.to_datetime(df[date_col])
    return (df.set_index(date_col)["close"].sort_index() / 100.0).rename("vix")


def _setup(config):
    opt_raw = dict(config.raw.get("options", {}))
    opt_cfg = OptionsConfig.from_dict(opt_raw)
    sizing_cfg = SizingConfig.from_dict({
        **dict(opt_raw.get("sizing", {})),
        "cycle_days": opt_cfg.cycle_days, "trading_days": opt_cfg.trading_days,
    })
    spot = load_data(config)["close"]
    vix_path = opt_raw.get("vix_csv", "external_data/india_vix.csv")
    if not os.path.exists(vix_path):
        sys.exit(f"VIX file not found: {vix_path}. Run scripts/fetch_external.py.")
    vix = _load_vix(vix_path)
    return opt_cfg, sizing_cfg, spot, vix, opt_raw


def _make_ticket(config):
    opt_cfg, sizing_cfg, spot, vix, opt_raw = _setup(config)
    live = config.raw.get("live", {})
    bt, _ = run_options_backtest(spot, vix.reindex(spot.index).ffill(), opt_cfg)
    return build_ticket(
        spot, vix, opt_cfg, sizing_cfg, per_cycle_ret=bt["ret"],
        lot_size=int(live.get("lot_size", 75)),
        vrp_min=float(live.get("vrp_min_to_trade", 0.02)),
        realized_window=int(opt_raw.get("realized_window", 20)),
    ), spot, vix, opt_cfg, sizing_cfg


def _ledger(config, opt_cfg, sizing_cfg) -> PaperLedger:
    live = config.raw.get("live", {})
    path = live.get("ledger_path", "paper_trading/ledger.json")
    return PaperLedger(path, capital=sizing_cfg.capital,
                       risk_free_rate=opt_cfg.risk_free_rate,
                       trading_days=opt_cfg.trading_days)


def cmd_ticket(config, args):
    ticket, *_ = _make_ticket(config)
    print(format_ticket(ticket))
    if args.json:
        print(json.dumps(ticket, indent=2))


def cmd_paper_open(config, args):
    ticket, spot, vix, opt_cfg, sizing_cfg = _make_ticket(config)
    print(format_ticket(ticket))
    if not ticket["action"].startswith("SELL") and not args.force:
        print("\nNo SELL signal today — not logging. Use --force to log anyway.")
        return
    if ticket["lots"] < 1 and not args.force:
        print("\nLots < 1 for your capital — not logging. Use --force to paper anyway.")
        return
    led = _ledger(config, opt_cfg, sizing_cfg)
    pos = led.open_from_ticket(ticket)
    print(f"\nLogged paper position #{pos['id']} ({pos['structure']}, "
          f"{pos['lots']} lot, expiry {pos['expiry_date']}).")


def cmd_paper_mark(config, args):
    opt_cfg, sizing_cfg, spot, vix, _ = _setup(config)
    led = _ledger(config, opt_cfg, sizing_cfg)
    settled = led.settle_expired(spot)
    if settled:
        print(f"Settled {len(settled)} expired position(s): "
              + ", ".join(f"#{s['id']} ₹{s['realized_rupees']:,.0f}" for s in settled))
    marks = led.mark(spot, vix)
    for m in marks:
        print(f"  open #{m['id']} {m['structure']:>14s} exp {m['expiry']} "
              f"T={m['T_years']:.3f}  unrealized ₹{m['unrealized_rupees']:,.0f}")
    _print_account(led, spot, vix)


def cmd_paper_status(config, args):
    opt_cfg, sizing_cfg, spot, vix, _ = _setup(config)
    led = _ledger(config, opt_cfg, sizing_cfg)
    _print_account(led, spot, vix)


def cmd_paper_report(config, args):
    opt_cfg, sizing_cfg, spot, vix, _ = _setup(config)
    led = _ledger(config, opt_cfg, sizing_cfg)
    df = led.realized_frame()
    if df.empty:
        print("No closed paper trades yet.")
        return
    wins = (df["realized_rupees"] > 0).mean()
    print(df.to_string())
    print(f"\nClosed trades: {len(df)} | win rate {wins:.1%} | "
          f"total realized ₹{df['realized_rupees'].sum():,.0f} | "
          f"final equity ₹{df['equity'].iloc[-1]:,.0f}")
    _plot_paper_equity(df, config)


def _print_account(led, spot, vix):
    a = led.account(spot, vix)
    print("-" * 56)
    print(f" As of {spot.index[-1].date()}  (start ₹{a['starting_capital']:,.0f})")
    print(f" Equity ₹{a['equity']:,.0f}  ({a['return_pct']:+.2f}%)  | "
          f"realized ₹{a['realized_rupees']:,.0f}  unrealized ₹{a['unrealized_rupees']:,.0f}")
    print(f" Open {a['open_positions']} | Closed {a['closed_positions']}")
    print("-" * 56)


def _plot_paper_equity(df, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    charts_dir = config.output["charts_dir"]
    os.makedirs(charts_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["equity"], marker="o")
    ax.set_title("Paper-trading account equity (realized)")
    ax.set_ylabel("₹")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(charts_dir, "paper_equity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Chart: {path}")


def main():
    p = argparse.ArgumentParser(description="Options tickets + paper trading")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("ticket"); t.add_argument("--json", action="store_true")
    o = sub.add_parser("paper-open"); o.add_argument("--force", action="store_true")
    sub.add_parser("paper-mark")
    sub.add_parser("paper-status")
    sub.add_parser("paper-report")
    args = p.parse_args()

    config = load_config(args.config)
    {
        "ticket": cmd_ticket, "paper-open": cmd_paper_open,
        "paper-mark": cmd_paper_mark, "paper-status": cmd_paper_status,
        "paper-report": cmd_paper_report,
    }[args.cmd](config, args)


if __name__ == "__main__":
    main()
