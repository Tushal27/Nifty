"""Tests for daily report rendering (network-free)."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load the script module directly (scripts/ isn't a package).
_spec = importlib.util.spec_from_file_location(
    "daily_report",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "daily_report.py"),
)
daily_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daily_report)


_DIRECTION = {"as_of": "2026-06-19", "prob_up": 0.554, "signal": "UP",
              "confidence": 0.054}


def _sell_ticket():
    return {
        "entry_date": "2026-06-19", "expiry_approx": "2026-06-26", "spot": 24013.1,
        "implied_vol_vix": 0.30, "realized_vol": 0.12, "variance_risk_premium": 0.18,
        "vrp_min_to_trade": 0.02, "action": "SELL short_strangle — 2 lot(s)",
        "structure": "short_strangle", "expected_move_pts": 440.0,
        "legs": [{"side": "SELL", "kind": "call", "strike": 24450, "lots": 2},
                 {"side": "SELL", "kind": "put", "strike": 23550, "lots": 2}],
        "lots": 2, "leverage_used": 2.7, "notional": 6e6, "margin_estimate": 720000,
        "credit_per_lot_pts": 70.8, "credit_total_rupees": 21240,
        "breakeven_upper": 24520.8, "breakeven_lower": 23479.2,
        "stop_loss_rupees": 42480, "stop_loss_rule": "exit if open loss reaches 2.0× credit",
        "max_risk_rupees": None, "risk_note": "UNDEFINED risk (naked short)",
    }


def _aside_ticket():
    t = _sell_ticket()
    t.update(action="STAND ASIDE — variance risk premium too thin to sell",
             variance_risk_premium=0.0018)
    return t


def test_render_html_sell_has_legs_and_credit():
    out = daily_report.render_html(_DIRECTION, _sell_ticket(), "live")
    assert "Nifty 50" in out
    assert "SELL" in out and "24450" in out and "23550" in out
    assert "21,240" in out  # credit formatted
    assert "Not financial advice" in out


def test_render_html_stand_aside():
    out = daily_report.render_html(_DIRECTION, _aside_ticket(), "cached CSV")
    assert "STAND ASIDE" in out
    assert "No favourable premium-selling edge" in out


def test_render_text_contains_direction_and_ticket():
    out = daily_report.render_text(_DIRECTION, _sell_ticket(), "live")
    assert "NEXT-DAY DIRECTION: UP" in out
    assert "OPTIONS TICKET: SELL" in out
    assert "Not financial advice" in out
