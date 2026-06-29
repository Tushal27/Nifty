"""Tests for the paper-trading ledger lifecycle."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.options.paper import PaperLedger


def _ticket(strike_call, strike_put, credit_pts, entry, expiry, lots=1):
    return {
        "entry_date": entry, "expiry_approx": expiry, "structure": "short_strangle",
        "legs": [{"side": "SELL", "kind": "call", "strike": strike_call, "lots": lots},
                 {"side": "SELL", "kind": "put", "strike": strike_put, "lots": lots}],
        "lots": lots, "lot_size": 75, "credit_per_lot_pts": credit_pts, "spot": 20000.0,
    }


def _flat_spot(value, start="2021-01-01", n=15):
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.Series(float(value), index=idx, name="close")


def test_open_persists(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = PaperLedger(path, capital=1_000_000)
    led.open_from_ticket(_ticket(20500, 19500, 100.0, "2021-01-01", "2021-01-08"))
    assert os.path.exists(path)
    # reload from disk
    led2 = PaperLedger(path, capital=1_000_000)
    assert len(led2.positions) == 1 and led2.positions[0]["status"] == "open"


def test_settle_win_inside_strikes(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = PaperLedger(path, capital=1_000_000)
    led.open_from_ticket(_ticket(20500, 19500, 100.0, "2021-01-01", "2021-01-08"))
    spot = _flat_spot(20000)  # expires between strikes -> both worthless
    settled = led.settle_expired(spot)
    assert len(settled) == 1
    # keep full credit: 100 pts * 75 * 1 lot
    assert settled[0]["realized_rupees"] == pytest.approx(100 * 75)


def test_settle_loss_beyond_strike(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = PaperLedger(path, capital=1_000_000)
    led.open_from_ticket(_ticket(20500, 19500, 100.0, "2021-01-01", "2021-01-08"))
    spot = _flat_spot(21000)  # blows through the call: intrinsic 500
    settled = led.settle_expired(spot)
    # realized = (credit 100 - intrinsic 500) * 75 = -30000
    assert settled[0]["realized_rupees"] == pytest.approx((100 - 500) * 75)


def test_mark_and_account(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = PaperLedger(path, capital=1_000_000)
    led.open_from_ticket(_ticket(20500, 19500, 100.0, "2021-01-01", "2021-01-15"))
    spot = _flat_spot(20000, n=5)  # mid-life, before expiry
    vix = pd.Series(0.15, index=spot.index)
    marks = led.mark(spot, vix)
    assert len(marks) == 1 and "unrealized_rupees" in marks[0]
    acct = led.account(spot, vix)
    assert acct["equity"] == pytest.approx(
        acct["starting_capital"] + acct["realized_rupees"] + acct["unrealized_rupees"])
    assert acct["open_positions"] == 1
