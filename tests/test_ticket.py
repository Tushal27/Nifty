"""Tests for the sized trade-ticket generator."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.options.backtest import OptionsConfig
from nifty.options.sizing import SizingConfig
from nifty.options.ticket import build_ticket


def _series(vix_level, n=120, spot0=20000.0, ret_vol=0.003, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n, name="date")
    spot = pd.Series(spot0 * np.exp(np.cumsum(rng.normal(0, ret_vol, n))),
                     index=idx, name="close")
    vix = pd.Series(vix_level, index=idx, name="vix")
    return spot, vix


def test_ticket_sells_when_vrp_high():
    spot, vix = _series(vix_level=0.30)        # implied 30% >> realized ~5%
    opt = OptionsConfig(structure="short_strangle")
    sz = SizingConfig(capital=5_000_000, cycle_days=opt.cycle_days)
    t = build_ticket(spot, vix, opt, sz, per_cycle_ret=None, lot_size=75, vrp_min=0.02)
    assert t["action"].startswith("SELL")
    assert t["lots"] >= 1
    assert len(t["legs"]) == 2 and all(l["side"] == "SELL" for l in t["legs"])
    assert t["credit_total_rupees"] > 0
    assert t["breakeven_lower"] < t["spot"] < t["breakeven_upper"]


def test_ticket_stands_aside_when_vrp_low():
    spot, vix = _series(vix_level=0.05)        # implied ≈ realized
    opt = OptionsConfig(structure="short_strangle")
    sz = SizingConfig(capital=5_000_000, cycle_days=opt.cycle_days)
    t = build_ticket(spot, vix, opt, sz, vrp_min=0.02)
    assert "STAND ASIDE" in t["action"]


def test_lots_and_credit_math():
    spot, vix = _series(vix_level=0.30)
    opt = OptionsConfig(structure="short_strangle")
    sz = SizingConfig(capital=5_000_000, cycle_days=opt.cycle_days)
    t = build_ticket(spot, vix, opt, sz, per_cycle_ret=None, lot_size=75, vrp_min=0.0)
    # leverage is 1.0 with no history -> notional ≤ capital
    assert t["notional"] <= sz.capital + 1e-6
    assert t["lots"] == int((1.0 * sz.capital) // (t["spot"] * 75))
    # total ≈ per-lot credit × lot_size × lots (small diff from rounding the field)
    assert t["credit_total_rupees"] == pytest.approx(
        t["credit_per_lot_pts"] * 75 * t["lots"], abs=75 * t["lots"])


def test_iron_condor_has_defined_max_risk():
    spot, vix = _series(vix_level=0.30)
    opt = OptionsConfig(structure="iron_condor")
    sz = SizingConfig(capital=5_000_000, cycle_days=opt.cycle_days)
    t = build_ticket(spot, vix, opt, sz, per_cycle_ret=None, lot_size=75, vrp_min=0.0)
    assert len(t["legs"]) == 4
    assert t["max_risk_rupees"] is not None and t["max_risk_rupees"] >= 0
