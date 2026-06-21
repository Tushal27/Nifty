"""Tests for Black-Scholes pricing and the options premium-selling backtest."""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.options.pricing import bs_price, bs_delta, expected_move
from nifty.options.backtest import (
    OptionsConfig, build_legs, run_options_backtest, recommend,
)


def test_put_call_parity():
    S, K, T, r, sig = 20000, 20000, 30 / 365, 0.065, 0.15
    call = bs_price(S, K, T, r, sig, "call")
    put = bs_price(S, K, T, r, sig, "put")
    # C - P == S - K e^{-rT}
    assert call - put == pytest.approx(S - K * math.exp(-r * T), rel=1e-6)


def test_atm_straddle_approx():
    """ATM straddle ≈ 0.8 * S * sigma * sqrt(T) (the classic approximation)."""
    S, T, sig, r = 20000, 30 / 365, 0.15, 0.0
    straddle = bs_price(S, S, T, r, sig, "call") + bs_price(S, S, T, r, sig, "put")
    approx = 0.8 * S * sig * math.sqrt(T)
    assert straddle == pytest.approx(approx, rel=0.05)


def test_intrinsic_at_expiry():
    assert bs_price(21000, 20000, 0.0, 0.065, 0.2, "call") == 1000
    assert bs_price(19000, 20000, 0.0, 0.065, 0.2, "put") == 1000
    assert bs_delta(21000, 20000, 0.0, 0.0, 0.2, "call") == 1.0


def test_legs_structure_counts():
    cfg = OptionsConfig(structure="iron_condor")
    legs = build_legs("iron_condor", 20000, 0.15, 30 / 365, cfg)
    assert len(legs) == 4
    assert sum(q for _, _, q in legs) == 0          # 2 short + 2 long, net 0 qty
    straddle = build_legs("short_straddle", 20000, 0.15, 30 / 365, cfg)
    assert all(q == -1 for _, _, q in straddle) and len(straddle) == 2


def _synthetic_series(n=600, drift=0.0003, vol=0.01, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(drift, vol, n)
    spot = pd.Series(20000 * np.exp(np.cumsum(rets)), index=dates, name="close")
    vix = pd.Series(np.clip(0.15 + 0.05 * rng.standard_normal(n), 0.08, 0.6),
                    index=dates, name="vix")
    return spot, vix


def test_backtest_runs_and_is_bounded():
    spot, vix = _synthetic_series()
    cfg = OptionsConfig(structure="short_strangle", cycle_days=5)
    bt, m = run_options_backtest(spot, vix, cfg)
    assert m["n_cycles"] > 50
    assert -1.0 < m["worst_cycle"] <= 0.0       # a cycle can't lose >100% of notional here
    assert set(["cagr", "sharpe", "max_drawdown", "win_rate"]).issubset(m)


def test_iron_condor_loss_is_capped():
    """Defined-risk: iron condor's worst cycle is far smaller than a naked straddle's."""
    # a violent up-move series to stress the short strikes
    dates = pd.bdate_range("2015-01-01", periods=120)
    spot = pd.Series(np.linspace(20000, 30000, 120), index=dates, name="close")
    vix = pd.Series(0.15, index=dates, name="vix")
    _, m_straddle = run_options_backtest(
        spot, vix, OptionsConfig(structure="short_straddle", cycle_days=5,
                                 stop_loss_mult=0))
    _, m_condor = run_options_backtest(
        spot, vix, OptionsConfig(structure="iron_condor", cycle_days=5,
                                 stop_loss_mult=0))
    assert m_condor["worst_cycle"] > m_straddle["worst_cycle"]  # condor less negative


def test_recommend_keys():
    spot, vix = _synthetic_series()
    rec = recommend(spot, vix, 20, OptionsConfig())
    for k in ("stance", "variance_risk_premium", "suggested_legs", "net_credit_pts"):
        assert k in rec
    assert len(rec["suggested_legs"]) == 2  # default short_strangle
