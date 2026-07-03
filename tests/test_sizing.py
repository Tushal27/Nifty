"""Tests for options position sizing and risk circuit breakers."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty.options.sizing import SizingConfig, apply_sizing, equity_stats


def _ret_series(vals):
    idx = pd.bdate_range("2015-01-01", periods=len(vals), freq="W-THU")
    return pd.Series(vals, index=idx)


def test_flat_method_is_full_notional():
    r = _ret_series([0.01, -0.02, 0.015] * 20)
    cfg = SizingConfig(method="flat", cycle_days=5)
    sized = apply_sizing(r, cfg)
    assert (sized["scale"] == 1.0).all()
    # account compounds exactly at the raw returns
    expected = cfg.capital * (1 + r).prod()
    assert sized["equity"].iloc[-1] == pytest.approx(expected)


def test_vol_target_scales_down_when_vol_high():
    rng = np.random.default_rng(0)
    calm = rng.normal(0.002, 0.005, 80)
    wild = rng.normal(0.002, 0.05, 80)
    r = _ret_series(np.concatenate([calm, wild]))
    cfg = SizingConfig(method="vol_target", target_vol=0.15, lookback=40,
                       cycle_days=5, max_leverage=10, margin_pct=0.01)
    sized = apply_sizing(r, cfg)
    early = sized["scale"].iloc[40:80].mean()   # calm regime
    late = sized["scale"].iloc[120:].mean()     # wild regime
    assert late < early  # leverage cut when realized vol rises


def test_monthly_stop_halts_within_month():
    # a big loss mid-month should halt the rest of THAT calendar month.
    idx = pd.bdate_range("2015-01-01", periods=21)  # ~one month of trading days
    vals = [-0.15] + [0.01] * 20                     # -15% on day 1, breaches 10%
    r = pd.Series(vals, index=idx)
    cfg = SizingConfig(method="flat", monthly_stop=0.10, cycle_days=1)
    sized = apply_sizing(r, cfg)
    halts = sized[sized["halted"] == "monthly_stop"]
    assert len(halts) > 0
    assert all(d.month == 1 for d in halts.index)  # only January halted


def test_drawdown_kill_switch():
    vals = [-0.2, -0.2, -0.2] + [0.05] * 10  # deep drawdown then recovery attempts
    r = _ret_series(vals)
    cfg = SizingConfig(method="flat", max_drawdown_stop=0.30, monthly_stop=1.0,
                       cycle_days=5)
    sized = apply_sizing(r, cfg)
    assert (sized["halted"] == "drawdown_kill").any()
    # once killed, no further trading -> equity flat thereafter
    killed_at = sized.index[sized["halted"] == "drawdown_kill"][0]
    after = sized.loc[killed_at:, "lev_ret"]
    assert (after == 0.0).all()


def test_risk_managed_beats_naive_leverage_on_drawdown():
    """Vs a naive high-leverage run, sizing + breakers cut the max drawdown.

    Realistic worst cycle (~-13%, like a short straddle), repeated losses to
    trigger the monthly breaker. Sizing must reduce the drawdown.
    """
    rng = np.random.default_rng(3)
    vals = list(rng.normal(0.004, 0.02, 300))
    # inject a cluster of bad weeks
    for j in (100, 101, 102, 200):
        vals[j] = -0.13
    r = _ret_series(vals)
    naive = apply_sizing(r, SizingConfig(method="flat", cycle_days=5,
                                         monthly_stop=1.0, max_drawdown_stop=1.0))
    sized = apply_sizing(r, SizingConfig(method="vol_target", lookback=40,
                                         cycle_days=5, max_leverage=1.0,
                                         margin_pct=1.0, monthly_stop=0.10,
                                         max_drawdown_stop=0.25))
    assert sized["drawdown"].min() > naive["drawdown"].min()


def test_equity_stats_keys():
    r = _ret_series([0.01, -0.02, 0.015] * 30)
    cfg = SizingConfig(method="vol_target", cycle_days=5)
    stats = equity_stats(apply_sizing(r, cfg), cfg, "x")
    for k in ("final_multiple", "cagr", "sharpe", "max_drawdown", "worst_month"):
        assert k in stats
