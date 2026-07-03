"""Position sizing and risk controls for the premium-selling backtest.

The options backtest produces per-cycle returns *on full notional*. That is the
raw edge; it says nothing about how much to bet. This module decides the bet size
each cycle from PAST data only (no look-ahead), simulates a real compounding
account at that leverage, and enforces risk circuit breakers:

  * sizing methods: vol-target, fixed-fraction, fractional Kelly
  * a per-month stop (halt new trades once the month's loss exceeds a limit)
  * a max-drawdown kill switch (stop trading if the account falls too far)

The point: a +1.1 Sharpe edge run at reckless leverage still goes to zero in one
crash week. Sizing is what keeps the tail survivable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SizingConfig:
    method: str = "vol_target"        # vol_target | fixed_fraction | kelly
    capital: float = 1_000_000.0      # starting account (currency units)
    target_vol: float = 0.15          # vol_target: annualised target vol
    risk_fraction: float = 0.02       # fixed_fraction: risk per trade (of capital)
    kelly_fraction: float = 0.25      # fraction of full Kelly to actually use
    max_leverage: float = 3.0         # cap on notional / capital
    margin_pct: float = 0.12          # margin as fraction of notional (caps leverage)
    lookback: int = 50                # cycles used for rolling estimates
    monthly_stop: float = 0.10        # halt new trades after this monthly loss
    max_drawdown_stop: float = 0.30   # kill switch: stop trading below this drawdown
    trading_days: int = 252
    cycle_days: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> "SizingConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})

    @property
    def cycles_per_year(self) -> float:
        return self.trading_days / self.cycle_days

    @property
    def leverage_cap(self) -> float:
        return min(self.max_leverage, 1.0 / max(self.margin_pct, 1e-6))


def _rolling_scale(returns: pd.Series, cfg: SizingConfig) -> pd.Series:
    """Leverage (notional/capital) per cycle from strictly-past returns."""
    cap = cfg.leverage_cap
    cpy = cfg.cycles_per_year
    scale = pd.Series(index=returns.index, dtype=float)
    vals = returns.values
    for k in range(len(vals)):
        past = vals[max(0, k - cfg.lookback):k]
        past = past[np.isfinite(past)]
        if cfg.method == "flat":
            s = 1.0  # always full notional — the no-sizing baseline
        elif len(past) < 10:
            s = 1.0  # conservative warm-up: 1x notional
        elif cfg.method == "vol_target":
            sd = past.std()
            tgt = cfg.target_vol / math.sqrt(cpy)
            s = tgt / sd if sd > 0 else cap
        elif cfg.method == "fixed_fraction":
            loss = -np.quantile(past, 0.05)  # typical bad-cycle loss (positive)
            s = cfg.risk_fraction / loss if loss > 0 else cap
        elif cfg.method == "kelly":
            mu, var = past.mean(), past.var()
            f = mu / var if var > 0 else 0.0
            s = cfg.kelly_fraction * f
        else:
            raise ValueError(f"unknown sizing method {cfg.method!r}")
        scale.iloc[k] = float(min(max(s, 0.0), cap))
    return scale


def apply_sizing(per_cycle_ret: pd.Series, cfg: SizingConfig) -> pd.DataFrame:
    """Simulate a compounding account with sizing + circuit breakers.

    ``per_cycle_ret`` is the return on full notional per cycle (from the backtest),
    indexed by entry date. Returns a frame with per-cycle leverage, leveraged
    return, account equity, drawdown, and whether a breaker halted the cycle.
    """
    r = per_cycle_ret.fillna(0.0)
    base_scale = _rolling_scale(r, cfg)

    equity = cfg.capital
    peak = equity
    cur_month = None
    month_start = equity
    rows = []
    for date, ri, sc in zip(r.index, r.values, base_scale.values):
        ym = (date.year, date.month)
        if ym != cur_month:
            cur_month, month_start = ym, equity

        halted = ""
        scale = sc
        month_ret = (equity / month_start - 1.0) if month_start > 0 else -1.0
        dd_ratio = (equity / peak - 1.0) if peak > 0 else -1.0
        if month_ret <= -cfg.monthly_stop:
            scale, halted = 0.0, "monthly_stop"
        if dd_ratio <= -cfg.max_drawdown_stop:
            scale, halted = 0.0, "drawdown_kill"

        lev_ret = float(ri) * scale
        equity *= (1.0 + lev_ret)
        peak = max(peak, equity)
        rows.append({
            "scale": scale, "lev_ret": lev_ret, "equity": equity,
            "drawdown": equity / peak - 1.0, "halted": halted,
        })
    return pd.DataFrame(rows, index=r.index)


def equity_stats(sized: pd.DataFrame, cfg: SizingConfig, label: str) -> dict:
    lev = sized["lev_ret"]
    eq = sized["equity"]
    cpy = cfg.cycles_per_year
    n = len(lev)
    years = n / cpy if n else float("nan")
    total = float(eq.iloc[-1] / cfg.capital - 1.0)
    cagr = float((eq.iloc[-1] / cfg.capital) ** (1 / years) - 1.0) if years > 0 else float("nan")
    sd = lev.std()
    sharpe = float(lev.mean() / sd * math.sqrt(cpy)) if sd > 0 else float("nan")
    max_dd = float(sized["drawdown"].min())
    # worst calendar-month return
    monthly = eq.resample("ME").last().pct_change().dropna()
    worst_month = float(monthly.min()) if len(monthly) else float("nan")
    halted = int((sized["halted"] != "").sum())
    return {
        "label": label, "method": cfg.method,
        "final_multiple": float(eq.iloc[-1] / cfg.capital),
        "total_return": total, "cagr": cagr, "sharpe": sharpe,
        "max_drawdown": max_dd, "worst_month": worst_month,
        "avg_leverage": float(sized.loc[sized["scale"] > 0, "scale"].mean()),
        "cycles_halted": halted,
    }
