"""Backtest premium-selling options strategies on the Nifty index.

The engine sells a chosen structure every cycle (e.g. weekly), prices each leg
with Black-Scholes using India VIX as the implied vol, marks the position daily
to expiry (for stop-loss), and settles at intrinsic value. P&L is expressed as a
return on the index notional so it compounds into an equity curve comparable to
buying the index.

Honest caveats baked into the docs/output:
  * BS-from-VIX ignores the volatility *skew* and term structure, so it slightly
    misprices OTM legs — real strangle/condor credits differ somewhat.
  * Short straddles/strangles have *undefined* tail risk; the stop-loss and VIX
    filter mitigate but do not remove it. Iron condors cap the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pricing import bs_price, expected_move

STRUCTURES = ("short_straddle", "short_strangle", "iron_condor")


@dataclass
class OptionsConfig:
    risk_free_rate: float = 0.065
    cycle_days: int = 5            # trading days per expiry cycle (5≈weekly)
    structure: str = "short_strangle"
    strangle_sd: float = 1.0       # short strikes at ±sd * expected move
    condor_wing_sd: float = 2.0    # long wings for iron condor
    strike_step: int = 50          # Nifty strike granularity
    cost_per_leg_pts: float = 1.0  # round-trip cost per leg, in index points
    vix_max: float = 1.50          # skip entries when VIX/100 above this (1.50≈off)
    stop_loss_mult: float = 2.0    # close early if loss exceeds mult × credit (0=off)
    trading_days: int = 252

    @classmethod
    def from_dict(cls, d: dict) -> "OptionsConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})


def _round_strike(x: float, step: int) -> float:
    return round(x / step) * step


def build_legs(structure: str, S: float, sigma: float, T: float,
               cfg: OptionsConfig) -> list[tuple[str, float, int]]:
    """Return legs as (kind, strike, qty); qty<0 short, qty>0 long."""
    em = expected_move(S, sigma, T)
    atm = _round_strike(S, cfg.strike_step)
    if structure == "short_straddle":
        return [("call", atm, -1), ("put", atm, -1)]
    kc = _round_strike(S + cfg.strangle_sd * em, cfg.strike_step)
    kp = _round_strike(S - cfg.strangle_sd * em, cfg.strike_step)
    if structure == "short_strangle":
        return [("call", kc, -1), ("put", kp, -1)]
    if structure == "iron_condor":
        kcw = _round_strike(S + cfg.condor_wing_sd * em, cfg.strike_step)
        kpw = _round_strike(S - cfg.condor_wing_sd * em, cfg.strike_step)
        return [("call", kc, -1), ("put", kp, -1),
                ("call", kcw, 1), ("put", kpw, 1)]
    raise ValueError(f"unknown structure {structure!r}; use one of {STRUCTURES}")


def _value_to_close(legs, S, T, r, sigma) -> float:
    """Cash needed to unwind the position now (credit collected uses entry T)."""
    return sum(-qty * bs_price(S, K, T, r, sigma, kind) for kind, K, qty in legs)


def _intrinsic_close(legs, S) -> float:
    val = 0.0
    for kind, K, qty in legs:
        intr = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        val += -qty * intr
    return val


def run_options_backtest(
    spot: pd.Series, vix: pd.Series, cfg: OptionsConfig
) -> tuple[pd.DataFrame, dict]:
    """Simulate the premium-selling strategy. ``vix`` is in decimals (VIX/100)."""
    spot = spot.dropna()
    vix = vix.reindex(spot.index).ffill()
    idx = spot.index
    r = cfg.risk_free_rate
    cyc = cfg.cycle_days

    rows = []
    i = 0
    while i + cyc < len(idx):
        S0 = float(spot.iloc[i])
        sig0 = float(vix.iloc[i])
        if not np.isfinite(sig0) or sig0 <= 0:
            i += cyc
            continue
        if sig0 > cfg.vix_max:          # regime filter: stand aside
            rows.append({"entry_date": idx[i], "traded": False, "ret": 0.0,
                         "pnl_pts": 0.0, "credit": 0.0, "exit": "filtered"})
            i += cyc
            continue

        T0 = cyc / cfg.trading_days
        legs = build_legs(cfg.structure, S0, sig0, T0, cfg)
        credit = _value_to_close(legs, S0, T0, r, sig0)  # premium received (net)

        # March day by day to expiry, applying an optional stop-loss.
        exit_reason = "expiry"
        realized_pnl = None
        for j in range(i + 1, i + cyc + 1):
            days_left = (i + cyc) - j
            Tj = days_left / cfg.trading_days
            Sj = float(spot.iloc[j])
            sigj = float(vix.iloc[j]) if np.isfinite(vix.iloc[j]) else sig0
            if days_left <= 0:
                close_cost = _intrinsic_close(legs, Sj)
            else:
                close_cost = _value_to_close(legs, Sj, Tj, r, sigj)
            pnl = credit - close_cost
            if cfg.stop_loss_mult > 0 and pnl <= -cfg.stop_loss_mult * credit:
                realized_pnl = pnl
                exit_reason = "stop"
                break
        if realized_pnl is None:
            realized_pnl = credit - _intrinsic_close(legs, float(spot.iloc[i + cyc]))

        costs = cfg.cost_per_leg_pts * len(legs)
        net_pnl = realized_pnl - costs
        rows.append({
            "entry_date": idx[i], "traded": True, "credit": credit,
            "pnl_pts": net_pnl, "ret": net_pnl / S0, "exit": exit_reason,
        })
        i += cyc

    bt = pd.DataFrame(rows).set_index("entry_date")
    bt["equity"] = (1 + bt["ret"]).cumprod()
    metrics = _options_stats(bt, spot, cfg)
    return bt, metrics


def _options_stats(bt: pd.DataFrame, spot: pd.Series, cfg: OptionsConfig) -> dict:
    rets = bt["ret"]
    traded = bt[bt["traded"]]
    cycles_per_year = cfg.trading_days / cfg.cycle_days
    n = len(rets)
    equity = bt["equity"]
    total_return = float(equity.iloc[-1] - 1.0) if n else float("nan")
    years = n / cycles_per_year if n else float("nan")
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years and years > 0 else float("nan")
    vol = rets.std()
    sharpe = float(rets.mean() / vol * math.sqrt(cycles_per_year)) if vol > 0 else float("nan")
    run_max = equity.cummax()
    max_dd = float((equity / run_max - 1.0).min()) if n else float("nan")
    wins = (traded["ret"] > 0).mean() if len(traded) else float("nan")

    # Index buy-and-hold over the same span, for context.
    px = spot.reindex(bt.index).ffill()
    bh_total = float(px.iloc[-1] / px.iloc[0] - 1.0) if len(px) > 1 else float("nan")

    return {
        "structure": cfg.structure,
        "cycle_days": cfg.cycle_days,
        "period_start": str(bt.index.min().date()),
        "period_end": str(bt.index.max().date()),
        "n_cycles": int(n),
        "n_traded": int(len(traded)),
        "win_rate": float(wins),
        "avg_credit_pts": float(traded["credit"].mean()) if len(traded) else float("nan"),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "worst_cycle": float(rets.min()) if n else float("nan"),
        "best_cycle": float(rets.max()) if n else float("nan"),
        "stops_hit": int((bt["exit"] == "stop").sum()),
        "cycles_filtered": int((bt["exit"] == "filtered").sum()),
        "index_buy_hold_return": bh_total,
    }


def recommend(spot: pd.Series, vix: pd.Series, realized_window: int,
              cfg: OptionsConfig) -> dict:
    """Strategy recommendation for the most recent day from the VRP signal."""
    spot = spot.dropna()
    vix = vix.reindex(spot.index).ffill()
    S = float(spot.iloc[-1])
    iv = float(vix.iloc[-1])  # implied (decimal)
    logret = np.log(spot).diff()
    rv = float(logret.iloc[-realized_window:].std() * math.sqrt(cfg.trading_days))
    vrp = iv - rv
    T = cfg.cycle_days / cfg.trading_days
    legs = build_legs(cfg.structure, S, iv, T, cfg)
    credit = _value_to_close(legs, S, T, cfg.risk_free_rate, iv)
    em = expected_move(S, iv, T)

    if vrp > 0.02:
        stance = "SELL premium — implied vol is richer than recent realized (positive VRP)."
    elif vrp < -0.02:
        stance = "AVOID selling / consider BUYING — implied is below realized (negative VRP)."
    else:
        stance = "NEUTRAL — implied ≈ realized; thin edge, trade small or stand aside."

    return {
        "as_of_date": str(spot.index[-1].date()),
        "spot": round(S, 2),
        "implied_vol_vix": round(iv, 4),
        "realized_vol": round(rv, 4),
        "variance_risk_premium": round(vrp, 4),
        "stance": stance,
        "suggested_structure": cfg.structure,
        "cycle_days": cfg.cycle_days,
        "expected_move_pts": round(em, 1),
        "suggested_legs": [
            {"kind": k, "strike": float(K), "side": "short" if q < 0 else "long"}
            for k, K, q in legs
        ],
        "net_credit_pts": round(credit, 1),
        "disclaimer": (
            "Educational simulation using Black-Scholes priced off VIX (no skew). "
            "Short premium carries large tail risk. Not financial advice."
        ),
    }
