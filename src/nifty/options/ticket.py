"""Turn the strategy recommendation into an actionable, sized trade ticket.

A ticket is what you'd actually place: exact legs, how many *lots*, the credit in
rupees, margin estimate, breakevens, stop-loss and max risk. Sizing reuses the
risk module so the lot count reflects your capital and risk settings — no naked
"sell 10 lots because it felt right".
"""

from __future__ import annotations

import math

import pandas as pd

from .backtest import OptionsConfig, _value_to_close, build_legs
from .pricing import expected_move
from .sizing import SizingConfig, _rolling_scale

# Standard NSE Nifty options lot size (update if the exchange changes it).
DEFAULT_LOT_SIZE = 75


def _latest_leverage(per_cycle_ret: pd.Series, sizing_cfg: SizingConfig) -> float:
    if per_cycle_ret is None or len(per_cycle_ret) == 0:
        return 1.0
    return float(_rolling_scale(per_cycle_ret.fillna(0.0), sizing_cfg).iloc[-1])


def build_ticket(
    spot: pd.Series,
    vix: pd.Series,
    opt_cfg: OptionsConfig,
    sizing_cfg: SizingConfig,
    per_cycle_ret: pd.Series | None = None,
    lot_size: int = DEFAULT_LOT_SIZE,
    vrp_min: float = 0.02,
    realized_window: int = 20,
) -> dict:
    """Build a sized trade ticket for the most recent day.

    ``vix`` is in decimals (VIX/100). Returns a dict ready to print or log.
    """
    spot = spot.dropna()
    vix = vix.reindex(spot.index).ffill()
    entry_date = spot.index[-1]
    S = float(spot.iloc[-1])
    iv = float(vix.iloc[-1])
    r = opt_cfg.risk_free_rate
    T = opt_cfg.cycle_days / opt_cfg.trading_days

    # Variance risk premium signal (implied vs recent realized).
    logret = (spot.apply(math.log)).diff()
    rv = float(logret.iloc[-realized_window:].std() * math.sqrt(opt_cfg.trading_days))
    vrp = iv - rv

    legs = build_legs(opt_cfg.structure, S, iv, T, opt_cfg)
    credit_pts = _value_to_close(legs, S, T, r, iv)
    em = expected_move(S, iv, T)

    # Position sizing -> lots.
    capital = sizing_cfg.capital
    leverage = _latest_leverage(per_cycle_ret, sizing_cfg)
    target_notional = leverage * capital
    contract_value = S * lot_size
    lots = int(target_notional // contract_value) if contract_value > 0 else 0
    actual_notional = lots * contract_value
    margin_est = sizing_cfg.margin_pct * actual_notional
    credit_rupees = credit_pts * lot_size * lots

    # Breakevens, stop-loss, max risk.
    calls = sorted(K for kind, K, q in legs if kind == "call" and q < 0)
    puts = sorted((K for kind, K, q in legs if kind == "put" and q < 0), reverse=True)
    upper_be = (calls[0] + credit_pts) if calls else None
    lower_be = (puts[0] - credit_pts) if puts else None

    stop_loss_rupees = (
        opt_cfg.stop_loss_mult * credit_rupees if opt_cfg.stop_loss_mult > 0 else None
    )
    if opt_cfg.structure == "iron_condor":
        long_calls = [K for kind, K, q in legs if kind == "call" and q > 0]
        long_puts = [K for kind, K, q in legs if kind == "put" and q > 0]
        call_width = (min(long_calls) - calls[0]) if (long_calls and calls) else 0.0
        put_width = (puts[0] - max(long_puts)) if (long_puts and puts) else 0.0
        width = max(call_width, put_width)
        max_loss_pts = max(width - credit_pts, 0.0)
        max_risk_rupees = max_loss_pts * lot_size * lots
        risk_note = f"defined risk: max loss ≈ ₹{max_risk_rupees:,.0f}"
    else:
        max_risk_rupees = None
        risk_note = ("UNDEFINED risk (naked short) — rely on the stop-loss and "
                     "leverage cap; a gap can exceed it")

    trade = vrp > vrp_min
    if trade and lots >= 1:
        action = f"SELL {opt_cfg.structure} — {lots} lot(s)"
    elif trade and lots < 1:
        action = "SIGNAL = SELL, but capital too small for 1 lot — stand aside / paper only"
    else:
        action = "STAND ASIDE — variance risk premium too thin to sell"

    return {
        "entry_date": str(entry_date.date()),
        "expiry_approx": str((entry_date + pd.offsets.BDay(opt_cfg.cycle_days)).date()),
        "cycle_days": opt_cfg.cycle_days,
        "spot": round(S, 2),
        "implied_vol_vix": round(iv, 4),
        "realized_vol": round(rv, 4),
        "variance_risk_premium": round(vrp, 4),
        "vrp_min_to_trade": vrp_min,
        "action": action,
        "structure": opt_cfg.structure,
        "expected_move_pts": round(em, 1),
        "legs": [
            {"side": "SELL" if q < 0 else "BUY", "kind": k, "strike": float(K),
             "lots": lots}
            for k, K, q in legs
        ],
        "lots": lots,
        "lot_size": lot_size,
        "leverage_used": round(leverage, 2),
        "notional": round(actual_notional, 0),
        "margin_estimate": round(margin_est, 0),
        "credit_per_lot_pts": round(credit_pts, 1),
        "credit_total_rupees": round(credit_rupees, 0),
        "breakeven_upper": round(upper_be, 1) if upper_be else None,
        "breakeven_lower": round(lower_be, 1) if lower_be else None,
        "stop_loss_rupees": round(stop_loss_rupees, 0) if stop_loss_rupees else None,
        "stop_loss_rule": (
            f"exit if open loss reaches {opt_cfg.stop_loss_mult}× credit"
            if opt_cfg.stop_loss_mult > 0 else "no stop configured"
        ),
        "max_risk_rupees": round(max_risk_rupees, 0) if max_risk_rupees is not None else None,
        "risk_note": risk_note,
        "disclaimer": (
            "Educational. Strikes/credit are Black-Scholes-from-VIX estimates (no "
            "skew); real fills differ. Short premium has large tail risk. Paper "
            "trade first. Not financial advice."
        ),
    }


def format_ticket(t: dict) -> str:
    """Human-readable ticket for the terminal."""
    L = []
    L.append("=" * 60)
    L.append(f" TRADE TICKET — {t['entry_date']}  (expiry ≈ {t['expiry_approx']})")
    L.append("=" * 60)
    L.append(f" Spot {t['spot']}  |  VIX {t['implied_vol_vix']*100:.1f}  "
             f"vs realized {t['realized_vol']*100:.1f}  ->  VRP {t['variance_risk_premium']:+.4f}")
    L.append(f" >>> {t['action']}")
    if t["lots"] >= 1 and t["action"].startswith("SELL"):
        L.append(f" Expected move ±{t['expected_move_pts']:.0f} pts  |  "
                 f"leverage {t['leverage_used']}x  |  notional ₹{t['notional']:,.0f}")
        L.append(" Legs:")
        for leg in t["legs"]:
            L.append(f"   {leg['side']:>4s} {leg['lots']} lot {leg['kind']:>4s} @ {leg['strike']:.0f}")
        L.append(f" Credit ≈ ₹{t['credit_total_rupees']:,.0f} "
                 f"({t['credit_per_lot_pts']} pts/lot)  |  margin ≈ ₹{t['margin_estimate']:,.0f}")
        if t["breakeven_lower"] and t["breakeven_upper"]:
            L.append(f" Breakevens: {t['breakeven_lower']:.0f}  /  {t['breakeven_upper']:.0f}")
        L.append(f" Stop: {t['stop_loss_rule']}"
                 + (f" (≈ ₹{t['stop_loss_rupees']:,.0f})" if t['stop_loss_rupees'] else ""))
        L.append(f" Risk: {t['risk_note']}")
    L.append("-" * 60)
    L.append(" ⚠ " + t["disclaimer"])
    L.append("=" * 60)
    return "\n".join(L)
