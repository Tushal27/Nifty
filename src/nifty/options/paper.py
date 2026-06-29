"""A tiny paper-trading ledger to validate the options edge before real money.

Records sold positions, marks them to market daily with Black-Scholes (off the
current VIX), and settles them at intrinsic value on expiry. The account equity =
starting capital + realised + unrealised, so you can watch the strategy work (or
not) on paper with zero risk.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date

import numpy as np
import pandas as pd

from .backtest import _intrinsic_close, _value_to_close


def _spot_on_or_before(spot: pd.Series, d: pd.Timestamp) -> float:
    s = spot.loc[:d]
    return float(s.iloc[-1]) if len(s) else float("nan")


class PaperLedger:
    def __init__(self, path: str, capital: float, risk_free_rate: float = 0.065,
                 trading_days: int = 252):
        self.path = path
        self.capital = capital
        self.r = risk_free_rate
        self.trading_days = trading_days
        self.positions: list[dict] = []
        self.load()

    # --- persistence --------------------------------------------------------
    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as fh:
                data = json.load(fh)
            self.positions = data.get("positions", [])
            self.capital = data.get("capital", self.capital)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"capital": self.capital, "positions": self.positions},
                      fh, indent=2)

    # --- trade lifecycle ----------------------------------------------------
    def open_from_ticket(self, ticket: dict) -> dict:
        legs = [[leg["kind"], leg["strike"], -1 if leg["side"] == "SELL" else 1]
                for leg in ticket["legs"]]
        pos = {
            "id": len(self.positions) + 1,
            "entry_date": ticket["entry_date"],
            "expiry_date": ticket["expiry_approx"],
            "structure": ticket["structure"],
            "legs": legs,
            "lots": ticket["lots"],
            "lot_size": ticket["lot_size"],
            "credit_pts": ticket["credit_per_lot_pts"],
            "spot_entry": ticket["spot"],
            "status": "open",
            "exit_date": None,
            "realized_rupees": None,
        }
        self.positions.append(pos)
        self.save()
        return pos

    def _remaining_T(self, expiry: str, today: pd.Timestamp) -> float:
        bdays = np.busday_count(today.date(), date.fromisoformat(expiry))
        return max(int(bdays), 0) / self.trading_days

    def mark(self, spot: pd.Series, vix: pd.Series) -> list[dict]:
        """Unrealised P&L for each open position at the latest date."""
        today = spot.index[-1]
        sig = float(vix.reindex(spot.index).ffill().iloc[-1])
        S = float(spot.iloc[-1])
        out = []
        for p in self.positions:
            if p["status"] != "open":
                continue
            T = self._remaining_T(p["expiry_date"], today)
            legs = [(k, K, q) for k, K, q in p["legs"]]
            if T <= 0:
                close_cost = _intrinsic_close(legs, S)
            else:
                close_cost = _value_to_close(legs, S, T, self.r, sig)
            unreal_pts = p["credit_pts"] - close_cost
            unreal = unreal_pts * p["lot_size"] * p["lots"]
            out.append({"id": p["id"], "structure": p["structure"],
                        "expiry": p["expiry_date"], "T_years": round(T, 4),
                        "unrealized_rupees": round(unreal, 0)})
        return out

    def settle_expired(self, spot: pd.Series) -> list[dict]:
        """Close any position whose expiry has passed, at intrinsic value."""
        today = spot.index[-1]
        settled = []
        for p in self.positions:
            if p["status"] != "open":
                continue
            exp = pd.Timestamp(p["expiry_date"])
            if today < exp:
                continue
            S_exp = _spot_on_or_before(spot, exp)
            legs = [(k, K, q) for k, K, q in p["legs"]]
            realized_pts = p["credit_pts"] - _intrinsic_close(legs, S_exp)
            realized = realized_pts * p["lot_size"] * p["lots"]
            p["status"] = "closed"
            p["exit_date"] = str(exp.date())
            p["realized_rupees"] = round(realized, 0)
            settled.append({"id": p["id"], "realized_rupees": p["realized_rupees"]})
        if settled:
            self.save()
        return settled

    # --- reporting ----------------------------------------------------------
    def account(self, spot: pd.Series, vix: pd.Series) -> dict:
        realized = sum(p["realized_rupees"] or 0 for p in self.positions
                       if p["status"] == "closed")
        marks = self.mark(spot, vix)
        unrealized = sum(m["unrealized_rupees"] for m in marks)
        n_open = sum(1 for p in self.positions if p["status"] == "open")
        n_closed = sum(1 for p in self.positions if p["status"] == "closed")
        return {
            "starting_capital": self.capital,
            "realized_rupees": round(realized, 0),
            "unrealized_rupees": round(unrealized, 0),
            "equity": round(self.capital + realized + unrealized, 0),
            "return_pct": round((realized + unrealized) / self.capital * 100, 2),
            "open_positions": n_open,
            "closed_positions": n_closed,
        }

    def realized_frame(self) -> pd.DataFrame:
        closed = [p for p in self.positions if p["status"] == "closed"]
        if not closed:
            return pd.DataFrame()
        df = pd.DataFrame(closed)[["exit_date", "structure", "realized_rupees"]]
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = df.sort_values("exit_date").set_index("exit_date")
        df["equity"] = self.capital + df["realized_rupees"].cumsum()
        return df
