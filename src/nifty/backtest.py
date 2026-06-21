"""Walk-forward backtest: turn model probabilities into a trade strategy.

The backtest is fully out-of-sample: the model is retrained on an expanding
window and only ever predicts the *held-out* tail, so the equity curve reflects
how the strategy would have traded on unseen data. Positions are applied to the
**next day's** return to avoid look-ahead bias.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import Config


def walk_forward_predict(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> pd.Series:
    """Expanding-window out-of-sample P(up) for the tail of the series."""
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=n_splits)
    Xv, yv = X.values, y.values
    proba = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in splitter.split(Xv):
        model = factory()
        model.fit(Xv[train_idx], yv[train_idx])
        p = np.asarray(model.predict_proba(Xv[test_idx]))
        proba.iloc[test_idx] = p
    return proba.dropna()


def _positions(proba: pd.Series, threshold: float, allow_short: bool) -> pd.Series:
    pos = (proba > threshold).astype(float)
    if allow_short:
        pos = pos - (proba < (1 - threshold)).astype(float)  # +1 / 0 / -1
    return pos


def run_backtest(
    proba: pd.Series,
    full: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict]:
    """Build an equity curve and performance metrics vs buy-and-hold.

    Returns ``(frame, metrics)`` where ``frame`` has per-day positions, returns
    and equity for both the strategy and buy-and-hold.
    """
    bt = config.backtest
    threshold = bt["threshold"]
    cost = bt["cost_bps"] / 1e4
    tdays = bt["trading_days"]

    # Next-day simple return realised by holding a position decided *today*.
    next_ret = full["close"].pct_change().shift(-1)

    df = pd.DataFrame(index=proba.index)
    df["proba"] = proba
    df["position"] = _positions(proba, threshold, bt["allow_short"])
    df["market_return"] = next_ret.reindex(df.index)
    df = df.dropna(subset=["market_return"])

    # Transaction cost charged whenever the position changes.
    turnover = df["position"].diff().abs().fillna(df["position"].abs())
    df["strategy_return"] = df["position"] * df["market_return"] - turnover * cost

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buyhold_equity"] = (1 + df["market_return"]).cumprod()

    metrics = {
        "period_start": str(df.index.min().date()),
        "period_end": str(df.index.max().date()),
        "n_days": int(len(df)),
        "n_trades": int(turnover[turnover > 0].count()),
        "strategy": _perf_stats(df["strategy_return"], df["strategy_equity"], tdays),
        "buy_and_hold": _perf_stats(df["market_return"], df["buyhold_equity"], tdays),
    }
    return df, metrics


def _perf_stats(returns: pd.Series, equity: pd.Series, tdays: int) -> dict:
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / tdays
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else float("nan")
    vol = returns.std()
    sharpe = float(returns.mean() / vol * np.sqrt(tdays)) if vol > 0 else float("nan")
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    # Win rate over *active* days only (flat days have a 0 return and would
    # otherwise be miscounted as losses).
    active = returns != 0
    win_rate = float((returns[active] > 0).mean()) if active.any() else float("nan")
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }


def latest_signal(
    model, X: pd.DataFrame, full: pd.DataFrame, config: Config
) -> dict:
    """Predict the direction for the most recent available day."""
    proba = float(np.asarray(model.predict_proba(X.values))[-1])
    threshold = config.backtest["threshold"]
    last_date = X.index[-1]
    direction = "UP" if proba > threshold else "DOWN/FLAT"
    return {
        "as_of_date": str(last_date.date()),
        "predicts_for": "next trading day",
        "probability_up": round(proba, 4),
        "signal": direction,
        "threshold": threshold,
        "last_close": float(full.loc[last_date, "close"]),
        "disclaimer": (
            "Educational output only. Daily direction is near-random; this is not "
            "financial advice."
        ),
    }
