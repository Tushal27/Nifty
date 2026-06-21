"""Black-Scholes pricing for European index options.

Self-contained (no SciPy): the normal CDF uses ``math.erf``. Prices are in the
same units as the underlying (Nifty index points). Dividends are ignored, which
is a fine approximation for the Nifty price index over short tenors.
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Black-Scholes price of a European call/put.

    ``T`` in years, ``sigma`` annualised (decimal), ``r`` annual risk-free rate.
    At/below zero time or vol the option is worth its intrinsic value.
    """
    if T <= 0.0 or sigma <= 0.0:
        return _intrinsic(S, K, kind)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    if kind == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Option delta (call in [0,1], put in [-1,0])."""
    if T <= 0.0 or sigma <= 0.0:
        if kind == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sd
    return norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0


def expected_move(S: float, sigma: float, T: float) -> float:
    """One-standard-deviation move of the underlying over tenor ``T``."""
    return S * sigma * math.sqrt(max(T, 0.0))
