"""Implied volatility: the vol that makes Black–Scholes match a given price.

The Black–Scholes price is strictly increasing in vol (vega > 0), so for any
price inside the no-arbitrage bounds there is exactly one implied vol. We find
it with Brent's method, a bracketing root-finder: it cannot diverge, which
matters far from the money where vega is tiny and Newton's method overshoots.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from optlab import black_scholes as bs
from optlab._inputs import OptionType, check_option_type, prepare_scalars

VOL_LOWER = 1e-4
VOL_UPPER = 5.0


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    vol_lower: float = VOL_LOWER,
    vol_upper: float = VOL_UPPER,
    tolerance: float = 1e-12,
) -> float:
    """Black–Scholes implied volatility of a European option price.

    Args:
        price: Observed option price.
        spot, strike, maturity, rate, div_yield, option_type: as in
            :func:`optlab.black_scholes.price`.
        vol_lower, vol_upper: Search bracket for the vol.
        tolerance: Absolute tolerance on the vol.

    Returns:
        The implied vol, or NaN if no vol in [vol_lower, vol_upper] reproduces
        the price. That happens when the price breaks a no-arbitrage bound
        (below intrinsic value, above the forward or strike) or implies a vol
        outside the bracket. Callers count NaNs as rejected quotes.
    """
    check_option_type(option_type)
    if not 0.0 < vol_lower < vol_upper:
        raise ValueError(f"need 0 < vol_lower < vol_upper, got {vol_lower}, {vol_upper}")
    s, k, t, r, _, q = prepare_scalars(spot, strike, maturity, rate, 1.0, div_yield)
    if not math.isfinite(price):
        return math.nan

    def pricing_error(vol: float) -> float:
        return float(bs.price(s, k, t, r, vol, q, option_type)) - price

    low, high = pricing_error(vol_lower), pricing_error(vol_upper)
    if low > 0.0 or high < 0.0:
        return math.nan
    if low == 0.0:
        return vol_lower
    return float(brentq(pricing_error, vol_lower, vol_upper, xtol=tolerance, rtol=4 * np.finfo(float).eps, maxiter=200))
