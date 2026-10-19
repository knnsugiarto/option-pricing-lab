"""Closed-form Black–Scholes–Merton prices and Greeks for European options.

This is the benchmark that the Monte Carlo and finite-difference pricers are
checked against. Every function broadcasts over NumPy arrays, so one call can
price a whole grid of strikes or maturities.

Conventions
-----------
- ``maturity`` is in years. ``rate`` and ``div_yield`` are continuously
  compounded annual rates. ``vol`` is annualised (0.20 means 20%).
- Vega and rho are per unit change: per 1.00 of vol (100 vol points) and per
  1.00 of rate (100%). Divide by 100 for "per point" quotes.
- Theta is the derivative with respect to calendar time, per year. Divide by
  365 for an approximate per-day figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from optlab._inputs import OptionType, check_option_type, prepare_arrays

FloatOrArray = float | NDArray[np.float64]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of an option price, in the units given in the module docstring."""

    delta: FloatOrArray
    gamma: FloatOrArray
    vega: FloatOrArray
    theta: FloatOrArray
    rho: FloatOrArray


def _normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard normal density."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _to_output(values: NDArray[np.float64]) -> FloatOrArray:
    """Return a Python float for scalar inputs and an array otherwise."""
    return float(values) if values.ndim == 0 else values


def _d1_d2_arrays(
    spot: NDArray[np.float64],
    strike: NDArray[np.float64],
    maturity: NDArray[np.float64],
    rate: NDArray[np.float64],
    vol: NDArray[np.float64],
    div_yield: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d1 and d2 on already-validated arrays."""
    total_vol = vol * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * maturity) / total_vol
    return d1, d1 - total_vol


def d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div_yield: ArrayLike = 0.0,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Return the Black–Scholes quantities d1 and d2.

    N(d2) is the risk-neutral probability that a call finishes in the money.
    N(d1) is the same probability under the measure that uses the stock as
    numeraire; e^(-qT) N(d1) is the call delta.
    """
    d1, d2 = _d1_d2_arrays(*prepare_arrays(spot, strike, maturity, rate, vol, div_yield))
    return _to_output(d1), _to_output(d2)


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div_yield: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> FloatOrArray:
    """Black–Scholes–Merton price of a European call or put.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        vol: Annualised volatility of the underlying.
        div_yield: Continuously compounded dividend yield.
        option_type: "call" or "put".

    Returns:
        The option price: a float for scalar inputs, otherwise an array of
        the broadcast shape of the inputs.
    """
    check_option_type(option_type)
    s, k, t, r, sigma, q = prepare_arrays(spot, strike, maturity, rate, vol, div_yield)
    d1, d2 = _d1_d2_arrays(s, k, t, r, sigma, q)
    prepaid_forward = s * np.exp(-q * t)  # value today of receiving the stock at expiry
    discounted_strike = k * np.exp(-r * t)
    if option_type == "call":
        value = prepaid_forward * ndtr(d1) - discounted_strike * ndtr(d2)
    else:
        value = discounted_strike * ndtr(-d2) - prepaid_forward * ndtr(-d1)
    return _to_output(value)


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div_yield: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> Greeks:
    """Analytical Greeks of a European call or put.

    Arguments are the same as :func:`price`. Units are in the module docstring.
    """
    check_option_type(option_type)
    s, k, t, r, sigma, q = prepare_arrays(spot, strike, maturity, rate, vol, div_yield)
    d1, d2 = _d1_d2_arrays(s, k, t, r, sigma, q)
    sqrt_t = np.sqrt(t)
    prepaid_forward = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    density = _normal_pdf(d1)

    # Gamma and vega are the same for calls and puts (put-call parity is linear in spot).
    gamma = prepaid_forward * density / (s * s * sigma * sqrt_t)
    vega = prepaid_forward * density * sqrt_t
    time_decay = -prepaid_forward * density * sigma / (2.0 * sqrt_t)

    if option_type == "call":
        delta = np.exp(-q * t) * ndtr(d1)
        theta = time_decay - r * discounted_strike * ndtr(d2) + q * prepaid_forward * ndtr(d1)
        rho = t * discounted_strike * ndtr(d2)
    else:
        delta = -np.exp(-q * t) * ndtr(-d1)
        theta = time_decay + r * discounted_strike * ndtr(-d2) - q * prepaid_forward * ndtr(-d1)
        rho = -t * discounted_strike * ndtr(-d2)

    return Greeks(
        delta=_to_output(delta),
        gamma=_to_output(gamma),
        vega=_to_output(vega),
        theta=_to_output(theta),
        rho=_to_output(rho),
    )
