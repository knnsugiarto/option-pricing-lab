"""Input validation shared by every pricer."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

OptionType = Literal["call", "put"]

_POSITIVE = ("spot", "strike", "maturity", "vol")


def check_option_type(option_type: str) -> None:
    """Raise ValueError unless option_type is 'call' or 'put'."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def prepare_arrays(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div_yield: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Convert pricing inputs to broadcast float arrays and validate them.

    Raises:
        ValueError: if any input is non-finite, or if spot, strike, maturity
            or vol is not strictly positive.
    """
    named = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": rate,
        "vol": vol,
        "div_yield": div_yield,
    }
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in named.items()}
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite, got {named[name]!r}")
    for name in _POSITIVE:
        if np.any(arrays[name] <= 0.0):
            raise ValueError(f"{name} must be strictly positive, got {named[name]!r}")
    return tuple(np.broadcast_arrays(*arrays.values()))


def prepare_scalars(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float,
) -> tuple[float, float, float, float, float, float]:
    """Validate scalar pricing inputs and return them as Python floats."""
    arrays = prepare_arrays(spot, strike, maturity, rate, vol, div_yield)
    if any(a.ndim != 0 for a in arrays):
        raise ValueError("this pricer takes scalar inputs; loop over parameter sets instead")
    s, k, t, r, sigma, q = (float(a) for a in arrays)
    return s, k, t, r, sigma, q
