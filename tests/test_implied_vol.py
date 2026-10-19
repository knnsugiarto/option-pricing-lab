"""Tests for the implied volatility solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optlab import black_scholes as bs
from optlab.implied_vol import implied_vol

SEED = 20261013


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_vol(option_type: str) -> None:
    """Round trip: price at a known vol, then solve back for it."""
    rng = np.random.default_rng(SEED)
    checked = 0
    for _ in range(500):
        contract = {
            "spot": 100.0,
            "strike": float(rng.uniform(60.0, 150.0)),
            "maturity": float(rng.uniform(0.02, 3.0)),
            "rate": float(rng.uniform(-0.01, 0.08)),
            "div_yield": float(rng.uniform(0.0, 0.05)),
        }
        vol = float(rng.uniform(0.05, 1.5))
        # Skip quotes so far out of the money that the price carries no information about vol.
        if bs.greeks(**contract, vol=vol, option_type=option_type).vega < 1e-3:
            continue
        price = bs.price(**contract, vol=vol, option_type=option_type)
        assert implied_vol(price, **contract, option_type=option_type) == pytest.approx(vol, abs=1e-8)
        checked += 1
    assert checked > 400


@pytest.mark.parametrize(
    ("price", "option_type"),
    [
        (0.5, "call"),  # below the intrinsic lower bound S e^(-qT) - K e^(-rT) of about 10.5
        (101.0, "call"),  # above the spot: no call is worth more than the stock
        (95.0, "put"),  # above the discounted strike
        (-1.0, "put"),
        (math.nan, "call"),
    ],
)
def test_arbitrage_violating_prices_return_nan(price: float, option_type: str) -> None:
    strike = 90.0 if option_type == "call" else 100.0
    result = implied_vol(price, spot=100.0, strike=strike, maturity=1.0, rate=0.05, option_type=option_type)  # type: ignore[arg-type]
    assert math.isnan(result)


def test_solution_is_unique_for_a_put_and_its_parity_call() -> None:
    """Put–call parity holds for any vol, so a call and the put implied by parity share one vol."""
    contract = {"spot": 100.0, "strike": 95.0, "maturity": 0.5, "rate": 0.03, "div_yield": 0.01}
    put_price = bs.price(**contract, vol=0.27, option_type="put")
    call_price = put_price + 100.0 * np.exp(-0.01 * 0.5) - 95.0 * np.exp(-0.03 * 0.5)
    assert implied_vol(call_price, **contract, option_type="call") == pytest.approx(0.27, abs=1e-10)
    assert implied_vol(put_price, **contract, option_type="put") == pytest.approx(0.27, abs=1e-10)


def test_invalid_bracket_raises() -> None:
    with pytest.raises(ValueError):
        implied_vol(10.0, 100.0, 100.0, 1.0, 0.05, vol_lower=0.5, vol_upper=0.1)
