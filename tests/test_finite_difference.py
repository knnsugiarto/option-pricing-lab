"""Tests for the Crank–Nicolson PDE pricer.

The closed form is exact under the same model, so every error here is
numerical. The tests check accuracy, the convergence order the theory
predicts, and the known failure of plain Crank–Nicolson that Rannacher
start-up exists to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from optlab import black_scholes as bs
from optlab import finite_difference as fd

SEED = 20261013
N_CASES = 40
ATM = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "vol": 0.20, "div_yield": 0.0}


@pytest.fixture(scope="module")
def random_contracts() -> list[dict[str, float]]:
    """Seeded contracts spanning the validated parameter range."""
    rng = np.random.default_rng(SEED)
    return [
        {
            "spot": 100.0,
            "strike": float(rng.uniform(70.0, 130.0)),
            "maturity": float(rng.uniform(0.1, 2.0)),
            "rate": float(rng.uniform(-0.01, 0.06)),
            "vol": float(rng.uniform(0.10, 0.60)),
            "div_yield": float(rng.uniform(0.0, 0.04)),
        }
        for _ in range(N_CASES)
    ]


def fitted_order(space_steps: list[int], errors: list[float]) -> float:
    """Convergence order p from a least-squares fit of log(error) = c - p log(N)."""
    slope = np.polyfit(np.log(space_steps), np.log(errors), 1)[0]
    return float(-slope)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_matches_closed_form_across_contracts(random_contracts: list[dict[str, float]], option_type: str) -> None:
    """Default settings (Rannacher start-up, Richardson extrapolation) are accurate everywhere tested."""
    for contract in random_contracts:
        numeric = fd.price(**contract, option_type=option_type)
        exact_price = bs.price(**contract, option_type=option_type)
        exact = bs.greeks(**contract, option_type=option_type)
        assert numeric.price == pytest.approx(exact_price, abs=1e-6), contract
        assert numeric.delta == pytest.approx(exact.delta, abs=1e-6), contract
        assert numeric.gamma == pytest.approx(exact.gamma, abs=5e-6), contract


def test_rannacher_crank_nicolson_is_second_order() -> None:
    """Price, delta and gamma errors at the strike fall like 1/N^2."""
    steps = [100, 200, 400, 800]
    exact_price = bs.price(**ATM)
    exact = bs.greeks(**ATM)
    results = [fd.price(**ATM, space_steps=n, time_steps=n // 8, extrapolate=False) for n in steps]
    assert fitted_order(steps, [abs(r.price - exact_price) for r in results]) == pytest.approx(2.0, abs=0.1)
    assert fitted_order(steps, [abs(r.delta - exact.delta) for r in results]) == pytest.approx(2.0, abs=0.1)
    assert fitted_order(steps, [abs(r.gamma - exact.gamma) for r in results]) == pytest.approx(2.0, abs=0.1)


def test_plain_crank_nicolson_gamma_diverges_with_large_time_steps() -> None:
    """The failure Rannacher start-up fixes: with 8x fewer time steps than space
    steps, plain Crank–Nicolson's gamma at the strike gets worse as the grid is
    refined, because the payoff kink's high-frequency error is barely damped."""
    exact_gamma = bs.greeks(**ATM).gamma
    errors = [
        abs(fd.price(**ATM, space_steps=n, time_steps=n // 8, rannacher_steps=0, extrapolate=False).gamma - exact_gamma)
        for n in (200, 400, 800)
    ]
    assert errors[0] < errors[1] < errors[2]
    fixed = fd.price(**ATM, space_steps=800, time_steps=100, rannacher_steps=2, extrapolate=False)
    assert abs(fixed.gamma - exact_gamma) < errors[2] / 1_000


def test_richardson_extrapolation_is_fourth_order() -> None:
    steps = [50, 100, 200, 400]
    exact_price = bs.price(**ATM)
    errors = [abs(fd.price(**ATM, space_steps=n, time_steps=n // 8).price - exact_price) for n in steps]
    assert fitted_order(steps, errors) == pytest.approx(4.0, abs=0.4)


def test_put_call_parity() -> None:
    contract = {"spot": 100.0, "strike": 110.0, "maturity": 0.5, "rate": 0.03, "vol": 0.3, "div_yield": 0.02}
    call = fd.price(**contract, option_type="call").price
    put = fd.price(**contract, option_type="put").price
    parity = 100.0 * np.exp(-0.02 * 0.5) - 110.0 * np.exp(-0.03 * 0.5)
    assert call - put == pytest.approx(parity, abs=1e-6)


def test_grid_greeks_match_closed_form_near_the_money() -> None:
    grid = fd.solve_grid(**ATM, space_steps=1600, time_steps=400)
    spots, delta, gamma = fd.grid_greeks(grid)
    near = (spots > 70.0) & (spots < 140.0)
    exact = bs.greeks(spots[near], ATM["strike"], ATM["maturity"], ATM["rate"], ATM["vol"], ATM["div_yield"])
    np.testing.assert_allclose(delta[near], exact.delta, atol=2e-5)
    np.testing.assert_allclose(gamma[near], exact.gamma, atol=2e-6)


def test_strike_is_a_grid_node() -> None:
    grid = fd.solve_grid(spot=93.0, strike=100.0, maturity=0.5, rate=0.02, vol=0.25, space_steps=200, time_steps=50)
    assert 0.0 in grid.log_moneyness
    assert grid.spots[np.argmin(np.abs(grid.log_moneyness))] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"space_steps": 101},
        {"space_steps": 2},
        {"time_steps": 0},
        {"rannacher_steps": 200, "time_steps": 100},
        {"n_std": 0.0},
        {"vol": 0.0},
        {"option_type": "digital"},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        fd.price(**{**ATM, **kwargs})  # type: ignore[arg-type]
