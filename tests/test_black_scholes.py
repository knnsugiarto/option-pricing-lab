"""Tests for the closed-form Black–Scholes–Merton pricer.

Each test checks the formulas against something independent of them: a
published textbook value, direct numerical integration of the risk-neutral
expectation, model-free no-arbitrage relations, or finite differences of the
price itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from optlab import black_scholes as bs

SEED = 20261012
N_RANDOM_CASES = 2_000


@pytest.fixture(scope="module")
def random_params() -> dict[str, np.ndarray]:
    """A seeded spread of realistic parameters, including negative rates."""
    rng = np.random.default_rng(SEED)
    return {
        "spot": rng.uniform(50.0, 150.0, N_RANDOM_CASES),
        "strike": rng.uniform(50.0, 150.0, N_RANDOM_CASES),
        "maturity": rng.uniform(0.05, 3.0, N_RANDOM_CASES),
        "rate": rng.uniform(-0.01, 0.08, N_RANDOM_CASES),
        "vol": rng.uniform(0.05, 0.80, N_RANDOM_CASES),
        "div_yield": rng.uniform(0.0, 0.05, N_RANDOM_CASES),
    }


def risk_neutral_expectation(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float,
    option_type: str,
) -> float:
    """Discounted expected payoff, integrated numerically over the terminal normal draw.

    Under the risk-neutral measure, S_T = S exp((r - q - vol^2/2) T + vol sqrt(T) Z)
    with Z standard normal. This does not use any Black–Scholes formula.
    """

    drift = (rate - div_yield - 0.5 * vol**2) * maturity
    diffusion = vol * np.sqrt(maturity)

    def integrand(z: float) -> float:
        log_density = -0.5 * z * z
        # S_T and K, each weighted by the Gaussian density. The stock term's
        # exponents are combined so exp() cannot overflow in the far tail.
        stock_term = spot * np.exp(drift + diffusion * z + log_density)
        strike_term = strike * np.exp(log_density)
        weighted = stock_term - strike_term if option_type == "call" else strike_term - stock_term
        return max(weighted, 0.0) / np.sqrt(2.0 * np.pi)

    # Split at the kink of the payoff so the quadrature sees two smooth pieces.
    z_kink = (np.log(strike / spot) - drift) / diffusion
    left, _ = quad(integrand, -np.inf, z_kink, epsabs=1e-13, epsrel=1e-12)
    right, _ = quad(integrand, z_kink, np.inf, epsabs=1e-13, epsrel=1e-12)
    return float(np.exp(-rate * maturity) * (left + right))


def test_matches_textbook_example() -> None:
    """Hull, Options, Futures, and Other Derivatives: S=42, K=40, r=10%, vol=20%, T=0.5."""
    call = bs.price(42.0, 40.0, 0.5, 0.10, 0.20, option_type="call")
    put = bs.price(42.0, 40.0, 0.5, 0.10, 0.20, option_type="put")
    assert call == pytest.approx(4.76, abs=0.005)
    assert put == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "vol", "div_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),  # at the money
        (100.0, 130.0, 0.25, 0.03, 0.35, 0.02),  # out of the money call, short expiry
        (100.0, 70.0, 2.0, 0.01, 0.15, 0.04),  # deep in the money call, dividends
        (50.0, 55.0, 0.5, -0.005, 0.60, 0.00),  # negative rate, high vol
    ],
)
def test_matches_numerical_risk_neutral_expectation(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float,
    option_type: str,
) -> None:
    closed_form = bs.price(spot, strike, maturity, rate, vol, div_yield, option_type)
    integrated = risk_neutral_expectation(spot, strike, maturity, rate, vol, div_yield, option_type)
    assert closed_form == pytest.approx(integrated, rel=1e-9, abs=1e-10)


def test_put_call_parity(random_params: dict[str, np.ndarray]) -> None:
    """C - P = S e^(-qT) - K e^(-rT), a model-free relation."""
    p = random_params
    call = bs.price(**p, option_type="call")
    put = bs.price(**p, option_type="put")
    parity = p["spot"] * np.exp(-p["div_yield"] * p["maturity"]) - p["strike"] * np.exp(
        -p["rate"] * p["maturity"]
    )
    np.testing.assert_allclose(call - put, parity, rtol=0.0, atol=1e-10)


def test_no_arbitrage_bounds(random_params: dict[str, np.ndarray]) -> None:
    """Model-free bounds on European option prices."""
    p = random_params
    prepaid_forward = p["spot"] * np.exp(-p["div_yield"] * p["maturity"])
    discounted_strike = p["strike"] * np.exp(-p["rate"] * p["maturity"])
    call = bs.price(**p, option_type="call")
    put = bs.price(**p, option_type="put")
    tolerance = 1e-12
    assert np.all(call >= np.maximum(prepaid_forward - discounted_strike, 0.0) - tolerance)
    assert np.all(call <= prepaid_forward + tolerance)
    assert np.all(put >= np.maximum(discounted_strike - prepaid_forward, 0.0) - tolerance)
    assert np.all(put <= discounted_strike + tolerance)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(
    random_params: dict[str, np.ndarray], option_type: str
) -> None:
    """Analytical Greeks agree with central differences of the price."""
    p = {name: values[:200] for name, values in random_params.items()}

    def priced(**overrides: np.ndarray) -> np.ndarray:
        return np.asarray(bs.price(**{**p, **overrides}, option_type=option_type))

    analytic = bs.greeks(**p, option_type=option_type)

    h_spot = 1e-4 * p["spot"]
    price_up, price_down = priced(spot=p["spot"] + h_spot), priced(spot=p["spot"] - h_spot)
    delta_fd = (price_up - price_down) / (2 * h_spot)
    gamma_fd = (price_up - 2 * priced() + price_down) / h_spot**2

    h = 1e-5
    vega_fd = (priced(vol=p["vol"] + h) - priced(vol=p["vol"] - h)) / (2 * h)
    rho_fd = (priced(rate=p["rate"] + h) - priced(rate=p["rate"] - h)) / (2 * h)
    # Theta is the derivative in calendar time, which shortens the time to expiry.
    theta_fd = -(priced(maturity=p["maturity"] + h) - priced(maturity=p["maturity"] - h)) / (2 * h)

    np.testing.assert_allclose(analytic.delta, delta_fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(analytic.gamma, gamma_fd, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(analytic.vega, vega_fd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(analytic.rho, rho_fd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(analytic.theta, theta_fd, rtol=1e-6, atol=1e-6)


def test_greek_signs(random_params: dict[str, np.ndarray]) -> None:
    p = random_params
    call = bs.greeks(**p, option_type="call")
    put = bs.greeks(**p, option_type="put")
    dividend_discount = np.exp(-p["div_yield"] * p["maturity"])
    assert np.all((call.delta >= 0.0) & (call.delta <= dividend_discount))
    assert np.all((put.delta <= 0.0) & (put.delta >= -dividend_discount))
    assert np.all(call.gamma > 0.0) and np.all(call.vega > 0.0)
    np.testing.assert_allclose(call.gamma, put.gamma, rtol=1e-12)
    np.testing.assert_allclose(call.vega, put.vega, rtol=1e-12)
    np.testing.assert_allclose(call.delta - put.delta, dividend_discount, atol=1e-12)


def test_vectorised_matches_scalar_loop() -> None:
    strikes = np.linspace(60.0, 140.0, 9)
    vectorised = bs.price(100.0, strikes, 1.0, 0.03, 0.25, 0.01, "put")
    looped = [bs.price(100.0, k, 1.0, 0.03, 0.25, 0.01, "put") for k in strikes]
    assert isinstance(vectorised, np.ndarray) and vectorised.shape == strikes.shape
    np.testing.assert_allclose(vectorised, looped, rtol=0.0, atol=1e-14)


def test_scalar_inputs_return_floats() -> None:
    assert isinstance(bs.price(100.0, 100.0, 1.0, 0.05, 0.2), float)
    result = bs.greeks(100.0, 100.0, 1.0, 0.05, 0.2)
    assert all(isinstance(value, float) for value in vars(result).values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"spot": 0.0},
        {"strike": -1.0},
        {"maturity": 0.0},
        {"vol": 0.0},
        {"vol": np.nan},
        {"rate": np.inf},
        {"strike": [100.0, -5.0]},
    ],
)
def test_invalid_inputs_raise(overrides: dict[str, object]) -> None:
    params = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "vol": 0.2}
    with pytest.raises(ValueError):
        bs.price(**{**params, **overrides})


def test_unknown_option_type_raises() -> None:
    with pytest.raises(ValueError, match="option_type"):
        bs.price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle")  # type: ignore[arg-type]
