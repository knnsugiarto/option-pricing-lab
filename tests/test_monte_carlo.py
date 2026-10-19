"""Tests for the Monte Carlo pricer.

Monte Carlo output is random, so the tests check statistical properties with
fixed seeds: the reported standard error must be honest (confidence intervals
cover the true price at their stated rate), variance reduction must reduce
variance, and exact identities must hold exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from optlab import black_scholes as bs
from optlab import monte_carlo as mc

SEED = 20261013
CONTRACT = {"spot": 100.0, "strike": 105.0, "maturity": 0.75, "rate": 0.04, "vol": 0.25, "div_yield": 0.01}


@pytest.mark.parametrize("estimator", mc.ESTIMATORS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_confidence_interval_coverage(estimator: mc.Estimator, option_type: str) -> None:
    """A 99% interval should contain the closed-form price in about 99% of runs.

    With 1,000 independent runs, the number of misses is Binomial(1000, 0.01):
    mean 10. Fewer than 3 misses (interval too wide, standard error overstated)
    or more than 25 (too narrow, understated) both mean the error bars are wrong.
    Each bound fails by chance with probability below 0.3%.
    """
    exact = bs.price(**CONTRACT, option_type=option_type)
    seeds = np.random.SeedSequence(SEED).spawn(1_000)
    misses = 0
    for child in seeds:
        result = mc.price(**CONTRACT, option_type=option_type, n_paths=10_000, seed=np.random.default_rng(child), estimator=estimator)
        low, high = result.confidence_interval(0.99)
        misses += not (low <= exact <= high)
    assert 3 <= misses <= 25


def test_variance_reduction_reduces_standard_error() -> None:
    """At an equal path count, every variance-reduced estimator beats plain Monte Carlo."""
    errors = {
        estimator: mc.price(**CONTRACT, n_paths=200_000, seed=SEED, estimator=estimator).std_error
        for estimator in mc.ESTIMATORS
    }
    assert errors["antithetic"] < errors["plain"]
    assert errors["control"] < errors["plain"]
    assert errors["antithetic_control"] < min(errors["antithetic"], errors["control"])


def test_error_shrinks_like_inverse_square_root_of_paths() -> None:
    small = mc.price(**CONTRACT, n_paths=40_000, seed=SEED, estimator="plain").std_error
    large = mc.price(**CONTRACT, n_paths=640_000, seed=SEED, estimator="plain").std_error
    assert small / large == pytest.approx(4.0, rel=0.05)


def test_control_variate_satisfies_put_call_parity_exactly() -> None:
    """With the same draws, call minus put equals the parity forward to rounding error.

    The call and put payoffs differ by e^(-rT) (S_T - K), which is linear in the
    control variate, so the fitted coefficients differ by exactly 1 and all the
    sampling noise cancels. A mistake in the beta estimate breaks this.
    """
    call = mc.price(**CONTRACT, option_type="call", n_paths=50_000, seed=SEED, estimator="control")
    put = mc.price(**CONTRACT, option_type="put", n_paths=50_000, seed=SEED, estimator="control")
    c = CONTRACT
    parity = c["spot"] * np.exp(-c["div_yield"] * c["maturity"]) - c["strike"] * np.exp(-c["rate"] * c["maturity"])
    assert call.price - put.price == pytest.approx(parity, abs=1e-10)


def test_same_seed_same_result() -> None:
    first = mc.price(**CONTRACT, n_paths=10_000, seed=7)
    second = mc.price(**CONTRACT, n_paths=10_000, seed=7)
    assert first == second


def test_confidence_interval_is_symmetric_and_widens_with_level() -> None:
    result = mc.MonteCarloResult(price=10.0, std_error=0.1, n_paths=100, estimator="plain")
    low95, high95 = result.confidence_interval(0.95)
    low99, high99 = result.confidence_interval(0.99)
    assert high95 - 10.0 == pytest.approx(10.0 - low95)
    assert high95 - 10.0 == pytest.approx(0.1 * 1.959964, rel=1e-6)
    assert low99 < low95 and high99 > high95


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_paths": 3},
        {"n_paths": 1_001, "estimator": "antithetic"},
        {"estimator": "quasi"},
        {"vol": -0.2},
        {"option_type": "straddle"},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, object]) -> None:
    params: dict[str, object] = {**CONTRACT, "n_paths": 1_000, "seed": SEED}
    with pytest.raises(ValueError):
        mc.price(**{**params, **kwargs})  # type: ignore[arg-type]
