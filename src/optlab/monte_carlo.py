"""Monte Carlo prices of European options under Black–Scholes dynamics.

A European payoff depends only on the terminal price, and under geometric
Brownian motion the terminal price has an exact closed form:

    S_T = S_0 exp((r - q - vol^2 / 2) T + vol sqrt(T) Z),    Z ~ N(0, 1)

so no time stepping is needed and there is no discretisation bias. The only
error is statistical, and its standard deviation shrinks like 1 / sqrt(n_paths).

Four estimators, all unbiased (the control variate up to an O(1/n) term from
estimating its coefficient on the same sample):

- ``plain``: the average discounted payoff.
- ``antithetic``: pairs each draw Z with -Z and averages the pair.
- ``control``: subtracts the error in a quantity whose mean is known exactly,
  the discounted terminal price, E[e^(-rT) S_T] = S_0 e^(-qT).
- ``antithetic_control``: both at once.

Memory grows linearly with ``n_paths``: about 50 bytes per path at peak.

Known limitation: the standard error only measures variation the sample saw.
If no path finishes in the money (a far out-of-the-money option), the estimate
is 0 with standard error 0, although the true price is small but positive.
The control variate has the mirror-image case: if every path finishes in the
money, the payoff is linear in the control and the error collapses to rounding.
Importance sampling is the standard fix; it is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from optlab._inputs import OptionType, check_option_type, prepare_scalars

Estimator = Literal["plain", "antithetic", "control", "antithetic_control"]
ESTIMATORS: tuple[Estimator, ...] = ("plain", "antithetic", "control", "antithetic_control")


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo price with its standard error."""

    price: float
    std_error: float
    n_paths: int
    estimator: Estimator

    def confidence_interval(self, level: float = 0.99) -> tuple[float, float]:
        """Two-sided normal confidence interval for the true price."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        half_width = float(ndtri(0.5 + level / 2.0)) * self.std_error
        return self.price - half_width, self.price + half_width


def terminal_prices(
    spot: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float,
    normals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact risk-neutral terminal prices for given standard normal draws."""
    drift = (rate - div_yield - 0.5 * vol**2) * maturity
    return spot * np.exp(drift + vol * np.sqrt(maturity) * normals)


def _discounted_payoffs(
    terminal: NDArray[np.float64],
    strike: float,
    discount: float,
    option_type: OptionType,
) -> NDArray[np.float64]:
    if option_type == "call":
        return discount * np.maximum(terminal - strike, 0.0)
    return discount * np.maximum(strike - terminal, 0.0)


def _mean_and_error(samples: NDArray[np.float64]) -> tuple[float, float]:
    """Sample mean and its standard error from independent samples."""
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size))


def _control_variate(
    targets: NDArray[np.float64],
    controls: NDArray[np.float64],
    control_mean: float,
) -> tuple[float, float]:
    """Control variate estimate and standard error.

    The coefficient beta = Cov(Y, X) / Var(X) minimises the variance of
    Y - beta (X - E[X]). It is estimated from the same sample, which makes the
    estimator biased by O(1/n): negligible next to the O(1/sqrt(n)) error.
    """
    centred_controls = controls - controls.mean()
    beta = float(np.dot(targets - targets.mean(), centred_controls) / np.dot(centred_controls, centred_controls))
    adjusted = targets - beta * (controls - control_mean)
    # Two parameters (the mean and beta) are fitted, hence ddof=2.
    std_error = float(adjusted.std(ddof=2) / np.sqrt(adjusted.size))
    return float(adjusted.mean()), std_error


def price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    n_paths: int,
    seed: int | np.random.Generator,
    estimator: Estimator = "antithetic_control",
) -> MonteCarloResult:
    """Monte Carlo price of a European call or put.

    Args:
        spot, strike, maturity, rate, vol, div_yield, option_type: as in
            :func:`optlab.black_scholes.price`, but scalars only.
        n_paths: Number of payoff evaluations. Antithetic estimators use
            n_paths / 2 independent draws, so n_paths must be even for them.
        seed: An integer seed or a NumPy Generator. Same seed, same result.
        estimator: One of "plain", "antithetic", "control", "antithetic_control".

    Returns:
        The price estimate, its standard error, the path count and the estimator.
    """
    check_option_type(option_type)
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    antithetic = estimator in ("antithetic", "antithetic_control")
    if n_paths < 4 or (antithetic and n_paths % 2):
        raise ValueError(f"n_paths must be at least 4, and even for antithetic estimators; got {n_paths}")
    s, k, t, r, sigma, q = prepare_scalars(spot, strike, maturity, rate, vol, div_yield)

    rng = np.random.default_rng(seed)
    discount = np.exp(-r * t)
    control_mean = s * np.exp(-q * t)  # E[e^(-rT) S_T] under the risk-neutral measure

    normals = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
    if antithetic:
        up = terminal_prices(s, t, r, sigma, q, normals)
        down = terminal_prices(s, t, r, sigma, q, -normals)
        # Each pair is one independent sample, so errors are computed on pair means.
        payoffs = 0.5 * (_discounted_payoffs(up, k, discount, option_type) + _discounted_payoffs(down, k, discount, option_type))
        controls = 0.5 * discount * (up + down)
    else:
        terminal = terminal_prices(s, t, r, sigma, q, normals)
        payoffs = _discounted_payoffs(terminal, k, discount, option_type)
        controls = discount * terminal

    if estimator in ("control", "antithetic_control"):
        estimate, std_error = _control_variate(payoffs, controls, control_mean)
    else:
        estimate, std_error = _mean_and_error(payoffs)
    return MonteCarloResult(price=estimate, std_error=std_error, n_paths=n_paths, estimator=estimator)
