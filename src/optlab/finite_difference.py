"""Crank–Nicolson finite-difference prices of European options.

The Black–Scholes PDE is solved after transforming it into the heat equation
u_tau = u_xx, the same equation as one-dimensional diffusion.

Transformation (Wilmott, Howison and Dewynne), with dividend yield q:

    S = K e^x,   tau = (vol^2 / 2) (T - t),   V(S, t) = K exp(alpha x + beta tau) u(x, tau)

    k     = r / (vol^2 / 2)
    k_q   = (r - q) / (vol^2 / 2)
    alpha = -(k_q - 1) / 2
    beta  = -(k_q - 1)^2 / 4 - k

x is log-moneyness, tau is time to expiry rescaled by the variance rate, and the
exponential factor absorbs the drift and discounting terms. The payoff becomes
the initial condition at tau = 0, and we march forward to tau = vol^2 T / 2.

Time stepping is Crank–Nicolson: second order in both dx and dtau and stable
for any step size. Its weakness is the kink in the payoff at the strike: for
large dtau / dx^2 the scheme damps high-frequency errors very slowly, so they
survive as oscillations in delta and gamma near the strike. Rannacher start-up
(Rannacher 1984; Giles and Carter 2006) replaces the first few steps with fully
implicit half-steps, which damp those components strongly and restore
second-order convergence.

With Rannacher start-up the error has a smooth expansion A h^2 + B h^4 + ...
in the grid spacing h, so Richardson extrapolation, (4 V(h/2) - V(h)) / 3,
cancels the h^2 term. Both grids must use the same *number* of Rannacher steps,
so that the start-up phase shrinks with the grid; a fixed start-up *duration*
leaves an h^2 term that extrapolation cannot remove.

Validated range: the exponential factor becomes extreme when |r - q| is large
relative to vol^2 / 2. Tests cover vol 0.10–0.60, rates -1% to 6%, dividend
yields 0–4%, maturities 0.1–2 years and strikes 70–130% of spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_solve_banded, cholesky_banded

from optlab._inputs import OptionType, check_option_type, prepare_scalars


@dataclass(frozen=True)
class PDEGrid:
    """Option values on the spatial grid at valuation time (t = 0)."""

    log_moneyness: NDArray[np.float64]  # x = ln(S / K) at each node
    spots: NDArray[np.float64]  # S = K e^x at each node
    values: NDArray[np.float64]  # option value V(S, 0) at each node
    strike: float
    diffusion_number: float  # lambda = dtau / dx^2, the step-size ratio that drives oscillations


@dataclass(frozen=True)
class PDEResult:
    """A finite-difference price and its grid Greeks, interpolated to the spot.

    ``space_steps``, ``time_steps`` and ``diffusion_number`` describe the
    coarser grid; an extrapolated result also used a grid twice as fine in
    both directions.
    """

    price: float
    delta: float
    gamma: float
    space_steps: int
    time_steps: int
    diffusion_number: float
    extrapolated: bool


@dataclass(frozen=True)
class _Transform:
    alpha: float
    beta: float
    half_var: float  # vol^2 / 2: converts tau back to calendar time


def _transform(rate: float, vol: float, div_yield: float) -> _Transform:
    half_var = 0.5 * vol**2
    k = rate / half_var
    k_q = (rate - div_yield) / half_var
    alpha = -0.5 * (k_q - 1.0)
    beta = -0.25 * (k_q - 1.0) ** 2 - k
    return _Transform(alpha=alpha, beta=beta, half_var=half_var)


def _payoff_over_strike(x: NDArray[np.float64], option_type: OptionType) -> NDArray[np.float64]:
    if option_type == "call":
        return np.maximum(np.exp(x) - 1.0, 0.0)
    return np.maximum(1.0 - np.exp(x), 0.0)


def _boundary_u(
    x: float,
    tau: float,
    rate: float,
    div_yield: float,
    option_type: OptionType,
    upper: bool,
    tr: _Transform,
) -> float:
    """Heat-equation value at a domain edge, from the option's far-field asymptotics.

    Deep in the money, a European option is worth its forward intrinsic value,
    S e^(-q t) - K e^(-r t) for a call; deep out of the money it is worth zero.
    """
    remaining = tau / tr.half_var  # calendar time to expiry
    deep_itm = (option_type == "call") == upper
    if not deep_itm:
        value_over_strike = 0.0
    elif option_type == "call":
        value_over_strike = np.exp(x - div_yield * remaining) - np.exp(-rate * remaining)
    else:
        value_over_strike = np.exp(-rate * remaining) - np.exp(x - div_yield * remaining)
    return float(value_over_strike * np.exp(-tr.alpha * x - tr.beta * tau))


def _banded_cholesky(diagonal: float, off_diagonal: float, size: int) -> NDArray[np.float64]:
    """Cholesky factor of a constant symmetric tridiagonal matrix, in banded storage."""
    bands = np.empty((2, size))
    bands[0, 0] = 0.0
    bands[0, 1:] = off_diagonal
    bands[1, :] = diagonal
    return cholesky_banded(bands, lower=False)


def _validate_grid(space_steps: int, time_steps: int, rannacher_steps: int, n_std: float) -> None:
    if space_steps < 4 or space_steps % 2:
        raise ValueError(f"space_steps must be an even integer >= 4 so the strike is a grid node, got {space_steps}")
    if time_steps < 1:
        raise ValueError(f"time_steps must be >= 1, got {time_steps}")
    if not 0 <= rannacher_steps <= time_steps:
        raise ValueError(f"rannacher_steps must be between 0 and time_steps, got {rannacher_steps}")
    if n_std <= 0:
        raise ValueError(f"n_std must be positive, got {n_std}")


def solve_grid(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    space_steps: int = 400,
    time_steps: int = 100,
    rannacher_steps: int = 2,
    n_std: float = 5.0,
) -> PDEGrid:
    """Solve for option values on the whole grid at valuation time.

    Args:
        spot, strike, maturity, rate, vol, div_yield, option_type: as in
            :func:`optlab.black_scholes.price`, scalars only.
        space_steps: Number of intervals in x. Even, so the strike (x = 0) is a node.
        time_steps: Number of steps in tau.
        rannacher_steps: How many initial steps are replaced by two fully
            implicit half-steps each. 0 gives plain Crank–Nicolson.
        n_std: Domain half-width beyond |ln(S/K)|, in standard deviations of
            the log price over the option's life (vol sqrt(T)).

    Returns:
        A :class:`PDEGrid` with values at every node.

    Raises:
        FloatingPointError: if the solution is not finite, which happens far
            outside the validated parameter range in the module docstring.
    """
    check_option_type(option_type)
    _validate_grid(space_steps, time_steps, rannacher_steps, n_std)
    s, k, t, r, sigma, q = prepare_scalars(spot, strike, maturity, rate, vol, div_yield)
    tr = _transform(r, sigma, q)

    half_width = abs(np.log(s / k)) + n_std * sigma * np.sqrt(t)
    dx = 2.0 * half_width / space_steps
    x = dx * (np.arange(space_steps + 1) - space_steps // 2)  # x = 0 exactly at the middle node
    tau_max = tr.half_var * t
    dtau = tau_max / time_steps
    lam = dtau / dx**2

    def edges(tau: float) -> tuple[float, float]:
        return (
            _boundary_u(x[0], tau, r, q, option_type, upper=False, tr=tr),
            _boundary_u(x[-1], tau, r, q, option_type, upper=True, tr=tr),
        )

    interior = space_steps - 1
    # Crank–Nicolson: (1 + lam) u_j - lam/2 (u_{j-1} + u_{j+1}) at the new level equals
    # (1 - lam) u_j + lam/2 (u_{j-1} + u_{j+1}) at the old level.
    cn_factor = _banded_cholesky(1.0 + lam, -0.5 * lam, interior)
    # Implicit Euler half-step with dtau/2: (1 + 2 mu) u_j - mu (u_{j-1} + u_{j+1}) = old u_j.
    mu = 0.5 * lam
    implicit_factor = _banded_cholesky(1.0 + 2.0 * mu, -mu, interior)

    u = np.exp(-tr.alpha * x) * _payoff_over_strike(x, option_type)
    tau = 0.0
    for step in range(time_steps):
        if step < rannacher_steps:
            for _ in range(2):
                tau += 0.5 * dtau
                low, high = edges(tau)
                rhs = u[1:-1].copy()
                rhs[0] += mu * low
                rhs[-1] += mu * high
                u[1:-1] = cho_solve_banded((implicit_factor, False), rhs)
                u[0], u[-1] = low, high
        else:
            # The old boundary values sit in u[0] and u[-1], so the explicit half
            # of the stencil picks them up through u[:-2] and u[2:].
            rhs = (1.0 - lam) * u[1:-1] + 0.5 * lam * (u[:-2] + u[2:])
            tau += dtau
            low, high = edges(tau)
            rhs[0] += 0.5 * lam * low
            rhs[-1] += 0.5 * lam * high
            u[1:-1] = cho_solve_banded((cn_factor, False), rhs)
            u[0], u[-1] = low, high

    values = k * np.exp(tr.alpha * x + tr.beta * tau_max) * u
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "non-finite PDE solution: parameters are outside the range where the heat-equation transform is well conditioned"
        )
    return PDEGrid(log_moneyness=x, spots=k * np.exp(x), values=values, strike=k, diffusion_number=lam)


def price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    div_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    space_steps: int = 400,
    time_steps: int = 100,
    rannacher_steps: int = 2,
    n_std: float = 5.0,
    extrapolate: bool = True,
) -> PDEResult:
    """Crank–Nicolson price, delta and gamma at the spot.

    Arguments are the same as :func:`solve_grid`, plus ``extrapolate``: when
    True (the default), the problem is also solved on a grid twice as fine in
    space and time, and the two results are combined by Richardson
    extrapolation, (4 fine - coarse) / 3. This costs about five coarse solves
    and raises the convergence order from 2 to about 4.

    The grid solution is interpolated to ln(S/K) with a cubic spline, whose
    error is O(dx^4). Delta and gamma come from the spline's derivatives,
    converted from x to S:

        dV/dS = V_x / S,    d2V/dS2 = (V_xx - V_x) / S^2
    """
    grid_args = (spot, strike, maturity, rate, vol, div_yield, option_type)
    coarse = _spot_greeks(
        solve_grid(*grid_args, space_steps=space_steps, time_steps=time_steps, rannacher_steps=rannacher_steps, n_std=n_std),
        spot,
    )
    if extrapolate:
        fine = _spot_greeks(
            solve_grid(
                *grid_args,
                space_steps=2 * space_steps,
                time_steps=2 * time_steps,
                rannacher_steps=rannacher_steps,
                n_std=n_std,
            ),
            spot,
        )
        estimates = tuple((4.0 * f - c) / 3.0 for f, c in zip(fine[:3], coarse[:3]))
    else:
        estimates = coarse[:3]
    value, delta, gamma = estimates
    return PDEResult(
        price=value,
        delta=delta,
        gamma=gamma,
        space_steps=space_steps,
        time_steps=time_steps,
        diffusion_number=coarse[3],
        extrapolated=extrapolate,
    )


def _spot_greeks(grid: PDEGrid, spot: float) -> tuple[float, float, float, float]:
    """Price, delta and gamma at the spot by cubic-spline interpolation, plus the grid's lambda."""
    spline = CubicSpline(grid.log_moneyness, grid.values)
    x0 = float(np.log(spot / grid.strike))
    value, v_x, v_xx = (float(spline(x0, order)) for order in (0, 1, 2))
    return value, v_x / spot, (v_xx - v_x) / spot**2, grid.diffusion_number


def grid_greeks(grid: PDEGrid) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Delta and gamma at interior nodes by central differences.

    Returns:
        (spots, delta, gamma) at the interior nodes. Used to plot gamma across
        spot levels, where scheme oscillations near the strike are visible.
    """
    x, v = grid.log_moneyness, grid.values
    dx = x[1] - x[0]
    v_x = (v[2:] - v[:-2]) / (2.0 * dx)
    v_xx = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / dx**2
    spots = grid.spots[1:-1]
    return spots, v_x / spots, (v_xx - v_x) / spots**2
