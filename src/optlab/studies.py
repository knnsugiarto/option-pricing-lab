"""Numerical studies: convergence, Greeks near the strike, and cross-method accuracy.

Every function is deterministic given its seed, returns plain data, and does
no plotting or file I/O, so tests can run each one on a small configuration.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optlab import black_scholes as bs
from optlab import finite_difference as fd
from optlab import monte_carlo as mc


@dataclass(frozen=True)
class Contract:
    """One European option."""

    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    div_yield: float
    option_type: str

    def kwargs(self) -> dict[str, Any]:
        return asdict(self)


def fitted_order(steps: NDArray[np.float64] | list[float], errors: NDArray[np.float64] | list[float]) -> float:
    """Convergence order p from a least-squares fit of log(error) = c - p log(steps)."""
    slope = np.polyfit(np.log(np.asarray(steps, dtype=float)), np.log(np.asarray(errors, dtype=float)), 1)[0]
    return float(-slope)


# --- Monte Carlo ------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloRow:
    estimator: str
    n_paths: int
    rmse: float  # root-mean-square error vs the closed form, across replications
    mean_std_error: float  # average reported standard error
    coverage: float  # share of runs whose confidence interval contains the closed form
    seconds_per_run: float


@dataclass(frozen=True)
class MonteCarloStudy:
    exact_price: float
    confidence_level: float
    replications: int
    rows: list[MonteCarloRow]
    orders: dict[str, float]  # fitted convergence order of the RMSE in n_paths
    variance_reduction: dict[str, float]  # plain variance / estimator variance at the largest path count
    pooled_coverage: dict[str, float]


def monte_carlo_convergence(
    contract: Contract,
    path_counts: list[int],
    replications: int,
    seed: int,
    confidence_level: float,
) -> MonteCarloStudy:
    """Measure how each Monte Carlo estimator's error shrinks with the path count.

    For each estimator and path count, the pricer is run ``replications`` times
    with independent seeds. The actual error (RMSE against the closed form) is
    compared with the error the pricer reports (its standard error).
    """
    exact = float(bs.price(**contract.kwargs()))
    seeds = iter(np.random.SeedSequence(seed).spawn(len(mc.ESTIMATORS) * len(path_counts) * replications))
    rows = []
    for estimator in mc.ESTIMATORS:
        for n_paths in path_counts:
            errors, std_errors, covered = [], [], 0
            start = time.perf_counter()
            for _ in range(replications):
                result = mc.price(**contract.kwargs(), n_paths=n_paths, seed=np.random.default_rng(next(seeds)), estimator=estimator)
                low, high = result.confidence_interval(confidence_level)
                errors.append(result.price - exact)
                std_errors.append(result.std_error)
                covered += low <= exact <= high
            elapsed = (time.perf_counter() - start) / replications
            rows.append(
                MonteCarloRow(
                    estimator=estimator,
                    n_paths=n_paths,
                    rmse=float(np.sqrt(np.mean(np.square(errors)))),
                    mean_std_error=float(np.mean(std_errors)),
                    coverage=covered / replications,
                    seconds_per_run=elapsed,
                )
            )

    def of(estimator: str) -> list[MonteCarloRow]:
        return [row for row in rows if row.estimator == estimator]

    largest = max(path_counts)
    plain_variance = next(r.mean_std_error for r in of("plain") if r.n_paths == largest) ** 2
    return MonteCarloStudy(
        exact_price=exact,
        confidence_level=confidence_level,
        replications=replications,
        rows=rows,
        orders={e: fitted_order([r.n_paths for r in of(e)], [r.rmse for r in of(e)]) for e in mc.ESTIMATORS},
        variance_reduction={
            e: plain_variance / next(r.mean_std_error for r in of(e) if r.n_paths == largest) ** 2 for e in mc.ESTIMATORS
        },
        pooled_coverage={e: float(np.mean([r.coverage for r in of(e)])) for e in mc.ESTIMATORS},
    )


# --- Crank–Nicolson -----------------------------------------------------------------

PDE_METHODS = {
    "Crank–Nicolson": {"rannacher": False, "extrapolate": False},
    "CN + Rannacher": {"rannacher": True, "extrapolate": False},
    "CN + Rannacher + Richardson": {"rannacher": True, "extrapolate": True},
}


@dataclass(frozen=True)
class PDERow:
    method: str
    space_steps: int
    time_steps: int
    diffusion_number: float
    price_error: float
    delta_error: float
    gamma_error: float
    seconds: float


@dataclass(frozen=True)
class PDEStudy:
    exact_price: float
    exact_gamma: float
    rows: list[PDERow]
    fit_from: int
    price_orders: dict[str, float]
    gamma_orders: dict[str, float]


def pde_convergence(
    contract: Contract,
    space_steps: list[int],
    time_steps_per_space_step: float,
    rannacher_steps: int,
    n_std: float,
    fit_from: int,
) -> PDEStudy:
    """Errors in price, delta and gamma at the spot as the grid is refined.

    The time grid is refined in proportion to the space grid, so dtau/dx stays
    fixed and the diffusion number dtau/dx^2 grows with refinement: the regime
    where plain Crank–Nicolson struggles.
    """
    exact_price = float(bs.price(**contract.kwargs()))
    exact = bs.greeks(**contract.kwargs())
    rows = []
    for method, flags in PDE_METHODS.items():
        for n in space_steps:
            m = max(1, round(n * time_steps_per_space_step))
            start = time.perf_counter()
            result = fd.price(
                **contract.kwargs(),
                space_steps=n,
                time_steps=m,
                rannacher_steps=rannacher_steps if flags["rannacher"] else 0,
                n_std=n_std,
                extrapolate=flags["extrapolate"],
            )
            rows.append(
                PDERow(
                    method=method,
                    space_steps=n,
                    time_steps=m,
                    diffusion_number=result.diffusion_number,
                    price_error=abs(result.price - exact_price),
                    delta_error=abs(result.delta - float(exact.delta)),
                    gamma_error=abs(result.gamma - float(exact.gamma)),
                    seconds=time.perf_counter() - start,
                )
            )

    def fit(method: str, attribute: str) -> float:
        selected = [r for r in rows if r.method == method and r.space_steps >= fit_from]
        return fitted_order([r.space_steps for r in selected], [getattr(r, attribute) for r in selected])

    return PDEStudy(
        exact_price=exact_price,
        exact_gamma=float(exact.gamma),
        rows=rows,
        fit_from=fit_from,
        price_orders={m: fit(m, "price_error") for m in PDE_METHODS},
        gamma_orders={m: fit(m, "gamma_error") for m in PDE_METHODS},
    )


@dataclass(frozen=True)
class GammaProfile:
    spots: NDArray[np.float64]
    exact: NDArray[np.float64]
    plain: NDArray[np.float64]
    rannacher: NDArray[np.float64]
    space_steps: int
    time_steps: int
    diffusion_number: float


def gamma_profile(
    contract: Contract,
    space_steps: int,
    time_steps: int,
    rannacher_steps: int,
    n_std: float,
    spot_min: float,
    spot_max: float,
) -> GammaProfile:
    """Grid gamma across spot levels, with and without Rannacher start-up, on one grid."""
    grid_kwargs = {"space_steps": space_steps, "time_steps": time_steps, "n_std": n_std}
    args = contract.kwargs()
    plain_grid = fd.solve_grid(**args, **grid_kwargs, rannacher_steps=0)
    spots, _, plain = fd.grid_greeks(plain_grid)
    _, _, rannacher = fd.grid_greeks(fd.solve_grid(**args, **grid_kwargs, rannacher_steps=rannacher_steps))
    keep = (spots >= spot_min) & (spots <= spot_max)
    exact = bs.greeks(spots[keep], args["strike"], args["maturity"], args["rate"], args["vol"], args["div_yield"], args["option_type"])
    return GammaProfile(
        spots=spots[keep],
        exact=np.asarray(exact.gamma),
        plain=plain[keep],
        rannacher=rannacher[keep],
        space_steps=space_steps,
        time_steps=time_steps,
        diffusion_number=plain_grid.diffusion_number,
    )


# --- Cross-method accuracy ----------------------------------------------------------


@dataclass(frozen=True)
class AccuracyStudy:
    n_cases: int  # per option type
    pde_max_error: float
    pde_median_error: float
    pde_max_relative_error: float
    pde_ms_per_price: float
    mc_paths: int
    mc_coverage: float  # share of Monte Carlo 99% intervals containing the closed form
    mc_max_abs_z: float  # largest |MC - exact| / standard error, over non-degenerate runs
    # Runs whose standard error is ~0 because no path reached the curved part of the
    # payoff (all paths out of the money, or all in the money with the control variate).
    mc_degenerate: int
    mc_degenerate_max_error: float
    mc_ms_per_price: float
    bs_microseconds_per_price: float


def random_contracts(n_cases: int, seed: int, ranges: dict[str, list[float]]) -> list[Contract]:
    """Seeded random contracts with spot 100 and parameters drawn uniformly from ``ranges``."""
    rng = np.random.default_rng(seed)
    contracts = []
    for option_type in ("call", "put"):
        for _ in range(n_cases):
            contracts.append(
                Contract(
                    spot=100.0,
                    strike=float(rng.uniform(*ranges["strike_range"])),
                    maturity=float(rng.uniform(*ranges["maturity_range"])),
                    rate=float(rng.uniform(*ranges["rate_range"])),
                    vol=float(rng.uniform(*ranges["vol_range"])),
                    div_yield=float(rng.uniform(*ranges["div_yield_range"])),
                    option_type=option_type,
                )
            )
    return contracts


def cross_method_accuracy(n_cases: int, seed: int, mc_paths: int, ranges: dict[str, list[float]], confidence_level: float) -> AccuracyStudy:
    """Compare the PDE and Monte Carlo pricers with the closed form on random contracts."""
    contracts = random_contracts(n_cases, seed, ranges)
    exact = np.array([float(bs.price(**c.kwargs())) for c in contracts])

    start = time.perf_counter()
    for c in contracts:
        bs.price(**c.kwargs())
    bs_seconds = (time.perf_counter() - start) / len(contracts)

    start = time.perf_counter()
    pde = np.array([fd.price(**c.kwargs()).price for c in contracts])
    pde_seconds = (time.perf_counter() - start) / len(contracts)

    seeds = np.random.SeedSequence(seed + 1).spawn(len(contracts))
    start = time.perf_counter()
    mc_results = [mc.price(**c.kwargs(), n_paths=mc_paths, seed=np.random.default_rng(s)) for c, s in zip(contracts, seeds)]
    mc_seconds = (time.perf_counter() - start) / len(contracts)
    intervals = [r.confidence_interval(confidence_level) for r in mc_results]
    mc_prices = np.array([r.price for r in mc_results])
    std_errors = np.array([r.std_error for r in mc_results])
    degenerate = std_errors <= 1e-12 * np.maximum(1.0, np.abs(mc_prices))
    z_scores = (mc_prices[~degenerate] - exact[~degenerate]) / std_errors[~degenerate]

    pde_errors = np.abs(pde - exact)
    return AccuracyStudy(
        n_cases=n_cases,
        pde_max_error=float(pde_errors.max()),
        pde_median_error=float(np.median(pde_errors)),
        pde_max_relative_error=float(np.max(pde_errors / np.maximum(exact, 1e-2))),
        pde_ms_per_price=1e3 * pde_seconds,
        mc_paths=mc_paths,
        mc_coverage=float(np.mean([low <= e <= high for (low, high), e in zip(intervals, exact)])),
        mc_max_abs_z=float(np.max(np.abs(z_scores))),
        mc_degenerate=int(degenerate.sum()),
        mc_degenerate_max_error=float(np.max(np.abs(mc_prices - exact)[degenerate], initial=0.0)),
        mc_ms_per_price=1e3 * mc_seconds,
        bs_microseconds_per_price=1e6 * bs_seconds,
    )
