"""Price a European option three ways and compare them with the closed form.

Example:
    python scripts/price.py --spot 42 --strike 40 --maturity 0.5 --rate 0.10 --vol 0.20
    python scripts/price.py --spot 100 --strike 110 --maturity 1 --rate 0.04 --vol 0.25 --div-yield 0.01 --type put
"""

from __future__ import annotations

import argparse
import time

from optlab import black_scholes as bs
from optlab import finite_difference as fd
from optlab import monte_carlo as mc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spot", type=float, required=True, help="underlying price")
    parser.add_argument("--strike", type=float, required=True, help="strike price")
    parser.add_argument("--maturity", type=float, required=True, help="time to expiry in years")
    parser.add_argument("--rate", type=float, required=True, help="risk-free rate, e.g. 0.05")
    parser.add_argument("--vol", type=float, required=True, help="annual volatility, e.g. 0.20")
    parser.add_argument("--div-yield", type=float, default=0.0, help="dividend yield, e.g. 0.01")
    parser.add_argument("--type", choices=("call", "put"), default="call", dest="option_type")
    parser.add_argument("--paths", type=int, default=1_000_000, help="Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=20261013, help="Monte Carlo seed")
    return parser.parse_args()


def timed(function, *args, **kwargs):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, 1e3 * (time.perf_counter() - start)


def main() -> None:
    a = parse_args()
    inputs = {"spot": a.spot, "strike": a.strike, "maturity": a.maturity, "rate": a.rate, "vol": a.vol, "div_yield": a.div_yield, "option_type": a.option_type}

    exact, exact_ms = timed(bs.price, **inputs)
    greeks = bs.greeks(**inputs)
    sim, sim_ms = timed(mc.price, **inputs, n_paths=a.paths, seed=a.seed)
    pde, pde_ms = timed(fd.price, **inputs)
    low, high = sim.confidence_interval(0.99)

    print(f"European {a.option_type}: S={a.spot:g}  K={a.strike:g}  T={a.maturity:g}y  r={a.rate:g}  q={a.div_yield:g}  vol={a.vol:g}\n")
    print(f"{'Method':<34}{'Price':>12}{'Error':>12}{'Time':>10}")
    print(f"{'Closed form (Black-Scholes)':<34}{exact:>12.6f}{'':>12}{exact_ms:>8.2f}ms")
    print(f"{'Monte Carlo (' + f'{a.paths:,} paths)':<34}{sim.price:>12.6f}{sim.price - exact:>12.2e}{sim_ms:>8.2f}ms")
    print(f"{'PDE (Crank-Nicolson, extrapolated)':<34}{pde.price:>12.6f}{pde.price - exact:>12.2e}{pde_ms:>8.2f}ms")
    inside = "contains" if low <= exact <= high else "MISSES"
    print(f"\nMonte Carlo 99% interval [{low:.6f}, {high:.6f}] {inside} the closed form (standard error {sim.std_error:.2e})")

    print(f"\n{'Greek':<20}{'Closed form':>14}{'PDE':>14}")
    print(f"{'delta':<20}{greeks.delta:>14.6f}{pde.delta:>14.6f}")
    print(f"{'gamma':<20}{greeks.gamma:>14.6f}{pde.gamma:>14.6f}")
    print(f"{'vega (per vol pt)':<20}{greeks.vega / 100:>14.6f}{'':>14}")
    print(f"{'theta (per day)':<20}{greeks.theta / 365:>14.6f}{'':>14}")
    print(f"{'rho (per 1% rate)':<20}{greeks.rho / 100:>14.6f}{'':>14}")


if __name__ == "__main__":
    main()
