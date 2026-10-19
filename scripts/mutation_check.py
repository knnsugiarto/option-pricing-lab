"""Test the tests: inject deliberate bugs and check that the suite catches each one.

Each mutation is applied to a temporary copy of the repository, never to your
files. A mutation "survives" if the tests still pass with the bug in place,
which would mean the suite cannot detect that mistake.

Example:
    python scripts/mutation_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    description: str
    module: str  # file under src/optlab/
    original: str
    mutated: str
    tests: str  # test file that should catch it


MUTATIONS = [
    Mutation("d1 adds the dividend yield instead of subtracting it", "black_scholes.py", "(rate - div_yield + 0.5 * vol**2)", "(rate + div_yield + 0.5 * vol**2)", "test_black_scholes.py"),
    Mutation("call rho is missing its factor T", "black_scholes.py", "rho = t * discounted_strike * ndtr(d2)", "rho = discounted_strike * ndtr(d2)", "test_black_scholes.py"),
    Mutation("gamma divides by S instead of S squared", "black_scholes.py", "(s * s * sigma * sqrt_t)", "(s * sigma * sqrt_t)", "test_black_scholes.py"),
    Mutation("Monte Carlo drift is missing the Ito correction", "monte_carlo.py", "(rate - div_yield - 0.5 * vol**2) * maturity", "(rate - div_yield) * maturity", "test_monte_carlo.py"),
    Mutation("control variate mean ignores dividends", "monte_carlo.py", "control_mean = s * np.exp(-q * t)", "control_mean = s", "test_monte_carlo.py"),
    Mutation("standard error understated by a factor of sqrt(2)", "monte_carlo.py", "/ np.sqrt(samples.size)", "/ np.sqrt(2 * samples.size)", "test_monte_carlo.py"),
    Mutation("control variate reports the unreduced error", "monte_carlo.py", "adjusted.std(ddof=2)", "targets.std(ddof=2)", "test_monte_carlo.py"),
    Mutation("PDE discounting uses r - q instead of r", "finite_difference.py", "beta = -0.25 * (k_q - 1.0) ** 2 - k", "beta = -0.25 * (k_q - 1.0) ** 2 - k_q", "test_finite_difference.py"),
    Mutation("deep in-the-money call boundary ignores dividends", "finite_difference.py", "np.exp(x - div_yield * remaining) - np.exp(-rate * remaining)", "np.exp(x) - np.exp(-rate * remaining)", "test_finite_difference.py"),
    Mutation("Crank-Nicolson boundary term weighted lambda, not lambda/2", "finite_difference.py", "rhs[-1] += 0.5 * lam * high", "rhs[-1] += lam * high", "test_finite_difference.py"),
    Mutation("Richardson extrapolation uses first-order weights", "finite_difference.py", "(4.0 * f - c) / 3.0", "(2.0 * f - c) / 1.0", "test_finite_difference.py"),
    Mutation("implied vol accepts prices outside the no-arbitrage bounds", "implied_vol.py", "if low > 0.0 or high < 0.0:", "if low > 0.0 and high < 0.0:", "test_implied_vol.py"),
    Mutation("forward multiplied by the discount factor instead of divided", "smile.py", "forward = float(fit.intercept) / discount_factor", "forward = float(fit.intercept) * discount_factor", "test_smile.py"),
    Mutation("puts above the forward treated as out of the money", "smile.py", '(quotes["strike"] < f)', '(quotes["strike"] > f)', "test_smile.py"),
    Mutation("settlement time read in UTC instead of New York time", "smile.py", "tzinfo=ZoneInfo(settlement_tz)", "tzinfo=UTC", "test_smile.py"),
]


def run_tests(workdir: Path, test_file: str) -> int:
    env = {**os.environ, "PYTHONPATH": str(workdir / "src")}
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"tests/{test_file}"]
    return subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True).returncode


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "repo"
        shutil.copytree(ROOT, workdir, ignore=shutil.ignore_patterns(".git", ".venv", "venv", "data", "results", "__pycache__", "*.egg-info"))
        env = {**os.environ, "PYTHONPATH": str(workdir / "src")}
        located = subprocess.run([sys.executable, "-c", "import optlab; print(optlab.__file__)"], env=env, capture_output=True, text=True).stdout
        if str(workdir) not in located:
            print(f"Could not import the temporary copy (got {located.strip()!r}); aborting.", file=sys.stderr)
            return 2

        survivors = 0
        print(f"{'Result':<10}Mutation")
        for mutation in MUTATIONS:
            path = workdir / "src" / "optlab" / mutation.module
            original_source = path.read_text(encoding="utf-8")
            if original_source.count(mutation.original) != 1:
                print(f"{'SKIPPED':<10}{mutation.description} (pattern not found exactly once)")
                survivors += 1
                continue
            path.write_text(original_source.replace(mutation.original, mutation.mutated), encoding="utf-8")
            caught = run_tests(workdir, mutation.tests) != 0
            path.write_text(original_source, encoding="utf-8")
            survivors += not caught
            print(f"{'caught' if caught else 'SURVIVED':<10}{mutation.description}")

    print(f"\n{len(MUTATIONS) - survivors} of {len(MUTATIONS)} injected bugs caught.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
