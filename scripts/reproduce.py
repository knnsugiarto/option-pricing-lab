"""Reproduce every result in the README with one command.

Steps: run the test suite; run the Monte Carlo and PDE studies; analyse the
latest option chain snapshot (downloading one if none exists); write figures
to results/figures/, numbers to results/metrics.json and results/summary.md,
and optionally refresh the README's results block.

Examples:
    python scripts/reproduce.py --update-readme
    python scripts/reproduce.py --fresh --update-readme     # download a new snapshot first
    python scripts/reproduce.py --skip-market               # numerical studies only, no network
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import time as clock_time
from pathlib import Path

from optlab import market_data, plots, report, smile, studies
from optlab.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce all results.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--snapshot", type=Path, help="analyse this snapshot CSV instead of the latest one")
    parser.add_argument("--fresh", action="store_true", help="download a new snapshot even if one exists")
    parser.add_argument("--skip-market", action="store_true", help="skip the market data study")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the test suite first")
    parser.add_argument("--update-readme", action="store_true", help="rewrite the README results block")
    return parser.parse_args()


def step(message: str) -> float:
    print(f"\n== {message}", flush=True)
    return time.perf_counter()


def done(start: float) -> None:
    print(f"   done in {time.perf_counter() - start:.1f} s", flush=True)


def run_market(config: dict, args: argparse.Namespace) -> tuple[smile.SmileAnalysis, str]:
    """Analyse a snapshot: the one given, else the latest saved, else a fresh download."""
    m = config["market"]
    snapshot_path = args.snapshot
    if snapshot_path is None and not args.fresh:
        snapshot_path = market_data.latest_snapshot(RAW_DATA, m["ticker"])
    if snapshot_path is None:
        print(f"   downloading {m['ticker']} option chain from Yahoo Finance")
        snapshot_path = market_data.download_snapshot(m["ticker"], m["target_days"], m["min_days"], RAW_DATA)
    print(f"   snapshot: {snapshot_path.name}")
    hour, minute = (int(part) for part in m["settlement_time"].split(":"))
    analysis = smile.analyse_snapshot(
        smile.load_snapshot(snapshot_path),
        preferred_root=m["preferred_root"],
        settlement_time=clock_time(hour, minute),
        settlement_tz=m["settlement_timezone"],
        min_bid=m["min_bid"],
        max_relative_spread=m["max_relative_spread"],
        parity_strikes=m["parity_strikes"],
        window_sd=m["window_sd"],
    )
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    analysis.smile.to_csv(PROCESSED_DATA / f"smile_{snapshot_path.stem}.csv", index=False)  # derived quotes stay out of git
    return analysis, snapshot_path.name


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = config["seed"]
    contract = studies.Contract(**config["contract"])

    if not args.skip_tests:
        start = step("Test suite")
        if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode != 0:
            print("Tests failed; fix them before trusting any result.", file=sys.stderr)
            return 1
        done(start)

    start = step("Monte Carlo convergence")
    mc_study = studies.monte_carlo_convergence(contract, seed=seed, **config["monte_carlo"])
    plots.plot_mc_convergence(mc_study, FIGURES / "mc_convergence.png")
    done(start)

    start = step("PDE convergence and gamma profile")
    cn = config["crank_nicolson"]
    pde_study = studies.pde_convergence(contract, cn["space_steps"], cn["time_steps_per_space_step"], cn["rannacher_steps"], cn["n_std"], cn["fit_from"])
    plots.plot_pde_convergence(pde_study, FIGURES / "pde_convergence.png")
    g = config["gamma_profile"]
    profile = studies.gamma_profile(contract, g["space_steps"], g["time_steps"], cn["rannacher_steps"], cn["n_std"], g["spot_min"], g["spot_max"])
    plots.plot_gamma_profile(profile, contract.strike, FIGURES / "gamma_profile.png")
    done(start)

    start = step("Cross-method accuracy on random contracts")
    a = config["accuracy"]
    accuracy = studies.cross_method_accuracy(a["n_cases"], seed, a["mc_paths"], a, config["monte_carlo"]["confidence_level"])
    done(start)

    analysis, snapshot_name, market_failed = None, None, False
    if not args.skip_market:
        start = step("Market implied vol smile")
        try:
            analysis, snapshot_name = run_market(config, args)
            plots.plot_smile(analysis, FIGURES / "smile.png")
            done(start)
        except Exception as error:  # report any data problem, keep the numerical results
            market_failed = True
            print(f"   market study failed: {error}", file=sys.stderr)
            print(
                "   Numerical results are still written. Check your internet connection; Yahoo may serve\n"
                "   zero bids and asks outside US market hours, so try again while US markets are open.",
                file=sys.stderr,
            )

    metrics = report.build_metrics(contract, mc_study, pde_study, accuracy, analysis, snapshot_name)
    results_dir = ROOT / "results"
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (results_dir / "summary.md").write_text(report.summary_markdown(metrics), encoding="utf-8")
    if args.update_readme:
        report.update_readme(ROOT / "README.md", report.results_block(metrics))
    print(f"\nWrote {results_dir / 'summary.md'}, metrics.json and figures in {FIGURES}")
    if args.update_readme:
        print("Updated the results block in README.md")
    return 1 if market_failed else 0


if __name__ == "__main__":
    sys.exit(main())
