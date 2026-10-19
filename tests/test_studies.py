"""Smoke tests: every study, figure and report function runs on a small configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synthetic_chain import PIPELINE, dirty_chain, snapshot_of

from optlab import plots, report, smile, studies
from optlab.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = studies.Contract(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2, div_yield=0.0, option_type="call")
RANGES = {
    "strike_range": [80.0, 120.0],
    "maturity_range": [0.25, 1.0],
    "rate_range": [0.0, 0.05],
    "vol_range": [0.15, 0.4],
    "div_yield_range": [0.0, 0.02],
}


@pytest.fixture(scope="module")
def small_results() -> tuple[studies.MonteCarloStudy, studies.PDEStudy, studies.AccuracyStudy, studies.GammaProfile]:
    mc_study = studies.monte_carlo_convergence(CONTRACT, [2_000, 8_000], replications=5, seed=1, confidence_level=0.99)
    pde_study = studies.pde_convergence(CONTRACT, [50, 100, 200], 0.125, rannacher_steps=2, n_std=5.0, fit_from=50)
    accuracy = studies.cross_method_accuracy(3, seed=2, mc_paths=4_000, ranges=RANGES, confidence_level=0.99)
    profile = studies.gamma_profile(CONTRACT, 100, 20, rannacher_steps=2, n_std=5.0, spot_min=80.0, spot_max=120.0)
    return mc_study, pde_study, accuracy, profile


def test_default_config_loads() -> None:
    config = load_config(ROOT / "configs" / "default.toml")
    studies.Contract(**config["contract"])
    assert config["crank_nicolson"]["space_steps"] == sorted(config["crank_nicolson"]["space_steps"])


def test_studies_produce_consistent_numbers(small_results) -> None:  # type: ignore[no-untyped-def]
    mc_study, pde_study, accuracy, profile = small_results
    assert len(mc_study.rows) == 4 * 2
    assert mc_study.variance_reduction["plain"] == pytest.approx(1.0)
    assert mc_study.variance_reduction["antithetic_control"] > 1.0
    by_method = {r.method: r for r in pde_study.rows if r.space_steps == 200}
    assert by_method["CN + Rannacher + Richardson"].price_error < by_method["CN + Rannacher"].price_error
    assert accuracy.pde_max_error < 1e-5 and 0.0 <= accuracy.mc_coverage <= 1.0
    assert profile.spots.min() >= 80.0 and profile.spots.max() <= 120.0


def test_random_contracts_are_seeded() -> None:
    assert studies.random_contracts(4, 7, RANGES) == studies.random_contracts(4, 7, RANGES)
    assert {c.option_type for c in studies.random_contracts(4, 7, RANGES)} == {"call", "put"}


def test_figures_are_written(small_results, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    mc_study, pde_study, _, profile = small_results
    analysis = smile.analyse_snapshot(snapshot_of(dirty_chain()[0]), **PIPELINE)
    plots.plot_mc_convergence(mc_study, tmp_path / "mc.png")
    plots.plot_pde_convergence(pde_study, tmp_path / "pde.png")
    plots.plot_gamma_profile(profile, CONTRACT.strike, tmp_path / "gamma.png")
    plots.plot_smile(analysis, tmp_path / "smile.png")
    for name in ("mc", "pde", "gamma", "smile"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000


@pytest.mark.parametrize("with_market", [False, True])
def test_report_round_trip(small_results, tmp_path: Path, with_market: bool) -> None:  # type: ignore[no-untyped-def]
    mc_study, pde_study, accuracy, _ = small_results
    analysis = smile.analyse_snapshot(snapshot_of(dirty_chain()[0]), **PIPELINE) if with_market else None
    metrics = report.build_metrics(CONTRACT, mc_study, pde_study, accuracy, analysis, "synthetic.csv" if with_market else None)
    json.dumps(metrics, default=str)  # serialisable
    block = report.results_block(metrics)
    assert ("Market test" in block) == with_market
    assert "## PDE convergence" in report.summary_markdown(metrics)

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.README_START}\nold\n{report.README_END}\noutro\n", encoding="utf-8")
    report.update_readme(readme, block)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and block in text and text.startswith("intro") and text.endswith("outro\n")


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        report.update_readme(readme, "block")
