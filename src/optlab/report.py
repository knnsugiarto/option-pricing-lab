"""Turn study results into metrics.json, a Markdown summary and the README results block."""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from optlab.smile import SmileAnalysis
from optlab.studies import AccuracyStudy, Contract, MonteCarloStudy, PDEStudy

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "yfinance")


def environment() -> dict[str, str]:
    """Where the numbers were produced. Timings depend on it."""
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        **versions,
    }


def smile_metrics(analysis: SmileAnalysis) -> dict[str, Any]:
    s, f = analysis.summary, analysis.forward
    return {
        "ticker": analysis.ticker,
        "root": analysis.root,
        "expiry": analysis.expiry.isoformat(),
        "quote_time_utc": analysis.quote_time.isoformat(),
        "days_to_expiry": analysis.maturity * 365.0,
        "spot": analysis.spot,
        "forward": f.forward,
        "forward_over_spot_minus_1": (f.forward / analysis.spot - 1.0) if analysis.spot else None,
        "implied_rate": f.implied_rate,
        "parity_strikes": f.n_strikes,
        "parity_residual_std": f.residual_std,
        "funnel": [{"step": step.name, "remaining": step.remaining} for step in analysis.funnel],
        **asdict(s),
    }


def build_metrics(
    contract: Contract,
    mc_study: MonteCarloStudy,
    pde_study: PDEStudy,
    accuracy: AccuracyStudy,
    analysis: SmileAnalysis | None,
    snapshot_file: str | None,
) -> dict[str, Any]:
    """Collect every reported number into one JSON-serialisable dictionary."""
    return {
        "environment": environment(),
        "contract": asdict(contract),
        "monte_carlo": {
            **{k: v for k, v in asdict(mc_study).items() if k != "rows"},
            "rows": [asdict(r) for r in mc_study.rows],
        },
        "pde": {
            **{k: v for k, v in asdict(pde_study).items() if k != "rows"},
            "rows": [asdict(r) for r in pde_study.rows],
        },
        "accuracy": asdict(accuracy),
        "market": None if analysis is None else {"snapshot_file": snapshot_file, **smile_metrics(analysis)},
    }


def _sci(x: float) -> str:
    return f"{x:.1e}"


def _ms(seconds: float) -> str:
    return f"{1e3 * seconds:.1f} ms"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def results_block(metrics: dict[str, Any]) -> str:
    """The Markdown that goes between the README results markers."""
    c, mcs, pde, acc, env = metrics["contract"], metrics["monte_carlo"], metrics["pde"], metrics["accuracy"], metrics["environment"]
    largest = max(r["n_paths"] for r in mcs["rows"])
    finest = max(r["space_steps"] for r in pde["rows"])

    def mc_row(estimator: str) -> dict[str, Any]:
        return next(r for r in mcs["rows"] if r["estimator"] == estimator and r["n_paths"] == largest)

    def pde_row(method: str) -> dict[str, Any]:
        return next(r for r in pde["rows"] if r["method"] == method and r["space_steps"] == finest)

    lines = [
        f"Benchmark contract: European {c['option_type']}, S = K = {c['spot']:g}, T = {c['maturity']:g} y, "
        f"r = {c['rate']:.0%}, σ = {c['vol']:.0%}, q = {c['div_yield']:.0%}. Closed-form price {mcs['exact_price']:.4f}.",
        "",
        _row(["Method", "Setting", "Error vs closed form", "Measured convergence order", "Time per price"]),
        _row(["---"] * 5),
    ]
    for estimator, label in (("plain", "Monte Carlo, plain (baseline)"), ("antithetic_control", "Monte Carlo, antithetic + control variate")):
        r = mc_row(estimator)
        lines.append(
            _row(
                [
                    label,
                    f"{largest:,} paths",
                    f"RMSE {r['rmse']:.4f}; variance ÷{mcs['variance_reduction'][estimator]:.1f}",
                    f"{mcs['orders'][estimator]:.2f} (theory 0.5)",
                    _ms(r["seconds_per_run"]),
                ]
            )
        )
    for method, label in (
        ("Crank–Nicolson", "Crank–Nicolson, plain (baseline)"),
        ("CN + Rannacher", "Crank–Nicolson + Rannacher start-up"),
        ("CN + Rannacher + Richardson", "+ Richardson extrapolation"),
    ):
        r = pde_row(method)
        grid = f"{r['space_steps']} × {r['time_steps']} grid" + (" and 2× finer" if "Richardson" in method else "")
        price_order, gamma_order = pde["price_orders"][method], pde["gamma_orders"][method]
        lines.append(
            _row(
                [
                    label,
                    grid,
                    f"price {_sci(r['price_error'])}; gamma {_sci(r['gamma_error'])}",
                    f"price {price_order:.1f}; gamma {gamma_order:.1f}" + (" (diverging)" if gamma_order < 0 else ""),
                    _ms(r["seconds"]),
                ]
            )
        )
    lines += [
        "",
        f"Cross-check on {2 * acc['n_cases']} random contracts (calls and puts; strikes 70–130% of spot, "
        "maturities 0.1–2 y, vols 10–60%, rates −1% to 6%, dividend yields 0–4%):",
        "",
        _row(["Check", "Result"]),
        _row(["---", "---"]),
        _row(["PDE pricer at default settings: max / median absolute error", f"{_sci(acc['pde_max_error'])} / {_sci(acc['pde_median_error'])} ({acc['pde_ms_per_price']:.1f} ms per price)"]),
        _row(
            [
                f"Monte Carlo ({acc['mc_paths']:,} paths): 99% intervals containing the closed form",
                f"{acc['mc_coverage']:.1%} ({acc['mc_degenerate']} degenerate runs with zero error bars; max error there {_sci(acc['mc_degenerate_max_error'])})",
            ]
        ),
        _row(["Closed form", f"{acc['bs_microseconds_per_price']:.0f} µs per price"]),
        "",
    ]
    market = metrics.get("market")
    if market is None:
        lines += ["Market smile: not run yet (see How to reproduce).", ""]
    else:
        m = market
        used = m["funnel"][-1]["remaining"]
        lines += [
            f"Market test: {m['ticker']} ({m['root']}) options expiring {m['expiry']}, quotes at {m['quote_time_utc'][:16].replace('T', ' ')} UTC "
            f"({m['days_to_expiry']:.0f} days to expiry).",
            "",
            _row(["Market test", "Result"]),
            _row(["---", "---"]),
            _row(["Quotes used / downloaded", f"{used} / {m['funnel'][0]['remaining']}"]),
            _row(["Forward from put–call parity; implied rate", f"{m['forward']:.2f}; {m['implied_rate']:.2%}"]),
            _row(["At-the-money implied vol", f"{m['atm_vol']:.2%}"]),
            _row([f"Flat-vol error vs market within ±{m['window_sd']:g} sd (RMSE / max)", f"{m['rmse_vs_flat_vol_points']:.2f} / {m['max_abs_dev_vol_points']:.2f} vol points"]),
            _row(["Quotes whose bid–ask band excludes the flat vol", f"{m['share_flat_outside_spread']:.0%} of {m['n_in_window']}"]),
            _row([f"Implied vol at −{m['window_sd']:g} sd / +{m['window_sd']:g} sd", f"{_pct(m['put_wing_vol'])} / {_pct(m['call_wing_vol'])}"]),
            "",
        ]
    lines.append(f"Timings: {env['processor']}, Python {env['python']}, generated {env['generated_at_utc'][:10]} UTC.")
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "not quoted" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2%}"


def summary_markdown(metrics: dict[str, Any]) -> str:
    """Full results, including every row of every study and the cleaning funnel."""
    lines = ["# Results", "", "Generated by `python scripts/reproduce.py`. Do not edit by hand.", "", results_block(metrics), ""]
    mcs = metrics["monte_carlo"]
    lines += [
        "## Monte Carlo convergence",
        "",
        _row(["Estimator", "Paths", "RMSE", "Mean reported SE", f"{mcs['confidence_level']:.0%} interval coverage", "Time per run"]),
        _row(["---"] * 6),
    ]
    for r in mcs["rows"]:
        lines.append(_row([r["estimator"], f"{r['n_paths']:,}", f"{r['rmse']:.5f}", f"{r['mean_std_error']:.5f}", f"{r['coverage']:.0%}", _ms(r["seconds_per_run"])]))
    pde = metrics["pde"]
    lines += [
        "",
        "## PDE convergence",
        "",
        f"Convergence orders fitted on grids with at least {pde['fit_from']} space steps.",
        "",
        _row(["Method", "Space steps", "Time steps", "λ = Δτ/Δx²", "Price error", "Delta error", "Gamma error", "Time"]),
        _row(["---"] * 8),
    ]
    for r in pde["rows"]:
        lines.append(
            _row(
                [
                    r["method"],
                    str(r["space_steps"]),
                    str(r["time_steps"]),
                    f"{r['diffusion_number']:.1f}",
                    _sci(r["price_error"]),
                    _sci(r["delta_error"]),
                    _sci(r["gamma_error"]),
                    _ms(r["seconds"]),
                ]
            )
        )
    market = metrics.get("market")
    if market is not None:
        lines += ["", "## Market data cleaning funnel", "", f"Snapshot: `{market['snapshot_file']}`", "", _row(["Step", "Quotes remaining"]), _row(["---", "---"])]
        lines += [_row([step["step"], str(step["remaining"])]) for step in market["funnel"]]
        if market["forward_over_spot_minus_1"] is not None:
            lines += ["", f"Forward / spot − 1 = {market['forward_over_spot_minus_1']:.3%} (should be close to (r − q) × T)."]
        lines += [f"Parity fit residual standard deviation: {market['parity_residual_std']:.3f} index points over {market['parity_strikes']} strikes."]
    env = metrics["environment"]
    lines += ["", "## Environment", ""] + [f"- {key}: {value}" for key, value in env.items()]
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, block: str) -> None:
    """Replace the text between the results markers in the README."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(f"{readme} has no {README_START} ... {README_END} block")
    updated = text[: start + len(README_START)] + "\n" + block + "\n" + text[end:]
    readme.write_text(updated, encoding="utf-8")
