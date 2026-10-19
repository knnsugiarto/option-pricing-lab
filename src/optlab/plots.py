"""Figures for the README. Each function takes study results and writes one PNG.

Styling follows one fixed palette (validated for colour-vision deficiency):
thin lines, distinct marker shapes as a second identity cue, hairline grids,
and text in ink colours rather than series colours.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from optlab.smile import SmileAnalysis  # noqa: E402
from optlab.studies import GammaProfile, MonteCarloStudy, PDEStudy  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order
MARKERS = ("o", "s", "^", "D")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_WIDTH = 1.6
MARKER_SIZE = 5.5
DPI = 160

ESTIMATOR_LABELS = {
    "plain": "Plain",
    "antithetic": "Antithetic",
    "control": "Control variate",
    "antithetic_control": "Antithetic + control",
}


@contextmanager
def _style() -> Iterator[None]:
    params = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_SECONDARY,
        "font.family": ["DejaVu Sans", "sans-serif"],
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
    with plt.rc_context(params):
        yield


def _series_line(ax: Axes, x: Sequence[float], y: Sequence[float], slot: int, label: str) -> None:
    ax.plot(
        x,
        y,
        color=SERIES[slot],
        linewidth=LINE_WIDTH,
        marker=MARKERS[slot],
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label=label,
        zorder=3,
    )


def _end_label(ax: Axes, x: float, y: float, text: str) -> None:
    """Direct label just right of a line's last point, in ink (not the series colour)."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_SECONDARY)


def _reference_slope(ax: Axes, x: Sequence[float], anchor_y: float, order: float, text: str) -> None:
    """Thin grey guide line of a given log-log slope through the first point."""
    xs = np.array([x[0], x[-1]], dtype=float)
    ys = anchor_y * (xs / xs[0]) ** (-order)
    ax.plot(xs, ys, color=INK_MUTED, linewidth=0.9, zorder=2)
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_MUTED)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_mc_convergence(study: MonteCarloStudy, path: Path) -> None:
    """RMSE against path count, and reported standard error against measured error."""
    with _style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
        for slot, (estimator, label) in enumerate(ESTIMATOR_LABELS.items()):
            rows = [r for r in study.rows if r.estimator == estimator]
            n = [r.n_paths for r in rows]
            rmse = [r.rmse for r in rows]
            _series_line(left, n, rmse, slot, label)
            _end_label(left, n[-1], rmse[-1], f"{label} ({study.variance_reduction[estimator]:.1f}x)")
            right.scatter(
                [r.rmse for r in rows],
                [r.mean_std_error for r in rows],
                s=34,
                color=SERIES[slot],
                marker=MARKERS[slot],
                edgecolors=SURFACE,
                linewidths=1.2,
                label=label,
                zorder=3,
            )
        plain = [r for r in study.rows if r.estimator == "plain"]
        largest = max(r.n_paths for r in plain)
        _reference_slope(left, [r.n_paths for r in plain], plain[0].rmse * 1.5, 0.5, "slope −½")
        left.set(xscale="log", yscale="log", xlabel="Paths", ylabel="RMSE vs closed form (price units)")
        left.set_title("Error falls like 1/√N; variance reduction shifts the line down")
        left.margins(x=0.05)
        left.set_xlim(right=largest * 9)
        left.text(
            0.98,
            0.97,
            f"In brackets: variance reduction vs plain\nat {largest:,} paths ({study.replications} runs per point)",
            transform=left.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_MUTED,
        )

        values = [r.rmse for r in study.rows] + [r.mean_std_error for r in study.rows]
        bounds = np.array([min(values) / 1.4, max(values) * 1.4])
        right.plot(bounds, bounds, color=INK_MUTED, linewidth=0.9, zorder=2)
        right.text(0.97, 0.05, "diagonal: reported = measured", transform=right.transAxes, ha="right", fontsize=8, color=INK_MUTED)
        right.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds, xlabel="Measured RMSE", ylabel="Mean reported standard error")
        right.set_title("Reported error bars are honest")
        handles, labels = right.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, path)


PDE_SLOTS = {"Crank–Nicolson": 0, "CN + Rannacher": 1, "CN + Rannacher + Richardson": 2}


def plot_pde_convergence(study: PDEStudy, path: Path) -> None:
    """Price and gamma errors at the strike against grid size, for each scheme."""
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
        for ax, attribute, orders, title in (
            (axes[0], "price_error", study.price_orders, "Price error at the strike"),
            (axes[1], "gamma_error", study.gamma_orders, "Gamma error at the strike"),
        ):
            for method, slot in PDE_SLOTS.items():
                rows = [r for r in study.rows if r.method == method]
                n = [r.space_steps for r in rows]
                err = [getattr(r, attribute) for r in rows]
                _series_line(ax, n, err, slot, method)
                _end_label(ax, n[-1], err[-1], f"order {orders[method]:.1f}" + (" (diverging)" if orders[method] < 0 else ""))
            ax.set(xscale="log", yscale="log", xlabel="Space steps N (time steps = N/8)", title=title)
            ax.set_xlim(right=max(r.space_steps for r in study.rows) * 4.0)
        axes[0].set_ylabel("Absolute error")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(
            "Plain Crank–Nicolson's gamma diverges with large time steps; Rannacher start-up restores second order",
            x=0.01,
            ha="left",
            y=0.995,
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.94))
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        _save(fig, path)


def plot_gamma_profile(profile: GammaProfile, strike: float, path: Path) -> None:
    """Grid gamma across spot levels: plain Crank–Nicolson against Rannacher start-up."""
    with _style():
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(profile.spots, profile.exact, color=INK, linewidth=2.6, label="Closed form", zorder=2)
        ax.plot(profile.spots, profile.plain, color=SERIES[0], linewidth=1.0, label="Crank–Nicolson", zorder=3)
        ax.plot(profile.spots, profile.rannacher, color=SERIES[1], linewidth=LINE_WIDTH, label="CN + Rannacher", zorder=4)
        ax.axvline(strike, color=AXIS, linewidth=0.8, zorder=1)
        ax.annotate("strike", (strike, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_MUTED)
        ax.set(xlabel="Spot", ylabel="Gamma")
        ax.set_title(
            f"Gamma on one grid ({profile.space_steps} space × {profile.time_steps} time steps, λ = {profile.diffusion_number:.0f}): "
            "plain CN oscillates at the strike"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, path)


def _smile_points(ax: Axes, smile: pd.DataFrame, with_bands: bool) -> None:
    """Puts and calls as separate series; optionally each quote's bid-ask vol band."""
    for slot, (option_type, label) in enumerate((("put", "Puts (K < F)"), ("call", "Calls (K ≥ F)"))):
        side = smile[smile["option_type"] == option_type]
        k = side["log_moneyness"].to_numpy()
        if with_bands:
            low = np.clip(side["iv_bid"].to_numpy(), 0.0, None)
            high = np.minimum(side["iv_ask"].to_numpy(), 1.5 * side["iv_mid"].max())
            ax.vlines(k, 100 * low, 100 * high, color=SERIES[slot], alpha=0.45, linewidth=1.6, zorder=2)
        ax.scatter(
            k,
            100 * side["iv_mid"],
            s=30 if with_bands else 22,
            color=SERIES[slot],
            marker=MARKERS[slot],
            edgecolors=SURFACE,
            linewidths=1.0,
            label=f"{label}: mid" + (", with bid–ask band" if with_bands else ""),
            zorder=3,
        )


def plot_smile(analysis: SmileAnalysis, path: Path) -> None:
    """Market implied vols against log-moneyness: the whole smile, and a zoom on the scored window.

    The left panel shows every quote that survived cleaning. The right panel
    zooms on the window where the flat-vol error is measured, where each
    quote's bid-ask vol band is wide enough to see against the flat vol.
    """
    smile, summary = analysis.smile, analysis.summary
    atm = 100 * summary.atm_vol
    edge = summary.window_sd * summary.atm_vol * np.sqrt(analysis.maturity)
    window = smile[np.abs(smile["log_moneyness"]) <= edge]
    with _style():
        fig, (full, zoom) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

        _smile_points(full, smile, with_bands=False)
        full.axvspan(-edge, edge, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        full.axhline(atm, color=INK_SECONDARY, linewidth=1.2, zorder=2)
        full.set(xlabel="Log-moneyness ln(K / F)", ylabel="Implied volatility (%)", title="Every quote kept after cleaning")
        full.annotate(
            f"flat vol {atm:.1f}%",
            (smile["log_moneyness"].min(), atm),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        full.annotate(
            f"scored window\n(±{summary.window_sd:g} sd)",
            (-edge, full.get_ylim()[1]),
            xytext=(4, -6),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=INK_MUTED,
        )

        _smile_points(zoom, window, with_bands=True)
        zoom.axhline(atm, color=INK_SECONDARY, linewidth=1.2, zorder=2)
        zoom.annotate(
            f"flat Black–Scholes vol = ATM {atm:.1f}%",
            (edge, atm),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        low = np.clip(window["iv_bid"].to_numpy(), 0.0, None).min()
        high = np.minimum(window["iv_ask"].to_numpy(), 1.5 * window["iv_mid"].max()).max()
        pad = 0.06 * (high - low)
        zoom.set(
            xlim=(-edge * 1.04, edge * 1.04),
            ylim=(100 * (low - pad), 100 * (high + pad)),
            xlabel="Log-moneyness ln(K / F)",
            title=f"Scored window: flat vol outside {summary.share_flat_outside_spread:.0%} of bid–ask bands",
        )
        zoom.legend(loc="upper right")

        days = analysis.maturity * 365
        fig.suptitle(
            f"{analysis.ticker} options expiring {analysis.expiry:%d %b %Y} ({days:.0f} days), quotes at "
            f"{analysis.quote_time:%d %b %Y %H:%M} UTC",
            x=0.01,
            ha="left",
            y=0.985,
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, path)
