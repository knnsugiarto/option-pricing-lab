"""From a raw option chain snapshot to an implied volatility smile.

Pipeline:

1. Keep one settlement root. SPX lists AM-settled (SPX) and PM-settled (SPXW)
   contracts with the same expiry date; mixing them double-counts strikes with
   different expiry times.
2. Quote filters: a two-sided market (bid at least ``min_bid``, ask above bid)
   and a bid-ask spread no wider than ``max_relative_spread`` of the mid.
3. Implied forward and discount factor from put–call parity. For European
   options C - P = DF (F - K) at every strike, so a straight-line fit of
   C - P against K gives slope -DF and intercept DF F. We use the Theil–Sen
   estimator (median of pairwise slopes), which ignores a minority of bad
   quotes. This needs no dividend forecast and no separate interest rate.
4. Keep out-of-the-money quotes (puts below the forward, calls at or above),
   which are the liquid side, and solve for each one's implied vol at the
   bid, mid and ask.

Implied vols use Black-76: with spot = F and div_yield = rate, the
Black–Scholes–Merton formula becomes DF [F N(d1) - K N(d2)], which depends on
the forward only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from optlab.implied_vol import implied_vol

_CONTRACT_SYMBOL = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<type>[CP])(?P<strike>\d{8})$")
_REQUIRED_COLUMNS = ("contractSymbol", "option_type", "strike", "bid", "ask")
_SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass(frozen=True)
class Snapshot:
    """An option chain for one expiry, as downloaded."""

    quotes: pd.DataFrame
    ticker: str
    expiry: date
    quote_time: datetime  # timezone-aware, UTC
    spot: float | None
    source_file: str


@dataclass(frozen=True)
class FunnelStep:
    """How many quotes remain after one cleaning step."""

    name: str
    remaining: int


@dataclass(frozen=True)
class ForwardEstimate:
    """Forward price and discount factor implied by put–call parity."""

    forward: float
    discount_factor: float
    implied_rate: float  # continuously compounded, -ln(DF) / T
    n_strikes: int
    residual_std: float  # scatter of C - P around the fitted line, in price units


@dataclass(frozen=True)
class SmileSummary:
    """How far the market smile is from a single flat Black–Scholes vol."""

    atm_vol: float
    window_sd: float  # analysis window: |ln(K/F)| <= window_sd * atm_vol * sqrt(T)
    n_in_window: int
    rmse_vs_flat_vol_points: float
    max_abs_dev_vol_points: float
    share_flat_outside_spread: float  # share of quotes whose bid-ask vol band excludes the flat vol
    put_wing_vol: float  # implied vol at ln(K/F) = -window_sd standard deviations (NaN if not quoted)
    call_wing_vol: float  # implied vol at ln(K/F) = +window_sd standard deviations


@dataclass(frozen=True)
class SmileAnalysis:
    """Everything produced from one snapshot."""

    ticker: str
    expiry: date
    quote_time: datetime
    spot: float | None
    root: str
    maturity: float
    forward: ForwardEstimate
    funnel: list[FunnelStep]
    smile: pd.DataFrame
    summary: SmileSummary


def load_snapshot(csv_path: Path) -> Snapshot:
    """Read a snapshot CSV and its JSON sidecar written by :mod:`optlab.market_data`."""
    quotes = pd.read_csv(csv_path)
    missing = set(_REQUIRED_COLUMNS) - set(quotes.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    meta = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    stamp = meta.get("quote_time_utc") or meta["downloaded_at_utc"]
    spot = meta.get("spot")
    return Snapshot(
        quotes=quotes,
        ticker=meta["ticker"],
        expiry=date.fromisoformat(meta["expiry"]),
        quote_time=datetime.fromisoformat(stamp).astimezone(UTC),
        spot=float(spot) if spot is not None else None,
        source_file=csv_path.name,
    )


def contract_root(symbol: str) -> str:
    """Root of an OCC option symbol, e.g. 'SPXW261120C05000000' -> 'SPXW'."""
    match = _CONTRACT_SYMBOL.match(symbol)
    if match is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    return match.group("root")


def years_to_expiry(quote_time: datetime, expiry: date, settlement_time: time, settlement_tz: str) -> float:
    """Calendar years (ACT/365) from the quote time to settlement on the expiry date.

    Settlement is a wall-clock time in the exchange's time zone, so daylight
    saving is handled by the time zone database.
    """
    if quote_time.tzinfo is None:
        raise ValueError("quote_time must be timezone-aware")
    settlement = datetime.combine(expiry, settlement_time, tzinfo=ZoneInfo(settlement_tz))
    seconds = (settlement - quote_time).total_seconds()
    if seconds <= 0:
        raise ValueError(f"quotes taken at {quote_time} are not before settlement at {settlement}")
    return seconds / _SECONDS_PER_YEAR


def select_root(quotes: pd.DataFrame, preferred_root: str | None) -> tuple[pd.DataFrame, str]:
    """Keep contracts of one root: the preferred one if present, else the most common."""
    roots = quotes["contractSymbol"].map(contract_root)
    counts = roots.value_counts()
    root = preferred_root if preferred_root in counts.index else str(counts.index[0])
    return quotes[roots == root].copy(), root


def clean_quotes(quotes: pd.DataFrame, min_bid: float, max_relative_spread: float) -> tuple[pd.DataFrame, list[FunnelStep]]:
    """Apply the quote-quality filters and add a ``mid`` column.

    Returns:
        The surviving quotes and the count after each filter.
    """
    funnel = []
    two_sided = quotes[(quotes["bid"] >= min_bid) & (quotes["ask"] > quotes["bid"])].copy()
    funnel.append(FunnelStep(f"two-sided quote (bid >= {min_bid:g}, ask > bid)", len(two_sided)))
    two_sided["mid"] = 0.5 * (two_sided["bid"] + two_sided["ask"])
    tight = two_sided[(two_sided["ask"] - two_sided["bid"]) <= max_relative_spread * two_sided["mid"]].copy()
    funnel.append(FunnelStep(f"spread <= {max_relative_spread:.0%} of mid", len(tight)))
    return tight, funnel


def implied_forward(quotes: pd.DataFrame, maturity: float, n_strikes: int) -> ForwardEstimate:
    """Forward and discount factor from put–call parity near the money.

    Uses the ``n_strikes`` strikes closest to the one where call and put
    prices are nearest each other (roughly the forward). Raises ValueError if
    there are too few strikes or the fit implies an implausible interest rate.
    """
    mids = quotes.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="first")
    if not {"call", "put"} <= set(mids.columns):
        raise ValueError("need both calls and puts to estimate the forward")
    paired = mids.dropna(subset=["call", "put"])
    if len(paired) < 5:
        raise ValueError(f"only {len(paired)} strikes have both a call and a put quote; need at least 5")
    difference = paired["call"] - paired["put"]
    anchor = float(difference.abs().idxmin())
    strikes = paired.index.to_numpy(dtype=float)
    nearest = strikes[np.argsort(np.abs(strikes - anchor), kind="stable")[:n_strikes]]
    y = difference.loc[nearest].to_numpy(dtype=float)
    fit = theilslopes(y, nearest)
    discount_factor = -float(fit.slope)
    if discount_factor <= 0.0:
        raise ValueError(f"put–call parity fit gives a non-positive discount factor ({discount_factor:.4f}); quotes are unusable")
    forward = float(fit.intercept) / discount_factor
    implied_rate = -math.log(discount_factor) / maturity
    if abs(implied_rate) > 0.20:
        raise ValueError(f"put–call parity implies a {implied_rate:.1%} interest rate; quotes are probably stale or mismatched")
    residuals = y - (fit.intercept + fit.slope * nearest)
    return ForwardEstimate(
        forward=forward,
        discount_factor=discount_factor,
        implied_rate=implied_rate,
        n_strikes=len(nearest),
        residual_std=float(np.std(residuals, ddof=2)) if len(nearest) > 2 else math.nan,
    )


def _black76_vol(price: float, forward: float, strike: float, maturity: float, rate: float, option_type: str) -> float:
    return implied_vol(price, forward, strike, maturity, rate, rate, option_type)  # type: ignore[arg-type]


def build_smile(quotes: pd.DataFrame, forward: ForwardEstimate, maturity: float) -> tuple[pd.DataFrame, list[FunnelStep]]:
    """Out-of-the-money quotes with log-moneyness and bid, mid and ask implied vols.

    A bid below the lowest price any vol in the search range can give is
    assigned vol 0, and an ask above the highest is assigned +inf, so the
    bid-ask vol band stays a valid (if wide) interval.
    """
    f, r = forward.forward, forward.implied_rate
    is_otm = ((quotes["option_type"] == "put") & (quotes["strike"] < f)) | ((quotes["option_type"] == "call") & (quotes["strike"] >= f))
    otm = quotes[is_otm].copy()
    funnel = [FunnelStep("out-of-the-money side of the forward", len(otm))]

    def vols(column: str) -> list[float]:
        return [
            _black76_vol(p, f, k, maturity, r, t)
            for p, k, t in zip(otm[column], otm["strike"], otm["option_type"])
        ]

    otm["log_moneyness"] = np.log(otm["strike"] / f)
    otm["iv_mid"] = vols("mid")
    otm["iv_bid"] = np.nan_to_num(vols("bid"), nan=0.0)
    otm["iv_ask"] = np.nan_to_num(vols("ask"), nan=np.inf)
    solved = otm[otm["iv_mid"].notna()].sort_values("log_moneyness").reset_index(drop=True)
    funnel.append(FunnelStep("implied vol solvable (inside no-arbitrage bounds)", len(solved)))
    return solved, funnel


def summarise_smile(smile: pd.DataFrame, maturity: float, window_sd: float) -> SmileSummary:
    """Compare the smile with a single flat vol, set at the at-the-money level.

    The ATM vol is interpolated at ln(K/F) = 0. Deviations are measured in vol
    points (1 point = 0.01 vol) over quotes within ``window_sd`` standard
    deviations of the forward, where one standard deviation is ATM vol x sqrt(T).
    """
    k = smile["log_moneyness"].to_numpy()
    iv = smile["iv_mid"].to_numpy()
    if len(k) < 3 or k.min() >= 0.0 or k.max() <= 0.0:
        raise ValueError("the smile must have quotes on both sides of the forward")
    atm_vol = float(np.interp(0.0, k, iv))
    half_width = window_sd * atm_vol * math.sqrt(maturity)
    window = smile[np.abs(k) <= half_width]
    deviation = window["iv_mid"].to_numpy() - atm_vol
    outside = (window["iv_bid"] > atm_vol) | (window["iv_ask"] < atm_vol)

    def wing(target: float) -> float:
        return float(np.interp(target, k, iv)) if k.min() <= target <= k.max() else math.nan

    return SmileSummary(
        atm_vol=atm_vol,
        window_sd=window_sd,
        n_in_window=len(window),
        rmse_vs_flat_vol_points=100.0 * float(np.sqrt(np.mean(deviation**2))),
        max_abs_dev_vol_points=100.0 * float(np.max(np.abs(deviation))),
        share_flat_outside_spread=float(outside.mean()),
        put_wing_vol=wing(-half_width),
        call_wing_vol=wing(half_width),
    )


def analyse_snapshot(
    snapshot: Snapshot,
    *,
    preferred_root: str | None,
    settlement_time: time,
    settlement_tz: str,
    min_bid: float,
    max_relative_spread: float,
    parity_strikes: int,
    window_sd: float,
) -> SmileAnalysis:
    """Run the full pipeline on one snapshot."""
    funnel = [FunnelStep("downloaded quotes", len(snapshot.quotes))]
    quotes, root = select_root(snapshot.quotes, preferred_root)
    funnel.append(FunnelStep(f"single settlement root ({root})", len(quotes)))
    quotes, quality_steps = clean_quotes(quotes, min_bid, max_relative_spread)
    funnel.extend(quality_steps)
    maturity = years_to_expiry(snapshot.quote_time, snapshot.expiry, settlement_time, settlement_tz)
    forward = implied_forward(quotes, maturity, parity_strikes)
    smile, smile_steps = build_smile(quotes, forward, maturity)
    funnel.extend(smile_steps)
    return SmileAnalysis(
        ticker=snapshot.ticker,
        expiry=snapshot.expiry,
        quote_time=snapshot.quote_time,
        spot=snapshot.spot,
        root=root,
        maturity=maturity,
        forward=forward,
        funnel=funnel,
        smile=smile,
        summary=summarise_smile(smile, maturity, window_sd),
    )
