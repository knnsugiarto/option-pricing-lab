"""A synthetic SPX-like option chain with a known forward, rate and smile.

Priced with Black-76 from known inputs, so the smile pipeline's output can be
checked against the truth. ``dirty_chain`` adds the kinds of bad quotes real
chains contain.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time

import numpy as np
import pandas as pd

from optlab import black_scholes as bs
from optlab import smile

QUOTE_TIME = datetime(2026, 10, 9, 19, 0, tzinfo=UTC)  # 15:00 New York time
EXPIRY = date(2026, 11, 20)  # after US clocks change on 1 November
SETTLEMENT = time(16, 0)
TZ = "America/New_York"
FORWARD = 5003.7
RATE = 0.04
STRIKES = np.arange(4200.0, 5601.0, 25.0)
PIPELINE = {
    "preferred_root": "SPXW",
    "settlement_time": SETTLEMENT,
    "settlement_tz": TZ,
    "min_bid": 0.10,
    "max_relative_spread": 0.25,
    "parity_strikes": 20,
    "window_sd": 2.0,
}


def true_vol(log_moneyness: np.ndarray | float) -> np.ndarray | float:
    """A downward-sloping, convex smile like an equity index's."""
    return 0.16 - 0.25 * log_moneyness + 0.6 * log_moneyness**2


def maturity() -> float:
    return smile.years_to_expiry(QUOTE_TIME, EXPIRY, SETTLEMENT, TZ)


def symbol(root: str, option_type: str, strike: float) -> str:
    return f"{root}{EXPIRY:%y%m%d}{'C' if option_type == 'call' else 'P'}{int(round(strike * 1000)):08d}"


def clean_chain() -> pd.DataFrame:
    """Calls and puts at every strike, quoted symmetrically around the exact Black-76 price."""
    t = maturity()
    rows = []
    for option_type in ("call", "put"):
        for strike in STRIKES:
            price = bs.price(FORWARD, strike, t, RATE, true_vol(math.log(strike / FORWARD)), RATE, option_type)
            half_spread = 0.05 + 0.01 * price
            rows.append(
                {
                    "contractSymbol": symbol("SPXW", option_type, strike),
                    "option_type": option_type,
                    "strike": strike,
                    "bid": price - half_spread,
                    "ask": price + half_spread,
                    "lastPrice": price,
                    "lastTradeDate": QUOTE_TIME.isoformat(),
                    "volume": 10,
                    "openInterest": 100,
                    "impliedVolatility": 0.0,
                }
            )
    return pd.DataFrame(rows)


def snapshot_of(quotes: pd.DataFrame) -> smile.Snapshot:
    return smile.Snapshot(quotes=quotes, ticker="^SPX", expiry=EXPIRY, quote_time=QUOTE_TIME, spot=4990.0, source_file="synthetic")


def dirty_chain() -> tuple[pd.DataFrame, dict[str, int]]:
    """The clean chain plus planted bad quotes, and how many of each were planted."""
    quotes = clean_chain()
    # Same strikes under the AM-settled SPX root: must be dropped, not double-counted.
    duplicates = quotes[quotes["strike"].between(4900.0, 5100.0)].copy()
    duplicates["contractSymbol"] = [symbol("SPX", t, k) for t, k in zip(duplicates["option_type"], duplicates["strike"])]
    duplicates[["bid", "ask"]] += 1.0
    # Bad quotes on otherwise good contracts, on the out-of-the-money side.
    otm_puts = quotes.index[(quotes["option_type"] == "put") & (quotes["strike"] < 4700.0)]
    zero_bid, crossed, wide = otm_puts[:3], otm_puts[3:5], otm_puts[5:7]
    quotes.loc[zero_bid, "bid"] = 0.0
    quotes.loc[crossed, "ask"] = quotes.loc[crossed, "bid"] - 0.10
    mid = quotes.loc[wide, "lastPrice"]
    quotes.loc[wide, "bid"], quotes.loc[wide, "ask"] = 0.5 * mid, 1.5 * mid
    planted = {"duplicates": len(duplicates), "untradeable": len(zero_bid) + len(crossed), "wide": len(wide)}
    return pd.concat([quotes, duplicates], ignore_index=True), planted
