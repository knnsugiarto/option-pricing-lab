"""Download one option chain snapshot and report its quote quality.

Example:
    python scripts/download_chain.py
    python scripts/download_chain.py --ticker SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from optlab import market_data
from optlab.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--ticker", help="override the ticker in the config")
    args = parser.parse_args()
    market = load_config(args.config)["market"]
    ticker = args.ticker or market["ticker"]

    try:
        path = market_data.download_snapshot(ticker, market["target_days"], market["min_days"], ROOT / "data" / "raw")
    except Exception as error:  # network failures surface as many different exception types
        print(f"Download failed: {error}", file=sys.stderr)
        print("Check your internet connection and that finance.yahoo.com opens in a browser.", file=sys.stderr)
        return 1
    quotes = pd.read_csv(path)
    two_sided = ((quotes["bid"] > 0) & (quotes["ask"] > quotes["bid"])).sum()
    print(f"Saved {path.relative_to(ROOT)}")
    print(f"{len(quotes)} quotes ({(quotes['option_type'] == 'call').sum()} calls, {(quotes['option_type'] == 'put').sum()} puts)")
    print(f"{two_sided} have a two-sided market (bid > 0 and ask > bid)")
    if two_sided < 0.2 * len(quotes):
        print(
            "Warning: few two-sided quotes. Yahoo often shows zero bids and asks outside US market hours\n"
            "(09:30-16:00 New York time). Download again while the market is open.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
