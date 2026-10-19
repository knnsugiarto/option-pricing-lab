"""Download and store option chain snapshots from Yahoo Finance (via yfinance).

Yahoo's data is for personal use only (see the yfinance README), so snapshots
are written under ``data/``, which git ignores, and are never committed. Each
snapshot is a CSV of quotes plus a JSON sidecar recording when and how it was
taken, so every later analysis can be re-run on exactly the same data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT_COLUMNS = [
    "contractSymbol",
    "option_type",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "lastTradeDate",
    "volume",
    "openInterest",
    "impliedVolatility",
]
_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def ticker_slug(ticker: str) -> str:
    """File-name-safe ticker, e.g. '^SPX' -> 'spx'."""
    return "".join(ch for ch in ticker.lower() if ch.isalnum())


def choose_expiry(expiries: Sequence[str], today: date, target_days: int, min_days: int) -> str:
    """Pick the listed expiry closest to ``target_days`` away, at least ``min_days`` away.

    Ties go to the earlier expiry. Raises ValueError if no expiry qualifies.
    """
    eligible = []
    for text in expiries:
        days = (date.fromisoformat(text) - today).days
        if days >= min_days:
            eligible.append((abs(days - target_days), days, text))
    if not eligible:
        raise ValueError(f"no expiry at least {min_days} days after {today} among {list(expiries)}")
    return min(eligible)[2]


def write_snapshot(quotes: pd.DataFrame, metadata: dict[str, Any], out_dir: Path) -> Path:
    """Write quotes to CSV and metadata to a JSON sidecar; return the CSV path."""
    missing = set(SNAPSHOT_COLUMNS) - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes are missing columns {sorted(missing)}")
    downloaded = datetime.fromisoformat(metadata["downloaded_at_utc"])
    stem = f"{ticker_slug(metadata['ticker'])}_{metadata['expiry']}_{downloaded.strftime(_STAMP_FORMAT)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    quotes[SNAPSHOT_COLUMNS].to_csv(csv_path, index=False)
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def latest_snapshot(directory: Path, ticker: str) -> Path | None:
    """Most recent snapshot CSV for a ticker, by the timestamp in its file name."""
    paths = sorted(directory.glob(f"{ticker_slug(ticker)}_*.csv"), key=lambda p: p.stem.rsplit("_", 1)[-1])
    return paths[-1] if paths else None


def _default_ticker_factory(ticker: str) -> Any:
    import yfinance  # imported here so the rest of the package works without network access

    return yfinance.Ticker(ticker)


def _yfinance_version() -> str:
    try:
        import yfinance

        return str(yfinance.__version__)
    except (ImportError, AttributeError):
        return "unknown"


def download_snapshot(
    ticker: str,
    target_days: int,
    min_days: int,
    out_dir: Path,
    *,
    ticker_factory: Callable[[str], Any] = _default_ticker_factory,
    now: datetime | None = None,
) -> Path:
    """Download one expiry's calls and puts and save them as a snapshot.

    Args:
        ticker: Yahoo ticker of the underlying, e.g. "^SPX" or "SPY".
        target_days, min_days: Expiry selection, see :func:`choose_expiry`.
        out_dir: Directory for the CSV and JSON files.
        ticker_factory: Builds the yfinance Ticker; replaced by a fake in tests.
        now: Download time (UTC); defaults to the current time.

    Returns:
        Path of the snapshot CSV.
    """
    now = now or datetime.now(UTC)
    source = ticker_factory(ticker)
    expiries = list(source.options)
    if not expiries:
        raise RuntimeError(
            f"Yahoo returned no option expiries for {ticker!r}. Check the network connection, "
            "or try ticker = \"SPY\" in configs/default.toml."
        )
    expiry = choose_expiry(expiries, now.date(), target_days, min_days)
    chain = source.option_chain(expiry)
    if chain.calls is None or chain.puts is None or chain.calls.empty or chain.puts.empty:
        raise RuntimeError(f"Yahoo returned an empty option chain for {ticker!r} expiring {expiry}")

    quotes = pd.concat([chain.calls.assign(option_type="call"), chain.puts.assign(option_type="put")], ignore_index=True)
    underlying = chain.underlying or {}
    market_time = underlying.get("regularMarketTime")
    metadata = {
        "ticker": ticker,
        "expiry": expiry,
        "downloaded_at_utc": now.isoformat(),
        # Time of the underlying's last quote. Outside US trading hours this is the
        # last close, which is the right clock for time to expiry, not the download time.
        "quote_time_utc": datetime.fromtimestamp(market_time, UTC).isoformat() if market_time else None,
        "spot": underlying.get("regularMarketPrice"),
        "source": "Yahoo Finance via yfinance (personal use only; not redistributed)",
        "yfinance_version": _yfinance_version(),
        "n_quotes": len(quotes),
    }
    return write_snapshot(quotes, metadata, out_dir)
