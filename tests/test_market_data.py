"""Tests for snapshot download and storage, with a fake Yahoo client (no network)."""

from __future__ import annotations

from collections import namedtuple
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from optlab import market_data

Chain = namedtuple("Chain", ["calls", "puts", "underlying"])
NOW = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
MARKET_CLOSE = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)


def yahoo_frame(kind: str, strikes: list[float]) -> pd.DataFrame:
    """A frame with the columns yfinance's option_chain returns."""
    letter = "C" if kind == "call" else "P"
    return pd.DataFrame(
        {
            "contractSymbol": [f"SPXW261120{letter}{int(k * 1000):08d}" for k in strikes],
            "lastTradeDate": pd.Timestamp(MARKET_CLOSE),
            "strike": strikes,
            "lastPrice": 10.0,
            "bid": 9.9,
            "ask": 10.1,
            "change": 0.0,
            "percentChange": 0.0,
            "volume": 5,
            "openInterest": 50,
            "impliedVolatility": 0.2,
            "inTheMoney": False,
            "contractSize": "REGULAR",
            "currency": "USD",
        }
    )


class FakeTicker:
    """Mimics the parts of yfinance.Ticker the downloader uses."""

    def __init__(self, ticker: str, expiries: tuple[str, ...] = ("2026-10-16", "2026-11-20", "2026-12-18")) -> None:
        self.ticker = ticker
        self.options = expiries
        self.requested: list[str] = []

    def option_chain(self, expiry: str) -> Chain:
        self.requested.append(expiry)
        strikes = [4900.0, 5000.0, 5100.0]
        underlying = {"regularMarketPrice": 5001.5, "regularMarketTime": int(MARKET_CLOSE.timestamp())}
        return Chain(yahoo_frame("call", strikes), yahoo_frame("put", strikes), underlying)


def test_download_writes_snapshot_with_quote_time(tmp_path: Path) -> None:
    fake = FakeTicker("^SPX")
    path = market_data.download_snapshot("^SPX", target_days=45, min_days=14, out_dir=tmp_path, ticker_factory=lambda _: fake, now=NOW)
    assert fake.requested == ["2026-11-20"]
    assert path.name == "spx_2026-11-20_20261010T080000Z.csv"
    quotes = pd.read_csv(path)
    assert list(quotes.columns) == market_data.SNAPSHOT_COLUMNS
    assert sorted(quotes["option_type"].unique()) == ["call", "put"] and len(quotes) == 6
    meta = path.with_suffix(".json").read_text(encoding="utf-8")
    assert '"quote_time_utc": "2026-10-09T20:00:00+00:00"' in meta
    assert '"spot": 5001.5' in meta


def test_download_fails_loudly_without_expiries(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no option expiries"):
        market_data.download_snapshot(
            "^SPX", 45, 14, tmp_path, ticker_factory=lambda t: FakeTicker(t, expiries=()), now=NOW
        )


@pytest.mark.parametrize(
    ("target", "minimum", "expected"),
    [(45, 14, "2026-11-20"), (7, 14, "2026-11-20"), (7, 1, "2026-10-16"), (90, 14, "2026-12-18")],
)
def test_choose_expiry(target: int, minimum: int, expected: str) -> None:
    expiries = ["2026-10-16", "2026-11-20", "2026-12-18"]
    assert market_data.choose_expiry(expiries, date(2026, 10, 10), target, minimum) == expected


def test_choose_expiry_raises_when_none_qualify() -> None:
    with pytest.raises(ValueError):
        market_data.choose_expiry(["2026-10-16"], date(2026, 10, 10), 45, 14)


def test_latest_snapshot_uses_timestamp_not_expiry(tmp_path: Path) -> None:
    for name in ("spx_2026-12-18_20261009T080000Z.csv", "spx_2026-11-20_20261010T080000Z.csv", "spy_2026-11-20_20261011T080000Z.csv"):
        (tmp_path / name).write_text("", encoding="utf-8")
    latest = market_data.latest_snapshot(tmp_path, "^SPX")
    assert latest is not None and latest.name == "spx_2026-11-20_20261010T080000Z.csv"
    assert market_data.latest_snapshot(tmp_path, "QQQ") is None


def test_ticker_slug() -> None:
    assert market_data.ticker_slug("^SPX") == "spx"
    assert market_data.ticker_slug("BRK-B") == "brkb"
