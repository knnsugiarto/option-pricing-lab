"""Tests for the smile pipeline, on synthetic chains where the right answer is known.

The synthetic chain is priced with Black-76 from a known forward, discount
factor and smile, then seeded with the kinds of bad quotes real chains have.
The pipeline must remove exactly the bad quotes and recover the inputs.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from synthetic_chain import (
    EXPIRY,
    FORWARD,
    PIPELINE,
    QUOTE_TIME,
    RATE,
    SETTLEMENT,
    STRIKES,
    TZ,
    clean_chain,
    dirty_chain,
    snapshot_of,
    true_vol,
)

from optlab import market_data, smile


def test_pipeline_removes_planted_quotes_and_recovers_the_smile() -> None:
    quotes, planted = dirty_chain()
    analysis = smile.analyse_snapshot(snapshot_of(quotes), **PIPELINE)
    counts = [step.remaining for step in analysis.funnel]
    n_clean = 2 * len(STRIKES)
    assert counts[0] == n_clean + planted["duplicates"]
    assert counts[1] == n_clean
    assert counts[2] == n_clean - planted["untradeable"]
    assert counts[3] == n_clean - planted["untradeable"] - planted["wide"]
    assert analysis.root == "SPXW"

    assert analysis.forward.forward == pytest.approx(FORWARD, rel=1e-10)
    assert analysis.forward.implied_rate == pytest.approx(RATE, abs=1e-8)
    assert analysis.forward.residual_std == pytest.approx(0.0, abs=1e-8)

    s = analysis.smile
    assert set(s.loc[s["option_type"] == "put", "strike"]) <= set(STRIKES[STRIKES < FORWARD])
    assert set(s.loc[s["option_type"] == "call", "strike"]) <= set(STRIKES[STRIKES >= FORWARD])
    np.testing.assert_allclose(s["iv_mid"], true_vol(s["log_moneyness"].to_numpy()), atol=1e-8)
    assert np.all(s["iv_bid"] < s["iv_mid"]) and np.all(s["iv_mid"] < s["iv_ask"])
    # Interpolating between strikes 25 points apart around the forward.
    assert analysis.summary.atm_vol == pytest.approx(true_vol(0.0), abs=1e-5)


def test_forward_estimate_ignores_a_stale_quote() -> None:
    """Theil–Sen tolerates a bad quote that passes the spread filter; least squares would not."""
    quotes = clean_chain()
    stale = quotes.index[(quotes["option_type"] == "call") & (quotes["strike"] == 5025.0)]
    quotes.loc[stale, ["bid", "ask"]] += 3.0
    analysis = smile.analyse_snapshot(snapshot_of(quotes), **PIPELINE)
    assert analysis.forward.forward == pytest.approx(FORWARD, rel=1e-6)
    assert analysis.forward.implied_rate == pytest.approx(RATE, abs=1e-4)


def test_summary_against_hand_computed_values() -> None:
    table = pd.DataFrame(
        {
            "log_moneyness": [-0.10, -0.05, 0.00, 0.05, 0.10],
            "iv_mid": [0.25, 0.20, 0.18, 0.17, 0.165],
            "iv_bid": [0.24, 0.17, 0.17, 0.16, 0.15],
            "iv_ask": [0.26, 0.21, 0.19, 0.175, 0.17],
        }
    )
    summary = smile.summarise_smile(table, maturity=1.0, window_sd=1.0)  # window |k| <= 0.18
    assert summary.atm_vol == pytest.approx(0.18)
    deviations = np.array([0.07, 0.02, 0.0, -0.01, -0.015])
    assert summary.rmse_vs_flat_vol_points == pytest.approx(100 * np.sqrt(np.mean(deviations**2)))
    assert summary.max_abs_dev_vol_points == pytest.approx(7.0)
    # Bands [0.24, 0.26], [0.16, 0.175] and [0.15, 0.17] exclude 0.18; the other two contain it.
    assert summary.share_flat_outside_spread == pytest.approx(0.6)
    assert summary.n_in_window == 5
    assert math.isnan(summary.put_wing_vol)  # -0.18 is beyond the lowest quoted strike


def test_smile_needs_quotes_on_both_sides() -> None:
    table = pd.DataFrame({"log_moneyness": [0.01, 0.02, 0.03], "iv_mid": [0.2] * 3, "iv_bid": [0.19] * 3, "iv_ask": [0.21] * 3})
    with pytest.raises(ValueError, match="both sides"):
        smile.summarise_smile(table, maturity=0.5, window_sd=2.0)


@pytest.mark.parametrize(
    ("quote_time", "expiry"),
    [
        (datetime(2026, 11, 19, 21, 0, tzinfo=UTC), date(2026, 11, 20)),  # 16:00 EST is 21:00 UTC
        (datetime(2026, 10, 29, 20, 0, tzinfo=UTC), date(2026, 10, 30)),  # 16:00 EDT is 20:00 UTC
    ],
)
def test_years_to_expiry_handles_daylight_saving(quote_time: datetime, expiry: date) -> None:
    assert smile.years_to_expiry(quote_time, expiry, SETTLEMENT, TZ) == pytest.approx(1.0 / 365.0)


def test_years_to_expiry_rejects_expired_and_naive_times() -> None:
    with pytest.raises(ValueError):
        smile.years_to_expiry(datetime(2026, 11, 21, tzinfo=UTC), EXPIRY, SETTLEMENT, TZ)
    with pytest.raises(ValueError):
        smile.years_to_expiry(datetime(2026, 11, 1), EXPIRY, SETTLEMENT, TZ)


def test_contract_root() -> None:
    assert smile.contract_root("SPXW261120C05000000") == "SPXW"
    assert smile.contract_root("SPY261120P00450000") == "SPY"
    with pytest.raises(ValueError):
        smile.contract_root("not-a-symbol")


def test_select_root_falls_back_to_most_common() -> None:
    quotes = pd.DataFrame({"contractSymbol": ["SPY261120C00450000", "SPY261120P00450000", "XYZ261120C00010000"]})
    kept, root = smile.select_root(quotes, preferred_root="SPXW")
    assert root == "SPY" and len(kept) == 2


def test_snapshot_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    metadata = {
        "ticker": "^SPX",
        "expiry": EXPIRY.isoformat(),
        "downloaded_at_utc": "2026-10-10T08:00:00+00:00",
        "quote_time_utc": QUOTE_TIME.isoformat(),
        "spot": 4990.0,
    }
    path = market_data.write_snapshot(clean_chain(), metadata, tmp_path)
    loaded = smile.load_snapshot(path)
    assert loaded.quote_time == QUOTE_TIME  # the quote time, not the download time
    assert loaded.expiry == EXPIRY and loaded.spot == 4990.0
    analysis = smile.analyse_snapshot(loaded, **PIPELINE)
    assert analysis.forward.forward == pytest.approx(FORWARD, rel=1e-10)
