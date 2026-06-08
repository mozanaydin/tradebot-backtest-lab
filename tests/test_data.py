from __future__ import annotations

import pandas as pd
import pytest

from tradebot_backtest.data import (
    FundingUnavailable,
    fetch_funding_history,
    normalize_candles,
)


def test_normalize_candles_parses_sorts_and_deduplicates_rows() -> None:
    raw = [
        {"t": 2000, "T": 2999, "s": "BTC", "i": "1h", "o": "101", "c": "102", "h": "103", "l": "100", "v": "2.5", "n": 10},
        {"t": 1000, "T": 1999, "s": "BTC", "i": "1h", "o": "99", "c": "100", "h": "101", "l": "98", "v": "1.5", "n": 8},
        {"t": 1000, "T": 1999, "s": "BTC", "i": "1h", "o": "99", "c": "100", "h": "101", "l": "98", "v": "1.5", "n": 8},
    ]

    candles = normalize_candles(raw)

    assert list(candles.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert candles["timestamp"].tolist() == [
        pd.Timestamp("1970-01-01 00:00:01", tz="UTC"),
        pd.Timestamp("1970-01-01 00:00:02", tz="UTC"),
    ]
    assert candles[["open", "high", "low", "close", "volume"]].dtypes.astype(str).tolist() == [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]


def test_normalize_candles_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        normalize_candles([{"t": 1000, "o": "1", "h": "2", "l": "0", "c": "1"}])


class FailingFundingClient:
    def post(self, *_args, **_kwargs):
        raise RuntimeError("rate limited")


def test_funding_fetch_failure_raises_typed_error() -> None:
    with pytest.raises(FundingUnavailable, match="rate limited"):
        fetch_funding_history(
            "BTC",
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
            client=FailingFundingClient(),
        )
