from __future__ import annotations

import pandas as pd
import pytest

from tradebot_backtest.data import (
    FundingUnavailable,
    fetch_candles,
    fetch_funding_history,
    normalize_binance_funding_history,
    normalize_binance_klines,
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


def test_normalize_binance_klines_parses_arrays_into_ohlcv_rows() -> None:
    raw = [
        [2000, "101", "103", "100", "102", "2.5", 2999, "0", 0, "0", "0", "0"],
        [1000, "99", "101", "98", "100", "1.5", 1999, "0", 0, "0", "0", "0"],
        [1000, "99", "101", "98", "100", "1.5", 1999, "0", 0, "0", "0", "0"],
    ]

    candles = normalize_binance_klines(raw)

    assert list(candles.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert candles["timestamp"].tolist() == [
        pd.Timestamp("1970-01-01 00:00:01", tz="UTC"),
        pd.Timestamp("1970-01-01 00:00:02", tz="UTC"),
    ]
    assert candles.iloc[0]["close"] == 100.0
    assert candles.iloc[1]["volume"] == 2.5


def test_normalize_binance_funding_history_uses_mark_price_for_premium_proxy() -> None:
    raw = [
        {"fundingTime": 7200000, "fundingRate": "0.0001", "markPrice": "105.0"},
        {"fundingTime": 0, "fundingRate": "-0.0002", "markPrice": "95.0"},
        {"fundingTime": 0, "fundingRate": "-0.0002", "markPrice": "95.0"},
    ]

    funding = normalize_binance_funding_history(raw)

    assert funding["timestamp"].tolist() == [
        pd.Timestamp("1970-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("1970-01-01 02:00:00", tz="UTC"),
    ]
    assert funding["funding_rate"].tolist() == [-0.0002, 0.0001]
    assert funding["mark_price"].tolist() == [95.0, 105.0]


class FailingFundingClient:
    def post(self, *_args, **_kwargs):
        raise RuntimeError("rate limited")

    def get(self, *_args, **_kwargs):
        raise RuntimeError("rate limited")


def test_funding_fetch_failure_raises_typed_error() -> None:
    with pytest.raises(FundingUnavailable, match="rate limited"):
        fetch_funding_history(
            "hyperliquid",
            "BTC",
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
            client=FailingFundingClient(),
        )


def test_binance_candle_fetch_failure_raises_provider_message() -> None:
    with pytest.raises(RuntimeError, match="rate limited"):
        fetch_candles(
            "binance",
            "BTCUSDT",
            "1h",
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
            client=FailingFundingClient(),
        )
