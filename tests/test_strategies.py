from __future__ import annotations

import pandas as pd

from tradebot_backtest.strategies import (
    bollinger_regime_reversion_signals,
    breakout_signals,
    compression_breakout_signals,
    ema_crossover_signals,
    funding_conditioned_momentum_signals,
    funding_crowding_reversal_signals,
    rsi_mean_reversion_signals,
    strategy_grid,
    volatility_scaled_momentum_signals,
)


def strategy_candles(closes: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def test_ema_crossover_generates_long_and_short_signals() -> None:
    candles = strategy_candles([10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8])

    signals = ema_crossover_signals(candles, fast=2, slow=3)

    assert any(signal.side == "long" for signal in signals)
    assert any(signal.side == "short" for signal in signals)
    assert all(signal.strategy_name == "ema_crossover" for signal in signals)


def test_rsi_mean_reversion_generates_recovery_signal() -> None:
    candles = strategy_candles([100, 96, 92, 88, 84, 86, 88, 90, 92, 94, 96, 98])

    signals = rsi_mean_reversion_signals(candles, length=3, lower=35, upper=65, invalidation_window=3)

    assert any(signal.side == "long" for signal in signals)
    assert all("rsi" in signal.entry_reason for signal in signals if signal.side != "flat")


def test_breakout_generates_channel_break_signals() -> None:
    candles = strategy_candles([100, 101, 102, 103, 104, 98, 97, 96, 105])

    signals = breakout_signals(candles, lookback=3)

    assert any(signal.side == "long" for signal in signals)
    assert any(signal.side == "short" for signal in signals)
    assert all(signal.strategy_name == "breakout" for signal in signals)


def test_volatility_scaled_momentum_generates_directional_signals() -> None:
    candles = strategy_candles([100, 101, 102, 103, 104, 105, 103, 101, 99, 97])

    signals = volatility_scaled_momentum_signals(candles, lookback=3, atr_length=3, atr_multiplier=2.0)

    assert any(signal.side == "long" for signal in signals)
    assert any(signal.side == "short" for signal in signals)
    assert all(signal.invalidation_price > 0 for signal in signals)


def test_bollinger_regime_reversion_enters_at_extremes_in_quiet_regime() -> None:
    candles = strategy_candles([100] * 20 + [90, 100, 110, 100])

    signals = bollinger_regime_reversion_signals(
        candles,
        length=20,
        entry_z=1.5,
        max_trend_strength=10.0,
        atr_length=3,
    )

    assert any(signal.side == "long" for signal in signals)
    assert any(signal.side == "short" for signal in signals)


def test_compression_breakout_requires_range_break_and_volume_confirmation() -> None:
    candles = strategy_candles([100] * 40 + [101, 102, 103])
    candles.loc[:39, "high"] = 100.0
    candles.loc[:39, "low"] = 100.0
    candles.loc[40:, "volume"] = 10.0

    signals = compression_breakout_signals(
        candles,
        lookback=20,
        bandwidth_window=20,
        compression_quantile=0.5,
        volume_multiplier=2.0,
        atr_length=3,
    )

    assert any(signal.side == "long" for signal in signals)


def test_compression_breakout_emits_exit_when_atr_trailing_level_breaks() -> None:
    candles = strategy_candles([100] * 40 + [105, 110, 100])
    candles.loc[:39, "high"] = 100.0
    candles.loc[:39, "low"] = 100.0
    candles.loc[40:, "volume"] = 10.0

    signals = compression_breakout_signals(
        candles,
        lookback=20,
        bandwidth_window=20,
        compression_quantile=0.5,
        volume_multiplier=2.0,
        atr_length=3,
    )

    assert any(signal.side == "long" for signal in signals)
    assert any(signal.side == "flat" and signal.entry_reason == "atr_trailing_exit" for signal in signals)


def test_funding_crowding_reversal_fades_extreme_funding_after_price_reversal() -> None:
    candles = strategy_candles([100, 101, 102, 103, 102, 101, 100, 99])
    candles["funding_rate"] = [0.0, 0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0]
    candles["premium"] = [0.0, 0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0]

    signals = funding_crowding_reversal_signals(candles, z_window=3, entry_z=1.0, invalidation_window=3)

    assert any(signal.side == "short" for signal in signals)


def test_funding_conditioned_momentum_rejects_crowded_long_but_allows_short() -> None:
    candles = strategy_candles([100, 101, 102, 103, 104, 103, 101, 99])
    candles["funding_rate"] = [0.0, 0.0, 0.01, 0.02, 0.03, 0.02, 0.03, 0.02]
    candles["premium"] = 0.0

    signals = funding_conditioned_momentum_signals(
        candles,
        lookback=2,
        funding_window=3,
        max_crowding_z=0.5,
        atr_length=3,
        atr_multiplier=2.0,
    )

    assert not any(signal.side == "long" and signal.timestamp == candles.loc[4, "timestamp"] for signal in signals)
    assert any(signal.side == "short" for signal in signals)


def test_strategy_grid_contains_all_eight_families() -> None:
    families = {name for name, _ in strategy_grid(include_funding=True)}

    assert families == {
        "ema_crossover",
        "rsi_mean_reversion",
        "breakout",
        "volatility_scaled_momentum",
        "bollinger_regime_reversion",
        "compression_breakout",
        "funding_crowding_reversal",
        "funding_conditioned_momentum",
    }
