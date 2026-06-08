from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import tradebot_backtest.regime as regime_module
from tradebot_backtest.engine import Signal
from tradebot_backtest.regime import (
    RegimeParams,
    classify_regime,
    regime_features,
    regime_parameter_grid,
    regime_switching_signals,
)


def synthetic_candles(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    trend = np.linspace(100.0, 140.0, periods)
    oscillation = np.sin(np.arange(periods) / 5.0) * 2.0
    close = trend + oscillation
    open_ = close - np.cos(np.arange(periods) / 7.0) * 0.4
    spread = 1.0 + np.abs(np.sin(np.arange(periods) / 9.0))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.linspace(10.0, 20.0, periods),
        }
    )


def constant_range_candles(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.0] * periods,
            "volume": [10.0] * periods,
        }
    )


def zero_range_candles(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * periods,
            "high": [100.0] * periods,
            "low": [100.0] * periods,
            "close": [100.0] * periods,
            "volume": [0.0] * periods,
        }
    )


def params() -> RegimeParams:
    return RegimeParams(0.90, 25.0, 20.0, 0.50, 50, 40, 2.0)


def trend_candles(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = np.linspace(100.0, 220.0, periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [10.0] * periods,
        }
    )


def volatility_shock_candles(periods: int = 200) -> pd.DataFrame:
    candles = constant_range_candles(periods)
    candles.loc[periods - 1, ["open", "high", "low", "close"]] = [100.0, 131.0, 99.0, 130.0]
    return candles


def ranging_extreme_candles(periods: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = 100.0 + np.sin(np.arange(periods) * np.pi / 2)
    close[-1] = 96.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [10.0] * periods,
        }
    )


def trend_then_range_candles() -> pd.DataFrame:
    trend = trend_candles(220)
    timestamps = pd.date_range(trend["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=120, freq="h")
    close = 220.0 + np.sin(np.arange(120) * np.pi / 2)
    ranging = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [10.0] * 120,
        }
    )
    return pd.concat([trend, ranging], ignore_index=True)


def router_candles(periods: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100.0, 101.0, periods),
            "high": np.linspace(101.0, 102.0, periods),
            "low": np.linspace(99.0, 100.0, periods),
            "close": np.linspace(100.0, 101.0, periods),
            "volume": [10.0] * periods,
        }
    )


def router_features(candles: pd.DataFrame, regimes: list[str]) -> pd.DataFrame:
    values = {
        "high_volatility": (0.95, 30.0, 1.0),
        "trending": (0.50, 30.0, 1.0),
        "ranging": (0.50, 10.0, 0.0),
        "unclear": (0.50, 22.0, 0.0),
    }
    rows = [values[regime] for regime in regimes]
    return pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "atr14": [1.0] * len(candles),
            "adx14": [row[1] for row in rows],
            "ema50": candles["close"],
            "normalized_ema_slope": [row[2] for row in rows],
            "atr_percentile": [row[0] for row in rows],
        }
    )


def test_regime_params_are_frozen() -> None:
    configured = params()

    with pytest.raises(FrozenInstanceError):
        configured.trend_adx = 30.0  # type: ignore[misc]


def test_regime_features_produce_expected_columns_and_ranges() -> None:
    features = regime_features(synthetic_candles())

    assert list(features.columns) == [
        "timestamp",
        "atr14",
        "adx14",
        "ema50",
        "normalized_ema_slope",
        "atr_percentile",
    ]
    mature = features.dropna()
    assert not mature.empty
    assert mature["atr14"].gt(0).all()
    assert mature["adx14"].between(0, 100).all()
    assert mature["atr_percentile"].between(0, 1).all()


def test_regime_features_do_not_change_when_future_candles_change() -> None:
    candles = synthetic_candles()
    original = regime_features(candles)
    changed = candles.copy()
    changed.loc[changed.index[-3:], ["high", "low", "close"]] *= 10

    recalculated = regime_features(changed)

    pd.testing.assert_frame_equal(original.iloc[:-3], recalculated.iloc[:-3])


def test_constant_atr_uses_average_rank_and_is_not_high_volatility() -> None:
    latest = regime_features(constant_range_candles()).iloc[-1]

    assert latest["atr_percentile"] == pytest.approx(0.5)
    assert classify_regime(
        latest["atr_percentile"],
        latest["adx14"],
        latest["normalized_ema_slope"],
        params(),
    ) != "high_volatility"


def test_atr_percentile_compares_current_atr_to_previous_168_values() -> None:
    features = regime_features(constant_range_candles(200))

    assert features["atr_percentile"].first_valid_index() == 181


def test_flat_market_has_zero_adx_and_classifies_as_ranging() -> None:
    latest = regime_features(constant_range_candles()).iloc[-1]

    assert latest["adx14"] == pytest.approx(0.0)
    assert classify_regime(
        latest["atr_percentile"],
        latest["adx14"],
        latest["normalized_ema_slope"],
        params(),
    ) == "ranging"


def test_zero_range_market_has_zero_adx_and_classifies_as_ranging() -> None:
    latest = regime_features(zero_range_candles()).iloc[-1]

    assert latest["atr14"] == pytest.approx(0.0)
    assert latest["atr_percentile"] == pytest.approx(0.5)
    assert latest["adx14"] == pytest.approx(0.0)
    assert classify_regime(
        latest["atr_percentile"],
        latest["adx14"],
        latest["normalized_ema_slope"],
        params(),
    ) == "ranging"


def test_high_volatility_has_priority_over_trending() -> None:
    regime = classify_regime(
        atr_percentile=0.95,
        adx=40.0,
        normalized_slope=1.0,
        params=params(),
    )

    assert regime == "high_volatility"


def test_classifier_covers_trending_ranging_and_unclear() -> None:
    configured = params()

    assert classify_regime(0.50, 30.0, 0.75, configured) == "trending"
    assert classify_regime(0.50, 15.0, 0.10, configured) == "ranging"
    assert classify_regime(0.50, 22.0, 0.10, configured) == "unclear"


def test_classifier_thresholds_are_inclusive() -> None:
    configured = params()

    assert classify_regime(0.90, 40.0, 1.0, configured) == "high_volatility"
    assert classify_regime(0.50, 25.0, 0.50, configured) == "trending"
    assert classify_regime(0.50, 20.0, 0.49, configured) == "ranging"


def test_high_volatility_routes_breakout_at_half_exposure() -> None:
    configured = RegimeParams(0.80, 25.0, 20.0, 0.50, 5, 20, 1.5)

    signals = regime_switching_signals(volatility_shock_candles(), configured)
    entries = [signal for signal in signals if signal.side in {"long", "short"}]

    assert entries
    assert entries[-1].strategy_name == "regime_switching"
    assert entries[-1].entry_reason.startswith("high_volatility:")
    assert entries[-1].exposure_multiplier == 0.5


def test_trending_routes_breakout_at_full_exposure() -> None:
    configured = RegimeParams(0.90, 20.0, 15.0, 0.25, 5, 20, 1.5)

    signals = regime_switching_signals(trend_candles(), configured)
    entries = [signal for signal in signals if signal.side in {"long", "short"}]

    assert entries
    assert entries[-1].entry_reason.startswith("trending:")
    assert entries[-1].exposure_multiplier == 1.0


def test_ranging_routes_bollinger_reversion_reason() -> None:
    configured = RegimeParams(1.10, 101.0, 100.0, 10.0, 5, 10, 1.0)

    signals = regime_switching_signals(ranging_extreme_candles(), configured)
    entries = [signal for signal in signals if signal.side in {"long", "short"}]

    assert entries
    assert entries[-1].entry_reason.startswith("ranging:")
    assert "band_extreme" in entries[-1].entry_reason
    assert entries[-1].exposure_multiplier == 1.0


def test_regime_change_emits_one_exit_without_repeated_flats() -> None:
    candles = router_candles(4)
    configured = params()
    entry = Signal(
        candles.loc[0, "timestamp"],
        "breakout",
        {},
        "long",
        "break_high",
        99.0,
    )
    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(candles, ["trending", "ranging", "ranging", "ranging"]),
        component_signals={"breakout": [entry], "bollinger": []},
    )
    regime_change_exits = [
        signal for signal in signals if signal.side == "flat" and signal.entry_reason == "regime_change"
    ]

    assert len(regime_change_exits) == 1


def test_inactive_component_signals_are_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = router_candles(2)
    configured = params()
    breakout = Signal(
        candles["timestamp"].iloc[1],
        "breakout",
        {"lookback": 50},
        "long",
        "close_above_channel_high",
        99.0,
    )
    monkeypatch.setattr(
        regime_module,
        "regime_features",
        lambda _: router_features(candles, ["ranging", "ranging"]),
    )

    signals = regime_switching_signals(
        candles,
        configured,
        component_signals={"breakout": [breakout], "bollinger": []},
    )

    assert signals == []


def test_component_exit_and_opposite_signals_are_routed_and_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = router_candles(4)
    configured = params()
    timestamps = candles["timestamp"].tolist()
    component_signals = [
        Signal(timestamps[0], "breakout", {"lookback": 50}, "long", "break_high", 99.0),
        Signal(timestamps[1], "breakout", {"lookback": 50}, "flat", "component_exit", 100.0),
        Signal(timestamps[2], "breakout", {"lookback": 50}, "long", "break_high", 100.0),
        Signal(timestamps[3], "breakout", {"lookback": 50}, "short", "break_low", 102.0),
    ]
    monkeypatch.setattr(
        regime_module,
        "regime_features",
        lambda _: router_features(candles, ["trending"] * 4),
    )

    signals = regime_switching_signals(
        candles,
        configured,
        component_signals={"breakout": component_signals, "bollinger": []},
    )

    assert [signal.side for signal in signals] == ["long", "flat", "long", "short"]
    assert all(signal.strategy_name == "regime_switching" for signal in signals)
    assert [signal.entry_reason for signal in signals] == [
        "trending:break_high",
        "trending:component_exit",
        "trending:break_high",
        "trending:break_low",
    ]


def test_unclear_regime_closes_once_and_does_not_repeat_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = router_candles(4)
    configured = params()
    entry = Signal(
        candles["timestamp"].iloc[0],
        "breakout",
        {"lookback": 50},
        "long",
        "break_high",
        99.0,
    )
    monkeypatch.setattr(
        regime_module,
        "regime_features",
        lambda _: router_features(candles, ["trending", "unclear", "unclear", "unclear"]),
    )

    signals = regime_switching_signals(
        candles,
        configured,
        component_signals={"breakout": [entry], "bollinger": []},
    )

    assert [signal.side for signal in signals] == ["long", "flat"]
    assert signals[-1].entry_reason == "regime_change"
    assert signals[-1].strategy_name == "regime_switching"


def test_regime_parameter_grid_uses_exact_approved_values() -> None:
    grid = regime_parameter_grid()

    assert len(grid) == 96
    assert all(configured.range_adx < configured.trend_adx for configured in grid)
    assert {configured.high_volatility_percentile for configured in grid} == {0.80, 0.90}
    assert {configured.trend_adx for configured in grid} == {20, 25}
    assert {configured.range_adx for configured in grid} == {15, 20}
    assert {configured.slope_threshold for configured in grid} == {0.25, 0.50}
    assert {configured.donchian_lookback for configured in grid} == {24, 50}
    assert {configured.bollinger_length for configured in grid} == {20, 40}
    assert {configured.bollinger_entry_z for configured in grid} == {1.5, 2.0}


def test_suppressed_bollinger_entry_does_not_block_later_range_entry() -> None:
    candles = router_candles(3)
    timestamps = candles["timestamp"].tolist()
    configured = params()
    bollinger_candidates = [
        Signal(timestamps[0], "bollinger", {}, "long", "lower_band_extreme", 99.0),
        Signal(timestamps[2], "bollinger", {}, "long", "lower_band_extreme", 99.0),
    ]

    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(candles, ["trending", "ranging", "ranging"]),
        component_signals={"breakout": [], "bollinger": bollinger_candidates},
    )

    assert [signal.timestamp for signal in signals] == [timestamps[2]]
    assert signals[0].side == "long"
    assert signals[0].entry_reason == "ranging:lower_band_extreme"


def test_default_bollinger_candidates_remain_stateless_when_entries_are_suppressed() -> None:
    candles = router_candles(6)
    candles["close"] = [100.0, 100.0, 90.0, 100.0, 100.0, 90.0]
    candles["open"] = candles["close"]
    candles["high"] = candles["close"] + 1.0
    candles["low"] = candles["close"] - 1.0
    configured = RegimeParams(0.90, 25.0, 20.0, 0.50, 50, 3, 0.8)
    features = router_features(
        candles,
        ["trending", "trending", "trending", "trending", "trending", "ranging"],
    )

    signals = regime_switching_signals(candles, configured, features=features)

    assert [(signal.timestamp, signal.side, signal.entry_reason) for signal in signals] == [
        (candles["timestamp"].iloc[5], "long", "ranging:lower_band_extreme"),
    ]


def test_regime_change_collision_routes_new_entry_on_same_signal_candle() -> None:
    candles = router_candles(3)
    timestamps = candles["timestamp"].tolist()
    configured = params()
    breakout = Signal(timestamps[0], "breakout", {}, "long", "break_high", 99.0)
    bollinger = Signal(timestamps[1], "bollinger", {}, "short", "upper_band_extreme", 102.0)

    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(candles, ["trending", "ranging", "ranging"]),
        component_signals={"breakout": [breakout], "bollinger": [bollinger]},
    )

    assert [(signal.timestamp, signal.side, signal.entry_reason) for signal in signals] == [
        (timestamps[0], "long", "trending:break_high"),
        (timestamps[1], "short", "ranging:upper_band_extreme"),
    ]


def test_breakout_exposure_transitions_close_and_reenter_at_new_size() -> None:
    candles = router_candles(5)
    timestamps = candles["timestamp"].tolist()
    configured = params()
    breakout = Signal(timestamps[0], "breakout", {}, "long", "break_high", 99.0)

    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(
            candles,
            ["trending", "high_volatility", "high_volatility", "trending", "trending"],
        ),
        component_signals={"breakout": [breakout], "bollinger": []},
    )

    assert [(signal.timestamp, signal.side, signal.exposure_multiplier) for signal in signals] == [
        (timestamps[0], "long", 1.0),
        (timestamps[1], "long", 0.5),
        (timestamps[3], "long", 1.0),
    ]
    assert all(signal.entry_reason == "regime_resize" for signal in signals[1:])


def test_precomputed_features_produce_equivalent_signals() -> None:
    candles = trend_candles()
    configured = RegimeParams(0.90, 20.0, 15.0, 0.25, 5, 20, 1.5)
    computed = regime_switching_signals(candles, configured)

    reused = regime_switching_signals(
        candles,
        configured,
        features=regime_features(candles),
    )

    assert reused == computed


def test_router_resets_state_at_requested_start_timestamp() -> None:
    candles = router_candles(4)
    timestamps = candles["timestamp"].tolist()
    configured = params()
    entries = [
        Signal(timestamps[0], "breakout", {}, "long", "old_entry", 99.0),
        Signal(timestamps[2], "breakout", {}, "long", "test_entry", 99.0),
    ]

    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(candles, ["trending"] * 4),
        component_signals={"breakout": entries, "bollinger": []},
        start_at=timestamps[2],
    )

    assert [(signal.timestamp, signal.entry_reason) for signal in signals] == [
        (timestamps[2], "trending:test_entry")
    ]


def test_router_tracks_close_invalidation_and_allows_reentry() -> None:
    candles = router_candles(4)
    timestamps = candles["timestamp"].tolist()
    candles.loc[1, "close"] = 98.0
    configured = params()
    entries = [
        Signal(timestamps[0], "breakout", {}, "long", "first", 99.0),
        Signal(timestamps[2], "breakout", {}, "long", "second", 97.0),
    ]

    signals = regime_switching_signals(
        candles,
        configured,
        features=router_features(candles, ["trending"] * 4),
        component_signals={"breakout": entries, "bollinger": []},
    )

    assert [(signal.side, signal.entry_reason) for signal in signals] == [
        ("long", "trending:first"),
        ("flat", "invalidation_close"),
        ("long", "trending:second"),
    ]
