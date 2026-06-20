from __future__ import annotations

import pandas as pd

from tradebot_backtest.engine import Signal

Params = dict[str, int | float]


def compression_breakout_exit_variant_signals(
    candles: pd.DataFrame,
    *,
    setup: str,
    lookback: int,
    bandwidth_window: int,
    compression_quantile: float,
    volume_multiplier: float,
    atr_length: int = 14,
    exit_style: str = "atr_trail",
    retest_atr_buffer: float = 0.75,
    retest_window_bars: int = 3,
    profit_take_r: float = 1.0,
    reduced_exposure: float = 0.5,
    post_take_trailing_atr: float = 0.5,
    max_hold_bars: int = 3,
) -> list[Signal]:
    if setup == "breakout":
        base_signals = compression_breakout_signals(
            candles,
            lookback=lookback,
            bandwidth_window=bandwidth_window,
            compression_quantile=compression_quantile,
            volume_multiplier=volume_multiplier,
            atr_length=atr_length,
        )
        strategy_name = "compression_breakout"
    elif setup == "retest":
        base_signals = compression_breakout_retest_signals(
            candles,
            lookback=lookback,
            bandwidth_window=bandwidth_window,
            compression_quantile=compression_quantile,
            volume_multiplier=volume_multiplier,
            atr_length=atr_length,
            retest_atr_buffer=retest_atr_buffer,
            retest_window_bars=retest_window_bars,
        )
        strategy_name = "compression_breakout_retest"
    else:
        raise ValueError(f"unsupported compression setup: {setup}")
    params: Params = {
        "setup": setup,
        "lookback": lookback,
        "bandwidth_window": bandwidth_window,
        "compression_quantile": compression_quantile,
        "volume_multiplier": volume_multiplier,
        "atr_length": atr_length,
        "exit_style": exit_style,
        "profit_take_r": profit_take_r,
        "reduced_exposure": reduced_exposure,
        "post_take_trailing_atr": post_take_trailing_atr,
        "max_hold_bars": max_hold_bars,
        "retest_atr_buffer": retest_atr_buffer,
        "retest_window_bars": retest_window_bars,
    }
    return _apply_compression_exit_style(
        candles,
        base_signals,
        strategy_name=strategy_name,
        params=params,
        exit_style=exit_style,
        profit_take_r=profit_take_r,
        reduced_exposure=reduced_exposure,
        post_take_trailing_atr=post_take_trailing_atr,
        max_hold_bars=max_hold_bars,
    )


def ema_crossover_signals(candles: pd.DataFrame, fast: int, slow: int) -> list[Signal]:
    frame = candles.copy()
    frame["fast"] = frame["close"].ewm(span=fast, adjust=False).mean()
    frame["slow"] = frame["close"].ewm(span=slow, adjust=False).mean()
    signals: list[Signal] = []
    params = {"fast": fast, "slow": slow}
    for idx in range(1, len(frame)):
        prev = frame.iloc[idx - 1]
        row = frame.iloc[idx]
        if prev["fast"] <= prev["slow"] and row["fast"] > row["slow"]:
            signals.append(Signal(row["timestamp"], "ema_crossover", params, "long", "bullish_ema_cross", float(row["slow"])))
        elif prev["fast"] >= prev["slow"] and row["fast"] < row["slow"]:
            signals.append(Signal(row["timestamp"], "ema_crossover", params, "short", "bearish_ema_cross", float(row["slow"])))
    return signals


def rsi_mean_reversion_signals(
    candles: pd.DataFrame,
    length: int,
    lower: int,
    upper: int,
    invalidation_window: int = 20,
) -> list[Signal]:
    frame = candles.copy()
    frame["rsi"] = _rsi(frame["close"], length)
    frame["rolling_low"] = frame["close"].rolling(invalidation_window, min_periods=1).min()
    frame["rolling_high"] = frame["close"].rolling(invalidation_window, min_periods=1).max()
    params = {"length": length, "lower": lower, "upper": upper}
    signals: list[Signal] = []
    for idx in range(1, len(frame)):
        prev = frame.iloc[idx - 1]
        row = frame.iloc[idx]
        if prev["rsi"] <= lower and row["rsi"] > lower:
            signals.append(Signal(row["timestamp"], "rsi_mean_reversion", params, "long", "rsi_recovered_from_oversold", float(row["rolling_low"])))
        elif prev["rsi"] >= upper and row["rsi"] < upper:
            signals.append(Signal(row["timestamp"], "rsi_mean_reversion", params, "short", "rsi_fell_from_overbought", float(row["rolling_high"])))
        elif prev["rsi"] < 50 <= row["rsi"] or prev["rsi"] > 50 >= row["rsi"]:
            signals.append(Signal(row["timestamp"], "rsi_mean_reversion", params, "flat", "rsi_neutral_exit", float(row["close"])))
    return signals


def breakout_signals(candles: pd.DataFrame, lookback: int) -> list[Signal]:
    frame = candles.copy()
    frame["channel_high"] = frame["close"].shift(1).rolling(lookback, min_periods=lookback).max()
    frame["channel_low"] = frame["close"].shift(1).rolling(lookback, min_periods=lookback).min()
    params = {"lookback": lookback}
    signals: list[Signal] = []
    for _, row in frame.dropna(subset=["channel_high", "channel_low"]).iterrows():
        if row["close"] > row["channel_high"]:
            signals.append(Signal(row["timestamp"], "breakout", params, "long", "close_above_channel_high", float(row["channel_high"])))
        elif row["close"] < row["channel_low"]:
            signals.append(Signal(row["timestamp"], "breakout", params, "short", "close_below_channel_low", float(row["channel_low"])))
    return signals


def breakout_with_funding_veto_signals(
    candles: pd.DataFrame,
    lookback: int,
    funding_window: int,
    max_adverse_funding_z: float,
) -> list[Signal]:
    if "funding_rate" not in candles.columns:
        raise ValueError("breakout funding veto strategy requires funding_rate column")
    frame = candles.copy()
    frame["channel_high"] = frame["close"].shift(1).rolling(lookback, min_periods=lookback).max()
    frame["channel_low"] = frame["close"].shift(1).rolling(lookback, min_periods=lookback).min()
    frame["funding_z"] = _rolling_zscore(frame["funding_rate"], funding_window)
    params: Params = {
        "lookback": lookback,
        "funding_window": funding_window,
        "max_adverse_funding_z": max_adverse_funding_z,
    }
    signals: list[Signal] = []
    for _, row in frame.dropna(subset=["channel_high", "channel_low", "funding_z"]).iterrows():
        if row["close"] > row["channel_high"] and row["funding_z"] <= max_adverse_funding_z:
            signals.append(
                Signal(row["timestamp"], "breakout_funding_veto", params, "long", "breakout_allowed_by_funding", float(row["channel_high"]))
            )
        elif row["close"] < row["channel_low"] and row["funding_z"] >= -max_adverse_funding_z:
            signals.append(
                Signal(row["timestamp"], "breakout_funding_veto", params, "short", "breakout_allowed_by_funding", float(row["channel_low"]))
            )
    return signals


def volatility_scaled_momentum_signals(
    candles: pd.DataFrame,
    lookback: int,
    atr_length: int,
    atr_multiplier: float,
) -> list[Signal]:
    frame = candles.copy()
    frame["momentum"] = frame["close"].pct_change(lookback)
    frame["atr"] = _atr(frame, atr_length)
    params: Params = {
        "lookback": lookback,
        "atr_length": atr_length,
        "atr_multiplier": atr_multiplier,
    }
    signals: list[Signal] = []
    previous_side = "flat"
    for _, row in frame.dropna(subset=["momentum", "atr"]).iterrows():
        side = "long" if row["momentum"] > 0 else "short" if row["momentum"] < 0 else "flat"
        if side == previous_side:
            continue
        if side == "long":
            invalidation = float(row["close"] - atr_multiplier * row["atr"])
            reason = "positive_momentum"
        elif side == "short":
            invalidation = float(row["close"] + atr_multiplier * row["atr"])
            reason = "negative_momentum"
        else:
            invalidation = float(row["close"])
            reason = "momentum_neutral"
        signals.append(Signal(row["timestamp"], "volatility_scaled_momentum", params, side, reason, invalidation))
        previous_side = side
    return signals


def bollinger_regime_reversion_signals(
    candles: pd.DataFrame,
    length: int,
    entry_z: float,
    max_trend_strength: float,
    atr_length: int = 14,
) -> list[Signal]:
    frame = candles.copy()
    frame["mean"] = frame["close"].rolling(length).mean()
    frame["std"] = frame["close"].rolling(length).std(ddof=0)
    frame["z"] = (frame["close"] - frame["mean"]) / frame["std"].replace(0, pd.NA)
    frame["atr"] = _atr(frame, atr_length)
    fast = frame["close"].ewm(span=max(3, length // 2), adjust=False).mean()
    slow = frame["close"].ewm(span=length, adjust=False).mean()
    frame["trend_strength"] = (fast - slow).abs() / frame["atr"].replace(0, pd.NA)
    params: Params = {
        "length": length,
        "entry_z": entry_z,
        "max_trend_strength": max_trend_strength,
        "atr_length": atr_length,
    }
    signals: list[Signal] = []
    active_side = "flat"
    for _, row in frame.dropna(subset=["z", "atr", "trend_strength"]).iterrows():
        if active_side == "flat" and row["trend_strength"] <= max_trend_strength:
            if row["z"] <= -entry_z:
                signals.append(
                    Signal(
                        row["timestamp"],
                        "bollinger_regime_reversion",
                        params,
                        "long",
                        "lower_band_extreme",
                        float(row["close"] - 2 * row["atr"]),
                    )
                )
                active_side = "long"
            elif row["z"] >= entry_z:
                signals.append(
                    Signal(
                        row["timestamp"],
                        "bollinger_regime_reversion",
                        params,
                        "short",
                        "upper_band_extreme",
                        float(row["close"] + 2 * row["atr"]),
                    )
                )
                active_side = "short"
        elif active_side == "long" and row["z"] >= 0:
            signals.append(Signal(row["timestamp"], "bollinger_regime_reversion", params, "flat", "mean_reached", float(row["close"])))
            active_side = "flat"
        elif active_side == "short" and row["z"] <= 0:
            signals.append(Signal(row["timestamp"], "bollinger_regime_reversion", params, "flat", "mean_reached", float(row["close"])))
            active_side = "flat"
    return signals


def pullback_in_trend_signals(
    candles: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    rsi_length: int,
    recovery_level: int,
    invalidation_window: int = 10,
) -> list[Signal]:
    frame = candles.copy()
    frame["fast"] = frame["close"].ewm(span=fast_length, adjust=False).mean()
    frame["slow"] = frame["close"].ewm(span=slow_length, adjust=False).mean()
    frame["rsi"] = _rsi(frame["close"], rsi_length)
    frame["rolling_low"] = frame["close"].rolling(invalidation_window, min_periods=1).min()
    frame["rolling_high"] = frame["close"].rolling(invalidation_window, min_periods=1).max()
    params: Params = {
        "fast_length": fast_length,
        "slow_length": slow_length,
        "rsi_length": rsi_length,
        "recovery_level": recovery_level,
        "invalidation_window": invalidation_window,
    }
    signals: list[Signal] = []
    active_side = "flat"
    for idx in range(1, len(frame)):
        previous = frame.iloc[idx - 1]
        row = frame.iloc[idx]
        in_uptrend = row["fast"] > row["slow"] and row["close"] >= row["slow"]
        in_downtrend = row["fast"] < row["slow"] and row["close"] <= row["slow"]
        reclaimed_fast = (previous["close"] <= previous["fast"] or previous["low"] <= previous["fast"]) and row["close"] > row["fast"]
        lost_fast = (previous["close"] >= previous["fast"] or previous["high"] >= previous["fast"]) and row["close"] < row["fast"]
        rsi_recovered_long = previous["rsi"] < recovery_level and row["rsi"] >= recovery_level
        short_recovery = 100 - recovery_level
        rsi_recovered_short = previous["rsi"] > short_recovery and row["rsi"] <= short_recovery

        if active_side == "flat" and in_uptrend and reclaimed_fast and rsi_recovered_long:
            signals.append(
                Signal(
                    row["timestamp"],
                    "pullback_in_trend",
                    params,
                    "long",
                    "uptrend_pullback_reclaimed_fast_ema",
                    float(row["rolling_low"]),
                )
            )
            active_side = "long"
        elif active_side == "flat" and in_downtrend and lost_fast and rsi_recovered_short:
            signals.append(
                Signal(
                    row["timestamp"],
                    "pullback_in_trend",
                    params,
                    "short",
                    "downtrend_pullback_lost_fast_ema",
                    float(row["rolling_high"]),
                )
            )
            active_side = "short"
        elif active_side == "long" and (row["close"] < row["slow"] or row["close"] < row["fast"]):
            signals.append(Signal(row["timestamp"], "pullback_in_trend", params, "flat", "trend_failed", float(row["close"])))
            active_side = "flat"
        elif active_side == "short" and (row["close"] > row["slow"] or row["close"] > row["fast"]):
            signals.append(Signal(row["timestamp"], "pullback_in_trend", params, "flat", "trend_failed", float(row["close"])))
            active_side = "flat"
    return signals


def compression_breakout_signals(
    candles: pd.DataFrame,
    lookback: int,
    bandwidth_window: int,
    compression_quantile: float,
    volume_multiplier: float,
    atr_length: int = 14,
) -> list[Signal]:
    frame = candles.copy()
    mean = frame["close"].rolling(lookback).mean()
    std = frame["close"].rolling(lookback).std(ddof=0)
    frame["bandwidth"] = (4 * std) / mean.replace(0, pd.NA)
    threshold = frame["bandwidth"].rolling(bandwidth_window, min_periods=max(5, bandwidth_window // 2)).quantile(compression_quantile)
    frame["was_compressed"] = (frame["bandwidth"] <= threshold).shift(1).fillna(False)
    frame["channel_high"] = frame["high"].shift(1).rolling(lookback).max()
    frame["channel_low"] = frame["low"].shift(1).rolling(lookback).min()
    frame["average_volume"] = frame["volume"].shift(1).rolling(lookback).mean()
    frame["atr"] = _atr(frame, atr_length)
    params: Params = {
        "lookback": lookback,
        "bandwidth_window": bandwidth_window,
        "compression_quantile": compression_quantile,
        "volume_multiplier": volume_multiplier,
        "atr_length": atr_length,
    }
    signals: list[Signal] = []
    active_side = "flat"
    trailing_stop: float | None = None
    for _, row in frame.dropna(subset=["channel_high", "channel_low", "average_volume", "atr"]).iterrows():
        if active_side == "long":
            trailing_stop = max(float(trailing_stop), float(row["close"] - 2 * row["atr"]))
            if row["close"] <= trailing_stop:
                signals.append(
                    Signal(row["timestamp"], "compression_breakout", params, "flat", "atr_trailing_exit", trailing_stop)
                )
                active_side = "flat"
                trailing_stop = None
            continue
        if active_side == "short":
            trailing_stop = min(float(trailing_stop), float(row["close"] + 2 * row["atr"]))
            if row["close"] >= trailing_stop:
                signals.append(
                    Signal(row["timestamp"], "compression_breakout", params, "flat", "atr_trailing_exit", trailing_stop)
                )
                active_side = "flat"
                trailing_stop = None
            continue
        if not row["was_compressed"] or row["volume"] < row["average_volume"] * volume_multiplier:
            continue
        if row["close"] > row["channel_high"]:
            trailing_stop = float(row["close"] - 2 * row["atr"])
            signals.append(
                Signal(
                    row["timestamp"],
                    "compression_breakout",
                    params,
                    "long",
                    "compressed_range_break_high",
                    trailing_stop,
                )
            )
            active_side = "long"
        elif row["close"] < row["channel_low"]:
            trailing_stop = float(row["close"] + 2 * row["atr"])
            signals.append(
                Signal(
                    row["timestamp"],
                    "compression_breakout",
                    params,
                    "short",
                    "compressed_range_break_low",
                    trailing_stop,
                )
            )
            active_side = "short"
    return signals


def compression_breakout_retest_signals(
    candles: pd.DataFrame,
    lookback: int,
    bandwidth_window: int,
    compression_quantile: float,
    volume_multiplier: float,
    atr_length: int = 14,
    retest_atr_buffer: float = 0.75,
    retest_window_bars: int = 3,
) -> list[Signal]:
    frame = candles.copy()
    mean = frame["close"].rolling(lookback).mean()
    std = frame["close"].rolling(lookback).std(ddof=0)
    frame["bandwidth"] = (4 * std) / mean.replace(0, pd.NA)
    threshold = frame["bandwidth"].rolling(bandwidth_window, min_periods=max(5, bandwidth_window // 2)).quantile(compression_quantile)
    frame["was_compressed"] = (frame["bandwidth"] <= threshold).shift(1).fillna(False)
    frame["channel_high"] = frame["high"].shift(1).rolling(lookback).max()
    frame["channel_low"] = frame["low"].shift(1).rolling(lookback).min()
    frame["average_volume"] = frame["volume"].shift(1).rolling(lookback).mean()
    frame["atr"] = _atr(frame, atr_length)
    params: Params = {
        "lookback": lookback,
        "bandwidth_window": bandwidth_window,
        "compression_quantile": compression_quantile,
        "volume_multiplier": volume_multiplier,
        "atr_length": atr_length,
        "retest_atr_buffer": retest_atr_buffer,
        "retest_window_bars": retest_window_bars,
    }
    signals: list[Signal] = []
    active_side = "flat"
    trailing_stop: float | None = None
    pending_retest: dict[str, float | int | str] | None = None

    for idx, row in frame.dropna(subset=["channel_high", "channel_low", "average_volume", "atr"]).iterrows():
        if active_side == "long":
            trailing_stop = max(float(trailing_stop), float(row["close"] - 2 * row["atr"]))
            if row["close"] <= trailing_stop:
                signals.append(Signal(row["timestamp"], "compression_breakout_retest", params, "flat", "atr_trailing_exit", trailing_stop))
                active_side = "flat"
                trailing_stop = None
            continue
        if active_side == "short":
            trailing_stop = min(float(trailing_stop), float(row["close"] + 2 * row["atr"]))
            if row["close"] >= trailing_stop:
                signals.append(Signal(row["timestamp"], "compression_breakout_retest", params, "flat", "atr_trailing_exit", trailing_stop))
                active_side = "flat"
                trailing_stop = None
            continue

        if pending_retest is not None:
            expires_at = int(pending_retest["expires_at"])
            level = float(pending_retest["level"])
            side = str(pending_retest["side"])
            if idx > expires_at:
                pending_retest = None
            elif side == "long":
                pulled_back = float(row["low"]) <= level + float(row["atr"]) * retest_atr_buffer
                reclaimed = float(row["close"]) > level
                if pulled_back and reclaimed:
                    trailing_stop = float(row["close"] - 2 * row["atr"])
                    signals.append(
                        Signal(
                            row["timestamp"],
                            "compression_breakout_retest",
                            params,
                            "long",
                            "compressed_range_breakout_retest_long",
                            trailing_stop,
                        )
                    )
                    active_side = "long"
                    pending_retest = None
                    continue
            else:
                pulled_back = float(row["high"]) >= level - float(row["atr"]) * retest_atr_buffer
                reclaimed = float(row["close"]) < level
                if pulled_back and reclaimed:
                    trailing_stop = float(row["close"] + 2 * row["atr"])
                    signals.append(
                        Signal(
                            row["timestamp"],
                            "compression_breakout_retest",
                            params,
                            "short",
                            "compressed_range_breakout_retest_short",
                            trailing_stop,
                        )
                    )
                    active_side = "short"
                    pending_retest = None
                    continue

        if not row["was_compressed"] or row["volume"] < row["average_volume"] * volume_multiplier:
            continue
        if row["close"] > row["channel_high"]:
            pending_retest = {"side": "long", "level": float(row["channel_high"]), "expires_at": idx + retest_window_bars}
        elif row["close"] < row["channel_low"]:
            pending_retest = {"side": "short", "level": float(row["channel_low"]), "expires_at": idx + retest_window_bars}
    return signals


def filter_signals_by_adverse_funding(
    candles: pd.DataFrame,
    signals: list[Signal],
    funding_window: int,
    max_adverse_funding_z: float,
) -> list[Signal]:
    _require_funding(candles)
    if not signals:
        return []
    frame = candles.copy()
    frame["funding_z"] = _rolling_zscore(frame["funding_rate"], funding_window)
    funding_by_time = {
        pd.Timestamp(row["timestamp"]): float(row["funding_z"])
        for _, row in frame.dropna(subset=["funding_z"]).iterrows()
    }
    filtered: list[Signal] = []
    for signal in signals:
        if signal.side == "flat":
            filtered.append(signal)
            continue
        funding_z = funding_by_time.get(pd.Timestamp(signal.timestamp))
        if funding_z is None:
            continue
        if signal.side == "long" and funding_z > max_adverse_funding_z:
            continue
        if signal.side == "short" and funding_z < -max_adverse_funding_z:
            continue
        filtered.append(signal)
    return filtered


def _apply_compression_exit_style(
    candles: pd.DataFrame,
    base_signals: list[Signal],
    *,
    strategy_name: str,
    params: Params,
    exit_style: str,
    profit_take_r: float,
    reduced_exposure: float,
    post_take_trailing_atr: float,
    max_hold_bars: int,
) -> list[Signal]:
    converted_base = [
        Signal(
            signal.timestamp,
            strategy_name,
            params,
            signal.side,
            signal.entry_reason,
            signal.invalidation_price,
            signal.exposure_multiplier,
        )
        for signal in base_signals
    ]
    if exit_style == "atr_trail":
        return converted_base

    frame = candles.copy().reset_index(drop=True)
    frame["atr"] = _atr(frame, int(params["atr_length"]))
    signals_by_time = {
        pd.Timestamp(signal.timestamp): signal
        for signal in converted_base
    }
    output: list[Signal] = []
    active_signal: Signal | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    entry_index: int | None = None
    original_risk: float | None = None
    partial_taken = False

    for idx, row in frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        close_price = float(row["close"])
        atr_value = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        base_signal = signals_by_time.get(timestamp)

        if active_signal is None:
            if base_signal is not None and base_signal.side in {"long", "short"}:
                output.append(base_signal)
                active_signal = base_signal
                entry_price = close_price
                stop_price = float(base_signal.invalidation_price)
                entry_index = idx
                original_risk = abs(entry_price - stop_price)
                partial_taken = False
            continue

        side = active_signal.side
        if side == "short":
            trail_multiple = post_take_trailing_atr if partial_taken and exit_style == "profit_trail" else 2.0
            stop_price = min(float(stop_price), close_price + trail_multiple * atr_value)
            if exit_style == "profit_trail" and not partial_taken and original_risk and entry_price is not None:
                if (entry_price - close_price) >= original_risk * profit_take_r:
                    resize = Signal(timestamp, strategy_name, params, "short", "partial_profit_take", float(stop_price), exposure_multiplier=reduced_exposure)
                    output.append(resize)
                    active_signal = resize
                    partial_taken = True
                    continue
            if exit_style == "time_stop_trail" and entry_index is not None and idx - entry_index >= max_hold_bars:
                output.append(Signal(timestamp, strategy_name, params, "flat", "time_stop_exit", close_price))
                active_signal = None
                entry_price = None
                stop_price = None
                entry_index = None
                original_risk = None
                partial_taken = False
                continue
            if close_price >= float(stop_price):
                output.append(Signal(timestamp, strategy_name, params, "flat", "atr_trailing_exit", float(stop_price)))
                active_signal = None
                entry_price = None
                stop_price = None
                entry_index = None
                original_risk = None
                partial_taken = False
                continue
        else:
            trail_multiple = post_take_trailing_atr if partial_taken and exit_style == "profit_trail" else 2.0
            stop_price = max(float(stop_price), close_price - trail_multiple * atr_value)
            if exit_style == "profit_trail" and not partial_taken and original_risk and entry_price is not None:
                if (close_price - entry_price) >= original_risk * profit_take_r:
                    resize = Signal(timestamp, strategy_name, params, "long", "partial_profit_take", float(stop_price), exposure_multiplier=reduced_exposure)
                    output.append(resize)
                    active_signal = resize
                    partial_taken = True
                    continue
            if exit_style == "time_stop_trail" and entry_index is not None and idx - entry_index >= max_hold_bars:
                output.append(Signal(timestamp, strategy_name, params, "flat", "time_stop_exit", close_price))
                active_signal = None
                entry_price = None
                stop_price = None
                entry_index = None
                original_risk = None
                partial_taken = False
                continue
            if close_price <= float(stop_price):
                output.append(Signal(timestamp, strategy_name, params, "flat", "atr_trailing_exit", float(stop_price)))
                active_signal = None
                entry_price = None
                stop_price = None
                entry_index = None
                original_risk = None
                partial_taken = False
                continue

        if base_signal is not None and base_signal.side == "flat":
            output.append(Signal(timestamp, strategy_name, params, "flat", base_signal.entry_reason, close_price))
            active_signal = None
            entry_price = None
            stop_price = None
            entry_index = None
            original_risk = None
            partial_taken = False

    return output


def funding_crowding_reversal_signals(
    candles: pd.DataFrame,
    z_window: int,
    entry_z: float,
    invalidation_window: int = 20,
) -> list[Signal]:
    _require_funding(candles)
    frame = candles.copy()
    frame["funding_z"] = _rolling_zscore(frame["funding_rate"], z_window)
    frame["premium_z"] = _rolling_zscore(frame["premium"], z_window)
    frame["crowding_z"] = (frame["funding_z"] + frame["premium_z"]) / 2
    frame["rolling_low"] = frame["close"].rolling(invalidation_window, min_periods=1).min()
    frame["rolling_high"] = frame["close"].rolling(invalidation_window, min_periods=1).max()
    params: Params = {"z_window": z_window, "entry_z": entry_z, "invalidation_window": invalidation_window}
    signals: list[Signal] = []
    active_side = "flat"
    for idx in range(1, len(frame)):
        row = frame.iloc[idx]
        previous = frame.iloc[idx - 1]
        if pd.isna(row["crowding_z"]):
            continue
        bearish_reversal = row["close"] < previous["close"]
        bullish_reversal = row["close"] > previous["close"]
        if active_side == "flat" and row["crowding_z"] >= entry_z and bearish_reversal:
            signals.append(
                Signal(row["timestamp"], "funding_crowding_reversal", params, "short", "crowded_longs_reversed", float(row["rolling_high"]))
            )
            active_side = "short"
        elif active_side == "flat" and row["crowding_z"] <= -entry_z and bullish_reversal:
            signals.append(
                Signal(row["timestamp"], "funding_crowding_reversal", params, "long", "crowded_shorts_reversed", float(row["rolling_low"]))
            )
            active_side = "long"
        elif active_side != "flat" and abs(row["crowding_z"]) < 0.25:
            signals.append(Signal(row["timestamp"], "funding_crowding_reversal", params, "flat", "crowding_normalized", float(row["close"])))
            active_side = "flat"
    return signals


def funding_conditioned_momentum_signals(
    candles: pd.DataFrame,
    lookback: int,
    funding_window: int,
    max_crowding_z: float,
    atr_length: int,
    atr_multiplier: float,
) -> list[Signal]:
    _require_funding(candles)
    frame = candles.copy()
    frame["momentum"] = frame["close"].pct_change(lookback)
    frame["funding_z"] = _rolling_zscore(frame["funding_rate"], funding_window)
    frame["atr"] = _atr(frame, atr_length)
    params: Params = {
        "lookback": lookback,
        "funding_window": funding_window,
        "max_crowding_z": max_crowding_z,
        "atr_length": atr_length,
        "atr_multiplier": atr_multiplier,
    }
    signals: list[Signal] = []
    previous_side = "flat"
    for _, row in frame.dropna(subset=["momentum", "funding_z", "atr"]).iterrows():
        side = "flat"
        if row["momentum"] > 0 and row["funding_z"] <= max_crowding_z:
            side = "long"
        elif row["momentum"] < 0 and row["funding_z"] >= -max_crowding_z:
            side = "short"
        if side == previous_side:
            continue
        if side == "long":
            invalidation = float(row["close"] - atr_multiplier * row["atr"])
            reason = "momentum_long_not_crowded"
        elif side == "short":
            invalidation = float(row["close"] + atr_multiplier * row["atr"])
            reason = "momentum_short_not_crowded"
        else:
            invalidation = float(row["close"])
            reason = "momentum_filtered_by_funding"
        signals.append(Signal(row["timestamp"], "funding_conditioned_momentum", params, side, reason, invalidation))
        previous_side = side
    return signals


def strategy_grid(include_funding: bool = False) -> list[tuple[str, Params]]:
    grid: list[tuple[str, Params]] = []
    for fast in [10, 20, 50]:
        for slow in [50, 100, 200]:
            if fast < slow:
                grid.append(("ema_crossover", {"fast": fast, "slow": slow}))
    for lower in [25, 30, 35]:
        for upper in [65, 70, 75]:
            grid.append(("rsi_mean_reversion", {"length": 14, "lower": lower, "upper": upper}))
    for lookback in [20, 50, 100]:
        grid.append(("breakout", {"lookback": lookback}))
    if include_funding:
        for lookback in [20, 50, 100]:
            for max_adverse_funding_z in [0.5, 1.0]:
                grid.append(
                    (
                        "breakout_funding_veto",
                        {
                            "lookback": lookback,
                            "funding_window": 24,
                            "max_adverse_funding_z": max_adverse_funding_z,
                        },
                    )
                )
    for lookback in [12, 24, 48]:
        for atr_multiplier in [1.5, 2.5]:
            grid.append(
                (
                    "volatility_scaled_momentum",
                    {"lookback": lookback, "atr_length": 14, "atr_multiplier": atr_multiplier},
                )
            )
    for length in [20, 40]:
        for entry_z in [1.5, 2.0]:
            grid.append(
                (
                    "bollinger_regime_reversion",
                    {"length": length, "entry_z": entry_z, "max_trend_strength": 0.75, "atr_length": 14},
                )
            )
    for fast_length in [20, 50]:
        for slow_length in [100, 200]:
            if fast_length < slow_length:
                grid.append(
                    (
                        "pullback_in_trend",
                        {
                            "fast_length": fast_length,
                            "slow_length": slow_length,
                            "rsi_length": 14,
                            "recovery_level": 45,
                            "invalidation_window": 10,
                        },
                    )
                )
    for lookback in [12, 24, 48]:
        for volume_multiplier in [1.25, 1.75]:
            grid.append(
                (
                    "compression_breakout",
                    {
                        "lookback": lookback,
                        "bandwidth_window": 100,
                        "compression_quantile": 0.2,
                        "volume_multiplier": volume_multiplier,
                        "atr_length": 14,
                    },
                )
            )
            grid.append(
                (
                    "compression_breakout_retest",
                    {
                        "lookback": lookback,
                        "bandwidth_window": 100,
                        "compression_quantile": 0.2,
                        "volume_multiplier": volume_multiplier,
                        "atr_length": 14,
                        "retest_atr_buffer": 0.75,
                        "retest_window_bars": 3,
                    },
                )
            )
    if include_funding:
        for z_window in [24, 72]:
            for entry_z in [1.5, 2.0]:
                grid.append(
                    (
                        "funding_crowding_reversal",
                        {"z_window": z_window, "entry_z": entry_z, "invalidation_window": 20},
                    )
                )
        for lookback in [12, 24, 48]:
            for max_crowding_z in [0.5, 1.0]:
                grid.append(
                    (
                        "funding_conditioned_momentum",
                        {
                            "lookback": lookback,
                            "funding_window": 72,
                            "max_crowding_z": max_crowding_z,
                            "atr_length": 14,
                            "atr_multiplier": 2.0,
                        },
                    )
                )
    return grid


def generate_signals(candles: pd.DataFrame, strategy_name: str, params: Params) -> list[Signal]:
    if strategy_name == "ema_crossover":
        return ema_crossover_signals(candles, fast=int(params["fast"]), slow=int(params["slow"]))
    if strategy_name == "rsi_mean_reversion":
        return rsi_mean_reversion_signals(
            candles,
            length=int(params["length"]),
            lower=int(params["lower"]),
            upper=int(params["upper"]),
        )
    if strategy_name == "breakout":
        return breakout_signals(candles, lookback=int(params["lookback"]))
    if strategy_name == "breakout_funding_veto":
        return breakout_with_funding_veto_signals(
            candles,
            lookback=int(params["lookback"]),
            funding_window=int(params["funding_window"]),
            max_adverse_funding_z=float(params["max_adverse_funding_z"]),
        )
    if strategy_name == "volatility_scaled_momentum":
        return volatility_scaled_momentum_signals(
            candles,
            lookback=int(params["lookback"]),
            atr_length=int(params["atr_length"]),
            atr_multiplier=float(params["atr_multiplier"]),
        )
    if strategy_name == "bollinger_regime_reversion":
        return bollinger_regime_reversion_signals(
            candles,
            length=int(params["length"]),
            entry_z=float(params["entry_z"]),
            max_trend_strength=float(params["max_trend_strength"]),
            atr_length=int(params["atr_length"]),
        )
    if strategy_name == "pullback_in_trend":
        return pullback_in_trend_signals(
            candles,
            fast_length=int(params["fast_length"]),
            slow_length=int(params["slow_length"]),
            rsi_length=int(params["rsi_length"]),
            recovery_level=int(params["recovery_level"]),
            invalidation_window=int(params["invalidation_window"]),
        )
    if strategy_name == "compression_breakout":
        return compression_breakout_signals(
            candles,
            lookback=int(params["lookback"]),
            bandwidth_window=int(params["bandwidth_window"]),
            compression_quantile=float(params["compression_quantile"]),
            volume_multiplier=float(params["volume_multiplier"]),
            atr_length=int(params["atr_length"]),
        )
    if strategy_name == "compression_breakout_retest":
        return compression_breakout_retest_signals(
            candles,
            lookback=int(params["lookback"]),
            bandwidth_window=int(params["bandwidth_window"]),
            compression_quantile=float(params["compression_quantile"]),
            volume_multiplier=float(params["volume_multiplier"]),
            atr_length=int(params["atr_length"]),
            retest_atr_buffer=float(params["retest_atr_buffer"]),
            retest_window_bars=int(params["retest_window_bars"]),
        )
    if strategy_name == "funding_crowding_reversal":
        return funding_crowding_reversal_signals(
            candles,
            z_window=int(params["z_window"]),
            entry_z=float(params["entry_z"]),
            invalidation_window=int(params["invalidation_window"]),
        )
    if strategy_name == "funding_conditioned_momentum":
        return funding_conditioned_momentum_signals(
            candles,
            lookback=int(params["lookback"]),
            funding_window=int(params["funding_window"]),
            max_crowding_z=float(params["max_crowding_z"]),
            atr_length=int(params["atr_length"]),
            atr_multiplier=float(params["atr_multiplier"]),
        )
    raise ValueError(f"unknown strategy: {strategy_name}")


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(frame: pd.DataFrame, length: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False).mean()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 3)).mean()
    std = series.rolling(window, min_periods=max(3, window // 3)).std(ddof=0)
    return (series - mean) / std.replace(0, pd.NA)


def _require_funding(candles: pd.DataFrame) -> None:
    missing = {"funding_rate", "premium"} - set(candles.columns)
    if missing:
        raise ValueError(f"funding strategy requires columns: {sorted(missing)}")
