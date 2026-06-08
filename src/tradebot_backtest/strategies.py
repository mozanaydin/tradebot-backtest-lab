from __future__ import annotations

import pandas as pd

from tradebot_backtest.engine import Signal

Params = dict[str, int | float]


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
    if strategy_name == "compression_breakout":
        return compression_breakout_signals(
            candles,
            lookback=int(params["lookback"]),
            bandwidth_window=int(params["bandwidth_window"]),
            compression_quantile=float(params["compression_quantile"]),
            volume_multiplier=float(params["volume_multiplier"]),
            atr_length=int(params["atr_length"]),
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
