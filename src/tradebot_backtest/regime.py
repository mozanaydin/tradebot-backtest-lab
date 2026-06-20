from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from tradebot_backtest.engine import Signal
from tradebot_backtest.strategies import breakout_signals

Regime = Literal["high_volatility", "trending", "ranging", "unclear"]
Component = Literal["breakout", "bollinger", "none"]


@dataclass(frozen=True)
class RegimeParams:
    high_volatility_percentile: float
    trend_adx: float
    range_adx: float
    slope_threshold: float
    donchian_lookback: int
    bollinger_length: int
    bollinger_entry_z: float


def regime_features(candles: pd.DataFrame) -> pd.DataFrame:
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = _wilder_average(true_range, 14)

    upward_move = high.diff()
    downward_move = -low.diff()
    plus_dm = upward_move.where((upward_move > downward_move) & (upward_move > 0), 0.0)
    minus_dm = downward_move.where((downward_move > upward_move) & (downward_move > 0), 0.0)
    smoothed_plus_dm = _wilder_average(plus_dm, 14)
    smoothed_minus_dm = _wilder_average(minus_dm, 14)
    plus_di = 100.0 * smoothed_plus_dm / atr14.replace(0, np.nan)
    minus_di = 100.0 * smoothed_minus_dm / atr14.replace(0, np.nan)
    plus_di = plus_di.mask(atr14.eq(0) & smoothed_plus_dm.eq(0), 0.0)
    minus_di = minus_di.mask(atr14.eq(0) & smoothed_minus_dm.eq(0), 0.0)
    directional_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / directional_sum.replace(0, np.nan)
    dx = dx.mask(directional_sum.eq(0), 0.0)
    adx14 = _wilder_average(dx, 14)

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    normalized_slope = (ema50 - ema50.shift(12)) / atr14.replace(0, np.nan)
    ema200_slope = (ema200 - ema200.shift(24)) / atr14.replace(0, np.nan)
    trend_gap_atr = (ema50 - ema200) / atr14.replace(0, np.nan)
    distance_from_ema50_atr = (close - ema50).abs() / atr14.replace(0, np.nan)
    atr_percentile = atr14.rolling(169, min_periods=169).apply(
        _rank_latest_against_previous,
        raw=True,
    )
    average_volume = candles["volume"].astype(float).rolling(20, min_periods=20).mean() if "volume" in candles.columns else pd.Series(np.nan, index=candles.index)
    volume_ratio = candles["volume"].astype(float) / average_volume.replace(0, np.nan) if "volume" in candles.columns else pd.Series(np.nan, index=candles.index)

    return pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "close": close,
            "atr14": atr14,
            "adx14": adx14,
            "ema50": ema50,
            "ema200": ema200,
            "normalized_ema_slope": normalized_slope,
            "normalized_ema200_slope": ema200_slope,
            "trend_gap_atr": trend_gap_atr,
            "distance_from_ema50_atr": distance_from_ema50_atr,
            "atr_percentile": atr_percentile,
            "volume_ratio": volume_ratio,
        },
        index=candles.index,
    )


def classify_regime(
    atr_percentile: float,
    adx: float,
    normalized_slope: float,
    params: RegimeParams,
) -> Regime:
    if atr_percentile >= params.high_volatility_percentile:
        return "high_volatility"
    if adx >= params.trend_adx and abs(normalized_slope) >= params.slope_threshold:
        return "trending"
    if adx <= params.range_adx:
        return "ranging"
    return "unclear"


def regime_switching_signals(
    candles: pd.DataFrame,
    params: RegimeParams,
    *,
    features: pd.DataFrame | None = None,
    component_signals: Mapping[Component, list[Signal]] | None = None,
    start_at: pd.Timestamp | None = None,
) -> list[Signal]:
    features = regime_features(candles) if features is None else features
    close_by_time = {
        pd.Timestamp(row["timestamp"]): float(row["close"])
        for _, row in candles.iterrows()
    }
    supplied_signals = component_signals or {}
    breakout_candidates = supplied_signals.get("breakout")
    if breakout_candidates is None:
        breakout_candidates = breakout_signals(candles, params.donchian_lookback)
    bollinger_candidates = supplied_signals.get("bollinger")
    if bollinger_candidates is None:
        bollinger_candidates = _bollinger_candidate_signals(
            candles,
            length=params.bollinger_length,
            entry_z=params.bollinger_entry_z,
        )
    breakout_by_time = {
        pd.Timestamp(signal.timestamp): signal
        for signal in breakout_candidates
    }
    bollinger_by_time = {
        pd.Timestamp(signal.timestamp): signal
        for signal in bollinger_candidates
    }
    strategy_params = asdict(params)
    signals: list[Signal] = []
    active_side: Literal["long", "short", "flat"] = "flat"
    active_component: Component = "none"
    active_exposure = 1.0
    active_entry: Signal | None = None

    for _, row in features.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if start_at is not None and timestamp < pd.Timestamp(start_at):
            continue
        regime = classify_regime(
            float(row["atr_percentile"]),
            float(row["adx14"]),
            float(row["normalized_ema_slope"]),
            params,
        )
        component = _component_for_regime(regime)
        exposure = _exposure_for_regime(regime)
        component_signal = (
            breakout_by_time.get(timestamp)
            if component == "breakout"
            else bollinger_by_time.get(timestamp)
            if component == "bollinger"
            else None
        )

        if active_side != "flat" and active_entry is not None:
            close_price = close_by_time[timestamp]
            invalidation = float(active_entry.invalidation_price)
            invalidated = (
                active_side == "long" and close_price <= invalidation
            ) or (
                active_side == "short" and close_price >= invalidation
            )
            if invalidated:
                signals.append(
                    Signal(
                        timestamp,
                        "regime_switching",
                        strategy_params,
                        "flat",
                        "invalidation_close",
                        invalidation,
                    )
                )
                active_side = "flat"
                active_component = "none"
                active_entry = None
                active_exposure = 1.0
                continue

        component_changed = active_side != "flat" and component != active_component
        exposure_changed = (
            active_side != "flat"
            and component == active_component == "breakout"
            and exposure != active_exposure
        )
        if component_changed or exposure_changed:
            if component_signal is not None and component_signal.side in {"long", "short"}:
                routed = _route_signal(
                    component_signal,
                    timestamp,
                    regime,
                    strategy_params,
                    exposure,
                )
            elif exposure_changed and active_entry is not None:
                routed = Signal(
                    timestamp,
                    "regime_switching",
                    strategy_params,
                    active_side,
                    "regime_resize",
                    active_entry.invalidation_price,
                    exposure_multiplier=exposure,
                )
            else:
                routed = Signal(
                    timestamp,
                    "regime_switching",
                    strategy_params,
                    "flat",
                    "regime_change",
                    close_by_time[timestamp],
                )
            signals.append(routed)
            active_side = routed.side
            active_component = component if routed.side != "flat" else "none"
            active_exposure = routed.exposure_multiplier
            active_entry = component_signal if routed.side != "flat" and component_signal is not None else active_entry
            continue

        if component_signal is None:
            continue
        if component_signal.side == "flat" and active_side == "flat":
            continue
        if component_signal.side == active_side:
            continue

        routed = _route_signal(
            component_signal,
            timestamp,
            regime,
            strategy_params,
            exposure,
        )
        signals.append(routed)
        active_side = routed.side
        active_component = component if active_side != "flat" else "none"
        active_exposure = routed.exposure_multiplier
        active_entry = component_signal if active_side != "flat" else None

    return signals


def regime_parameter_grid() -> list[RegimeParams]:
    grid: list[RegimeParams] = []
    for values in product(
        [0.80, 0.90],
        [20, 25],
        [15, 20],
        [0.25, 0.50],
        [24, 50],
        [20, 40],
        [1.5, 2.0],
    ):
        configured = RegimeParams(*values)
        if configured.range_adx < configured.trend_adx:
            grid.append(configured)
    return grid


def filter_entry_signals_by_regime(
    signals: list[Signal],
    features: pd.DataFrame,
    *,
    min_adx: float,
    min_abs_slope: float,
    min_atr_percentile: float,
    max_atr_percentile: float,
    require_trend_alignment: bool,
    require_price_above_ema200: bool = False,
    require_ema_stack: bool = False,
    max_distance_from_ema50_atr: float | None = None,
    min_volume_ratio: float | None = None,
) -> list[Signal]:
    if not signals:
        return []
    feature_by_time = {
        pd.Timestamp(row["timestamp"]): row
        for _, row in features.iterrows()
    }
    filtered: list[Signal] = []
    for signal in signals:
        if signal.side == "flat":
            filtered.append(signal)
            continue
        row = feature_by_time.get(pd.Timestamp(signal.timestamp))
        if row is None:
            continue
        atr_percentile = float(row["atr_percentile"])
        adx = float(row["adx14"])
        slope = float(row["normalized_ema_slope"])
        close_price = float(row["close"])
        ema50 = float(row["ema50"])
        ema200 = float(row["ema200"])
        distance_from_ema50_atr = float(row["distance_from_ema50_atr"])
        volume_ratio = float(row["volume_ratio"]) if not pd.isna(row["volume_ratio"]) else float("nan")
        if pd.isna(atr_percentile) or pd.isna(adx) or pd.isna(slope) or pd.isna(close_price) or pd.isna(ema50) or pd.isna(ema200):
            continue
        if adx < min_adx or abs(slope) < min_abs_slope:
            continue
        if atr_percentile < min_atr_percentile or atr_percentile > max_atr_percentile:
            continue
        if require_trend_alignment:
            if signal.side == "long" and slope <= 0:
                continue
            if signal.side == "short" and slope >= 0:
                continue
        if require_price_above_ema200:
            if signal.side == "long" and close_price <= ema200:
                continue
            if signal.side == "short" and close_price >= ema200:
                continue
        if require_ema_stack:
            if signal.side == "long" and not (ema50 > ema200):
                continue
            if signal.side == "short" and not (ema50 < ema200):
                continue
        if max_distance_from_ema50_atr is not None:
            if pd.isna(distance_from_ema50_atr) or distance_from_ema50_atr > max_distance_from_ema50_atr:
                continue
        if min_volume_ratio is not None:
            if pd.isna(volume_ratio) or volume_ratio < min_volume_ratio:
                continue
        filtered.append(signal)
    return filtered


def _component_for_regime(regime: Regime) -> Component:
    if regime in {"high_volatility", "trending"}:
        return "breakout"
    if regime == "ranging":
        return "bollinger"
    return "none"


def _exposure_for_regime(regime: Regime) -> float:
    return 0.5 if regime == "high_volatility" else 1.0


def _route_signal(
    signal: Signal,
    timestamp: pd.Timestamp,
    regime: Regime,
    params: dict[str, int | float | str],
    exposure_multiplier: float,
) -> Signal:
    return Signal(
        timestamp,
        "regime_switching",
        params,
        signal.side,
        f"{regime}:{signal.entry_reason}",
        signal.invalidation_price,
        exposure_multiplier=exposure_multiplier if signal.side != "flat" else 1.0,
    )


def _bollinger_candidate_signals(
    candles: pd.DataFrame,
    length: int,
    entry_z: float,
) -> list[Signal]:
    frame = candles.copy()
    frame["mean"] = frame["close"].rolling(length).mean()
    frame["std"] = frame["close"].rolling(length).std(ddof=0)
    frame["z"] = (frame["close"] - frame["mean"]) / frame["std"].replace(0, pd.NA)
    frame["atr"] = _exponential_atr(frame, 14)
    params = {"length": length, "entry_z": entry_z, "atr_length": 14}
    signals: list[Signal] = []

    for idx in range(1, len(frame)):
        previous = frame.iloc[idx - 1]
        row = frame.iloc[idx]
        if pd.isna(row["z"]) or pd.isna(row["atr"]):
            continue
        if row["z"] <= -entry_z:
            signals.append(
                Signal(
                    row["timestamp"],
                    "bollinger",
                    params,
                    "long",
                    "lower_band_extreme",
                    float(row["close"] - 2 * row["atr"]),
                )
            )
        elif row["z"] >= entry_z:
            signals.append(
                Signal(
                    row["timestamp"],
                    "bollinger",
                    params,
                    "short",
                    "upper_band_extreme",
                    float(row["close"] + 2 * row["atr"]),
                )
            )
        elif previous["z"] < 0 <= row["z"] or previous["z"] > 0 >= row["z"]:
            signals.append(
                Signal(
                    row["timestamp"],
                    "bollinger",
                    params,
                    "flat",
                    "mean_reached",
                    float(row["close"]),
                )
            )
    return signals


def _exponential_atr(frame: pd.DataFrame, length: int) -> pd.Series:
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


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid_positions = np.flatnonzero(values.notna().to_numpy())
    if len(valid_positions) < period:
        return result

    seed_position = int(valid_positions[period - 1])
    seed_values = values.iloc[valid_positions[:period]].astype(float)
    result.iloc[seed_position] = seed_values.mean()
    previous = float(result.iloc[seed_position])

    for position in range(seed_position + 1, len(values)):
        current = values.iloc[position]
        if pd.isna(current):
            continue
        previous = (previous * (period - 1) + float(current)) / period
        result.iloc[position] = previous
    return result


def _rank_latest_against_previous(window: np.ndarray) -> float:
    latest = window[-1]
    previous = window[:-1]
    count_less = np.count_nonzero(previous < latest)
    count_equal = np.count_nonzero(previous == latest)
    return float((count_less + 0.5 * count_equal) / len(previous))
