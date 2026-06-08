from __future__ import annotations

import math

import pandas as pd
import pytest

from tradebot_backtest.engine import (
    BacktestConfig,
    CostModel,
    Signal,
    apply_funding,
    calculate_position_notional,
    run_backtest,
    score_result,
)


def sample_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 104.0, 103.0, 101.0],
            "high": [100.0, 200.0, 103.0, 105.0, 104.0, 102.0],
            "low": [100.0, 1.0, 101.0, 103.0, 100.0, 99.0],
            "close": [100.0, 102.0, 104.0, 103.0, 101.0, 100.0],
            "volume": [1.0] * 6,
        }
    )


def test_cost_model_applies_fee_and_slippage_on_entry_and_exit() -> None:
    model = CostModel(fee_rate=0.00045, slippage_rate=0.0002)

    assert model.round_trip_cost(1000.0) == 1.3


def test_position_sizing_respects_risk_and_max_leverage() -> None:
    notional = calculate_position_notional(equity=1000.0, risk_fraction=0.05, stop_distance_pct=0.01, max_leverage=3.0)

    assert notional == 3000.0


def test_signal_exposure_multiplier_scales_position_notional() -> None:
    candles = sample_candles()
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=0.05,
        max_leverage=3.0,
        cost_model=CostModel(0.0, 0.0),
    )
    full_exposure = run_backtest(
        candles,
        [Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0, 1.0)],
        config,
    )
    half_exposure = run_backtest(
        candles,
        [Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0, 0.5)],
        config,
    )

    assert half_exposure.trades[0].notional == full_exposure.trades[0].notional / 2


def test_same_side_signal_with_new_exposure_rebalances_at_next_open() -> None:
    candles = sample_candles()
    signals = [
        Signal(candles.loc[0, "timestamp"], "test", {}, "long", "entry", 90.0, 1.0),
        Signal(candles.loc[2, "timestamp"], "test", {}, "long", "regime_resize", 90.0, 0.5),
    ]
    config = BacktestConfig(cost_model=CostModel(0.0, 0.0))

    result = run_backtest(candles, signals, config)

    assert len(result.trades) == 2
    assert result.trades[0].exit_time == candles.loc[3, "timestamp"]
    assert result.trades[1].entry_time == candles.loc[3, "timestamp"]
    assert result.trades[1].notional < result.trades[0].notional


def test_invalid_signal_exposure_multiplier_skips_entry() -> None:
    candles = sample_candles()
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=0.05,
        max_leverage=3.0,
        cost_model=CostModel(0.0, 0.0),
    )

    for multiplier in (0.0, -0.5, 1.01):
        result = run_backtest(
            candles,
            [Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0, multiplier)],
            config,
        )

        assert result.trades == []


@pytest.mark.parametrize(
    "multiplier",
    [True, "half", None, object(), complex(0.5, 0.0), float("nan"), float("inf"), float("-inf")],
)
def test_non_real_or_non_finite_signal_exposure_multiplier_skips_entry_without_exception(multiplier: object) -> None:
    candles = sample_candles()
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=0.05,
        max_leverage=3.0,
        cost_model=CostModel(0.0, 0.0),
    )

    result = run_backtest(
        candles,
        [Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0, multiplier)],  # type: ignore[arg-type]
        config,
    )

    assert result.trades == []


def test_positive_funding_means_long_pays_and_short_receives() -> None:
    assert apply_funding("long", 1000.0, 0.0001) == -0.1
    assert apply_funding("short", 1000.0, 0.0001) == 0.1


def test_backtest_enters_after_signal_candle_and_ignores_intrabar_stop_hits() -> None:
    candles = sample_candles()
    signals = [
        Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0),
        Signal(candles.loc[3, "timestamp"], "test", {"id": 1}, "flat", "exit", 90.0),
    ]
    config = BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, cost_model=CostModel(0.0, 0.0))

    result = run_backtest(candles, signals, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_time == candles.loc[1, "timestamp"]
    assert trade.entry_price == 101.0
    assert trade.exit_time == candles.loc[4, "timestamp"]
    assert trade.exit_price == 103.0
    assert trade.exit_reason == "exit"


def test_trade_pnl_reports_gross_less_both_entry_and_exit_fees() -> None:
    candles = sample_candles()
    signals = [
        Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0),
        Signal(candles.loc[3, "timestamp"], "test", {"id": 1}, "flat", "exit", 90.0),
    ]
    config = BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, cost_model=CostModel(0.001, 0.0))

    result = run_backtest(candles, signals, config)

    trade = result.trades[0]
    gross = (103.0 - 101.0) * (trade.notional / 101.0)
    assert trade.fees == trade.notional * 0.001 * 2
    assert trade.pnl == gross - trade.fees


def test_close_beyond_invalidation_exits_at_next_candle_open() -> None:
    candles = sample_candles()
    signals = [
        Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 101.5),
    ]
    config = BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, cost_model=CostModel(0.0, 0.0))

    result = run_backtest(candles, signals, config)

    assert len(result.trades) == 1
    assert result.trades[0].exit_time == candles.loc[5, "timestamp"]
    assert result.trades[0].exit_price == candles.loc[5, "open"]
    assert result.trades[0].exit_reason == "invalidation_close"


def test_equity_curve_marks_open_position_to_market() -> None:
    candles = sample_candles()
    signals = [
        Signal(candles.loc[0, "timestamp"], "test", {"id": 1}, "long", "entry", 90.0),
    ]
    config = BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, cost_model=CostModel(0.0, 0.0))

    result = run_backtest(candles, signals, config)

    assert result.equity_curve["equity"].max() > 1000.0
    assert result.max_drawdown_pct > 0


def test_ranking_excludes_results_with_fewer_than_ten_trades() -> None:
    low_count = type("Result", (), {"total_return_pct": 10.0, "max_drawdown_pct": 5.0, "trade_count": 9})()
    enough = type("Result", (), {"total_return_pct": 6.0, "max_drawdown_pct": 3.0, "trade_count": 10})()

    assert math.isinf(score_result(low_count))
    assert score_result(low_count) < 0
    assert score_result(enough) == 2.0
