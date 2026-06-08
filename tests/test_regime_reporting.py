from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradebot_backtest.engine import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    Signal,
    buy_and_hold_result,
    run_backtest,
)
from tradebot_backtest.reporting import regime_distribution_frame, result_summary_frame, write_reports


def candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [10.0] * 5,
        }
    )


def test_backtest_reports_exposure_time() -> None:
    frame = candles()
    signals = [
        Signal(frame.loc[0, "timestamp"], "test", {}, "long", "entry", 95.0),
        Signal(frame.loc[2, "timestamp"], "test", {}, "flat", "exit", 95.0),
    ]
    result = run_backtest(
        frame,
        signals,
        BacktestConfig(cost_model=CostModel(0.0, 0.0)),
    )

    assert "exposed" in result.equity_curve
    assert result.exposure_time_pct == pytest.approx(
        result.equity_curve.groupby("timestamp")["exposed"].max().mean() * 100
    )
    assert 0 < result.exposure_time_pct < 100
    assert "exposure_time_pct" in result_summary_frame([result]).columns


def test_exposure_time_counts_each_timestamp_once() -> None:
    result = BacktestResult(
        strategy_name="test",
        params={},
        trades=[],
        equity_curve=pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                    pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                    pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                    pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                ],
                "equity": [1000.0, 1000.0, 1010.0, 1010.0],
                "exposed": [False, False, True, False],
            }
        ),
    )

    assert result.exposure_time_pct == 50.0


def test_buy_and_hold_uses_one_times_starting_equity_and_costs() -> None:
    frame = candles()
    config = BacktestConfig(
        starting_balance=1000.0,
        cost_model=CostModel(fee_rate=0.001, slippage_rate=0.0),
    )

    result = buy_and_hold_result(frame, config)

    trade = result.trades[0]
    gross = (105.0 - 100.0) * 10.0
    assert result.strategy_name == "buy_and_hold"
    assert trade.notional == 1000.0
    assert trade.fees == 2.0
    assert trade.pnl == gross - 2.0
    assert result.final_equity == 1000.0 + gross - 2.0
    assert result.total_return_pct == pytest.approx((gross - 2.0) / 1000.0 * 100)
    assert result.exposure_time_pct == 100.0


def test_regime_distribution_contains_all_regimes_and_sums_to_100() -> None:
    labels = pd.Series(
        ["trending", "trending", "ranging", "high_volatility", "unclear"]
    )

    distribution = regime_distribution_frame(labels)

    assert set(distribution["regime"]) == {
        "high_volatility",
        "trending",
        "ranging",
        "unclear",
    }
    assert distribution["hours"].sum() == 5
    assert distribution["percent"].sum() == pytest.approx(100.0)


def test_write_reports_writes_regime_distribution(tmp_path: Path) -> None:
    result = buy_and_hold_result(
        candles(),
        BacktestConfig(cost_model=CostModel(0.0, 0.0)),
    )

    latest = write_reports(
        [result],
        tmp_path,
        regime_labels=pd.Series(["trending", "ranging", "unclear"]),
    )

    assert (latest / "regime_distribution.csv").exists()


def test_report_scores_do_not_require_ten_test_trades() -> None:
    result = buy_and_hold_result(
        candles(),
        BacktestConfig(cost_model=CostModel(0.0, 0.0)),
    )

    summary = result_summary_frame([result])

    assert summary.loc[0, "score"] != float("-inf")
