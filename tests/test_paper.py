from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradebot_backtest.engine import BacktestConfig, Signal
from tradebot_backtest.paper import format_paper_cli_summary, simulate_paper_snapshot, write_paper_artifacts


def paper_fixture_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "open": [100.0, 100.0, 95.0, 96.0, 98.0],
            "high": [101.0, 101.0, 97.0, 99.0, 100.0],
            "low": [99.0, 99.0, 94.0, 95.0, 97.0],
            "close": [100.0, 95.0, 96.0, 98.0, 97.0],
            "volume": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


def test_simulate_paper_snapshot_keeps_open_position_without_forcing_end_close() -> None:
    candles = paper_fixture_candles()
    signals = [
        Signal(
            candles.iloc[1]["timestamp"],
            "compression_breakout",
            {"lookback": 48},
            "short",
            "compressed_range_break_low",
            105.0,
        )
    ]
    snapshot = simulate_paper_snapshot(
        candles,
        signals,
        BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, allowed_directions=("short",)),
    )

    assert snapshot.closed_trade_count == 0
    assert snapshot.open_position is not None
    assert snapshot.open_position.side == "short"
    assert snapshot.pending_signal is None


def test_simulate_paper_snapshot_preserves_pending_entry_from_last_closed_candle() -> None:
    candles = paper_fixture_candles()
    signals = [
        Signal(
            candles.iloc[-1]["timestamp"],
            "compression_breakout",
            {"lookback": 48},
            "short",
            "compressed_range_break_low",
            100.0,
        )
    ]
    snapshot = simulate_paper_snapshot(
        candles,
        signals,
        BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, allowed_directions=("short",)),
    )

    assert snapshot.open_position is None
    assert snapshot.pending_signal is not None
    assert snapshot.pending_signal.side == "short"


def test_write_paper_artifacts_generates_dashboard_and_trade_files(tmp_path: Path) -> None:
    candles = paper_fixture_candles()
    signals = [
        Signal(
            candles.iloc[1]["timestamp"],
            "compression_breakout",
            {"lookback": 48},
            "short",
            "compressed_range_break_low",
            105.0,
        ),
        Signal(
            candles.iloc[3]["timestamp"],
            "compression_breakout",
            {"lookback": 48},
            "flat",
            "atr_trailing_exit",
            97.0,
        ),
    ]
    snapshot = simulate_paper_snapshot(
        candles,
        signals,
        BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, allowed_directions=("short",)),
    )

    output_dir = write_paper_artifacts(snapshot, tmp_path)

    assert (output_dir / "snapshot.json").exists()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "index.html").exists()
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Best Strategy Live Check" in html
    assert "Recent Paper Trades" in html


def test_format_paper_cli_summary_mentions_dashboard() -> None:
    candles = paper_fixture_candles()
    snapshot = simulate_paper_snapshot(
        candles,
        [],
        BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, allowed_directions=("short",)),
    )

    summary = format_paper_cli_summary(snapshot)

    assert "Paper trader status" in summary
    assert "Dashboard: reports/paper/index.html" in summary
