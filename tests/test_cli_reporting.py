from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import tradebot_backtest.cli as cli_module
from tradebot_backtest.cli import app, select_best_per_family
from tradebot_backtest.engine import BacktestResult, Signal
from tradebot_backtest.regime import RegimeParams


def fixture_candles(path: Path) -> None:
    closes = [100 + i * 0.4 for i in range(80)] + [132 - i * 0.5 for i in range(80)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [10.0] * len(closes),
        }
    )
    df.to_csv(path, index=False)


def fixture_funding(path: Path) -> None:
    timestamps = pd.date_range("2026-01-01", periods=160, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "funding_rate": [0.00001 + (index % 12) * 0.000001 for index in range(160)],
            "premium": [((index % 20) - 10) * 0.00001 for index in range(160)],
        }
    ).to_csv(path, index=False)


def test_cli_runs_from_fixture_and_generates_reports(tmp_path: Path) -> None:
    candle_file = tmp_path / "candles.csv"
    report_dir = tmp_path / "reports"
    fixture_candles(candle_file)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--data-file",
            str(candle_file),
            "--reports-dir",
            str(report_dir),
            "--skip-funding",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Best valid strategy" in result.output
    assert (report_dir / "latest" / "summary.csv").exists()
    assert (report_dir / "latest" / "trades.csv").exists()
    assert (report_dir / "latest" / "equity_curves.html").exists()


def test_cli_includes_funding_strategies_when_funding_file_is_supplied(tmp_path: Path) -> None:
    candle_file = tmp_path / "candles.csv"
    funding_file = tmp_path / "funding.csv"
    report_dir = tmp_path / "reports"
    fixture_candles(candle_file)
    fixture_funding(funding_file)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--data-file",
            str(candle_file),
            "--funding-file",
            str(funding_file),
            "--reports-dir",
            str(report_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = pd.read_csv(report_dir / "latest" / "summary.csv")
    assert set(summary["strategy"]) == {
        "ema_crossover",
        "rsi_mean_reversion",
        "breakout",
        "volatility_scaled_momentum",
        "bollinger_regime_reversion",
        "compression_breakout",
        "funding_crowding_reversal",
        "funding_conditioned_momentum",
    }


def test_select_best_per_family_uses_training_score() -> None:
    def result(name: str, params: dict[str, int], end_equity: float, drawdown_equity: float) -> BacktestResult:
        equity = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
                "equity": [1000.0, drawdown_equity, end_equity],
            }
        )
        return BacktestResult(name, params, [object()] * 10, equity)  # type: ignore[list-item]

    selected = select_best_per_family(
        [
            result("momentum", {"lookback": 12}, 1100.0, 950.0),
            result("momentum", {"lookback": 24}, 1080.0, 990.0),
            result("breakout", {"lookback": 20}, 1050.0, 980.0),
        ]
    )

    assert selected == {
        "momentum": {"lookback": 24},
        "breakout": {"lookback": 20},
    }


def test_regime_command_generates_comparison_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candle_file = tmp_path / "candles.csv"
    funding_file = tmp_path / "funding.csv"
    report_dir = tmp_path / "reports"
    periods = 240
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    closes = [100.0 + (index % 12) for index in range(periods)]
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [10.0] * periods,
        }
    ).to_csv(candle_file, index=False)
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "funding_rate": [0.0] * periods,
            "premium": [0.0] * periods,
        }
    ).to_csv(funding_file, index=False)
    configured = RegimeParams(0.9, 25, 20, 0.5, 24, 20, 1.5)

    def deterministic_regime_signals(candles, params, **_kwargs):
        signals = []
        for index in range(0, len(candles) - 3, 6):
            signals.append(
                Signal(
                    candles.iloc[index]["timestamp"],
                    "regime_switching",
                    params.__dict__,
                    "long",
                    "trending:test_entry",
                    float(candles.iloc[index]["close"]) - 5,
                )
            )
            signals.append(
                Signal(
                    candles.iloc[index + 3]["timestamp"],
                    "regime_switching",
                    params.__dict__,
                    "flat",
                    "regime_change",
                    float(candles.iloc[index + 3]["close"]),
                )
            )
        return signals

    monkeypatch.setattr(cli_module, "regime_parameter_grid", lambda: [configured], raising=False)
    monkeypatch.setattr(
        cli_module,
        "regime_switching_signals",
        deterministic_regime_signals,
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "regime",
            "--data-file",
            str(candle_file),
            "--funding-file",
            str(funding_file),
            "--reports-dir",
            str(report_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Selected regime parameters" in result.output
    assert "Training selection metrics" in result.output
    assert "Untouched test-period results" in result.output
    summary = pd.read_csv(report_dir / "latest" / "summary.csv")
    assert set(summary["strategy"]) == {
        "regime_switching",
        "breakout_baseline",
        "bollinger_baseline",
        "buy_and_hold",
    }
    assert (report_dir / "latest" / "regime_distribution.csv").exists()
    assert (report_dir / "latest" / "training_selection.csv").exists()
