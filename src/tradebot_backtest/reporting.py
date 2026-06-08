from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from tradebot_backtest.engine import BacktestResult, Trade, score_result


def result_summary_frame(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "strategy": result.strategy_name,
                "params": result.params,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate_pct": result.win_rate_pct,
                "trade_count": result.trade_count,
                "exposure_time_pct": result.exposure_time_pct,
                "score": score_result(result, minimum_trades=0),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def trades_frame(results: list[BacktestResult]) -> pd.DataFrame:
    trades: list[Trade] = [trade for result in results for trade in result.trades]
    if not trades:
        return pd.DataFrame(
            columns=[
                "strategy_name",
                "params",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "notional",
                "pnl",
                "fees",
                "funding",
                "return_pct",
                "exit_reason",
            ]
        )
    return pd.DataFrame([trade.__dict__ for trade in trades])


def write_reports(
    results: list[BacktestResult],
    reports_dir: Path,
    regime_labels: pd.Series | None = None,
    training_selection: pd.DataFrame | None = None,
) -> Path:
    latest = reports_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    summary = result_summary_frame(results)
    trades = trades_frame(results)
    summary.to_csv(latest / "summary.csv", index=False)
    trades.to_csv(latest / "trades.csv", index=False)
    if regime_labels is not None:
        regime_distribution_frame(regime_labels).to_csv(
            latest / "regime_distribution.csv",
            index=False,
        )
    if training_selection is not None:
        training_selection.to_csv(latest / "training_selection.csv", index=False)
    _write_equity_html(results, latest / "equity_curves.html")
    return latest


def regime_distribution_frame(labels: pd.Series) -> pd.DataFrame:
    regimes = ["high_volatility", "trending", "ranging", "unclear"]
    counts = labels.value_counts().reindex(regimes, fill_value=0)
    total = int(counts.sum())
    percent = counts.astype(float) * 100.0 / total if total else counts.astype(float)
    return pd.DataFrame(
        {
            "regime": regimes,
            "hours": counts.to_numpy(),
            "percent": percent.to_numpy(),
        }
    )


def format_cli_summary(results: list[BacktestResult], warnings: list[str] | None = None) -> str:
    summary = result_summary_frame(results)
    valid = summary[summary["trade_count"] >= 10]
    if valid.empty:
        best_line = "Best valid strategy: none (no variant reached 10 trades)"
    else:
        best = valid.iloc[0]
        best_line = f"Best valid strategy: {best['strategy']} {best['params']} score={best['score']:.4f}"
    lines = [best_line, "", "Top strategies:"]
    display = summary.head(10).copy()
    for column in ["total_return_pct", "max_drawdown_pct", "win_rate_pct", "exposure_time_pct", "score"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    lines.append(display.to_string(index=False))
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _write_equity_html(results: list[BacktestResult], path: Path) -> None:
    figure = go.Figure()
    for result in results:
        label = f"{result.strategy_name} {result.params}"
        figure.add_trace(go.Scatter(x=result.equity_curve["timestamp"], y=result.equity_curve["equity"], mode="lines", name=label))
    figure.update_layout(title="Strategy Equity Curves", xaxis_title="Time", yaxis_title="Equity")
    figure.write_html(path)
