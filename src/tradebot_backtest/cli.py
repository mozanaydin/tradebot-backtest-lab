from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from tradebot_backtest.data import FundingUnavailable, fetch_funding_history, load_or_fetch_candles, read_funding_csv
from tradebot_backtest.engine import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    buy_and_hold_result,
    run_backtest,
    score_result,
)
from tradebot_backtest.regime import (
    RegimeParams,
    _bollinger_candidate_signals,
    classify_regime,
    regime_features,
    regime_parameter_grid,
    regime_switching_signals,
)
from tradebot_backtest.reporting import (
    format_cli_summary,
    regime_distribution_frame,
    write_reports,
)
from tradebot_backtest.strategies import (
    bollinger_regime_reversion_signals,
    breakout_signals,
    generate_signals,
    strategy_grid,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Research backtester for Hyperliquid strategies."""


@app.command()
def run(
    exchange: Annotated[str, typer.Option(help="Exchange adapter name.")] = "hyperliquid",
    symbol: Annotated[str, typer.Option(help="Perpetual coin symbol.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of candle history to use.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    data_file: Annotated[Path | None, typer.Option(help="Use an existing candle CSV instead of fetching.")] = None,
    funding_file: Annotated[Path | None, typer.Option(help="Use an existing funding CSV.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated reports.")] = Path("reports"),
    skip_funding: Annotated[bool, typer.Option(help="Disable optional funding fetch.")] = False,
) -> None:
    if exchange != "hyperliquid":
        raise typer.BadParameter("v1 supports only exchange=hyperliquid")

    candles = load_or_fetch_candles(symbol, interval, days, data_dir, data_file=data_file)
    config = BacktestConfig(starting_balance=1000.0, risk_fraction=0.05, max_leverage=3.0, cost_model=CostModel())
    warnings: list[str] = []
    funding_rates = None
    if not skip_funding:
        try:
            cached_funding = funding_file or (data_dir / f"hyperliquid_{symbol}_funding.csv")
            if cached_funding.exists():
                funding_rates = read_funding_csv(cached_funding)
            else:
                funding_rates = fetch_funding_history(symbol, candles["timestamp"].min(), candles["timestamp"].max())
                data_dir.mkdir(parents=True, exist_ok=True)
                funding_rates.to_csv(cached_funding, index=False)
        except FundingUnavailable as exc:
            warnings.append(f"Funding excluded: {exc}")

    feature_frame = _merge_funding(candles, funding_rates)
    train, test = _train_test_split(feature_frame, train_fraction=0.7)
    include_funding = funding_rates is not None and not funding_rates.empty
    train_results = _run_grid(train, config, include_funding=include_funding)
    selected = select_best_per_family(train_results)
    test_results = _run_selected(feature_frame, test, selected, config)
    write_reports(test_results, reports_dir)
    typer.echo(format_cli_summary(test_results, warnings=warnings))


@app.command()
def regime(
    exchange: Annotated[str, typer.Option(help="Exchange adapter name.")] = "hyperliquid",
    symbol: Annotated[str, typer.Option(help="Perpetual coin symbol.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of candle history to use.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    data_file: Annotated[Path | None, typer.Option(help="Use an existing candle CSV instead of fetching.")] = None,
    funding_file: Annotated[Path | None, typer.Option(help="Use an existing funding CSV.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated reports.")] = Path("reports"),
) -> None:
    if exchange != "hyperliquid":
        raise typer.BadParameter("v1 supports only exchange=hyperliquid")

    candles = load_or_fetch_candles(symbol, interval, days, data_dir, data_file=data_file)
    funding_rates = None
    cached_funding = funding_file or (data_dir / f"hyperliquid_{symbol}_funding.csv")
    if cached_funding.exists():
        funding_rates = read_funding_csv(cached_funding)
    feature_frame = _merge_funding(candles, funding_rates)
    train, test = _train_test_split(feature_frame, train_fraction=0.7)
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=0.05,
        max_leverage=3.0,
        cost_model=CostModel(),
    )

    configurations = regime_parameter_grid()
    train_features = regime_features(train)
    breakout_cache = {
        lookback: breakout_signals(train, lookback)
        for lookback in {configured.donchian_lookback for configured in configurations}
    }
    bollinger_cache = {
        (length, entry_z): _bollinger_candidate_signals(train, length, entry_z)
        for length in {configured.bollinger_length for configured in configurations}
        for entry_z in {configured.bollinger_entry_z for configured in configurations}
    }
    training_results: list[BacktestResult] = []
    for configured in configurations:
        signals = regime_switching_signals(
            train,
            configured,
            features=train_features,
            component_signals={
                "breakout": breakout_cache[configured.donchian_lookback],
                "bollinger": bollinger_cache[
                    (configured.bollinger_length, configured.bollinger_entry_z)
                ],
            },
        )
        result = run_backtest(
            train,
            signals,
            config,
            funding_rates=_funding_rates(train),
        )
        if not signals:
            result.strategy_name = "regime_switching"
            result.params = configured.__dict__
        training_results.append(result)

    valid = [result for result in training_results if result.trade_count >= 10]
    if not valid:
        raise typer.BadParameter("no regime configuration reached 10 training trades")
    selected_result = max(
        valid,
        key=lambda result: (
            score_result(result),
            result.total_return_pct,
            -result.max_drawdown_pct,
        ),
    )
    selected = RegimeParams(**selected_result.params)

    full_features = regime_features(feature_frame)
    test_start = pd.Timestamp(test["timestamp"].min())
    all_regime_signals = regime_switching_signals(
        feature_frame,
        selected,
        features=full_features,
        start_at=test_start,
    )
    test_regime_signals = all_regime_signals
    regime_result = run_backtest(
        test,
        test_regime_signals,
        config,
        funding_rates=_funding_rates(test),
    )
    if not test_regime_signals:
        regime_result.strategy_name = "regime_switching"
        regime_result.params = selected.__dict__

    breakout_test_signals = [
        signal
        for signal in breakout_signals(feature_frame, selected.donchian_lookback)
        if signal.timestamp >= test_start
    ]
    breakout_result = run_backtest(
        test,
        breakout_test_signals,
        config,
        funding_rates=_funding_rates(test),
    )
    breakout_result.strategy_name = "breakout_baseline"
    breakout_result.params = {"lookback": selected.donchian_lookback}

    bollinger_test_signals = [
        signal
        for signal in _bollinger_candidate_signals(
            feature_frame,
            length=selected.bollinger_length,
            entry_z=selected.bollinger_entry_z,
        )
        if signal.timestamp >= test_start
    ]
    bollinger_result = run_backtest(
        test,
        bollinger_test_signals,
        config,
        funding_rates=_funding_rates(test),
    )
    bollinger_result.strategy_name = "bollinger_baseline"
    bollinger_result.params = {
        "length": selected.bollinger_length,
        "entry_z": selected.bollinger_entry_z,
    }
    hold_result = buy_and_hold_result(test, config)

    test_features = full_features[full_features["timestamp"] >= test_start]
    regime_labels = test_features.apply(
        lambda row: classify_regime(
            float(row["atr_percentile"]),
            float(row["adx14"]),
            float(row["normalized_ema_slope"]),
            selected,
        ),
        axis=1,
    )
    results = [regime_result, breakout_result, bollinger_result, hold_result]
    training_selection = pd.DataFrame(
        [
            {
                "strategy": "regime_switching",
                "params": selected.__dict__,
                "total_return_pct": selected_result.total_return_pct,
                "max_drawdown_pct": selected_result.max_drawdown_pct,
                "win_rate_pct": selected_result.win_rate_pct,
                "trade_count": selected_result.trade_count,
                "exposure_time_pct": selected_result.exposure_time_pct,
                "score": score_result(selected_result),
            }
        ]
    )
    write_reports(
        results,
        reports_dir,
        regime_labels=regime_labels,
        training_selection=training_selection,
    )
    typer.echo(f"Selected regime parameters: {selected}")
    typer.echo(
        "Training selection metrics: "
        f"return={selected_result.total_return_pct:.4f}% "
        f"drawdown={selected_result.max_drawdown_pct:.4f}% "
        f"trades={selected_result.trade_count} "
        f"score={score_result(selected_result):.4f}"
    )
    typer.echo("\nTest-period regime distribution:")
    typer.echo(regime_distribution_frame(regime_labels).to_string(index=False))
    typer.echo("\nUntouched test-period results:")
    typer.echo(format_cli_summary(results))


def _train_test_split(candles: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = max(1, min(len(candles) - 1, int(len(candles) * train_fraction)))
    return candles.iloc[:split].reset_index(drop=True), candles.iloc[split:].reset_index(drop=True)


def select_best_per_family(results: list[BacktestResult]) -> dict[str, dict[str, int | float | str]]:
    by_family: dict[str, list[BacktestResult]] = {}
    for result in results:
        by_family.setdefault(result.strategy_name, []).append(result)
    selected = {}
    for family, family_results in by_family.items():
        best = max(family_results, key=lambda result: (score_result(result), result.total_return_pct))
        selected[family] = best.params
    return selected


def _merge_funding(candles: pd.DataFrame, funding_rates: pd.DataFrame | None) -> pd.DataFrame:
    if funding_rates is None or funding_rates.empty:
        return candles.copy()
    return candles.merge(funding_rates[["timestamp", "funding_rate", "premium"]], on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)


def _funding_rates(candles: pd.DataFrame) -> pd.DataFrame | None:
    if "funding_rate" not in candles.columns:
        return None
    return candles[["timestamp", "funding_rate"]].dropna().reset_index(drop=True)


def _run_grid(candles: pd.DataFrame, config: BacktestConfig, include_funding: bool) -> list[BacktestResult]:
    results = []
    funding_rates = _funding_rates(candles)
    for strategy_name, params in strategy_grid(include_funding=include_funding):
        signals = generate_signals(candles, strategy_name, params)
        result = run_backtest(candles, signals, config, funding_rates=funding_rates)
        if not signals:
            result.strategy_name = strategy_name
            result.params = params
        results.append(result)
    return results


def _run_selected(
    full_frame: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, dict[str, int | float | str]],
    config: BacktestConfig,
) -> list[BacktestResult]:
    test_start = test["timestamp"].min()
    funding_rates = _funding_rates(test)
    results = []
    for strategy_name, params in selected.items():
        all_signals = generate_signals(full_frame, strategy_name, params)
        test_signals = [signal for signal in all_signals if signal.timestamp >= test_start]
        result = run_backtest(test, test_signals, config, funding_rates=funding_rates)
        if not test_signals:
            result.strategy_name = strategy_name
            result.params = params
        results.append(result)
    return results
