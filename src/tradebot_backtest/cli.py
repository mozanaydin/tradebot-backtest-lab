from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from tradebot_backtest.data import (
    FundingUnavailable,
    fetch_funding_history,
    load_or_fetch_candles,
    read_funding_csv,
    refresh_candles_cache,
)
from tradebot_backtest.engine import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    buy_and_hold_result,
    run_backtest,
    score_result,
)
from tradebot_backtest.paper import (
    best_strategy_config,
    format_paper_cli_summary,
    run_paper_loop,
)
from tradebot_backtest.testnet import (
    approve_testnet_agent,
    format_testnet_cli_summary,
    format_worker_final_summary,
    load_testnet_credentials_from_env,
    load_telegram_config_from_env,
    run_testnet_worker,
    sync_best_strategy_to_testnet,
    write_testnet_dashboard,
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
    symbol: Annotated[str, typer.Option(help="Perpetual symbol or coin.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of candle history to use.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    data_file: Annotated[Path | None, typer.Option(help="Use an existing candle CSV instead of fetching.")] = None,
    funding_file: Annotated[Path | None, typer.Option(help="Use an existing funding CSV.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated reports.")] = Path("reports"),
    skip_funding: Annotated[bool, typer.Option(help="Disable optional funding fetch.")] = False,
    risk_fraction: Annotated[float, typer.Option(help="Fraction of equity risked per trade.")] = 0.05,
    direction_mode: Annotated[str, typer.Option(help="both, long_only, or short_only.")] = "both",
    session_preset: Annotated[str, typer.Option(help="all, asia, europe_us, or us.")] = "all",
    cooldown_bars_after_stop: Annotated[int, typer.Option(help="Bars to wait after an invalidation stop before re-entry.")] = 0,
    minimum_stop_distance_to_cost: Annotated[float, typer.Option(help="Skip entries whose stop distance is too small relative to round-trip cost.")] = 0.0,
) -> None:
    exchange = exchange.lower()
    _validate_exchange(exchange)
    candles = load_or_fetch_candles(exchange, symbol, interval, days, data_dir, data_file=data_file)
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=risk_fraction,
        max_leverage=3.0,
        cost_model=CostModel(),
        allowed_directions=_direction_tuple(direction_mode),
        allowed_entry_hours=_session_hours(session_preset),
        cooldown_bars_after_stop=cooldown_bars_after_stop,
        minimum_stop_distance_to_cost=minimum_stop_distance_to_cost,
    )
    warnings: list[str] = []
    funding_rates = None
    if not skip_funding:
        try:
            cached_funding = funding_file or _funding_path(exchange, symbol, data_dir)
            if cached_funding.exists():
                funding_rates = read_funding_csv(cached_funding)
            else:
                funding_rates = fetch_funding_history(exchange, symbol, candles["timestamp"].min(), candles["timestamp"].max())
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
    symbol: Annotated[str, typer.Option(help="Perpetual symbol or coin.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of candle history to use.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    data_file: Annotated[Path | None, typer.Option(help="Use an existing candle CSV instead of fetching.")] = None,
    funding_file: Annotated[Path | None, typer.Option(help="Use an existing funding CSV.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated reports.")] = Path("reports"),
    risk_fraction: Annotated[float, typer.Option(help="Fraction of equity risked per trade.")] = 0.05,
    direction_mode: Annotated[str, typer.Option(help="both, long_only, or short_only.")] = "both",
    session_preset: Annotated[str, typer.Option(help="all, asia, europe_us, or us.")] = "all",
    cooldown_bars_after_stop: Annotated[int, typer.Option(help="Bars to wait after an invalidation stop before re-entry.")] = 0,
    minimum_stop_distance_to_cost: Annotated[float, typer.Option(help="Skip entries whose stop distance is too small relative to round-trip cost.")] = 0.0,
) -> None:
    exchange = exchange.lower()
    _validate_exchange(exchange)
    candles = load_or_fetch_candles(exchange, symbol, interval, days, data_dir, data_file=data_file)
    funding_rates = None
    cached_funding = funding_file or _funding_path(exchange, symbol, data_dir)
    if cached_funding.exists():
        funding_rates = read_funding_csv(cached_funding)
    feature_frame = _merge_funding(candles, funding_rates)
    train, test = _train_test_split(feature_frame, train_fraction=0.7)
    config = BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=risk_fraction,
        max_leverage=3.0,
        cost_model=CostModel(),
        allowed_directions=_direction_tuple(direction_mode),
        allowed_entry_hours=_session_hours(session_preset),
        cooldown_bars_after_stop=cooldown_bars_after_stop,
        minimum_stop_distance_to_cost=minimum_stop_distance_to_cost,
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


@app.command()
def paper(
    exchange: Annotated[str, typer.Option(help="Exchange adapter name.")] = "hyperliquid",
    symbol: Annotated[str, typer.Option(help="Perpetual symbol or coin.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of fresh candle history to fetch each run. Keep 1h runs at 180 or below for Hyperliquid's candle snapshot limit.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    funding_file: Annotated[Path | None, typer.Option(help="Use an existing funding CSV if you want funding reflected in paper PnL.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated paper-trading artifacts.")] = Path("reports/paper"),
    starting_balance: Annotated[float, typer.Option(help="Paper account starting balance.")] = 1000.0,
    poll_seconds: Annotated[int, typer.Option(help="Seconds between refreshes. Use 0 for a one-shot run.")] = 0,
    iterations: Annotated[int, typer.Option(help="How many refresh cycles to run before stopping.")] = 1,
) -> None:
    exchange = exchange.lower()
    _validate_exchange(exchange)
    config = best_strategy_config(starting_balance=starting_balance)

    def loader() -> tuple[pd.DataFrame, pd.DataFrame | None]:
        candles = refresh_candles_cache(exchange, symbol, interval, days, data_dir)
        funding_rates = None
        if funding_file is not None and funding_file.exists():
            funding_rates = read_funding_csv(funding_file)
        return candles, funding_rates

    snapshot = run_paper_loop(
        loader=loader,
        output_dir=reports_dir,
        config=config,
        poll_seconds=poll_seconds,
        iterations=iterations,
    )
    typer.echo(format_paper_cli_summary(snapshot))


@app.command("testnet-bootstrap-agent")
def testnet_bootstrap_agent(
    master_secret_env: Annotated[str, typer.Option(help="Environment variable containing the testnet master wallet private key.")] = "HL_TESTNET_MASTER_SECRET",
    account_address_env: Annotated[str, typer.Option(help="Environment variable containing the onchain account address to trade for.")] = "HL_TESTNET_ACCOUNT_ADDRESS",
    agent_name: Annotated[str | None, typer.Option(help="Optional name for the API wallet on Hyperliquid.")] = None,
) -> None:
    master_secret = _required_env(master_secret_env)
    account_address = _required_env(account_address_env)
    approved = approve_testnet_agent(master_secret, account_address, name=agent_name)
    typer.echo("Testnet API wallet approved.")
    typer.echo(f"Account address: {approved.account_address}")
    typer.echo(f"Agent address: {approved.agent_address}")
    typer.echo("Agent secret (store this safely and export it as HL_TESTNET_API_SECRET):")
    typer.echo(approved.agent_secret)


@app.command("testnet-sync")
def testnet_sync(
    symbol: Annotated[str, typer.Option(help="Perpetual symbol or coin.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of fresh candle history to fetch for the strategy state.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    funding_file: Annotated[Path | None, typer.Option(help="Optional cached funding CSV to reflect funding in the paper side of the model.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated testnet dashboard artifacts.")] = Path("reports/testnet"),
    execute: Annotated[bool, typer.Option(help="If true, place or close Hyperliquid testnet orders. Otherwise only simulate the decision.")] = False,
    leverage: Annotated[int, typer.Option(help="Cross leverage to set before a trade action.")] = 3,
    slippage: Annotated[float, typer.Option(help="Market-order slippage guard, expressed as a fraction.")] = 0.02,
    account_address_env: Annotated[str, typer.Option(help="Environment variable containing the target testnet account address.")] = "HL_TESTNET_ACCOUNT_ADDRESS",
    api_secret_env: Annotated[str, typer.Option(help="Environment variable containing the approved testnet API wallet private key.")] = "HL_TESTNET_API_SECRET",
    vault_address_env: Annotated[str, typer.Option(help="Optional environment variable containing a vault or subaccount address.")] = "HL_TESTNET_VAULT_ADDRESS",
) -> None:
    if symbol.upper() != "BTC":
        raise typer.BadParameter("this first version only supports BTC")
    candles = refresh_candles_cache("hyperliquid", symbol.upper(), interval, days, data_dir)
    funding_rates = read_funding_csv(funding_file) if funding_file is not None and funding_file.exists() else None
    try:
        credentials = load_testnet_credentials_from_env(
            account_address_env=account_address_env,
            api_secret_env=api_secret_env,
            vault_address_env=vault_address_env,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = sync_best_strategy_to_testnet(
        credentials,
        candles,
        funding_rates=funding_rates,
        execute=execute,
        leverage=leverage,
        slippage=slippage,
    )
    write_testnet_dashboard(result, reports_dir)
    typer.echo(format_testnet_cli_summary(result))


@app.command("testnet-worker")
def testnet_worker(
    symbol: Annotated[str, typer.Option(help="Perpetual symbol or coin.")] = "BTC",
    interval: Annotated[str, typer.Option(help="Candle interval.")] = "1h",
    days: Annotated[int, typer.Option(help="Days of fresh candle history to fetch for the strategy state.")] = 180,
    data_dir: Annotated[Path, typer.Option(help="Directory for candle cache.")] = Path("data"),
    funding_file: Annotated[Path | None, typer.Option(help="Optional cached funding CSV to reflect funding in the paper side of the model.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Directory for generated testnet dashboard artifacts.")] = Path("reports/testnet"),
    execute: Annotated[bool, typer.Option(help="If true, place or close Hyperliquid testnet orders. Otherwise only simulate the decision.")] = False,
    leverage: Annotated[int, typer.Option(help="Cross leverage to set before a trade action.")] = 3,
    slippage: Annotated[float, typer.Option(help="Market-order slippage guard, expressed as a fraction.")] = 0.02,
    poll_seconds: Annotated[int, typer.Option(help="Seconds between checks.")] = 300,
    duration_hours: Annotated[float, typer.Option(help="How long the worker should run before stopping.")] = 48.0,
    extend_hours_if_no_trades: Annotated[float, typer.Option(help="One-time extension if no trades happened in the first window.")] = 48.0,
    min_account_value: Annotated[float, typer.Option(help="Safety floor in USDC; stop if account value drops below this.")] = 10.0,
    account_address_env: Annotated[str, typer.Option(help="Environment variable containing the target testnet account address.")] = "HL_TESTNET_ACCOUNT_ADDRESS",
    api_secret_env: Annotated[str, typer.Option(help="Environment variable containing the approved testnet API wallet private key.")] = "HL_TESTNET_API_SECRET",
    vault_address_env: Annotated[str, typer.Option(help="Optional environment variable containing a vault or subaccount address.")] = "HL_TESTNET_VAULT_ADDRESS",
    telegram_token_env: Annotated[str, typer.Option(help="Environment variable containing the Telegram bot token.")] = "HL_TELEGRAM_BOT_TOKEN",
    telegram_chat_id_env: Annotated[str, typer.Option(help="Environment variable containing the Telegram chat id.")] = "HL_TELEGRAM_CHAT_ID",
) -> None:
    if symbol.upper() != "BTC":
        raise typer.BadParameter("this first version only supports BTC")
    try:
        credentials = load_testnet_credentials_from_env(
            account_address_env=account_address_env,
            api_secret_env=api_secret_env,
            vault_address_env=vault_address_env,
        )
        telegram = load_telegram_config_from_env(
            token_env=telegram_token_env,
            chat_id_env=telegram_chat_id_env,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = run_testnet_worker(
        credentials,
        execute=execute,
        leverage=leverage,
        slippage=slippage,
        symbol=symbol.upper(),
        interval=interval,
        days=days,
        data_dir=data_dir,
        funding_file=funding_file,
        reports_dir=reports_dir,
        poll_seconds=poll_seconds,
        duration_hours=duration_hours,
        extend_hours_if_no_trades=extend_hours_if_no_trades,
        min_account_value=min_account_value,
        telegram=telegram,
    )
    typer.echo(format_worker_final_summary(summary))


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
    columns = [column for column in ["timestamp", "funding_rate", "premium", "mark_price"] if column in funding_rates.columns]
    merged = candles.merge(funding_rates[columns], on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
    for column in ["funding_rate", "premium", "mark_price"]:
        if column in merged.columns:
            merged[column] = merged[column].ffill()
    if "premium" not in merged.columns:
        merged["premium"] = pd.NA
    if "mark_price" in merged.columns:
        premium_proxy = merged["mark_price"] / merged["close"].replace(0, pd.NA) - 1
        merged["premium"] = merged["premium"].fillna(premium_proxy)
    return merged


def _funding_rates(candles: pd.DataFrame) -> pd.DataFrame | None:
    if "funding_rate" not in candles.columns:
        return None
    return candles[["timestamp", "funding_rate"]].dropna().reset_index(drop=True)


def _run_grid(candles: pd.DataFrame, config: BacktestConfig, include_funding: bool) -> list[BacktestResult]:
    results = []
    funding_rates = _funding_rates(candles)
    for strategy_name, params in strategy_grid(include_funding=include_funding):
        signals = generate_signals(candles, strategy_name, params)
        strategy_config = config
        if strategy_name == "breakout_funding_veto":
            strategy_config = BacktestConfig(
                starting_balance=config.starting_balance,
                risk_fraction=config.risk_fraction,
                max_leverage=config.max_leverage,
                cost_model=config.cost_model,
                include_funding=False,
                allowed_directions=config.allowed_directions,
                allowed_entry_hours=config.allowed_entry_hours,
                cooldown_bars_after_stop=config.cooldown_bars_after_stop,
                minimum_stop_distance_to_cost=config.minimum_stop_distance_to_cost,
            )
        result = run_backtest(candles, signals, strategy_config, funding_rates=funding_rates)
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


def _validate_exchange(exchange: str) -> None:
    if exchange not in {"hyperliquid", "binance"}:
        raise typer.BadParameter("supported exchanges: hyperliquid, binance")


def _funding_path(exchange: str, symbol: str, data_dir: Path) -> Path:
    return data_dir / f"{exchange}_{symbol}_funding.csv"


def _direction_tuple(direction_mode: str) -> tuple[str, ...]:
    normalized = direction_mode.lower()
    if normalized == "both":
        return ("long", "short")
    if normalized == "long_only":
        return ("long",)
    if normalized == "short_only":
        return ("short",)
    raise typer.BadParameter("direction-mode must be one of: both, long_only, short_only")


def _session_hours(session_preset: str) -> tuple[int, ...] | None:
    normalized = session_preset.lower()
    if normalized == "all":
        return None
    if normalized == "asia":
        return tuple(range(0, 8))
    if normalized == "europe_us":
        return tuple(range(7, 17))
    if normalized == "us":
        return tuple(range(13, 22))
    raise typer.BadParameter("session-preset must be one of: all, asia, europe_us, us")


def _required_env(name: str) -> str:
    value = __import__("os").getenv(name, "").strip()
    if not value:
        raise typer.BadParameter(f"missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    app()
