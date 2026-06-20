from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradebot_backtest.cli import _merge_funding
from tradebot_backtest.data import read_candles_csv, read_funding_csv
from tradebot_backtest.engine import BacktestConfig, CostModel, run_backtest
from tradebot_backtest.regime import filter_entry_signals_by_regime, regime_features
from tradebot_backtest.reporting import result_summary_frame, trades_frame, write_reports
from tradebot_backtest.strategies import (
    compression_breakout_exit_variant_signals,
    filter_signals_by_adverse_funding,
    generate_signals,
)

MINIMUM_VALIDATION_TRADES = 20


@dataclass(frozen=True)
class Experiment:
    strategy_name: str
    strategy_params: dict[str, int | float]
    risk_fraction: float
    direction_mode: str
    session_preset: str
    cooldown_bars_after_stop: int
    minimum_stop_distance_to_cost: float
    regime_filter: dict[str, int | float | bool | str] | None = None

    def label(self) -> str:
        regime_label = self.regime_filter if self.regime_filter is not None else "none"
        return (
            f"{self.strategy_name}|{self.strategy_params}|risk={self.risk_fraction}"
            f"|dir={self.direction_mode}|session={self.session_preset}"
            f"|cooldown={self.cooldown_bars_after_stop}|cost_buffer={self.minimum_stop_distance_to_cost}"
            f"|regime={regime_label}"
        )


def direction_tuple(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return ("long", "short")
    if mode == "long_only":
        return ("long",)
    if mode == "short_only":
        return ("short",)
    raise ValueError(f"unsupported direction mode: {mode}")


def session_hours(preset: str) -> tuple[int, ...] | None:
    if preset == "all":
        return None
    if preset == "europe_us":
        return tuple(range(7, 17))
    if preset == "us":
        return tuple(range(13, 22))
    raise ValueError(f"unsupported session preset: {preset}")


def build_experiments() -> list[Experiment]:
    strategy_variants = [
        ("breakout", {"lookback": 20}, ("both", "long_only", "short_only")),
        ("breakout", {"lookback": 50}, ("both", "long_only", "short_only")),
        ("breakout", {"lookback": 100}, ("both", "long_only", "short_only")),
        ("breakout", {"lookback": 20, "funding_window": 72, "max_adverse_funding_z": 1.0}, ("both", "long_only", "short_only")),
        ("breakout", {"lookback": 100, "funding_window": 72, "max_adverse_funding_z": 1.0}, ("both", "long_only", "short_only")),
        ("pullback_in_trend", {"fast_length": 20, "slow_length": 200, "rsi_length": 14, "recovery_level": 45, "invalidation_window": 10}, ("both", "long_only", "short_only")),
        ("pullback_in_trend", {"fast_length": 50, "slow_length": 200, "rsi_length": 14, "recovery_level": 45, "invalidation_window": 10}, ("both", "long_only", "short_only")),
        ("compression_breakout", {"lookback": 24, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14}, ("both", "long_only", "short_only")),
        ("compression_breakout", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14}, ("both", "long_only", "short_only")),
        ("compression_breakout", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14, "exit_style": "profit_trail", "profit_take_r": 1.0, "reduced_exposure": 0.5, "post_take_trailing_atr": 0.5}, ("short_only",)),
        ("compression_breakout", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14, "exit_style": "time_stop_trail", "max_hold_bars": 3}, ("short_only",)),
        ("compression_breakout_retest", {"lookback": 24, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14, "retest_atr_buffer": 0.75, "retest_window_bars": 3}, ("both",)),
        ("compression_breakout_retest", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14, "retest_atr_buffer": 0.75, "retest_window_bars": 3}, ("both",)),
        ("compression_breakout_retest", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.25, "atr_length": 14, "retest_atr_buffer": 1.0, "retest_window_bars": 4, "funding_window": 72, "max_adverse_funding_z": 1.0}, ("both",)),
        ("compression_breakout_retest", {"lookback": 48, "bandwidth_window": 100, "compression_quantile": 0.2, "volume_multiplier": 1.35, "atr_length": 14, "retest_atr_buffer": 0.6, "retest_window_bars": 2}, ("long_only",)),
        ("compression_breakout_retest", {"lookback": 48, "bandwidth_window": 120, "compression_quantile": 0.25, "volume_multiplier": 1.1, "atr_length": 14, "retest_atr_buffer": 1.25, "retest_window_bars": 5, "funding_window": 72, "max_adverse_funding_z": 0.75}, ("short_only",)),
        ("compression_breakout_retest", {"lookback": 24, "bandwidth_window": 100, "compression_quantile": 0.25, "volume_multiplier": 1.0, "atr_length": 14, "retest_atr_buffer": 1.0, "retest_window_bars": 4}, ("short_only",)),
        ("compression_breakout_retest", {"lookback": 24, "bandwidth_window": 100, "compression_quantile": 0.25, "volume_multiplier": 1.0, "atr_length": 14, "retest_atr_buffer": 1.0, "retest_window_bars": 4, "exit_style": "profit_trail", "profit_take_r": 1.0, "reduced_exposure": 0.5, "post_take_trailing_atr": 0.5}, ("short_only",)),
        ("compression_breakout_retest", {"lookback": 24, "bandwidth_window": 100, "compression_quantile": 0.25, "volume_multiplier": 1.0, "atr_length": 14, "retest_atr_buffer": 1.0, "retest_window_bars": 4, "exit_style": "time_stop_trail", "max_hold_bars": 3}, ("short_only",)),
        ("volatility_scaled_momentum", {"lookback": 24, "atr_length": 14, "atr_multiplier": 2.0}, ("both", "long_only", "short_only")),
        ("volatility_scaled_momentum", {"lookback": 48, "atr_length": 14, "atr_multiplier": 2.5}, ("both", "long_only", "short_only")),
    ]
    experiments: list[Experiment] = []
    for strategy_name, strategy_params, direction_modes in strategy_variants:
        for risk_fraction in [0.005]:
            for direction_mode in direction_modes:
                for session_preset in ["all", "europe_us"]:
                    for cooldown_bars_after_stop in [0]:
                        for minimum_stop_distance_to_cost in [1.5]:
                            experiments.append(
                                Experiment(
                                    strategy_name=strategy_name,
                                    strategy_params=strategy_params,
                                    risk_fraction=risk_fraction,
                                    direction_mode=direction_mode,
                                    session_preset=session_preset,
                                    cooldown_bars_after_stop=cooldown_bars_after_stop,
                                    minimum_stop_distance_to_cost=minimum_stop_distance_to_cost,
                                )
                            )
    return apply_regime_filters_to_experiments(experiments)


def apply_regime_filters_to_experiments(experiments: list[Experiment]) -> list[Experiment]:
    filter_presets = [
        None,
        {
            "label": "trend_quality",
            "min_adx": 20.0,
            "min_abs_slope": 0.35,
            "min_atr_percentile": 0.15,
            "max_atr_percentile": 0.85,
            "require_trend_alignment": True,
            "require_price_above_ema200": True,
            "require_ema_stack": False,
            "max_distance_from_ema50_atr": 2.5,
            "min_volume_ratio": 0.9,
        },
        {
            "label": "stacked_trend",
            "min_adx": 25.0,
            "min_abs_slope": 0.5,
            "min_atr_percentile": 0.3,
            "max_atr_percentile": 0.7,
            "require_trend_alignment": True,
            "require_price_above_ema200": True,
            "require_ema_stack": True,
            "max_distance_from_ema50_atr": 1.75,
            "min_volume_ratio": 1.0,
        },
        {
            "label": "quiet_trend",
            "min_adx": 18.0,
            "min_abs_slope": 0.3,
            "min_atr_percentile": 0.1,
            "max_atr_percentile": 0.6,
            "require_trend_alignment": True,
            "require_price_above_ema200": True,
            "require_ema_stack": False,
            "max_distance_from_ema50_atr": 1.5,
            "min_volume_ratio": 0.8,
        },
    ]
    supported = {"breakout", "pullback_in_trend", "compression_breakout", "compression_breakout_retest", "volatility_scaled_momentum"}
    expanded: list[Experiment] = []
    for experiment in experiments:
        if experiment.strategy_name not in supported:
            expanded.append(experiment)
            continue
        for filter_preset in filter_presets:
            expanded.append(
                Experiment(
                    strategy_name=experiment.strategy_name,
                    strategy_params=dict(experiment.strategy_params),
                    risk_fraction=experiment.risk_fraction,
                    direction_mode=experiment.direction_mode,
                    session_preset=experiment.session_preset,
                    cooldown_bars_after_stop=experiment.cooldown_bars_after_stop,
                    minimum_stop_distance_to_cost=experiment.minimum_stop_distance_to_cost,
                    regime_filter=dict(filter_preset) if filter_preset is not None else None,
                )
            )
    return expanded


def chronological_holdout(frame: pd.DataFrame, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(frame) * (1 - holdout_fraction))
    return frame.iloc[:split].reset_index(drop=True), frame.iloc[split:].reset_index(drop=True)


def prepare_frame_bundle(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "candles": frame,
        "regime_features": regime_features(frame),
    }


def build_validation_folds(frame: pd.DataFrame) -> list[tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]]:
    fold_boundaries = [
        (0.50, 0.10),
        (0.60, 0.10),
        (0.70, 0.10),
    ]
    folds: list[tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = []
    for train_end_fraction, validation_fraction in fold_boundaries:
        train_end = int(len(frame) * train_end_fraction)
        validation_end = int(len(frame) * (train_end_fraction + validation_fraction))
        train = prepare_frame_bundle(frame.iloc[:train_end].reset_index(drop=True))
        validation = prepare_frame_bundle(frame.iloc[train_end:validation_end].reset_index(drop=True))
        folds.append((train, validation))
    return folds


def experiment_config(experiment: Experiment, cost_multiplier: float = 1.0) -> BacktestConfig:
    cost_model = CostModel(
        fee_rate=0.00045 * cost_multiplier,
        slippage_rate=0.0002 * cost_multiplier,
    )
    return BacktestConfig(
        starting_balance=1000.0,
        risk_fraction=experiment.risk_fraction,
        max_leverage=3.0,
        cost_model=cost_model,
        include_funding=False,
        allowed_directions=direction_tuple(experiment.direction_mode),  # type: ignore[arg-type]
        allowed_entry_hours=session_hours(experiment.session_preset),
        cooldown_bars_after_stop=experiment.cooldown_bars_after_stop,
        minimum_stop_distance_to_cost=experiment.minimum_stop_distance_to_cost,
    )


def evaluate_experiment(frame_bundle: dict[str, pd.DataFrame], experiment: Experiment, cost_multiplier: float = 1.0):
    frame = frame_bundle["candles"]
    if experiment.strategy_name in {"compression_breakout", "compression_breakout_retest"} and "exit_style" in experiment.strategy_params:
        signals = compression_breakout_exit_variant_signals(
            frame,
            setup="retest" if experiment.strategy_name == "compression_breakout_retest" else "breakout",
            lookback=int(experiment.strategy_params["lookback"]),
            bandwidth_window=int(experiment.strategy_params["bandwidth_window"]),
            compression_quantile=float(experiment.strategy_params["compression_quantile"]),
            volume_multiplier=float(experiment.strategy_params["volume_multiplier"]),
            atr_length=int(experiment.strategy_params["atr_length"]),
            exit_style=str(experiment.strategy_params["exit_style"]),
            retest_atr_buffer=float(experiment.strategy_params.get("retest_atr_buffer", 0.75)),
            retest_window_bars=int(experiment.strategy_params.get("retest_window_bars", 3)),
            profit_take_r=float(experiment.strategy_params.get("profit_take_r", 1.0)),
            reduced_exposure=float(experiment.strategy_params.get("reduced_exposure", 0.5)),
            post_take_trailing_atr=float(experiment.strategy_params.get("post_take_trailing_atr", 0.5)),
            max_hold_bars=int(experiment.strategy_params.get("max_hold_bars", 3)),
        )
    else:
        signals = generate_signals(frame, experiment.strategy_name, experiment.strategy_params)
    if (
        experiment.strategy_name in {"breakout", "compression_breakout", "compression_breakout_retest"}
        and "funding_window" in experiment.strategy_params
        and "max_adverse_funding_z" in experiment.strategy_params
    ):
        signals = filter_signals_by_adverse_funding(
            frame,
            signals,
            funding_window=int(experiment.strategy_params["funding_window"]),
            max_adverse_funding_z=float(experiment.strategy_params["max_adverse_funding_z"]),
        )
    if experiment.regime_filter is not None:
        signals = filter_entry_signals_by_regime(
            signals,
            frame_bundle["regime_features"],
            min_adx=float(experiment.regime_filter["min_adx"]),
            min_abs_slope=float(experiment.regime_filter["min_abs_slope"]),
            min_atr_percentile=float(experiment.regime_filter["min_atr_percentile"]),
            max_atr_percentile=float(experiment.regime_filter["max_atr_percentile"]),
            require_trend_alignment=bool(experiment.regime_filter["require_trend_alignment"]),
            require_price_above_ema200=bool(experiment.regime_filter.get("require_price_above_ema200", False)),
            require_ema_stack=bool(experiment.regime_filter.get("require_ema_stack", False)),
            max_distance_from_ema50_atr=(
                float(experiment.regime_filter["max_distance_from_ema50_atr"])
                if experiment.regime_filter.get("max_distance_from_ema50_atr") is not None
                else None
            ),
            min_volume_ratio=(
                float(experiment.regime_filter["min_volume_ratio"])
                if experiment.regime_filter.get("min_volume_ratio") is not None
                else None
            ),
        )
    result = run_backtest(frame, signals, experiment_config(experiment, cost_multiplier=cost_multiplier))
    if not signals:
        result.strategy_name = experiment.strategy_name
        result.params = dict(experiment.strategy_params)
    return result


def rank_leaderboard(leaderboard: pd.DataFrame, minimum_validation_trades: int = MINIMUM_VALIDATION_TRADES) -> pd.DataFrame:
    ranked = leaderboard.copy()
    ranked["meets_minimum_validation_trades"] = ranked["validation_total_trades"] >= minimum_validation_trades
    return ranked.sort_values(
        by=[
            "meets_minimum_validation_trades",
            "cost_robustness_ratio",
            "stability_ratio",
            "validation_median_return_pct",
            "validation_median_drawdown_pct",
            "validation_total_trades",
        ],
        ascending=[False, False, False, False, True, False],
    ).reset_index(drop=True)


def select_short_compression_showdown(leaderboard: pd.DataFrame, max_retests: int = 3) -> pd.DataFrame:
    short_only = leaderboard[
        leaderboard["direction_mode"].eq("short_only")
        & leaderboard["meets_minimum_validation_trades"].fillna(False)
    ].copy()
    if short_only.empty:
        return short_only
    plain = short_only[short_only["strategy"].eq("compression_breakout")].head(1)
    retests = short_only[short_only["strategy"].eq("compression_breakout_retest")].head(max_retests)
    return pd.concat([plain, retests], ignore_index=True)


def select_short_compression_regime_showdown(leaderboard: pd.DataFrame, max_filtered: int = 3) -> pd.DataFrame:
    short_only = leaderboard[
        leaderboard["strategy"].eq("compression_breakout")
        & leaderboard["direction_mode"].eq("short_only")
        & leaderboard["meets_minimum_validation_trades"].fillna(False)
    ].copy()
    if short_only.empty:
        return short_only
    baseline = short_only[short_only["regime_filter"].isna()].head(1)
    filtered = short_only[short_only["regime_filter"].notna()].head(max_filtered)
    return pd.concat([baseline, filtered], ignore_index=True)


def leaderboard_row(experiment: Experiment, base_results, stressed_results) -> dict[str, object]:
    base_returns = [result.total_return_pct for result in base_results]
    base_drawdowns = [result.max_drawdown_pct for result in base_results]
    base_trades = [result.trade_count for result in base_results]
    stressed_returns = [result.total_return_pct for result in stressed_results]
    positive_base = sum(1 for value in base_returns if value > 0)
    positive_stressed = sum(1 for value in stressed_returns if value > 0)
    return {
        "strategy": experiment.strategy_name,
        "strategy_params": experiment.strategy_params,
        "risk_fraction": experiment.risk_fraction,
        "direction_mode": experiment.direction_mode,
        "session_preset": experiment.session_preset,
        "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
        "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
        "regime_filter": experiment.regime_filter,
        "validation_median_return_pct": float(pd.Series(base_returns).median()),
        "validation_median_drawdown_pct": float(pd.Series(base_drawdowns).median()),
        "validation_total_trades": int(sum(base_trades)),
        "validation_positive_folds": positive_base,
        "cost_robust_positive_folds": positive_stressed,
        "stability_ratio": positive_base / len(base_results),
        "cost_robustness_ratio": positive_stressed / len(stressed_results),
        "selection_score": (
            positive_stressed / len(stressed_results),
            positive_base / len(base_results),
            float(pd.Series(base_returns).median()),
            -float(pd.Series(base_drawdowns).median()),
            int(sum(base_trades)),
        ),
    }


def format_experiment_headline(row: pd.Series) -> str:
    strategy = str(row["strategy"]).replace("_", " ").title()
    params = row["strategy_params"]
    if isinstance(params, str):
        return f"{strategy} strategy"
    if isinstance(params, dict):
        if "lookback" in params:
            return f"{strategy} with {params['lookback']}-bar setup"
        if "fast_length" in params and "slow_length" in params:
            return f"{strategy} with {params['fast_length']}/{params['slow_length']} trend lengths"
    return f"{strategy} strategy"


def format_holdout_headline(row: pd.Series) -> str:
    strategy = str(row["strategy"]).replace("_", " ").title()
    params = _parse_mapping(row.get("params"))
    if params is None:
        return f"{strategy} strategy"
    if "lookback" in params:
        return f"{strategy} with {params['lookback']}-bar setup"
    if "fast_length" in params and "slow_length" in params:
        return f"{strategy} with {params['fast_length']}/{params['slow_length']} trend lengths"
    return f"{strategy} strategy"


def explain_verdict(best_holdout_return_pct: float) -> tuple[str, str]:
    if best_holdout_return_pct > 5:
        return ("Promising", "This version held up reasonably well on fresh unseen data.")
    if best_holdout_return_pct > 0:
        return ("Mixed", "This version made a little money on unseen data, but not enough to trust yet.")
    if best_holdout_return_pct > -10:
        return ("Not Ready", "This looked somewhat interesting in testing, but it still lost money in the final reality check.")
    return ("Weak", "The bot still lost meaningful money on fresh unseen data, so it is not ready for real trading.")


def explain_next_step(best_row: pd.Series, best_holdout_return_pct: float) -> str:
    if best_holdout_return_pct > 0:
        return "Keep this family, then test tighter market-condition rules before risking real money."
    if pd.notna(best_row.get("regime_filter")):
        return "The market filter helped the practice rounds, so the next step is testing even stricter structure rules instead of adding more random settings."
    return "The next step is adding smarter market-condition rules, because changing trade size alone is not fixing the core idea."


def summarize_best_setup(row: pd.Series) -> str:
    direction = str(row["direction_mode"]).replace("_", " ")
    session = str(row["session_preset"]).replace("_", " ")
    risk = float(row["risk_fraction"]) * 100
    return f"{format_experiment_headline(row)}, trading {direction}, during {session} hours, risking about {risk:.2f}% per trade."


def _parse_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return None
        parsed = literal_eval(stripped)
        if isinstance(parsed, dict):
            return dict(parsed)
    return None


def experiment_from_row(row: pd.Series) -> Experiment:
    return Experiment(
        strategy_name=str(row["strategy"]),
        strategy_params=_parse_mapping(row["strategy_params"]) or {},
        risk_fraction=float(row["risk_fraction"]),
        direction_mode=str(row["direction_mode"]),
        session_preset=str(row["session_preset"]),
        cooldown_bars_after_stop=int(row["cooldown_bars_after_stop"]),
        minimum_stop_distance_to_cost=float(row["minimum_stop_distance_to_cost"]),
        regime_filter=_parse_mapping(row.get("regime_filter")),
    )


def build_top_candidates_table(leaderboard: pd.DataFrame) -> str:
    table = leaderboard.head(8).copy()
    table["Strategy Idea"] = table.apply(format_experiment_headline, axis=1)
    table["Practice Result"] = table["validation_median_return_pct"].map(lambda value: f"{value:.1f}%")
    table["Worst Pain"] = table["validation_median_drawdown_pct"].map(lambda value: f"{value:.1f}%")
    table["Trades"] = table["validation_total_trades"].astype(int)
    table["Consistency"] = table.apply(
        lambda row: f"{int(row['validation_positive_folds'])}/3 good rounds, {int(row['cost_robust_positive_folds'])}/3 after higher costs",
        axis=1,
    )
    table["Why It Ranked Here"] = table.apply(
        lambda row: "Stable across all practice rounds" if float(row["cost_robustness_ratio"]) == 1.0 else "Good, but less stable under tougher assumptions",
        axis=1,
    )
    display = table[["Strategy Idea", "Practice Result", "Worst Pain", "Trades", "Consistency", "Why It Ranked Here"]]
    return display.to_html(index=False, border=0, classes="friendly-table")


def build_holdout_table(holdout_summary: pd.DataFrame) -> str:
    table = holdout_summary.copy()
    table["Strategy Idea"] = table.apply(
        lambda row: str(row["strategy"]).replace("_", " ").title(),
        axis=1,
    )
    table["Final Reality Check"] = table["total_return_pct"].map(lambda value: f"{value:.1f}%")
    table["Worst Drop"] = table["max_drawdown_pct"].map(lambda value: f"{value:.1f}%")
    table["Win Rate"] = table["win_rate_pct"].map(lambda value: f"{value:.1f}%")
    table["Trades"] = table["trade_count"].astype(int)
    table["Verdict"] = table["total_return_pct"].map(
        lambda value: "Best of the bunch" if value == table["total_return_pct"].max() else "Worse than the leader"
    )
    display = table[["Strategy Idea", "Final Reality Check", "Worst Drop", "Win Rate", "Trades", "Verdict"]].head(8)
    return display.to_html(index=False, border=0, classes="friendly-table")


def build_holdout_chart(holdout_summary: pd.DataFrame) -> str:
    chart = holdout_summary.head(6).copy()
    if chart.empty:
        return "<p class='note'>No holdout results available yet.</p>"
    chart["label"] = chart["strategy"].astype(str).str.replace("_", " ", regex=False).str.title()
    max_abs = max(1.0, float(chart["total_return_pct"].abs().max()))
    bars: list[str] = []
    for _, row in chart.iterrows():
        value = float(row["total_return_pct"])
        width = max(6.0, abs(value) / max_abs * 100.0)
        offset = 50.0 if value >= 0 else 50.0 - width
        bar_class = "pos" if value >= 0 else "neg"
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{row['label']}</div>
              <div class="bar-track">
                <div class="bar bar-mid"></div>
                <div class="bar {bar_class}" style="left:{offset:.2f}%; width:{width:.2f}%"></div>
              </div>
              <div class="bar-value">{value:.1f}%</div>
            </div>
            """
        )
    return "".join(bars)


def build_trade_examples(holdout_trades: pd.DataFrame) -> str:
    if holdout_trades.empty:
        return "<p class='note'>No trades to show yet.</p>"
    table = holdout_trades.copy()
    for column in ["entry_time", "exit_time"]:
        table[column] = pd.to_datetime(table[column], utc=True, errors="coerce")
    table["duration_hours"] = ((table["exit_time"] - table["entry_time"]).dt.total_seconds().fillna(0) / 3600.0)
    sample = pd.concat(
        [
            table.nlargest(2, "pnl"),
            table.nsmallest(2, "pnl"),
            table.sort_values("entry_time").tail(2),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["strategy_name", "entry_time", "exit_time"]).head(6)
    cards: list[str] = []
    for _, row in sample.iterrows():
        pnl = float(row["pnl"])
        return_pct = float(row["return_pct"])
        side = str(row["side"]).title()
        entry_time = pd.Timestamp(row["entry_time"]).strftime("%Y-%m-%d %H:%M UTC")
        exit_time = pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d %H:%M UTC")
        cards.append(
            f"""
            <article class="trade-card {'win' if pnl >= 0 else 'loss'}">
              <div class="trade-card-top">
                <div>
                  <div class="trade-title">{str(row['strategy_name']).replace('_', ' ').title()} • {side}</div>
                  <div class="trade-subtitle">{entry_time} → {exit_time}</div>
                </div>
                <div class="trade-pnl">{pnl:+.2f} USD</div>
              </div>
              <div class="trade-grid">
                <div><span>Investment size</span><strong>{float(row['notional']):,.0f} USD</strong></div>
                <div><span>Trade return</span><strong>{return_pct:+.2f}%</strong></div>
                <div><span>Entry → exit</span><strong>{float(row['entry_price']):,.0f} → {float(row['exit_price']):,.0f}</strong></div>
                <div><span>Held for</span><strong>{float(row['duration_hours']):.1f} hours</strong></div>
                <div><span>Fees + funding</span><strong>{float(row['fees']):.2f} + {float(row['funding']):.2f}</strong></div>
                <div><span>Exit reason</span><strong>{str(row['exit_reason']).replace('_', ' ')}</strong></div>
              </div>
            </article>
            """
        )
    return "".join(cards)


def build_showdown_table(showdown_summary: pd.DataFrame) -> str:
    if showdown_summary.empty:
        return "<p class='note'>No short-side showdown data available yet.</p>"
    table = showdown_summary.copy()
    table["Strategy Idea"] = table.apply(
        lambda row: format_holdout_headline(row),
        axis=1,
    )
    table["Holdout Return"] = table["total_return_pct"].map(lambda value: f"{value:.1f}%")
    table["Worst Drop"] = table["max_drawdown_pct"].map(lambda value: f"{value:.1f}%")
    table["Win Rate"] = table["win_rate_pct"].map(lambda value: f"{value:.1f}%")
    table["Trades"] = table["trade_count"].astype(int)
    display = table[["Strategy Idea", "Holdout Return", "Worst Drop", "Win Rate", "Trades"]]
    return display.to_html(index=False, border=0, classes="friendly-table")


def build_regime_showdown_table(showdown_summary: pd.DataFrame) -> str:
    if showdown_summary.empty:
        return "<p class='note'>No regime-filter showdown data available yet.</p>"
    table = showdown_summary.copy()
    table["Strategy Idea"] = table.apply(lambda row: format_holdout_headline(row), axis=1)
    table["Filter"] = table["params"].apply(
        lambda value: (_parse_mapping(value) or {}).get("regime_filter", {}).get("label", "none")
        if isinstance((_parse_mapping(value) or {}).get("regime_filter"), dict)
        else "none"
    )
    table["Holdout Return"] = table["total_return_pct"].map(lambda value: f"{value:.1f}%")
    table["Worst Drop"] = table["max_drawdown_pct"].map(lambda value: f"{value:.1f}%")
    table["Trades"] = table["trade_count"].astype(int)
    display = table[["Strategy Idea", "Filter", "Holdout Return", "Worst Drop", "Trades"]]
    return display.to_html(index=False, border=0, classes="friendly-table")


def describe_keep_focus(best_holdout: pd.Series) -> str:
    return f"{format_holdout_headline(best_holdout)} is the first setup here that passed the final unseen-data check with a positive result."


def describe_caution(best_validation: pd.Series, best_holdout: pd.Series) -> str:
    if float(best_holdout["total_return_pct"]) > 0:
        if str(best_validation["strategy"]) != str(best_holdout["strategy"]):
            return "The final test finally made money, but the practice-round winner and the final winner were different, so we still need one more round of confirmation."
        return "The final test made money, but we still need to confirm that this result survives fresh data and tougher execution assumptions."
    return "The final unseen-data test was still negative, which matters more than pretty practice results."


def write_dashboard(
    output_dir: Path,
    leaderboard: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    holdout_trades: pd.DataFrame,
    showdown_summary: pd.DataFrame | None = None,
    regime_showdown_summary: pd.DataFrame | None = None,
) -> None:
    best_validation = leaderboard.iloc[0]
    best_holdout = holdout_summary.iloc[0]
    best_holdout_return_pct = float(best_holdout["total_return_pct"])
    verdict_label, verdict_text = explain_verdict(best_holdout_return_pct)
    next_step_text = explain_next_step(best_validation, best_holdout_return_pct)
    summary_sentence = summarize_best_setup(best_validation)
    holdout_winner_sentence = format_holdout_headline(best_holdout)
    caution_text = describe_caution(best_validation, best_holdout)
    keep_focus_text = describe_keep_focus(best_holdout)
    final_metric_class = "good" if best_holdout_return_pct > 0 else "warn" if best_holdout_return_pct > -10 else "bad"
    top_candidates_table = build_top_candidates_table(leaderboard)
    holdout_table = build_holdout_table(holdout_summary)
    holdout_chart = build_holdout_chart(holdout_summary)
    trade_examples = build_trade_examples(holdout_trades)
    showdown_table = build_showdown_table(showdown_summary if showdown_summary is not None else pd.DataFrame())
    regime_showdown_table = build_regime_showdown_table(regime_showdown_summary if regime_showdown_summary is not None else pd.DataFrame())
    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tradebot Long History Lab</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07111f;
      --bg-soft: #101b2d;
      --panel: #122033;
      --panel-strong: #16263b;
      --border: #24364f;
      --text: #e5eef9;
      --muted: #9eb0c8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #f97316;
      --accent: #60a5fa;
      --accent-soft: #dbeafe;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Inter, Arial, sans-serif; background: linear-gradient(180deg, #07111f 0%, #0d1727 100%); color: var(--text); margin: 0; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 80px; }}
    h1, h2, h3 {{ margin: 0; }}
    p {{ margin: 0; line-height: 1.5; }}
    .hero {{ display:grid; grid-template-columns: 1.3fr .7fr; gap: 18px; margin-bottom: 20px; }}
    .hero-card, .panel {{ background: rgba(18, 32, 51, 0.92); border: 1px solid var(--border); border-radius: 8px; padding: 22px; }}
    .eyebrow {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--accent); margin-bottom: 12px; }}
    .hero h1 {{ font-size: 34px; margin-bottom: 12px; }}
    .hero p {{ color: var(--muted); }}
    .pill-row {{ display:flex; flex-wrap:wrap; gap:10px; margin-top: 16px; }}
    .pill {{ padding:8px 12px; border-radius:999px; background: #0e1930; border:1px solid var(--border); color: var(--muted); font-size: 13px; }}
    .tabs {{ display:flex; gap:10px; margin-bottom: 18px; }}
    .tab {{ padding:10px 14px; border-radius:999px; background:#0d1930; border:1px solid var(--border); color:var(--muted); cursor:pointer; font-size:14px; }}
    .tab.active {{ background: var(--accent); color:#08111e; border-color: transparent; font-weight: 600; }}
    .view {{ display:none; }}
    .view.active {{ display:block; }}
    .summary-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .metric {{ background: var(--panel-strong); border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-height: 120px; }}
    .metric-label {{ color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
    .metric strong {{ display:block; font-size: 30px; margin-bottom: 8px; }}
    .metric.good strong {{ color: var(--good); }}
    .metric.warn strong {{ color: var(--warn); }}
    .metric.bad strong {{ color: var(--bad); }}
    .metric p {{ color: var(--muted); font-size: 13px; }}
    .explain-grid {{ display:grid; grid-template-columns: 1.1fr .9fr; gap: 14px; margin-bottom: 18px; }}
    .stack {{ display:grid; gap:14px; }}
    .panel h2 {{ font-size: 22px; margin-bottom: 12px; }}
    .panel h3 {{ font-size: 17px; margin-bottom: 8px; }}
    .panel-copy {{ color: var(--muted); margin-bottom: 14px; }}
    .signal-list {{ display:grid; gap:10px; }}
    .signal-item {{ padding:12px 14px; border-radius:8px; background:#0e1930; border:1px solid var(--border); }}
    .signal-item strong {{ display:block; margin-bottom:4px; }}
    .friendly-table {{ width:100%; border-collapse: collapse; font-size:14px; }}
    .friendly-table th, .friendly-table td {{ padding:12px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align: top; }}
    .friendly-table th {{ color: var(--accent-soft); font-weight: 600; }}
    .friendly-table td {{ color: var(--text); }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
    .link-row {{ display:flex; flex-wrap:wrap; gap:12px; margin-top: 14px; }}
    .link-button {{ display:inline-flex; align-items:center; justify-content:center; padding:10px 14px; border-radius:8px; background:#0d1930; border:1px solid var(--border); color:var(--text); text-decoration:none; }}
    .deep-grid {{ display:grid; grid-template-columns: 1fr; gap: 14px; }}
    .warning-box {{ background: rgba(249, 115, 22, 0.1); border:1px solid rgba(249, 115, 22, 0.35); border-radius:8px; padding:14px; color:#ffd3b0; }}
    .bar-stack {{ display:grid; gap:10px; margin-top: 8px; }}
    .bar-row {{ display:grid; grid-template-columns: 180px 1fr 80px; gap:12px; align-items:center; }}
    .bar-label, .bar-value {{ font-size: 13px; color: var(--muted); }}
    .bar-track {{ position:relative; height:18px; border-radius:999px; background:#091425; border:1px solid var(--border); overflow:hidden; }}
    .bar {{ position:absolute; top:0; bottom:0; border-radius:999px; }}
    .bar-mid {{ left:50%; width:1px; background: rgba(255,255,255,0.2); }}
    .bar.pos {{ background: linear-gradient(90deg, #34d399, #22c55e); }}
    .bar.neg {{ background: linear-gradient(90deg, #fb923c, #f97316); }}
    .trade-card-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }}
    .trade-card {{ border:1px solid var(--border); border-radius:8px; background:#0e1930; padding:16px; }}
    .trade-card.win {{ box-shadow: inset 0 0 0 1px rgba(34,197,94,0.12); }}
    .trade-card.loss {{ box-shadow: inset 0 0 0 1px rgba(249,115,22,0.12); }}
    .trade-card-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:14px; }}
    .trade-title {{ font-size:15px; font-weight:600; }}
    .trade-subtitle {{ font-size:13px; color:var(--muted); margin-top:4px; }}
    .trade-pnl {{ font-size:22px; font-weight:700; white-space:nowrap; }}
    .trade-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }}
    .trade-grid span {{ display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }}
    .trade-grid strong {{ display:block; font-size:14px; }}
    @media (max-width: 980px) {{
      .hero, .summary-grid, .explain-grid {{ grid-template-columns: 1fr; }}
      .trade-card-grid, .trade-grid, .bar-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Tradebot Backtest Report</div>
        <h1>Simple first, deep detail when you want it</h1>
        <p>This page explains our 3-year BTC strategy test in normal language. You do not need to know trading terms to understand the main result.</p>
        <div class="pill-row">
          <div class="pill">Data source: Binance BTC perpetual proxy</div>
          <div class="pill">Timeframe: 1 hour candles</div>
          <div class="pill">Window: 3 years</div>
          <div class="pill">Experiments tested: {len(leaderboard)}</div>
        </div>
      </div>
      <div class="hero-card">
        <div class="eyebrow">Quick Verdict</div>
        <h2 style="font-size:28px; margin-bottom:10px;">{verdict_label}</h2>
        <p>{verdict_text}</p>
        <div class="link-row">
          <a class="link-button" href="equity_curves.html">Open Equity Curves</a>
        </div>
      </div>
    </section>

    <div class="tabs">
      <button class="tab active" data-view-target="simple-view" onclick="setView('simple-view', this)">Simple View</button>
      <button class="tab" data-view-target="deep-view" onclick="setView('deep-view', this)">Deep View</button>
    </div>

    <section id="simple-view" class="view active">
      <div class="summary-grid">
        <div class="metric {final_metric_class}">
          <div class="metric-label">Final reality check</div>
          <strong>{best_holdout_return_pct:.1f}%</strong>
          <p>This is the best unseen-data result from the finalists. Negative means it still lost money.</p>
        </div>
        <div class="metric warn">
          <div class="metric-label">Best-looking idea in practice</div>
          <strong>{format_experiment_headline(best_validation)}</strong>
          <p>{summary_sentence}</p>
        </div>
        <div class="metric {'good' if float(best_validation['cost_robustness_ratio']) == 1.0 else 'warn'}">
          <div class="metric-label">How stable it looked in practice</div>
          <strong>{int(best_validation['validation_positive_folds'])}/3 good rounds</strong>
          <p>{int(best_validation['cost_robust_positive_folds'])}/3 still looked okay after higher trading costs.</p>
        </div>
        <div class="metric warn">
          <div class="metric-label">Worst drop during final test</div>
          <strong>{float(best_holdout['max_drawdown_pct']):.1f}%</strong>
          <p>This is how far the account sank from a high point during the final check.</p>
        </div>
      </div>

      <div class="explain-grid">
        <div class="stack">
          <section class="panel">
            <h2>What this means in plain English</h2>
            <p class="panel-copy">{verdict_text}</p>
            <div class="signal-list">
              <div class="signal-item">
                <strong>Best practice-round setup</strong>
                <span>{summary_sentence}</span>
              </div>
              <div class="signal-item">
                <strong>Why that looked promising</strong>
                <span>It stayed positive across all 3 practice rounds and also survived tougher cost assumptions.</span>
              </div>
              <div class="signal-item">
                <strong>Why we are still cautious</strong>
                <span>{caution_text}</span>
              </div>
            </div>
          </section>

          <section class="panel">
            <h2>What should we do next?</h2>
            <p class="panel-copy">{next_step_text}</p>
            <div class="signal-list">
              <div class="signal-item">
                <strong>Keep</strong>
                <span>{keep_focus_text}</span>
              </div>
              <div class="signal-item">
                <strong>Change</strong>
                <span>Add smarter market-structure rules on top of {holdout_winner_sentence.lower()} instead of endlessly tweaking trade size and settings.</span>
              </div>
              <div class="signal-item">
                <strong>Avoid</strong>
                <span>Do not treat the current bot as live-trading ready yet.</span>
              </div>
            </div>
          </section>

          <section class="panel">
            <h2>How We Keep Ourselves Honest</h2>
            <p class="panel-copy">We now ignore flashy low-sample ideas unless they trade enough times to be worth taking seriously.</p>
            <div class="signal-list">
              <div class="signal-item">
                <strong>Minimum validation trades</strong>
                <span>{MINIMUM_VALIDATION_TRADES} trades across the validation rounds.</span>
              </div>
              <div class="signal-item">
                <strong>Why this matters</strong>
                <span>A setup that looks great on only a handful of trades can be luck, not a real edge.</span>
              </div>
            </div>
          </section>
        </div>

        <div class="stack">
          <section class="panel">
            <h2>Beginner Cheat Sheet</h2>
            <div class="signal-list">
              <div class="signal-item">
                <strong>Practice rounds</strong>
                <span>Older chunks of history we used to rank ideas before the final test.</span>
              </div>
              <div class="signal-item">
                <strong>Final reality check</strong>
                <span>Fresh unseen history. This is the score that matters most.</span>
              </div>
              <div class="signal-item">
                <strong>Worst drop</strong>
                <span>How painful the ride got before recovering, or before ending.</span>
              </div>
              <div class="signal-item">
                <strong>Win rate</strong>
                <span>How often trades made money. A high win rate alone does not guarantee a good strategy.</span>
              </div>
            </div>
          </section>

          <section class="panel">
            <h2>Fast Links</h2>
            <p class="panel-copy">Open the raw details only if you want to dig deeper.</p>
            <div class="link-row">
              <a class="link-button" href="equity_curves.html">Equity Curves</a>
              <a class="link-button" href="../leaderboard.csv">Leaderboard CSV</a>
              <a class="link-button" href="../holdout_summary.csv">Holdout Summary CSV</a>
            </div>
          </section>
        </div>
      </div>

      <section class="panel">
        <h2>Top Ideas, Explained Simply</h2>
        <p class="panel-copy">These are the best-looking setups from the practice rounds. This is not the same as “best live strategy.”</p>
        {top_candidates_table}
      </section>

      <section class="panel">
        <h2>Validation Winner vs Final Winner</h2>
        <p class="panel-copy">The best practice-round setup and the best fresh-data setup were not the same. This section keeps that difference obvious.</p>
        <div class="signal-list">
          <div class="signal-item">
            <strong>Validation winner</strong>
            <span>{summary_sentence}</span>
          </div>
          <div class="signal-item">
            <strong>Final holdout winner</strong>
            <span>{keep_focus_text}</span>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Short-Side Showdown</h2>
        <p class="panel-copy">This compares the strongest short compression idea against the short retest candidates so we can see whether waiting for the retest actually helps in the final unseen period.</p>
        {showdown_table}
      </section>

      <section class="panel">
        <h2>Short-Side Regime Filter Showdown</h2>
        <p class="panel-copy">This compares the plain short compression winner against regime-gated short versions, so we can see whether trading only in certain market states improves the fresh-data result.</p>
        {regime_showdown_table}
      </section>

      <section class="panel">
        <h2>How The Finalists Performed</h2>
        <p class="panel-copy">Bars to the right mean profit. Bars to the left mean loss. This lets you see quickly whether any finalist really held up on fresh unseen data.</p>
        <div class="bar-stack">
          {holdout_chart}
        </div>
      </section>

      <section class="panel">
        <h2>Trade Examples From Testing</h2>
        <p class="panel-copy">A few real trades from the final test set, including investment size, outcome, and why the bot got out.</p>
        <div class="trade-card-grid">
          {trade_examples}
        </div>
      </section>
    </section>

    <section id="deep-view" class="view">
      <div class="deep-grid">
        <section class="panel">
          <h2>Finalist Holdout Results</h2>
          <p class="panel-copy">This is the final unseen-data comparison for the top-ranked ideas.</p>
          <div class="warning-box">Important: even the best finalist still lost money in the final reality check. That means the research got better, but the bot is not ready yet.</div>
          <div style="margin-top:14px;">
            {holdout_table}
          </div>
        </section>

        <section class="panel">
          <h2>How The Ranking Worked</h2>
          <div class="signal-list">
            <div class="signal-item">
              <strong>Step 1: Practice rounds</strong>
              <span>Each setup was tested on multiple historical slices to see whether it behaved consistently.</span>
            </div>
            <div class="signal-item">
              <strong>Step 2: Tougher costs</strong>
              <span>We re-tested with harsher trading-cost assumptions to see which ideas were fragile.</span>
            </div>
            <div class="signal-item">
              <strong>Step 3: Final unseen period</strong>
              <span>We then checked the finalists on fresh data they had not been ranked on.</span>
            </div>
          </div>
        </section>
      </div>
    </section>
  </main>
  <script>
    function setView(viewId, button) {{
      document.querySelectorAll('.view').forEach((node) => node.classList.remove('active'));
      document.querySelectorAll('.tab').forEach((node) => node.classList.remove('active'));
      document.getElementById(viewId).classList.add('active');
      button.classList.add('active');
    }}
  </script>
</body>
</html>"""
    (output_dir / "index.html").write_text(dashboard, encoding="utf-8")


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("reports/long_history_lab")
    reports_dir.mkdir(parents=True, exist_ok=True)

    candles = read_candles_csv(data_dir / "binance_BTCUSDT_1h.csv")
    funding = read_funding_csv(data_dir / "binance_BTCUSDT_funding.csv")
    feature_frame = _merge_funding(candles, funding)
    research_frame, holdout = chronological_holdout(feature_frame, holdout_fraction=0.2)
    folds = build_validation_folds(research_frame)
    holdout_bundle = prepare_frame_bundle(holdout)
    experiments = build_experiments()

    fold_rows: list[dict[str, object]] = []
    leaderboard_rows: list[dict[str, object]] = []
    for experiment in experiments:
        base_results = []
        stressed_results = []
        for fold_index, (_train, validation) in enumerate(folds, start=1):
            base_result = evaluate_experiment(validation, experiment, cost_multiplier=1.0)
            stressed_result = evaluate_experiment(validation, experiment, cost_multiplier=1.5)
            base_results.append(base_result)
            stressed_results.append(stressed_result)
            fold_rows.append(
                {
                    "experiment": experiment.label(),
                    "fold": fold_index,
                    "strategy": experiment.strategy_name,
                    "params": experiment.strategy_params,
                    "risk_fraction": experiment.risk_fraction,
                    "direction_mode": experiment.direction_mode,
                    "session_preset": experiment.session_preset,
                    "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
                    "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
                    "cost_multiplier": 1.0,
                    "total_return_pct": base_result.total_return_pct,
                    "max_drawdown_pct": base_result.max_drawdown_pct,
                    "trade_count": base_result.trade_count,
                }
            )
            fold_rows.append(
                {
                    "experiment": experiment.label(),
                    "fold": fold_index,
                    "strategy": experiment.strategy_name,
                    "params": experiment.strategy_params,
                    "risk_fraction": experiment.risk_fraction,
                    "direction_mode": experiment.direction_mode,
                    "session_preset": experiment.session_preset,
                    "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
                    "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
                    "cost_multiplier": 1.5,
                    "total_return_pct": stressed_result.total_return_pct,
                    "max_drawdown_pct": stressed_result.max_drawdown_pct,
                    "trade_count": stressed_result.trade_count,
                }
            )
        leaderboard_rows.append(leaderboard_row(experiment, base_results, stressed_results))

    fold_frame = pd.DataFrame(fold_rows)
    leaderboard = rank_leaderboard(pd.DataFrame(leaderboard_rows))
    fold_frame.to_csv(reports_dir / "fold_results.csv", index=False)
    leaderboard.to_csv(reports_dir / "leaderboard.csv", index=False)

    holdout_results = []
    for _, row in leaderboard.head(10).iterrows():
        experiment = experiment_from_row(row)
        holdout_result = evaluate_experiment(holdout_bundle, experiment, cost_multiplier=1.0)
        holdout_result.strategy_name = experiment.strategy_name
        holdout_result.params = {
            **experiment.strategy_params,
            "risk_fraction": experiment.risk_fraction,
            "direction_mode": experiment.direction_mode,
            "session_preset": experiment.session_preset,
            "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
            "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
            "regime_filter": experiment.regime_filter,
        }
        holdout_results.append(holdout_result)

    showdown_results = []
    showdown_frame = select_short_compression_showdown(leaderboard)
    for _, row in showdown_frame.iterrows():
        experiment = experiment_from_row(row)
        holdout_result = evaluate_experiment(holdout_bundle, experiment, cost_multiplier=1.0)
        holdout_result.strategy_name = experiment.strategy_name
        holdout_result.params = {
            **experiment.strategy_params,
            "risk_fraction": experiment.risk_fraction,
            "direction_mode": experiment.direction_mode,
            "session_preset": experiment.session_preset,
            "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
            "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
            "regime_filter": experiment.regime_filter,
        }
        showdown_results.append(holdout_result)

    regime_showdown_results = []
    regime_showdown_frame = select_short_compression_regime_showdown(leaderboard)
    for _, row in regime_showdown_frame.iterrows():
        experiment = experiment_from_row(row)
        holdout_result = evaluate_experiment(holdout_bundle, experiment, cost_multiplier=1.0)
        holdout_result.strategy_name = experiment.strategy_name
        holdout_result.params = {
            **experiment.strategy_params,
            "risk_fraction": experiment.risk_fraction,
            "direction_mode": experiment.direction_mode,
            "session_preset": experiment.session_preset,
            "cooldown_bars_after_stop": experiment.cooldown_bars_after_stop,
            "minimum_stop_distance_to_cost": experiment.minimum_stop_distance_to_cost,
            "regime_filter": experiment.regime_filter,
        }
        regime_showdown_results.append(holdout_result)

    report_latest = write_reports(holdout_results, reports_dir)
    holdout_summary = result_summary_frame(holdout_results)
    holdout_summary.to_csv(reports_dir / "holdout_summary.csv", index=False)
    holdout_trades = trades_frame(holdout_results)
    holdout_trades.to_csv(reports_dir / "holdout_trades.csv", index=False)
    showdown_summary = result_summary_frame(showdown_results) if showdown_results else pd.DataFrame()
    showdown_summary.to_csv(reports_dir / "short_side_showdown.csv", index=False)
    regime_showdown_summary = result_summary_frame(regime_showdown_results) if regime_showdown_results else pd.DataFrame()
    regime_showdown_summary.to_csv(reports_dir / "short_regime_showdown.csv", index=False)
    write_dashboard(reports_dir, leaderboard, holdout_summary, holdout_trades, showdown_summary, regime_showdown_summary)

    best = leaderboard.iloc[0]
    print("Best validation experiment:")
    print(best.drop(labels=["selection_score"]).to_string())
    print(f"\nDashboard: {reports_dir / 'index.html'}")
    print(f"Equity report: {report_latest / 'equity_curves.html'}")


if __name__ == "__main__":
    main()
