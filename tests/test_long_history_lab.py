from __future__ import annotations

import pandas as pd

from scripts.long_history_lab import (
    Experiment,
    apply_regime_filters_to_experiments,
    build_experiments,
    experiment_from_row,
    rank_leaderboard,
    select_short_compression_regime_showdown,
    select_short_compression_showdown,
    prepare_frame_bundle,
    write_dashboard,
)


def test_apply_regime_filters_to_experiments_adds_filtered_variants_for_supported_families() -> None:
    experiments = [
        Experiment("breakout", {"lookback": 20}, 0.005, "both", "all", 0, 0.0),
        Experiment("pullback_in_trend", {"fast_length": 50, "slow_length": 200, "rsi_length": 14, "recovery_level": 45, "invalidation_window": 10}, 0.005, "both", "all", 0, 0.0),
        Experiment("ema_crossover", {"fast": 10, "slow": 50}, 0.005, "both", "all", 0, 0.0),
    ]

    expanded = apply_regime_filters_to_experiments(experiments)

    breakout_variants = [experiment for experiment in expanded if experiment.strategy_name == "breakout"]
    pullback_variants = [experiment for experiment in expanded if experiment.strategy_name == "pullback_in_trend"]
    ema_variants = [experiment for experiment in expanded if experiment.strategy_name == "ema_crossover"]

    assert len(breakout_variants) > 1
    assert len(pullback_variants) > 1
    assert len(ema_variants) == 1
    assert any(experiment.regime_filter is not None for experiment in breakout_variants)


def test_experiment_label_includes_regime_filter_details() -> None:
    experiment = Experiment(
        "breakout",
        {"lookback": 20},
        0.005,
        "both",
        "all",
        0,
        0.0,
        regime_filter={
            "min_adx": 20.0,
            "min_abs_slope": 0.5,
            "min_atr_percentile": 0.2,
            "max_atr_percentile": 0.8,
            "require_trend_alignment": True,
        },
    )

    label = experiment.label()

    assert "regime=" in label
    assert "min_adx" in label


def test_prepare_frame_bundle_caches_regime_features() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=220, freq="h", tz="UTC"),
            "open": [100.0 + index for index in range(220)],
            "high": [101.0 + index for index in range(220)],
            "low": [99.0 + index for index in range(220)],
            "close": [100.0 + index for index in range(220)],
            "volume": [1.0] * 220,
        }
    )

    bundle = prepare_frame_bundle(frame)

    assert bundle["candles"] is frame
    assert "regime_features" in bundle
    assert len(bundle["regime_features"]) == len(frame)


def test_write_dashboard_creates_beginner_friendly_hybrid_sections(tmp_path) -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "strategy": "breakout",
                "strategy_params": {"lookback": 50},
                "risk_fraction": 0.005,
                "direction_mode": "long_only",
                "session_preset": "europe_us",
                "cooldown_bars_after_stop": 0,
                "minimum_stop_distance_to_cost": 0.0,
                "regime_filter": {"min_adx": 20.0},
                "validation_median_return_pct": 11.49,
                "validation_median_drawdown_pct": 14.31,
                "validation_total_trades": 26,
                "validation_positive_folds": 3,
                "cost_robust_positive_folds": 3,
                "stability_ratio": 1.0,
                "cost_robustness_ratio": 1.0,
                "selection_score": (1.0, 1.0, 11.49, -14.31, 26),
            }
        ]
    )
    holdout_summary = pd.DataFrame(
        [
            {
                "strategy": "breakout",
                "params": {"lookback": 50},
                "total_return_pct": -13.24,
                "max_drawdown_pct": 28.30,
                "win_rate_pct": 15.09,
                "trade_count": 53,
                "score": -0.46,
            }
        ]
    )
    holdout_trades = pd.DataFrame(
        [
            {
                "strategy_name": "breakout",
                "params": {"lookback": 50},
                "side": "long",
                "entry_time": "2026-01-01 10:00:00+00:00",
                "exit_time": "2026-01-01 15:00:00+00:00",
                "entry_price": 100000.0,
                "exit_price": 101500.0,
                "notional": 1200.0,
                "pnl": 14.25,
                "fees": 1.4,
                "funding": 0.0,
                "return_pct": 1.19,
                "exit_reason": "trend_failed",
            }
        ]
    )
    showdown_summary = pd.DataFrame(
        [
            {
                "strategy": "compression_breakout",
                "params": {"lookback": 48, "direction_mode": "short_only"},
                "total_return_pct": 8.1,
                "max_drawdown_pct": 2.2,
                "win_rate_pct": 52.6,
                "trade_count": 19,
                "score": 3.6,
            },
            {
                "strategy": "compression_breakout_retest",
                "params": {"lookback": 48, "direction_mode": "short_only", "retest_atr_buffer": 1.25},
                "total_return_pct": 4.2,
                "max_drawdown_pct": 3.1,
                "win_rate_pct": 48.0,
                "trade_count": 21,
                "score": 1.3,
            },
        ]
    )
    regime_showdown_summary = pd.DataFrame(
        [
            {
                "strategy": "compression_breakout",
                "params": {"lookback": 48, "direction_mode": "short_only", "regime_filter": None},
                "total_return_pct": 8.1,
                "max_drawdown_pct": 2.2,
                "win_rate_pct": 52.6,
                "trade_count": 19,
                "score": 3.6,
            },
            {
                "strategy": "compression_breakout",
                "params": {"lookback": 48, "direction_mode": "short_only", "regime_filter": {"label": "trend_quality"}},
                "total_return_pct": 6.9,
                "max_drawdown_pct": 1.8,
                "win_rate_pct": 57.0,
                "trade_count": 14,
                "score": 3.8,
            },
        ]
    )

    write_dashboard(tmp_path, leaderboard, holdout_summary, holdout_trades, showdown_summary, regime_showdown_summary)

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Simple View" in html
    assert "Deep View" in html
    assert "What this means in plain English" in html
    assert "What should we do next?" in html
    assert "Trade Examples From Testing" in html
    assert "Investment size" in html
    assert "Minimum validation trades" in html
    assert "20" in html
    assert "Short-Side Showdown" in html
    assert "Short-Side Regime Filter Showdown" in html
    assert "Validation Winner vs Final Winner" in html
    assert "Top Validation Configs" not in html
    assert "strategy_params" not in html
    assert "regime_filter" not in html


def test_experiment_from_row_preserves_regime_filter() -> None:
    row = pd.Series(
        {
            "strategy": "breakout",
            "strategy_params": {"lookback": 50},
            "risk_fraction": 0.005,
            "direction_mode": "long_only",
            "session_preset": "europe_us",
            "cooldown_bars_after_stop": 2,
            "minimum_stop_distance_to_cost": 1.5,
            "regime_filter": {
                "label": "stacked_trend",
                "min_adx": 25.0,
                "min_abs_slope": 0.5,
                "min_atr_percentile": 0.2,
                "max_atr_percentile": 0.8,
                "require_trend_alignment": True,
                "require_price_above_ema200": True,
            },
        }
    )

    experiment = experiment_from_row(row)

    assert experiment.strategy_name == "breakout"
    assert experiment.regime_filter is not None
    assert experiment.regime_filter["label"] == "stacked_trend"
    assert experiment.regime_filter["require_price_above_ema200"] is True


def test_rank_leaderboard_pushes_low_trade_variants_below_valid_setups() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "strategy": "compression_breakout_retest",
                "validation_total_trades": 48,
                "cost_robustness_ratio": 0.66,
                "stability_ratio": 0.66,
                "validation_median_return_pct": 4.5,
                "validation_median_drawdown_pct": 3.0,
            },
            {
                "strategy": "pullback_in_trend",
                "validation_total_trades": 8,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "validation_median_return_pct": 10.0,
                "validation_median_drawdown_pct": 1.0,
            },
        ]
    )

    ranked = rank_leaderboard(leaderboard, minimum_validation_trades=20)

    assert list(ranked["strategy"]) == ["compression_breakout_retest", "pullback_in_trend"]
    assert ranked.loc[0, "meets_minimum_validation_trades"]
    assert not ranked.loc[1, "meets_minimum_validation_trades"]


def test_build_experiments_includes_directional_splits_and_retest_variants() -> None:
    experiments = build_experiments()

    assert any(experiment.strategy_name == "compression_breakout_retest" for experiment in experiments)
    assert any(experiment.direction_mode == "short_only" for experiment in experiments)
    assert not any(experiment.strategy_name == "breakout_funding_veto" for experiment in experiments)
    assert any(
        experiment.strategy_name == "compression_breakout_retest"
        and experiment.direction_mode == "short_only"
        and experiment.strategy_params.get("retest_atr_buffer") == 1.25
        and experiment.strategy_params.get("volume_multiplier") == 1.1
        for experiment in experiments
    )
    assert any(
        experiment.strategy_name == "compression_breakout_retest"
        and experiment.direction_mode == "long_only"
        and experiment.strategy_params.get("retest_atr_buffer") == 0.6
        and experiment.strategy_params.get("volume_multiplier") == 1.35
        for experiment in experiments
    )
    assert any(
        experiment.direction_mode == "short_only"
        and experiment.strategy_params.get("exit_style") == "profit_trail"
        for experiment in experiments
    )
    assert any(
        experiment.direction_mode == "short_only"
        and experiment.strategy_params.get("exit_style") == "time_stop_trail"
        for experiment in experiments
    )


def test_select_short_compression_showdown_keeps_plain_short_and_best_retests() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "strategy": "compression_breakout",
                "strategy_params": {"lookback": 48},
                "direction_mode": "short_only",
                "session_preset": "all",
                "validation_median_return_pct": 1.9,
                "validation_median_drawdown_pct": 1.9,
                "validation_total_trades": 23,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout_retest",
                "strategy_params": {"lookback": 48, "retest_atr_buffer": 1.25},
                "direction_mode": "short_only",
                "session_preset": "all",
                "validation_median_return_pct": 2.4,
                "validation_median_drawdown_pct": 2.5,
                "validation_total_trades": 25,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout_retest",
                "strategy_params": {"lookback": 24, "retest_atr_buffer": 1.0},
                "direction_mode": "short_only",
                "session_preset": "all",
                "validation_median_return_pct": 1.7,
                "validation_median_drawdown_pct": 1.8,
                "validation_total_trades": 22,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout_retest",
                "strategy_params": {"lookback": 48, "retest_atr_buffer": 0.8},
                "direction_mode": "both",
                "session_preset": "all",
                "validation_median_return_pct": 9.9,
                "validation_median_drawdown_pct": 1.0,
                "validation_total_trades": 30,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
        ]
    )

    showdown = select_short_compression_showdown(leaderboard)

    assert list(showdown["strategy"]) == [
        "compression_breakout",
        "compression_breakout_retest",
        "compression_breakout_retest",
    ]
    assert all(showdown["direction_mode"] == "short_only")


def test_select_short_compression_regime_showdown_keeps_baseline_and_filtered_short_breakouts() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "strategy": "compression_breakout",
                "strategy_params": {"lookback": 48},
                "direction_mode": "short_only",
                "session_preset": "all",
                "regime_filter": None,
                "validation_median_return_pct": 1.9,
                "validation_median_drawdown_pct": 1.9,
                "validation_total_trades": 23,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout",
                "strategy_params": {"lookback": 48},
                "direction_mode": "short_only",
                "session_preset": "all",
                "regime_filter": {"label": "trend_quality"},
                "validation_median_return_pct": 2.2,
                "validation_median_drawdown_pct": 1.7,
                "validation_total_trades": 21,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout",
                "strategy_params": {"lookback": 48},
                "direction_mode": "short_only",
                "session_preset": "all",
                "regime_filter": {"label": "stacked_trend"},
                "validation_median_return_pct": 2.0,
                "validation_median_drawdown_pct": 1.6,
                "validation_total_trades": 20,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
            {
                "strategy": "compression_breakout_retest",
                "strategy_params": {"lookback": 24},
                "direction_mode": "short_only",
                "session_preset": "all",
                "regime_filter": {"label": "trend_quality"},
                "validation_median_return_pct": 9.9,
                "validation_median_drawdown_pct": 1.0,
                "validation_total_trades": 30,
                "cost_robustness_ratio": 1.0,
                "stability_ratio": 1.0,
                "meets_minimum_validation_trades": True,
            },
        ]
    )

    showdown = select_short_compression_regime_showdown(leaderboard)

    assert list(showdown["strategy"]) == [
        "compression_breakout",
        "compression_breakout",
        "compression_breakout",
    ]
    assert showdown.iloc[0]["regime_filter"] is None
    assert all(showdown["direction_mode"] == "short_only")
