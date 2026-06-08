# Regime-Switching Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, validate, and run a deterministic BTC 1h regime-switching strategy that selects breakout, mean reversion, reduced-size breakout, or cash from lagged ATR, ADX, and EMA-slope features.

**Architecture:** Put regime feature calculation, classification, routing, and its parameter grid in a new `regime.py` module. Extend `Signal` with an exposure multiplier consumed by the existing engine, then add a dedicated CLI experiment that selects one regime configuration on the training segment and compares it with matching breakout, Bollinger, and buy-and-hold baselines on the untouched test segment.

**Tech Stack:** Python 3.13, pandas, NumPy, Typer, Plotly, pytest, uv

---

### Task 1: Regime Feature Calculation And Classification

**Files:**
- Create: `src/tradebot_backtest/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: Write failing classifier tests**

Add synthetic fixtures and tests proving:

```python
def test_regime_features_do_not_change_when_future_candles_change() -> None:
    original = regime_features(candles)
    changed = candles.copy()
    changed.loc[changed.index[-3:], ["high", "low", "close"]] *= 10
    recalculated = regime_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:-3], recalculated.iloc[:-3])


def test_high_volatility_has_priority_over_trending() -> None:
    row = classify_regime(
        atr_percentile=0.95,
        adx=40.0,
        normalized_slope=1.0,
        params=RegimeParams(0.90, 25, 20, 0.50, 50, 40, 2.0),
    )
    assert row == "high_volatility"


def test_classifier_covers_trending_ranging_and_unclear() -> None:
    params = RegimeParams(0.90, 25, 20, 0.50, 50, 40, 2.0)
    assert classify_regime(0.50, 30, 0.75, params) == "trending"
    assert classify_regime(0.50, 15, 0.10, params) == "ranging"
    assert classify_regime(0.50, 22, 0.10, params) == "unclear"
```

- [ ] **Step 2: Run tests and confirm the red phase**

Run:

```bash
.venv/bin/pytest tests/test_regime.py -q
```

Expected: collection fails because `tradebot_backtest.regime` does not exist.

- [ ] **Step 3: Implement regime types and lagged features**

Create:

```python
@dataclass(frozen=True)
class RegimeParams:
    high_volatility_percentile: float
    trend_adx: float
    range_adx: float
    slope_threshold: float
    donchian_lookback: int
    bollinger_length: int
    bollinger_entry_z: float


Regime = Literal["high_volatility", "trending", "ranging", "unclear"]
```

Implement Wilder ATR(14), Wilder ADX(14), EMA(50), 12-hour normalized EMA slope, and ATR percentile rank against the current and previous 167 ATR observations. All outputs at timestamp `t` may use candle `t`, because signals execute at candle `t+1` open; they must never use later candles.

Implement `classify_regime()` using the approved priority:

```python
if atr_percentile >= params.high_volatility_percentile:
    return "high_volatility"
if adx >= params.trend_adx and abs(normalized_slope) >= params.slope_threshold:
    return "trending"
if adx <= params.range_adx:
    return "ranging"
return "unclear"
```

- [ ] **Step 4: Run classifier tests**

Run:

```bash
.venv/bin/pytest tests/test_regime.py -q
```

Expected: all classifier and no-lookahead tests pass.

### Task 2: Exposure-Aware Position Sizing

**Files:**
- Modify: `src/tradebot_backtest/engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write a failing half-exposure test**

Add:

```python
def test_signal_exposure_multiplier_reduces_position_notional() -> None:
    timestamp = candles.loc[0, "timestamp"]
    full_signal = Signal(
        timestamp,
        "test",
        {"variant": "full"},
        "long",
        "entry",
        90.0,
        exposure_multiplier=1.0,
    )
    half_signal = Signal(
        timestamp,
        "test",
        {"variant": "half"},
        "long",
        "entry",
        90.0,
        exposure_multiplier=0.5,
    )
    full = run_backtest(candles, [full_signal], config)
    half = run_backtest(candles, [half_signal], config)
    assert half.trades[0].notional == full.trades[0].notional * 0.5
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/pytest tests/test_engine.py::test_signal_exposure_multiplier_reduces_position_notional -q
```

Expected: failure because `Signal` does not accept `exposure_multiplier`.

- [ ] **Step 3: Extend the signal contract and sizing**

Add a backwards-compatible field:

```python
@dataclass(frozen=True)
class Signal:
    ...
    exposure_multiplier: float = 1.0
```

Validate the multiplier inside `_notional_for_signal` and return zero for values outside `(0, 1]`. Calculate:

```python
base_notional = calculate_position_notional(...)
return base_notional * signal.exposure_multiplier
```

Keep the existing risk and leverage caps unchanged.

- [ ] **Step 4: Run engine tests**

Run:

```bash
.venv/bin/pytest tests/test_engine.py -q
```

Expected: all engine tests pass, including half exposure.

### Task 3: Fixed Regime Router

**Files:**
- Modify: `src/tradebot_backtest/regime.py`
- Modify: `tests/test_regime.py`

- [ ] **Step 1: Write failing routing tests**

Test these behaviors:

```python
def test_high_volatility_routes_breakout_at_half_exposure() -> None:
    signals = regime_switching_signals(candles, params)
    entries = [s for s in signals if s.side in {"long", "short"}]
    assert entries[0].strategy_name == "regime_switching"
    assert entries[0].exposure_multiplier == 0.5


def test_trending_routes_breakout_at_full_exposure() -> None:
    assert trending_entry.exposure_multiplier == 1.0


def test_ranging_routes_bollinger_reversion() -> None:
    assert ranging_entry.entry_reason.startswith("ranging:")


def test_regime_change_emits_exit() -> None:
    assert any(s.side == "flat" and s.entry_reason == "regime_change" for s in signals)
```

- [ ] **Step 2: Confirm the routing tests fail**

Run:

```bash
.venv/bin/pytest tests/test_regime.py -q
```

Expected: failures because `regime_switching_signals` is not implemented.

- [ ] **Step 3: Implement signal routing**

Generate component signals using existing:

```python
breakout_signals(candles, params.donchian_lookback)
bollinger_regime_reversion_signals(
    candles,
    length=params.bollinger_length,
    entry_z=params.bollinger_entry_z,
    max_trend_strength=float("inf"),
)
```

At each timestamp:

- Route breakout entry/opposite/exit signals only in `trending` and `high_volatility`.
- Route Bollinger signals only in `ranging`.
- Rewrite routed signals to `strategy_name="regime_switching"` and prefix their reason with the regime.
- Set high-volatility breakout entries to `exposure_multiplier=0.5`; all other entries use `1.0`.
- Emit one flat `regime_change` signal when an active position’s mapped component changes or the regime becomes unclear.
- Do not emit repeated flat signals while already flat.

- [ ] **Step 4: Implement the approved grid**

Create `regime_parameter_grid()` with:

```python
high_volatility_percentile = [0.80, 0.90]
trend_adx = [20, 25]
range_adx = [15, 20]
slope_threshold = [0.25, 0.50]
donchian_lookback = [24, 50]
bollinger_length = [20, 40]
bollinger_entry_z = [1.5, 2.0]
```

Exclude configurations where `range_adx >= trend_adx`. This produces 96 valid configurations.

- [ ] **Step 5: Run regime tests**

Run:

```bash
.venv/bin/pytest tests/test_regime.py -q
```

Expected: all classifier, routing, grid-size, and exposure tests pass.

### Task 4: Metrics And Baselines

**Files:**
- Modify: `src/tradebot_backtest/engine.py`
- Modify: `src/tradebot_backtest/reporting.py`
- Create: `tests/test_regime_reporting.py`

- [ ] **Step 1: Write failing metrics tests**

Add tests for:

```python
assert result.exposure_time_pct == expected
assert buy_and_hold_result.strategy_name == "buy_and_hold"
assert set(regime_counts) == {"high_volatility", "trending", "ranging", "unclear"}
```

- [ ] **Step 2: Verify the failures**

Run:

```bash
.venv/bin/pytest tests/test_regime_reporting.py -q
```

Expected: failures because exposure and baseline helpers do not exist.

- [ ] **Step 3: Add exposure-time tracking**

Record a boolean `exposed` column in each equity point. Calculate:

```python
exposure_time_pct = equity_curve["exposed"].mean() * 100
```

Add `exposure_time_pct` to `result_summary_frame()`.

- [ ] **Step 4: Add buy-and-hold baseline**

Create a `buy_and_hold_result(candles, config)` helper that:

- buys at the first test candle open;
- uses one-times starting equity, not leveraged risk sizing;
- deducts configured entry and exit fees/slippage;
- marks equity to market through the test segment;
- exits at the final close.

- [ ] **Step 5: Add regime-distribution reporting**

Write `reports/latest/regime_distribution.csv` with columns:

```text
regime,hours,percent
```

Add regime counts and selected parameters to the CLI output.

- [ ] **Step 6: Run reporting tests**

Run:

```bash
.venv/bin/pytest tests/test_regime_reporting.py -q
```

Expected: all metrics and baseline tests pass.

### Task 5: Training Selection And CLI Experiment

**Files:**
- Modify: `src/tradebot_backtest/cli.py`
- Modify: `tests/test_cli_reporting.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing CLI integration test**

Invoke:

```bash
tradebot-backtest regime \
  --data-file fixture.csv \
  --funding-file funding.csv \
  --reports-dir reports
```

Assert:

```python
assert "Selected regime parameters" in result.output
assert set(summary["strategy"]) == {
    "regime_switching",
    "breakout_baseline",
    "bollinger_baseline",
    "buy_and_hold",
}
assert (reports / "latest" / "regime_distribution.csv").exists()
```

- [ ] **Step 2: Confirm the CLI test fails**

Run:

```bash
.venv/bin/pytest tests/test_cli_reporting.py::test_regime_command_generates_comparison_report -q
```

Expected: failure because the `regime` command does not exist.

- [ ] **Step 3: Implement training-only configuration selection**

Add a `regime` command that:

1. Loads and merges candles/funding using existing helpers.
2. Splits candles 70/30 chronologically.
3. Runs all 96 regime configurations on training candles.
4. Retains configurations with at least 10 training trades.
5. Selects by `(score, total_return_pct, -max_drawdown_pct)`.
6. Raises a clear CLI error if no configuration reaches 10 trades.
7. Generates selected regime signals from the full feature frame for indicator warm-up, then filters signals to the test timestamps.
8. Runs matching breakout and Bollinger baselines on the same test period.
9. Runs buy-and-hold on the same test period.
10. Writes the comparison and regime distribution reports.

- [ ] **Step 4: Document the command**

Add:

```bash
uv run tradebot-backtest regime \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

State that configuration is selected on 70% and reported on the untouched final 30%.

- [ ] **Step 5: Run CLI integration tests**

Run:

```bash
.venv/bin/pytest tests/test_cli_reporting.py -q
```

Expected: all existing and regime CLI tests pass.

### Task 6: Full Verification And Real Experiment

**Files:**
- Verify: `data/hyperliquid_BTC_1h.csv`
- Generate: `reports/latest/summary.csv`
- Generate: `reports/latest/trades.csv`
- Generate: `reports/latest/equity_curves.html`
- Generate: `reports/latest/regime_distribution.csv`

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run the approved experiment**

Run:

```bash
.venv/bin/tradebot-backtest regime \
  --exchange hyperliquid \
  --symbol BTC \
  --interval 1h \
  --days 180 \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv \
  --reports-dir reports
```

Expected: command exits zero, prints selected parameters and four test-period comparisons, and writes all report files.

- [ ] **Step 3: Validate report integrity**

Run:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

summary = pd.read_csv("reports/latest/summary.csv")
regimes = pd.read_csv("reports/latest/regime_distribution.csv")
assert set(summary["strategy"]) == {
    "regime_switching",
    "breakout_baseline",
    "bollinger_baseline",
    "buy_and_hold",
}
assert regimes["hours"].sum() > 0
assert abs(regimes["percent"].sum() - 100) < 0.01
print(summary.to_string(index=False))
print(regimes.to_string(index=False))
PY
```

Expected: assertions pass and both comparison tables print.

- [ ] **Step 4: Record the result honestly**

Report:

- the selected training parameters;
- untouched test-period return, drawdown, trades, win rate, score, and exposure;
- comparison against all three baselines;
- regime distribution;
- whether the regime strategy improved risk-adjusted performance;
- that one 70/30 split is insufficient evidence for live trading.

No Git commit steps are included because `/Users/mozanaydin/Desktop/trade-bot` is not a Git repository.
