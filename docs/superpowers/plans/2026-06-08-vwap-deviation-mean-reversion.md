# VWAP Deviation Mean-Reversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently selected Balanced and Quick Profit rolling-VWAP mean-reversion strategies, evaluate them against existing BTC 1h strategies, and generate a polished standalone dashboard from real backtest results.

**Architecture:** Extend the existing signal contract and close-only engine without introducing a second execution model. `strategies.py` will calculate VWAP features and emit next-open entry/exit signals, `cli.py` will select each exit family on training data and evaluate a common test period, and reporting will serialize richer metrics and delegate the standalone HTML experience to a focused `dashboard.py` renderer.

**Tech Stack:** Python 3.11+, pandas, NumPy, Plotly, Typer, pytest, uv, Hyperliquid cached CSV data.

---

## File Map

- Modify `src/tradebot_backtest/strategies.py`: VWAP feature calculation, state machine, parameter grid integration, and dispatch.
- Modify `src/tradebot_backtest/engine.py`: additive trade/result metrics only; preserve execution semantics.
- Modify `src/tradebot_backtest/cli.py`: VWAP experiment command, family selection, comparison orchestration, and warnings.
- Modify `src/tradebot_backtest/reporting.py`: richer summary/training tables and dashboard handoff.
- Create `src/tradebot_backtest/dashboard.py`: standalone responsive HTML dashboard renderer.
- Modify `tests/test_strategies.py`: deterministic VWAP calculation and state-machine tests.
- Modify `tests/test_engine.py`: aggregate metric tests.
- Modify `tests/test_cli_reporting.py`: selection, common-boundary, output, and CLI integration tests.
- Create `tests/test_dashboard.py`: dashboard content and empty-state tests.
- Modify `README.md`: VWAP command and output instructions.

### Task 1: Rolling VWAP Features

**Files:**
- Modify: `src/tradebot_backtest/strategies.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Add failing tests for typical-price VWAP, full-window warm-up, and invalid rows**

Add tests that construct a four-row frame with unequal volume and assert:

```python
features = vwap_features(candles, window=3, atr_length=2)

expected = (
    (((candles.loc[0:2, ["high", "low", "close"]].sum(axis=1) / 3) * candles.loc[0:2, "volume"]).sum())
    / candles.loc[0:2, "volume"].sum()
)
assert features.loc[2, "vwap"] == pytest.approx(expected)
assert features.loc[:1, "vwap"].isna().all()
assert pd.isna(features.loc[2, "vwap_z"]) is False
```

Add a zero-volume window fixture and assert its `vwap`, `vwap_deviation_std`,
and `vwap_z` are non-finite or missing and therefore unusable.

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```bash
uv run pytest tests/test_strategies.py -k vwap_features -v
```

Expected: FAIL because `vwap_features` does not exist.

- [ ] **Step 3: Implement the feature frame**

Add:

```python
def vwap_features(candles: pd.DataFrame, window: int, atr_length: int = 14) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"VWAP strategy requires columns: {sorted(missing)}")

    frame = candles.copy()
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    rolling_volume = frame["volume"].rolling(window, min_periods=window).sum()
    weighted_price = (typical_price * frame["volume"]).rolling(window, min_periods=window).sum()
    frame["vwap"] = weighted_price / rolling_volume.replace(0, pd.NA)
    frame["vwap_deviation_std"] = typical_price.rolling(
        window,
        min_periods=window,
    ).std(ddof=0)
    frame["vwap_z"] = (
        (frame["close"] - frame["vwap"])
        / frame["vwap_deviation_std"].replace(0, pd.NA)
    )
    frame["atr"] = _atr(frame, atr_length)
    return frame
```

Keep the columns additive so tests and dashboard diagnostics can inspect them.

- [ ] **Step 4: Run focused and existing strategy tests**

Run:

```bash
uv run pytest tests/test_strategies.py -v
```

Expected: all strategy tests PASS.

- [ ] **Step 5: Commit the feature calculation**

```bash
git add src/tradebot_backtest/strategies.py tests/test_strategies.py
git commit -m "Add rolling VWAP feature calculation"
```

### Task 2: Shared VWAP Entry State Machine

**Files:**
- Modify: `src/tradebot_backtest/strategies.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Add failing long and short confirmation tests**

Use monkeypatched or directly supplied deterministic feature rows to cover:

```python
# Lower excursion arms at z <= -entry_z, then confirms only when close rises.
assert entry_signals[0].side == "long"
assert entry_signals[0].timestamp == candles.loc[confirmation_index, "timestamp"]
assert entry_signals[0].invalidation_price == pytest.approx(
    confirmation_close - 2.0 * confirmation_atr
)

# Upper excursion mirrors the behavior.
assert short_signals[0].side == "short"
```

Add cases proving that a same-direction continuation does not confirm and a
direct close through VWAP cancels the armed setup.

- [ ] **Step 2: Run the confirmation tests and verify red**

Run:

```bash
uv run pytest tests/test_strategies.py -k "vwap and (confirmation or cancel)" -v
```

Expected: FAIL because the VWAP signal generator does not exist.

- [ ] **Step 3: Implement shared setup state and immutable entry metadata**

Add a generator with this public shape:

```python
def vwap_mean_reversion_signals(
    candles: pd.DataFrame,
    window: int,
    entry_z: float,
    exit_variant: str,
    atr_length: int = 14,
    atr_multiplier: float = 2.0,
    max_holding_hours: int | None = None,
) -> list[Signal]:
```

Validate `exit_variant in {"balanced", "quick_profit"}`. Store params:

```python
params = {
    "window": window,
    "entry_z": entry_z,
    "exit_variant": exit_variant,
    "atr_length": atr_length,
    "atr_multiplier": atr_multiplier,
    "max_holding_hours": max_holding_hours or (24 if exit_variant == "balanced" else 12),
}
```

Track `setup_side`, `active_side`, `confirmation_vwap`, `target_price`,
`invalidation_price`, and the entry signal index. Emit entries only after a
directional close back toward VWAP. Clear an unconfirmed setup if price crosses
VWAP.

- [ ] **Step 4: Add failing re-entry tests**

Create a fixture containing an entry, exit, continued extreme prices,
normalization, and a second excursion. Assert:

```python
entries = [signal for signal in signals if signal.side != "flat"]
assert len(entries) == 2
assert entries[1].timestamp == fresh_excursion_confirmation_time
```

Also assert no entry occurs while price remains beyond the original band.

- [ ] **Step 5: Implement normalization gating**

After an exit, set `awaiting_normalization = True`. Clear it only on a usable
row where `abs(vwap_z) < entry_z`. Do not arm a new setup on the same row that
first satisfies normalization; require a subsequent fresh excursion.

- [ ] **Step 6: Run VWAP entry tests**

Run:

```bash
uv run pytest tests/test_strategies.py -k vwap -v
```

Expected: all VWAP feature and entry-state tests PASS.

- [ ] **Step 7: Commit the shared entry state machine**

```bash
git add src/tradebot_backtest/strategies.py tests/test_strategies.py
git commit -m "Add VWAP reversal entry state machine"
```

### Task 3: Balanced And Quick Profit Exits

**Files:**
- Modify: `src/tradebot_backtest/strategies.py`
- Test: `tests/test_strategies.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Add failing Balanced exit tests**

Cover three distinct fixtures and assert flat-signal reasons:

```python
assert mean_exit.entry_reason == "vwap_mean_reached"
assert timeout_exit.entry_reason == "max_holding_24h"
assert stop_exit.entry_reason == "invalidation_close"
```

Set an intrabar low below the long invalidation while close remains above it;
assert no stop exit is emitted on that row.

- [ ] **Step 2: Add failing Quick Profit exit tests**

Freeze target from the confirmation row:

```python
expected_target = confirmation_close + (
    confirmation_vwap - confirmation_close
) / 2.0
assert target_exit.entry_reason == "halfway_target_reached"
```

Mutate later VWAP values to prove the target does not move. Assert the timeout
reason is `max_holding_12h`, and mirror the target behavior for shorts.

- [ ] **Step 3: Run exit tests and verify red**

Run:

```bash
uv run pytest tests/test_strategies.py -k "vwap and (exit or target or holding or intrabar)" -v
```

Expected: FAIL because exit handling is incomplete.

- [ ] **Step 4: Implement close-only exit priority**

For each active position, evaluate in this deterministic order:

1. Invalidation close.
2. Variant target.
3. Holding timeout.

Use the signal candle count after the actual next-open entry:

```python
entry_candle_index = confirmation_index + 1
held_candles = current_index - entry_candle_index + 1
```

Balanced compares close to the current VWAP. Quick Profit compares close to
the frozen halfway target. Emit exactly one flat signal per position.

- [ ] **Step 5: Verify next-open timing through the engine**

Add an engine-backed test:

```python
result = run_backtest(candles, signals, BacktestConfig(include_funding=False))
assert result.trades[0].entry_time == candles.loc[confirmation_index + 1, "timestamp"]
assert result.trades[0].exit_time == candles.loc[exit_signal_index + 1, "timestamp"]
```

- [ ] **Step 6: Run strategy and engine suites**

Run:

```bash
uv run pytest tests/test_strategies.py tests/test_engine.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit both exit variants**

```bash
git add src/tradebot_backtest/strategies.py tests/test_strategies.py tests/test_engine.py
git commit -m "Add VWAP balanced and quick profit exits"
```

### Task 4: Metrics And Validity Columns

**Files:**
- Modify: `src/tradebot_backtest/engine.py`
- Modify: `src/tradebot_backtest/reporting.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_cli_reporting.py`

- [ ] **Step 1: Add failing aggregate metric tests**

Construct `BacktestResult` instances with winning and losing trades and assert:

```python
assert result.gross_profit == pytest.approx(120.0)
assert result.gross_loss == pytest.approx(-40.0)
assert result.profit_factor == pytest.approx(3.0)
assert result.total_fees == pytest.approx(sum(trade.fees for trade in result.trades))
assert result.total_funding == pytest.approx(sum(trade.funding for trade in result.trades))
assert result.average_trade_return_pct == pytest.approx(...)
```

Assert `profit_factor == float("inf")` for winners with no losses and `0.0`
for no trades or losses without winners.

- [ ] **Step 2: Run metric tests and verify red**

Run:

```bash
uv run pytest tests/test_engine.py -k "profit_factor or aggregate_metrics" -v
```

Expected: FAIL because the properties do not exist.

- [ ] **Step 3: Add read-only result properties**

Implement:

```python
@property
def gross_profit(self) -> float:
    return sum(max(trade.pnl, 0.0) for trade in self.trades)

@property
def gross_loss(self) -> float:
    return sum(min(trade.pnl, 0.0) for trade in self.trades)

@property
def profit_factor(self) -> float:
    if not self.trades:
        return 0.0
    if self.gross_loss == 0:
        return float("inf") if self.gross_profit > 0 else 0.0
    return self.gross_profit / abs(self.gross_loss)
```

Add `total_fees`, `total_funding`, and `average_trade_return_pct` using the
same empty-result convention.

- [ ] **Step 4: Extend summary rows**

Add these columns to `result_summary_frame`:

```python
"profit_factor": result.profit_factor,
"gross_profit": result.gross_profit,
"gross_loss": result.gross_loss,
"total_fees": result.total_fees,
"total_funding": result.total_funding,
"average_trade_return_pct": result.average_trade_return_pct,
"valid": result.trade_count >= 10,
```

Keep `score_result(result, minimum_trades=0)` for visible raw score and use the
`valid` column for winner eligibility.

- [ ] **Step 5: Run engine and reporting tests**

Run:

```bash
uv run pytest tests/test_engine.py tests/test_cli_reporting.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the richer metrics**

```bash
git add src/tradebot_backtest/engine.py src/tradebot_backtest/reporting.py tests/test_engine.py tests/test_cli_reporting.py
git commit -m "Add backtest aggregate performance metrics"
```

### Task 5: VWAP Grid Selection And Test Evaluation

**Files:**
- Modify: `src/tradebot_backtest/strategies.py`
- Modify: `src/tradebot_backtest/cli.py`
- Test: `tests/test_cli_reporting.py`

- [ ] **Step 1: Add failing grid and dispatch tests**

Assert exactly twelve VWAP candidates:

```python
grid = vwap_strategy_grid()
assert len(grid) == 12
assert {
    (name, params["window"], params["entry_z"])
    for name, params in grid
} == {
    (name, window, entry_z)
    for name in {
        "vwap_mean_reversion_balanced",
        "vwap_mean_reversion_quick_profit",
    }
    for window in {24, 48}
    for entry_z in {1.5, 2.0, 2.5}
}
```

Assert `generate_signals` dispatches both names to the shared generator with
the correct exit variant and holding limit.

- [ ] **Step 2: Add failing independent-family selection tests**

Create synthetic `BacktestResult` rows where each exit family has a different
winner. Assert:

```python
selected = select_best_per_family(results, minimum_trades=10)
assert selected["vwap_mean_reversion_balanced"] == balanced_params
assert selected["vwap_mean_reversion_quick_profit"] == quick_params
```

Add a tie case proving order is score, return, then lower drawdown. Add a
family with nine trades and assert it is omitted.

- [ ] **Step 3: Run selection tests and verify red**

Run:

```bash
uv run pytest tests/test_cli_reporting.py -k "vwap or select_best" -v
```

Expected: FAIL because the grid and minimum-trade selector are incomplete.

- [ ] **Step 4: Implement grid, dispatch, and strict selector**

Add:

```python
def vwap_strategy_grid() -> list[tuple[str, Params]]:
    return [
        (
            f"vwap_mean_reversion_{variant}",
            {
                "window": window,
                "entry_z": entry_z,
                "exit_variant": variant,
                "atr_length": 14,
                "atr_multiplier": 2.0,
                "max_holding_hours": 24 if variant == "balanced" else 12,
            },
        )
        for variant in ("balanced", "quick_profit")
        for window in (24, 48)
        for entry_z in (1.5, 2.0, 2.5)
    ]
```

Update the selector signature to accept `minimum_trades: int = 0`, omit
families without a valid candidate when the threshold is positive, and use:

```python
key=lambda result: (
    score_result(result, minimum_trades=0),
    result.total_return_pct,
    -result.max_drawdown_pct,
)
```

- [ ] **Step 5: Add a failing common-boundary integration test**

Run a saved synthetic fixture through the new experiment helper and assert:

```python
assert all(
    trade.entry_time >= test_start
    for result in test_results
    for trade in result.trades
)
assert selected.keys() >= {
    "vwap_mean_reversion_balanced",
    "vwap_mean_reversion_quick_profit",
}
```

The fixture must contain pre-boundary history sufficient for 48-hour warm-up.

- [ ] **Step 6: Add the `vwap` CLI command and orchestration**

Create a Typer command:

```bash
uv run tradebot-backtest vwap \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

The command must:

1. Load candles and optional funding with the same warning behavior as `run`.
2. Split 70/30 chronologically.
3. Run all VWAP candidates and existing strategy candidates on training data.
4. Run the regime grid on training data using the existing cached-feature
   approach.
5. Select one valid candidate per family with at least 10 training trades.
6. Generate signals on the full frame and retain only signals confirmed at or
   after `test_start`.
7. Evaluate every selected strategy and buy-and-hold on the same test frame.
8. Pass all candidate rows, selected rows, warnings, and dataset metadata to
   reporting.

- [ ] **Step 7: Run CLI integration tests**

Run:

```bash
uv run pytest tests/test_cli_reporting.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit experiment orchestration**

```bash
git add src/tradebot_backtest/strategies.py src/tradebot_backtest/cli.py tests/test_cli_reporting.py
git commit -m "Add VWAP training and test experiment"
```

### Task 6: Standalone Results Dashboard

**Files:**
- Create: `src/tradebot_backtest/dashboard.py`
- Modify: `src/tradebot_backtest/reporting.py`
- Create: `tests/test_dashboard.py`
- Modify: `tests/test_cli_reporting.py`

- [ ] **Step 1: Add failing dashboard structure tests**

Build two tiny results and training rows, render to a temporary path, and
assert the HTML contains:

```python
html = output.read_text()
assert "Tradebot Strategy Lab" in html
assert "Balanced vs Quick Profit" in html
assert "Training Parameter Heatmaps" in html
assert "Equity Curves" in html
assert "Drawdown" in html
assert "Trade Ledger" in html
assert "Funding included" in html
assert "Plotly.newPlot" in html
```

Add empty-trade and funding-warning cases and assert their visible messages.

- [ ] **Step 2: Run dashboard tests and verify red**

Run:

```bash
uv run pytest tests/test_dashboard.py -v
```

Expected: FAIL because `dashboard.py` does not exist.

- [ ] **Step 3: Implement serializable chart helpers**

In `dashboard.py`, add focused helpers:

```python
def drawdown_curve(result: BacktestResult) -> pd.DataFrame:
    equity = result.equity_curve[["timestamp", "equity"]].copy()
    equity["drawdown_pct"] = (equity["equity"] / equity["equity"].cummax() - 1.0) * 100.0
    return equity

def finite_number(value: float | int) -> float | int | None:
    return value if math.isfinite(float(value)) else None
```

Use `plotly.io.to_html(..., include_plotlyjs=True, full_html=False)` once for
the first figure and `include_plotlyjs=False` for subsequent figures so the
file remains standalone without duplicating the library.

- [ ] **Step 4: Implement the responsive HTML shell**

Create:

```python
def write_dashboard(
    results: list[BacktestResult],
    training_selection: pd.DataFrame,
    path: Path,
    metadata: dict[str, str | int | float | bool],
    warnings: list[str],
) -> None:
```

Render:

- Header with symbol, interval, train/test dates, fees, slippage, and funding.
- KPI band using the best valid test result.
- Ranked table with valid/invalid badges.
- Balanced-versus-Quick-Profit comparison.
- Equity and drawdown Plotly figures.
- Two training heatmaps using score by `window` and `entry_z`.
- Exit-reason and win/loss figures.
- Escaped HTML trade ledger.

Use CSS variables with a neutral near-black/white foundation plus green,
coral, cyan, and amber accents. Keep card radii at 8px or less, use full-width
sections, stable chart heights, horizontal table scrolling, and mobile
breakpoints. Do not add marketing copy or decorative gradients.

- [ ] **Step 5: Wire dashboard generation into reports**

Extend `write_reports` with optional `metadata` and `warnings`, preserve all
existing callers, and call:

```python
write_dashboard(
    results,
    training_selection if training_selection is not None else pd.DataFrame(),
    latest / "dashboard.html",
    metadata or {},
    warnings or [],
)
```

Write all training candidate rows to `training_selection.csv`, including
`family`, `valid`, and `selected`.

- [ ] **Step 6: Run dashboard and reporting tests**

Run:

```bash
uv run pytest tests/test_dashboard.py tests/test_cli_reporting.py tests/test_regime_reporting.py -v
```

Expected: all tests PASS and existing regime report calls remain compatible.

- [ ] **Step 7: Commit the dashboard**

```bash
git add src/tradebot_backtest/dashboard.py src/tradebot_backtest/reporting.py tests/test_dashboard.py tests/test_cli_reporting.py
git commit -m "Add interactive strategy results dashboard"
```

### Task 7: Full Verification And Real BTC Backtest

**Files:**
- Modify: `README.md`
- Generated: `reports/latest/summary.csv`
- Generated: `reports/latest/trades.csv`
- Generated: `reports/latest/training_selection.csv`
- Generated: `reports/latest/equity_curves.html`
- Generated: `reports/latest/dashboard.html`

- [ ] **Step 1: Document the VWAP experiment**

Add a `VWAP Mean-Reversion Experiment` section containing this command:

```bash
uv run tradebot-backtest vwap \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

State that the command selects Balanced and Quick Profit variants
independently on the first 70% of candles, evaluates the selected strategies
on the final 30%, and writes `reports/latest/dashboard.html`.

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
uv run pytest
```

Expected: all tests PASS with zero failures.

- [ ] **Step 3: Run the real cached-data experiment**

Run:

```bash
uv run tradebot-backtest vwap \
  --exchange hyperliquid \
  --symbol BTC \
  --interval 1h \
  --days 180 \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

Expected:

- Exit code 0.
- Both VWAP families appear in the training selection file.
- Both selected VWAP families appear in the test summary when valid.
- All comparison rows share the same test-period timestamps.
- `reports/latest/dashboard.html` exists and is non-empty.

- [ ] **Step 4: Inspect report consistency**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd

summary = pd.read_csv("reports/latest/summary.csv")
trades = pd.read_csv("reports/latest/trades.csv")
training = pd.read_csv("reports/latest/training_selection.csv")
dashboard = Path("reports/latest/dashboard.html")

assert {"valid", "profit_factor", "total_fees", "total_funding"} <= set(summary.columns)
assert {
    "vwap_mean_reversion_balanced",
    "vwap_mean_reversion_quick_profit",
} <= set(training["strategy"])
assert dashboard.stat().st_size > 10_000
print(summary[["strategy", "total_return_pct", "max_drawdown_pct", "trade_count", "valid", "score"]])
PY
```

Expected: assertions pass and the ranked result table prints.

- [ ] **Step 5: Verify the dashboard in the in-app browser**

Start a local server:

```bash
uv run python -m http.server 8765
```

Open `http://localhost:8765/reports/latest/dashboard.html` with the Browser
plugin. Inspect desktop at approximately 1440x900 and mobile at approximately
390x844. Verify:

- No overlapping text or clipped KPI values.
- Charts contain visible traces.
- Tables scroll horizontally on mobile.
- Balanced and Quick Profit are distinguishable.
- Empty/warning states are legible.
- Browser console has no uncaught errors.

- [ ] **Step 6: Run final tests after report generation**

Run:

```bash
uv run pytest
git status --short
```

Expected: all tests PASS; only intended source, docs, and regenerated report
files are modified.

- [ ] **Step 7: Commit verified implementation and outputs**

```bash
git add README.md src tests reports/latest docs/superpowers/plans/2026-06-08-vwap-deviation-mean-reversion.md
git commit -m "Complete VWAP strategy experiment"
```

- [ ] **Step 8: Push the completed work**

Run:

```bash
git push origin main
```

Expected: `main` and `origin/main` resolve to the same commit.
