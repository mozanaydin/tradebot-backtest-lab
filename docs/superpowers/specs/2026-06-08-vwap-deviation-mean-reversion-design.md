# VWAP Deviation Mean-Reversion Strategy Design

## Objective

Add a long/short VWAP deviation mean-reversion experiment to the existing
Hyperliquid BTC perpetual 1h backtester. The experiment will share one entry
model across two separately named and ranked exit variants:

- `vwap_mean_reversion_balanced`
- `vwap_mean_reversion_quick_profit`

The first 70% of the current dataset is the training period. The final 30% is
an untouched test period. The final deliverable includes automated tests,
fresh backtest outputs, and a polished static HTML dashboard built from the
real test results.

## Scope

The implementation covers:

- BTC perpetual 1h candles.
- Rolling 24-hour and 48-hour VWAP configurations.
- Long and short entries.
- Entry thresholds of 1.5, 2.0, and 2.5 standard deviations.
- Balanced and Quick Profit exit families.
- Hyperliquid fees, slippage, and available funding data.
- Chronological train/test selection and comparison with existing strategies
  and buy-and-hold.
- CSV outputs and a standalone HTML results dashboard.

It does not add live trading, intrabar execution, daily-session VWAP, anchored
VWAP, or the four-fixed-strategy presentation discussed as a possible future
enhancement.

## Indicator Calculation

For every candle:

1. Calculate typical price as `(high + low + close) / 3`.
2. Calculate rolling VWAP over the configured 24- or 48-candle window:
   `sum(typical_price * volume) / sum(volume)`.
3. Calculate the rolling population standard deviation of
   `typical_price - VWAP` over the same window.
4. Calculate deviation z-score as
   `(close - VWAP) / deviation_standard_deviation`.
5. Calculate 14-period ATR using the existing ATR convention.

All rolling values use the current and previous completed candles only. A
window is unusable until it has the full requested number of observations.
Rows with zero volume, zero deviation, zero ATR, or non-finite indicator
values cannot produce entries.

## Entry State Machine

The strategy maintains one of three setup states: normalized, lower excursion,
or upper excursion.

- A lower excursion is armed when z-score closes at or below `-entry_z`.
- An upper excursion is armed when z-score closes at or above `entry_z`.
- An armed lower excursion confirms a long when a later candle closes above
  the immediately preceding close.
- An armed upper excursion confirms a short when a later candle closes below
  the immediately preceding close.
- If an armed excursion crosses directly through VWAP before confirmation,
  discard it without entering.

The confirmation candle emits a signal at its close. The backtest engine
enters at the next candle open. The signal's fixed close-based invalidation is:

- Long: confirmation close minus `2 * ATR`.
- Short: confirmation close plus `2 * ATR`.

After an entry, no additional entry signal is allowed until the position exits
and price normalizes. Normalization means the close has returned inside the
configured deviation band: `abs(z-score) < entry_z`. A fresh excursion beyond
the band is then required before re-entry.

## Exit Variants

Both variants use the same entries and fixed `2 * ATR` close-based
invalidation. Intrabar highs and lows never trigger an exit. An exit condition
observed at candle close emits a flat signal and executes at the next candle
open.

### Balanced

Exit on the first of:

- Mean reversion: a long closes at or above the current rolling VWAP, or a
  short closes at or below it.
- Time limit: 24 completed hourly candles have elapsed since entry.
- Invalidation: a long closes at or below its fixed invalidation, or a short
  closes at or above it.
- End of available data.

### Quick Profit

At confirmation, freeze the halfway target between the confirmation close and
the confirmation candle's VWAP:

- Long: `confirmation_close + (VWAP - confirmation_close) / 2`.
- Short: `confirmation_close - (confirmation_close - VWAP) / 2`.

Exit on the first of:

- Quick target: a long closes at or above its fixed halfway target, or a short
  closes at or below it.
- Time limit: 12 completed hourly candles have elapsed since entry.
- The same fixed ATR invalidation used by Balanced.
- End of available data.

Exit reasons remain visible in the trade CSV and dashboard. Existing engine
behavior applies entry and exit fees and slippage, funding during exposure,
and next-open execution.

## Parameter Selection

Each exit family has six training candidates:

- VWAP window: `24`, `48`.
- Entry z-score: `1.5`, `2.0`, `2.5`.

Balanced and Quick Profit are selected independently. Within each family:

1. Run all six candidates on the first 70% of candles.
2. Exclude candidates with fewer than 10 training trades from valid
   selection.
3. Rank valid candidates by risk-adjusted score
   `return / abs(max_drawdown)`.
4. Break ties by higher return, then lower drawdown.
5. If no candidate reaches 10 training trades, report the family as having no
   valid selection rather than silently relaxing the threshold.

For zero drawdown, preserve the engine's existing rule: use positive return as
the score when return is positive, otherwise use zero.

Only the selected configuration from each family is evaluated on the final
30%. Indicators and setup state are calculated on the full chronological
frame so the test period has warm-up history, but only signals confirmed at or
after the test boundary can trade in the test backtest. Training outcomes do
not alter test candles or execution.

## Comparison Set

The final test report includes:

- Selected VWAP Balanced variant.
- Selected VWAP Quick Profit variant.
- One training-selected representative from each existing standard strategy
  family: EMA crossover, RSI mean reversion, breakout, volatility-scaled
  momentum, Bollinger regime reversion, and compression breakout.
- Funding crowding reversal and funding-conditioned momentum when valid
  funding features are available.
- The training-selected deterministic regime-switching strategy.
- Buy-and-hold.

All comparison rows use the same test boundary, starting equity, risk sizing,
fees, slippage, and available funding data. Any result with fewer than 10 test
trades is marked invalid for winner selection but remains visible.

## Metrics And Outputs

Metrics include:

- Total return.
- Maximum drawdown.
- Risk-adjusted score.
- Win rate.
- Profit factor, with zero-loss handling.
- Trade count.
- Exposure time.
- Gross profit and gross loss.
- Total fees and funding.
- Average trade return.

The run writes:

- `reports/latest/summary.csv`
- `reports/latest/trades.csv`
- `reports/latest/training_selection.csv`
- `reports/latest/equity_curves.html`
- `reports/latest/dashboard.html`

Training selection records all VWAP candidates, their family, parameters,
metrics, validity, and whether they were selected. It also records the selected
candidate for every comparison family. Existing report files gain the stated
metric and validity columns while retaining their current paths.

## Dashboard

`dashboard.html` is a standalone Plotly-backed report generated only after
real backtests finish. It opens without a server and uses a restrained trading
analytics layout rather than a marketing page.

The first view contains:

- Dataset and test-period context.
- Best valid test strategy.
- Compact KPI strip for return, drawdown, score, trade count, and costs.
- Ranked strategy table with clear valid/invalid status.

Interactive sections contain:

- Balanced versus Quick Profit comparison.
- Selectable equity curves.
- Drawdown curves.
- Training candidate heatmaps for VWAP window and entry threshold, separated
  by exit family.
- Trade outcome and exit-reason breakdowns.
- Searchable or scrollable trade ledger.
- Visible fee, slippage, funding, and warning status.

The dashboard must remain readable at common desktop and mobile widths, avoid
overlapping labels, and use stable chart dimensions. Empty strategies and
missing funding produce explicit empty or warning states instead of broken
charts.

## Architecture

The change follows existing package boundaries:

- `strategies.py` owns VWAP/ATR feature calculation and the shared entry and
  exit state machine. A small configuration object or explicit parameters
  distinguish exit variants without duplicating entry logic.
- `cli.py` owns the VWAP grid, independent family selection, chronological
  evaluation, and comparison orchestration.
- `engine.py` remains the source of position sizing, costs, funding, and
  next-open execution. It receives ordinary entry and flat signals; only
  narrowly scoped metric additions are expected.
- `reporting.py` owns new metric columns, training candidate output, and the
  standalone dashboard.

The public `Signal` and `Trade` structures remain compatible. Strategy
parameters include the VWAP window, entry threshold, ATR settings, exit
variant, and holding limit so reports can reconstruct each experiment.

## Failure Handling

- Missing funding warns and continues with funding excluded.
- Missing or invalid OHLCV columns fail before strategy generation.
- Invalid rolling calculations suppress signals rather than creating
  nonsensical position sizing.
- An empty strategy result is retained in reports with zero trades and a clear
  invalid status.
- A VWAP family with no valid training candidate does not enter the test
  winner ranking.
- Dashboard generation must tolerate empty trades and absent optional regime
  artifacts.

## Test Plan

### Unit Tests

- Rolling VWAP uses typical price and volume weighting correctly.
- VWAP and deviation require the full configured window.
- Lower and upper excursions arm correctly.
- Confirmation requires the next directional close toward VWAP.
- A direct VWAP crossing cancels an unconfirmed excursion.
- Signals enter at the candle after confirmation.
- Re-entry requires normalization and a fresh excursion.
- Balanced exits at VWAP, 24 completed candles, and close-based ATR
  invalidation.
- Quick Profit freezes and exits at the halfway target, 12 completed candles,
  and close-based ATR invalidation.
- Intrabar high or low alone does not trigger any exit.
- Long and short paths are symmetric.
- Independent family selection enforces 10 training trades and deterministic
  tie-breaking.
- Profit factor handles no trades, no winners, and no losing trades.

### Integration Tests

- Run the complete VWAP grid against a saved candle fixture without network
  access.
- Confirm one selected candidate per valid exit family.
- Confirm warm-up candles cannot create pre-boundary test trades.
- Generate all CSV and HTML outputs.
- Confirm dashboard generation succeeds with trades, without trades, and
  without funding.
- Confirm the comparison report applies a common test period and cost model.

### Manual Acceptance

- `uv run pytest` passes.
- The fresh-data backtest completes against the current Hyperliquid BTC 1h
  dataset.
- Both VWAP exit families appear separately in terminal and CSV results.
- The dashboard opens locally and displays real results, interactive curves,
  candidate heatmaps, and the trade ledger without layout overlap.

## Success Criteria

The experiment is complete when both exit variants have been selected without
test-period leakage, evaluated on the untouched test segment, compared under
identical costs, and presented in a verified standalone dashboard. Success
does not require either strategy to be profitable; an honest negative or
insufficient-trade result is a valid outcome.
