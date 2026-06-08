# Regime-Switching Strategy Design

## Goal

Add and test a deterministic BTC 1h regime-switching strategy that routes
between breakout, mean reversion, reduced-risk breakout, and cash using only
completed-candle information.

## Regime Classifier

Calculate these features without lookahead:

- ATR(14) percentile rank against the previous 168 completed candles.
- ADX(14) using Wilder smoothing.
- EMA(50) normalized slope over 12 completed candles:
  `(ema_50[t] - ema_50[t-12]) / atr_14[t]`.

Classify each completed candle in this priority order:

1. **High volatility:** ATR percentile is at or above the configured threshold.
2. **Trending:** ADX is at or above the trend threshold and absolute normalized
   EMA slope is at or above the slope threshold.
3. **Ranging:** ADX is at or below the range threshold.
4. **Unclear:** all other candles.

The high-volatility rule has priority over trend and range classifications.

## Fixed Strategy Mapping

- **High volatility:** Donchian breakout with 50% exposure.
- **Trending:** Donchian breakout with 100% exposure.
- **Ranging:** Bollinger mean reversion with 100% exposure.
- **Unclear:** flat.

The routed strategy may open a position only when its regime is active. A
position exits at the next candle open when:

- its underlying strategy emits an exit or opposite signal;
- the regime changes to one mapped to a different strategy;
- the regime becomes unclear; or
- its close-based invalidation is breached.

Signals carry an `exposure_multiplier`. Position sizing multiplies the normal
risk-derived notional by this value while preserving the existing leverage cap.

## Parameter Selection

Evaluate this deliberately small grid on the first 70% of candles:

- High-volatility ATR percentile: `0.80`, `0.90`.
- Trend ADX threshold: `20`, `25`.
- Range ADX threshold: `15`, `20`, constrained below the trend threshold.
- Normalized EMA slope threshold: `0.25`, `0.50`.
- Donchian lookback: `24`, `50`.
- Bollinger length: `20`, `40`.
- Bollinger entry z-score: `1.5`, `2.0`.

Select one complete configuration using the existing risk-adjusted score,
requiring at least 10 training trades. Break ties by total return, then lower
maximum drawdown.

Evaluate the selected configuration once on the untouched final 30%.

## Comparisons And Reporting

Report the regime strategy alongside:

- standalone Donchian breakout using the selected Donchian lookback;
- standalone Bollinger reversion using the selected Bollinger settings;
- BTC buy-and-hold over the same test period.

Include return, maximum drawdown, trade count, win rate, score, exposure time,
and hours assigned to each regime. Existing fees, slippage, funding accounting,
next-open execution, and close-based invalidation remain active.

## Tests

- Feature calculations use only current and prior completed candles.
- Synthetic trending, ranging, high-volatility, and unclear fixtures receive
  the expected labels.
- High volatility takes precedence over trending.
- Regime changes close incompatible open positions at the next candle open.
- High-volatility entries use half the normal notional.
- Training selection never reads test-period results.
- Reports contain the regime strategy and all three baselines.

## Acceptance Criteria

- The complete automated test suite passes.
- The strategy runs against the refreshed December 9, 2025 through June 7,
  2026 Hyperliquid BTC 1h dataset.
- The final report clearly separates training-selected parameters from
  untouched test-period metrics.
- No claim of profitability is made solely from this single split.
