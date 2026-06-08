# Tradebot Backtest

Research backtester for Hyperliquid BTC perpetual 1h strategies.

## Strategy Families

- EMA crossover
- RSI mean reversion
- Donchian breakout
- Volatility-scaled momentum
- Bollinger mean reversion with a trend-regime filter
- Volatility-compression breakout with an ATR trailing exit
- Funding-crowding reversal
- Funding-conditioned momentum

Parameters are selected independently for each family on the first 70% of
the data. Only the final 30% is used for the reported comparison.

## Run

```bash
uv sync --no-editable
uv run tradebot-backtest run \
  --exchange hyperliquid \
  --symbol BTC \
  --interval 1h \
  --days 180 \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

Reports are written to `reports/latest/`.

## Regime-Switching Experiment

```bash
uv run tradebot-backtest regime \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

The command selects one deterministic regime configuration on the first 70%
of the candle history, then compares it against matching breakout, Bollinger,
and buy-and-hold baselines on the untouched final 30%.
