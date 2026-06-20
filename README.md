# Tradebot Backtest

Research backtester for BTC perpetual 1h strategies, with Hyperliquid-native
short-history support and Binance USD-M futures as the long-history proxy.

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

The CLI also accepts `--exchange binance --symbol BTCUSDT` for longer history,
plus engine filters such as `--direction-mode`, `--session-preset`,
`--cooldown-bars-after-stop`, and `--minimum-stop-distance-to-cost`.

## Regime-Switching Experiment

```bash
uv run tradebot-backtest regime \
  --data-file data/hyperliquid_BTC_1h.csv \
  --funding-file data/hyperliquid_BTC_funding.csv
```

The command selects one deterministic regime configuration on the first 70%
of the candle history, then compares it against matching breakout, Bollinger,
and buy-and-hold baselines on the untouched final 30%.

## Long-History Lab

Fetch 3 years of Binance BTCUSDT perpetual history into `data/`:

```bash
uv sync
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from tradebot_backtest.data import load_or_fetch_candles, fetch_funding_history

data_dir = Path("data")
candles = load_or_fetch_candles("binance", "BTCUSDT", "1h", 1095, data_dir)
funding = fetch_funding_history("binance", "BTCUSDT", candles["timestamp"].min(), candles["timestamp"].max())
funding.to_csv(data_dir / "binance_BTCUSDT_funding.csv", index=False)
PY
```

Run the long-history experiment sweep across plain breakout,
funding-veto breakout, and pullback-in-trend:

```bash
PYTHONPATH=src .venv/bin/python scripts/long_history_lab.py
```

This writes:

- `reports/long_history_lab/index.html`
- `reports/long_history_lab/leaderboard.csv`
- `reports/long_history_lab/holdout_summary.csv`
- `reports/long_history_lab/latest/equity_curves.html`

## Paper Trading

Run the current best strategy locally against fresh Hyperliquid BTC 1h data without sending orders:

```bash
uv run tradebot-backtest paper \
  --exchange hyperliquid \
  --symbol BTC \
  --interval 1h \
  --days 180
```

This writes a status page to `reports/paper/index.html`.

## Hyperliquid Testnet

The official Hyperliquid docs say:

- use `https://api.hyperliquid-testnet.xyz` for testnet,
- use the official Python SDK for signing,
- use a separate API wallet per trading process.

Sources:

- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/signing
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/nonces-and-api-wallets

Environment variables:

```bash
export HL_TESTNET_ACCOUNT_ADDRESS=0x...
export HL_TESTNET_MASTER_SECRET=0x...      # only needed once to approve an API wallet
export HL_TESTNET_API_SECRET=0x...         # approved API wallet secret used by the bot
```

Approve a dedicated API wallet on testnet:

```bash
uv run tradebot-backtest testnet-bootstrap-agent
```

Dry-run the best strategy against the current testnet account state:

```bash
uv run tradebot-backtest testnet-sync
```

Actually place or close testnet orders:

```bash
uv run tradebot-backtest testnet-sync --execute
```

This writes a status page to `reports/testnet/index.html`.
