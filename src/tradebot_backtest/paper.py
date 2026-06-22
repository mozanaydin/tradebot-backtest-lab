from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from tradebot_backtest.engine import (
    BacktestConfig,
    Signal,
    Trade,
    _close_position,
    _entry_allowed,
    _funding_rate_at,
    _marked_equity,
    _notional_for_signal,
)
from tradebot_backtest.reporting import trades_frame
from tradebot_backtest.strategies import compression_breakout_signals


@dataclass(frozen=True)
class StrategyPreset:
    strategy_name: str
    params: dict[str, int | float | str | None]
    direction_mode: str
    risk_fraction: float
    session_preset: str
    cooldown_bars_after_stop: int
    minimum_stop_distance_to_cost: float
    notes: str


@dataclass(frozen=True)
class PaperPosition:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    invalidation_price: float
    notional: float
    units: float
    unrealized_pnl: float
    realized_fees: float
    accumulated_funding: float


@dataclass(frozen=True)
class PaperSnapshot:
    strategy_name: str
    params: dict[str, int | float | str | None]
    generated_at: pd.Timestamp
    candle_count: int
    last_candle_time: pd.Timestamp
    last_close: float
    cash_equity: float
    marked_equity: float
    closed_trade_count: int
    wins: int
    losses: int
    open_position: PaperPosition | None
    pending_signal: Signal | None
    latest_signal: Signal | None
    trades: list[Trade]


BEST_STRATEGY_PRESET = StrategyPreset(
    strategy_name="compression_breakout",
    params={
        "lookback": 48,
        "bandwidth_window": 100,
        "compression_quantile": 0.2,
        "volume_multiplier": 1.25,
        "atr_length": 14,
        "regime_filter": None,
    },
    direction_mode="both",
    risk_fraction=0.005,
    session_preset="all",
    cooldown_bars_after_stop=0,
    minimum_stop_distance_to_cost=1.5,
    notes="Current live preset uses the best compression breakout configuration with both long and short entries enabled.",
)


def best_strategy_config(starting_balance: float = 1000.0) -> BacktestConfig:
    return BacktestConfig(
        starting_balance=starting_balance,
        risk_fraction=BEST_STRATEGY_PRESET.risk_fraction,
        max_leverage=3.0,
        include_funding=True,
        allowed_directions=("long", "short"),
        allowed_entry_hours=None,
        cooldown_bars_after_stop=BEST_STRATEGY_PRESET.cooldown_bars_after_stop,
        minimum_stop_distance_to_cost=BEST_STRATEGY_PRESET.minimum_stop_distance_to_cost,
    )


def best_strategy_signals(candles: pd.DataFrame) -> list[Signal]:
    return compression_breakout_signals(
        candles,
        lookback=int(BEST_STRATEGY_PRESET.params["lookback"]),
        bandwidth_window=int(BEST_STRATEGY_PRESET.params["bandwidth_window"]),
        compression_quantile=float(BEST_STRATEGY_PRESET.params["compression_quantile"]),
        volume_multiplier=float(BEST_STRATEGY_PRESET.params["volume_multiplier"]),
        atr_length=int(BEST_STRATEGY_PRESET.params["atr_length"]),
    )


def simulate_paper_snapshot(
    candles: pd.DataFrame,
    signals: list[Signal],
    config: BacktestConfig,
    funding_rates: pd.DataFrame | None = None,
) -> PaperSnapshot:
    if candles.empty:
        raise ValueError("paper trading requires at least one candle")
    ordered = candles.sort_values("timestamp").reset_index(drop=True)
    signal_by_time = {
        pd.Timestamp(signal.timestamp): signal
        for signal in sorted(signals, key=lambda item: item.timestamp)
    }
    equity = config.starting_balance
    pending_signal: Signal | None = None
    latest_signal: Signal | None = None
    open_position: dict[str, object] | None = None
    trades: list[Trade] = []
    cooldown_until_index = -1

    for idx, candle in ordered.iterrows():
        timestamp = pd.Timestamp(candle["timestamp"])
        open_price = float(candle["open"])
        close_price = float(candle["close"])

        if pending_signal is not None:
            if open_position is not None and pending_signal.side == "flat":
                equity, trade = _close_position(
                    open_position,
                    pending_signal,
                    timestamp,
                    open_price,
                    equity,
                    config,
                )
                trades.append(trade)
                if trade.exit_reason == "invalidation_close":
                    cooldown_until_index = idx + config.cooldown_bars_after_stop
                open_position = None
            elif pending_signal.side in {"long", "short"}:
                if open_position is not None:
                    equity, trade = _close_position(
                        open_position,
                        pending_signal,
                        timestamp,
                        open_price,
                        equity,
                        config,
                    )
                    trades.append(trade)
                    if trade.exit_reason == "invalidation_close":
                        cooldown_until_index = idx + config.cooldown_bars_after_stop
                    open_position = None
                notional = _notional_for_signal(
                    equity,
                    float(pending_signal.invalidation_price),
                    open_price,
                    config,
                    pending_signal.exposure_multiplier,
                )
                if notional > 0:
                    entry_fee = config.cost_model.one_way_cost(notional)
                    equity -= entry_fee
                    open_position = {
                        "signal": pending_signal,
                        "side": pending_signal.side,
                        "entry_time": timestamp,
                        "entry_price": open_price,
                        "notional": notional,
                        "entry_fee": entry_fee,
                        "funding": 0.0,
                    }
            pending_signal = None

        if open_position is not None and config.include_funding and funding_rates is not None:
            funding_rate = _funding_rate_at(funding_rates, timestamp)
            if funding_rate is not None:
                funding = (
                    float(open_position["notional"]) * funding_rate
                    if open_position["side"] == "short"
                    else -float(open_position["notional"]) * funding_rate
                )
                open_position["funding"] = float(open_position["funding"]) + funding
                equity += funding

        current_signal = signal_by_time.get(timestamp)
        if current_signal is not None:
            latest_signal = current_signal
        if open_position is None:
            if (
                current_signal is not None
                and current_signal.side in {"long", "short"}
                and idx > cooldown_until_index
                and _entry_allowed(current_signal, timestamp, config)
            ):
                pending_signal = current_signal
        else:
            side = str(open_position["side"])
            entry_signal = open_position["signal"]
            invalidation = float(entry_signal.invalidation_price)  # type: ignore[union-attr]
            invalidated = (side == "long" and close_price <= invalidation) or (
                side == "short" and close_price >= invalidation
            )
            if invalidated:
                pending_signal = Signal(
                    timestamp,
                    entry_signal.strategy_name,  # type: ignore[union-attr]
                    entry_signal.params,  # type: ignore[union-attr]
                    "flat",
                    "invalidation_close",
                    invalidation,
                )
            elif current_signal is not None:
                opposite = current_signal.side in {"long", "short"} and current_signal.side != side
                explicit_exit = current_signal.side == "flat"
                entry_exposure = float(entry_signal.exposure_multiplier)  # type: ignore[union-attr]
                exposure_change = (
                    current_signal.side == side
                    and current_signal.exposure_multiplier != entry_exposure
                )
                if opposite or explicit_exit or exposure_change:
                    pending_signal = current_signal

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    last_candle_time = pd.Timestamp(ordered.iloc[-1]["timestamp"])
    last_close = float(ordered.iloc[-1]["close"])
    marked_equity = _marked_equity(equity, open_position, last_close)
    live_position = _snapshot_position(open_position, last_close)
    return PaperSnapshot(
        strategy_name=BEST_STRATEGY_PRESET.strategy_name,
        params=BEST_STRATEGY_PRESET.params,
        generated_at=pd.Timestamp.now(tz="UTC"),
        candle_count=len(ordered),
        last_candle_time=last_candle_time,
        last_close=last_close,
        cash_equity=equity,
        marked_equity=marked_equity,
        closed_trade_count=len(trades),
        wins=wins,
        losses=losses,
        open_position=live_position,
        pending_signal=pending_signal,
        latest_signal=latest_signal,
        trades=trades,
    )


def write_paper_artifacts(snapshot: PaperSnapshot, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_name": snapshot.strategy_name,
        "params": snapshot.params,
        "generated_at": snapshot.generated_at.isoformat(),
        "candle_count": snapshot.candle_count,
        "last_candle_time": snapshot.last_candle_time.isoformat(),
        "last_close": snapshot.last_close,
        "cash_equity": snapshot.cash_equity,
        "marked_equity": snapshot.marked_equity,
        "closed_trade_count": snapshot.closed_trade_count,
        "wins": snapshot.wins,
        "losses": snapshot.losses,
        "open_position": _jsonify(snapshot.open_position),
        "pending_signal": _jsonify(snapshot.pending_signal),
        "latest_signal": _jsonify(snapshot.latest_signal),
        "notes": BEST_STRATEGY_PRESET.notes,
    }
    (output_dir / "snapshot.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    trades = trades_frame([_snapshot_to_result(snapshot)])
    trades.to_csv(output_dir / "trades.csv", index=False)
    _write_paper_html(snapshot, output_dir / "index.html")
    return output_dir


def run_paper_loop(
    loader,
    output_dir: Path,
    config: BacktestConfig,
    poll_seconds: int,
    iterations: int,
) -> PaperSnapshot:
    latest: PaperSnapshot | None = None
    remaining = max(1, iterations)
    while remaining > 0:
        candles, funding_rates = loader()
        signals = best_strategy_signals(candles)
        latest = simulate_paper_snapshot(candles, signals, config, funding_rates=funding_rates)
        write_paper_artifacts(latest, output_dir)
        remaining -= 1
        if remaining > 0 and poll_seconds > 0:
            time.sleep(poll_seconds)
    assert latest is not None
    return latest


def format_paper_cli_summary(snapshot: PaperSnapshot) -> str:
    stance = "flat"
    if snapshot.pending_signal is not None and snapshot.pending_signal.side in {"long", "short"}:
        stance = f"pending {snapshot.pending_signal.side} next candle"
    elif snapshot.open_position is not None:
        stance = f"open {snapshot.open_position.side}"
    latest_reason = snapshot.latest_signal.entry_reason if snapshot.latest_signal is not None else "none"
    lines = [
        "Paper trader status",
        f"Strategy: {snapshot.strategy_name} {snapshot.params}",
        f"Last completed candle: {snapshot.last_candle_time.isoformat()} close={snapshot.last_close:.2f}",
        f"Current stance: {stance}",
        f"Marked equity: {snapshot.marked_equity:.2f} | Cash equity: {snapshot.cash_equity:.2f}",
        f"Closed trades: {snapshot.closed_trade_count} | Wins: {snapshot.wins} | Losses: {snapshot.losses}",
        f"Latest signal reason: {latest_reason}",
    ]
    if snapshot.open_position is not None:
        position = snapshot.open_position
        lines.append(
            "Open position: "
            f"{position.side} notional={position.notional:.2f} "
            f"entry={position.entry_price:.2f} stop={position.invalidation_price:.2f} "
            f"unrealized={position.unrealized_pnl:.2f}"
        )
    if snapshot.pending_signal is not None and snapshot.pending_signal.side in {"long", "short"}:
        lines.append(
            "Pending entry: "
            f"{snapshot.pending_signal.side} because {snapshot.pending_signal.entry_reason}"
        )
    lines.append("Dashboard: reports/paper/index.html")
    return "\n".join(lines)


def _snapshot_position(position: dict[str, object] | None, mark_price: float) -> PaperPosition | None:
    if position is None:
        return None
    entry_price = float(position["entry_price"])
    notional = float(position["notional"])
    units = notional / entry_price
    unrealized = (entry_price - mark_price) * units if position["side"] == "short" else (mark_price - entry_price) * units
    signal = position["signal"]
    return PaperPosition(
        side=str(position["side"]),
        entry_time=position["entry_time"],  # type: ignore[arg-type]
        entry_price=entry_price,
        invalidation_price=float(signal.invalidation_price),  # type: ignore[union-attr]
        notional=notional,
        units=units,
        unrealized_pnl=unrealized,
        realized_fees=float(position["entry_fee"]),
        accumulated_funding=float(position["funding"]),
    )


def _snapshot_to_result(snapshot: PaperSnapshot):
    from tradebot_backtest.engine import BacktestResult

    if snapshot.trades:
        start_time = snapshot.trades[0].entry_time
    else:
        start_time = snapshot.last_candle_time
    equity_curve = pd.DataFrame(
        [
            {
                "timestamp": start_time,
                "equity": snapshot.cash_equity,
                "exposed": snapshot.open_position is not None,
            },
            {
                "timestamp": snapshot.last_candle_time,
                "equity": snapshot.marked_equity,
                "exposed": snapshot.open_position is not None,
            },
        ]
    )
    return BacktestResult(snapshot.strategy_name, snapshot.params, snapshot.trades, equity_curve)


def _jsonify(value):
    if value is None:
        return None
    if isinstance(value, Signal):
        return {
            "timestamp": value.timestamp.isoformat(),
            "strategy_name": value.strategy_name,
            "params": value.params,
            "side": value.side,
            "entry_reason": value.entry_reason,
            "invalidation_price": value.invalidation_price,
            "exposure_multiplier": value.exposure_multiplier,
        }
    if hasattr(value, "__dataclass_fields__"):
        payload = asdict(value)
        for key, field_value in list(payload.items()):
            if isinstance(field_value, pd.Timestamp):
                payload[key] = field_value.isoformat()
        return payload
    return value


def _write_paper_html(snapshot: PaperSnapshot, path: Path) -> None:
    stance = "Flat"
    stance_class = "flat"
    if snapshot.pending_signal is not None and snapshot.pending_signal.side in {"long", "short"}:
        stance = f"Pending {snapshot.pending_signal.side.title()}"
        stance_class = "pending"
    elif snapshot.open_position is not None:
        stance = f"Open {snapshot.open_position.side.title()}"
        stance_class = "open"
    open_position_html = "<p class='muted'>No open paper position right now.</p>"
    if snapshot.open_position is not None:
        position = snapshot.open_position
        open_position_html = f"""
        <div class="stat-grid">
          <div class="stat"><span>Side</span><strong>{position.side}</strong></div>
          <div class="stat"><span>Entry</span><strong>{position.entry_price:.2f}</strong></div>
          <div class="stat"><span>Stop</span><strong>{position.invalidation_price:.2f}</strong></div>
          <div class="stat"><span>Size</span><strong>${position.notional:.2f}</strong></div>
          <div class="stat"><span>Unrealized</span><strong>{position.unrealized_pnl:.2f}</strong></div>
        </div>
        """
    recent_trades = snapshot.trades[-8:]
    trade_rows = []
    for trade in reversed(recent_trades):
        trade_rows.append(
            f"""
            <tr>
              <td>{trade.entry_time.isoformat()}</td>
              <td>{trade.side}</td>
              <td>${trade.notional:.2f}</td>
              <td>{trade.entry_price:.2f}</td>
              <td>{trade.exit_price:.2f}</td>
              <td class="{'good' if trade.pnl > 0 else 'bad'}">{trade.pnl:.2f}</td>
              <td>{trade.exit_reason}</td>
            </tr>
            """
        )
    trade_table = """
      <p class="muted">No closed paper trades yet.</p>
    """
    if trade_rows:
        trade_table = f"""
        <table>
          <thead>
            <tr>
              <th>Entry Time</th>
              <th>Side</th>
              <th>Size</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>PnL</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody>
            {''.join(trade_rows)}
          </tbody>
        </table>
        """
    pending_text = "No pending entry."
    if snapshot.pending_signal is not None and snapshot.pending_signal.side in {"long", "short"}:
        pending_text = (
            f"Next candle open would enter {snapshot.pending_signal.side} because "
            f"{snapshot.pending_signal.entry_reason}."
        )
    latest_signal = "No recent signal."
    if snapshot.latest_signal is not None:
        latest_signal = (
            f"{snapshot.latest_signal.timestamp.isoformat()} | "
            f"{snapshot.latest_signal.side} | {snapshot.latest_signal.entry_reason}"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Paper Trader Status</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #08111f;
        --panel: #111d31;
        --panel-soft: #182840;
        --text: #f4f7fb;
        --muted: #9eb0cc;
        --good: #46d59f;
        --bad: #ff7b8b;
        --accent: #66b3ff;
        --pending: #ffd166;
        --border: rgba(255,255,255,0.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        background:
          radial-gradient(circle at top left, rgba(102,179,255,0.12), transparent 28%),
          radial-gradient(circle at top right, rgba(70,213,159,0.08), transparent 24%),
          var(--bg);
        color: var(--text);
      }}
      .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
      .hero {{
        display: grid;
        gap: 18px;
        grid-template-columns: 1.4fr 1fr;
        align-items: stretch;
      }}
      .panel {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        backdrop-filter: blur(12px);
      }}
      .eyebrow {{ color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
      h1, h2, p {{ margin: 0; }}
      h1 {{ font-size: 34px; line-height: 1.1; margin-top: 8px; }}
      h2 {{ font-size: 20px; margin-bottom: 14px; }}
      .muted {{ color: var(--muted); line-height: 1.5; }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        font-weight: 700;
        margin-top: 16px;
        background: rgba(255,255,255,0.06);
      }}
      .status-pill.open {{ color: var(--good); }}
      .status-pill.pending {{ color: var(--pending); }}
      .status-pill.flat {{ color: var(--muted); }}
      .metric-grid, .stat-grid {{
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      }}
      .metric, .stat {{
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
      }}
      .metric span, .stat span {{
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
      }}
      .metric strong, .stat strong {{
        font-size: 20px;
        line-height: 1.2;
      }}
      .stack {{ display: grid; gap: 18px; margin-top: 20px; }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }}
      th, td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--border);
      }}
      th {{ color: var(--muted); font-weight: 600; }}
      .good {{ color: var(--good); }}
      .bad {{ color: var(--bad); }}
      @media (max-width: 860px) {{
        .hero {{ grid-template-columns: 1fr; }}
        h1 {{ font-size: 28px; }}
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <div class="panel">
          <div class="eyebrow">Paper Trading</div>
          <h1>Best Strategy Live Check</h1>
          <p class="muted" style="margin-top: 12px;">
            This page follows the current best backtest winner on fresh 1h Hyperliquid BTC data.
            It is a local paper trader, so it watches the market and simulates entries, exits, fees, and PnL without sending real orders.
          </p>
          <div class="status-pill {stance_class}">{stance}</div>
          <p class="muted" style="margin-top: 14px;">Last completed candle: {snapshot.last_candle_time.isoformat()} | Last close: {snapshot.last_close:.2f}</p>
          <p class="muted" style="margin-top: 8px;">Latest signal: {latest_signal}</p>
        </div>
        <div class="panel">
          <h2>What The Bot Would Do Next</h2>
          <p class="muted">{pending_text}</p>
          <div class="stack">
            <div class="metric-grid">
              <div class="metric"><span>Marked Equity</span><strong>${snapshot.marked_equity:.2f}</strong></div>
              <div class="metric"><span>Cash Equity</span><strong>${snapshot.cash_equity:.2f}</strong></div>
              <div class="metric"><span>Closed Trades</span><strong>{snapshot.closed_trade_count}</strong></div>
              <div class="metric"><span>Win / Loss</span><strong>{snapshot.wins} / {snapshot.losses}</strong></div>
            </div>
          </div>
        </div>
      </section>

      <section class="stack">
        <div class="panel">
          <h2>Open Position</h2>
          {open_position_html}
        </div>
        <div class="panel">
          <h2>Recent Paper Trades</h2>
          {trade_table}
        </div>
      </section>
    </div>
  </body>
</html>
"""
    path.write_text(html, encoding="utf-8")
