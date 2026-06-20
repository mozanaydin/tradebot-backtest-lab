from __future__ import annotations

import json
import os
import time
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import TESTNET_API_URL

from tradebot_backtest.data import read_funding_csv, refresh_candles_cache
from tradebot_backtest.engine import BacktestConfig, Signal, _notional_for_signal
from tradebot_backtest.paper import (
    PaperSnapshot,
    best_strategy_config,
    best_strategy_signals,
    simulate_paper_snapshot,
)


@dataclass(frozen=True)
class TestnetCredentials:
    __test__ = False
    account_address: str
    api_secret: str
    vault_address: str | None = None


@dataclass(frozen=True)
class ApprovedAgent:
    account_address: str
    agent_address: str
    agent_secret: str
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class LivePosition:
    side: str
    size: float
    entry_price: float | None
    position_value: float | None
    unrealized_pnl: float | None
    leverage: float | None


@dataclass(frozen=True)
class TestnetAccountSnapshot:
    __test__ = False
    account_address: str
    trade_address: str
    account_value: float
    withdrawable: float
    mark_price: float
    open_orders: list[dict[str, Any]]
    recent_fills: list[dict[str, Any]]
    live_position: LivePosition | None


@dataclass(frozen=True)
class TestnetSyncResult:
    __test__ = False
    generated_at: pd.Timestamp
    paper_snapshot: PaperSnapshot
    account_snapshot: TestnetAccountSnapshot
    desired_state: str
    action: str
    action_reason: str
    executed: bool
    order_size_btc: float | None
    order_notional_usd: float | None
    response: dict[str, Any] | None


@dataclass(frozen=True)
class TelegramConfig:
    __test__ = False
    bot_token: str
    chat_id: str


@dataclass(frozen=True)
class TestnetWorkerSummary:
    __test__ = False
    started_at: pd.Timestamp
    ended_at: pd.Timestamp
    initial_account_value: float
    final_account_value: float
    cycles: int
    executed_actions: int
    extension_used: bool
    stop_reason: str
    event_log_path: str


def load_testnet_credentials_from_env(
    account_address_env: str = "HL_TESTNET_ACCOUNT_ADDRESS",
    api_secret_env: str = "HL_TESTNET_API_SECRET",
    vault_address_env: str = "HL_TESTNET_VAULT_ADDRESS",
    dotenv_path: Path | None = None,
) -> TestnetCredentials:
    _load_dotenv(dotenv_path or Path(".env"))
    account_address = os.getenv(account_address_env, "").strip()
    api_secret = os.getenv(api_secret_env, "").strip()
    vault_address = os.getenv(vault_address_env, "").strip() or None
    if not account_address:
        raise ValueError(f"missing required environment variable: {account_address_env}")
    if not api_secret:
        raise ValueError(f"missing required environment variable: {api_secret_env}")
    return TestnetCredentials(
        account_address=account_address.lower(),
        api_secret=api_secret,
        vault_address=vault_address.lower() if vault_address else None,
    )


def approve_testnet_agent(master_secret: str, account_address: str, name: str | None = None) -> ApprovedAgent:
    master_wallet = _wallet_from_secret(master_secret)
    exchange = Exchange(master_wallet, base_url=TESTNET_API_URL, account_address=account_address.lower())
    response, agent_secret = exchange.approve_agent(name=name)
    agent_wallet = _wallet_from_secret(agent_secret)
    return ApprovedAgent(
        account_address=account_address.lower(),
        agent_address=agent_wallet.address.lower(),
        agent_secret=agent_secret,
        raw_response=response,
    )


def load_telegram_config_from_env(
    token_env: str = "HL_TELEGRAM_BOT_TOKEN",
    chat_id_env: str = "HL_TELEGRAM_CHAT_ID",
    dotenv_path: Path | None = None,
) -> TelegramConfig:
    _load_dotenv(dotenv_path or Path(".env"))
    bot_token = os.getenv(token_env, "").strip()
    chat_id = os.getenv(chat_id_env, "").strip()
    if not bot_token:
        raise ValueError(f"missing required environment variable: {token_env}")
    if not chat_id:
        raise ValueError(f"missing required environment variable: {chat_id_env}")
    return TelegramConfig(bot_token=bot_token, chat_id=chat_id)


def send_telegram_message(
    config: TelegramConfig,
    text: str,
    client: httpx.Client | None = None,
) -> None:
    owns_client = client is None
    client = client or httpx.Client()
    try:
        response = client.post(
            f"https://api.telegram.org/bot{config.bot_token}/sendMessage",
            json={
                "chat_id": config.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30.0,
        )
        response.raise_for_status()
    finally:
        if owns_client:
            client.close()


def build_paper_snapshot_for_best_strategy(
    candles: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    starting_balance: float = 1000.0,
) -> PaperSnapshot:
    config = best_strategy_config(starting_balance=starting_balance)
    signals = best_strategy_signals(candles)
    return simulate_paper_snapshot(candles, signals, config, funding_rates=funding_rates)


def sync_best_strategy_to_testnet(
    credentials: TestnetCredentials,
    candles: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    execute: bool = False,
    leverage: int = 3,
    slippage: float = 0.02,
    fills_lookback_days: int = 14,
) -> TestnetSyncResult:
    paper_snapshot = build_paper_snapshot_for_best_strategy(candles, funding_rates=funding_rates)
    info = Info(TESTNET_API_URL, skip_ws=True)
    account_snapshot = fetch_testnet_account_snapshot(info, credentials, fills_lookback_days=fills_lookback_days)
    desired_state, action_reason, reference_signal = desired_testnet_state(paper_snapshot)
    live_side = account_snapshot.live_position.side if account_snapshot.live_position else "flat"
    action = "hold"
    response: dict[str, Any] | None = None
    order_size_btc: float | None = None
    order_notional_usd: float | None = None

    if live_side == "long":
        action = "close_long"
        action_reason = "best strategy is short-only, so any live long is flattened"
    elif live_side == "short" and desired_state == "flat":
        action = "close_short"
    elif live_side == "flat" and desired_state == "short":
        action = "open_short"

    if action == "hold":
        return TestnetSyncResult(
            generated_at=pd.Timestamp.now(tz="UTC"),
            paper_snapshot=paper_snapshot,
            account_snapshot=account_snapshot,
            desired_state=desired_state,
            action=action,
            action_reason=action_reason,
            executed=execute,
            order_size_btc=None,
            order_notional_usd=None,
            response=None,
        )

    exchange = _build_exchange(credentials)
    if action in {"open_short", "close_short", "close_long"}:
        exchange.update_leverage(leverage, "BTC", is_cross=True)

    if action == "open_short":
        if reference_signal is None:
            raise RuntimeError("cannot size open_short action without a reference signal")
        config = best_strategy_config(starting_balance=account_snapshot.account_value)
        current_price = account_snapshot.mark_price
        order_notional_usd = _notional_for_signal(
            account_snapshot.account_value,
            float(reference_signal.invalidation_price),
            current_price,
            config,
            reference_signal.exposure_multiplier,
        )
        if order_notional_usd <= 0:
            return TestnetSyncResult(
                generated_at=pd.Timestamp.now(tz="UTC"),
                paper_snapshot=paper_snapshot,
                account_snapshot=account_snapshot,
                desired_state=desired_state,
                action="skip_open_short",
                action_reason="risk sizing rejected this entry because the stop distance is too small versus costs",
                executed=execute,
                order_size_btc=None,
                order_notional_usd=0.0,
                response=None,
            )
        order_size_btc = _round_size(order_notional_usd / current_price, info, "BTC")
        if order_size_btc <= 0:
            return TestnetSyncResult(
                generated_at=pd.Timestamp.now(tz="UTC"),
                paper_snapshot=paper_snapshot,
                account_snapshot=account_snapshot,
                desired_state=desired_state,
                action="skip_open_short",
                action_reason="size rounded down to zero at the exchange lot size",
                executed=execute,
                order_size_btc=0.0,
                order_notional_usd=order_notional_usd,
                response=None,
            )
        if execute:
            response = exchange.market_open("BTC", is_buy=False, sz=order_size_btc, slippage=slippage)
    elif action in {"close_short", "close_long"} and execute:
        response = exchange.market_close("BTC", slippage=slippage)

    return TestnetSyncResult(
        generated_at=pd.Timestamp.now(tz="UTC"),
        paper_snapshot=paper_snapshot,
        account_snapshot=account_snapshot,
        desired_state=desired_state,
        action=action,
        action_reason=action_reason,
        executed=execute,
        order_size_btc=order_size_btc,
        order_notional_usd=order_notional_usd,
        response=response,
    )


def run_testnet_worker(
    credentials: TestnetCredentials,
    *,
    execute: bool,
    leverage: int,
    slippage: float,
    symbol: str,
    interval: str,
    days: int,
    data_dir: Path,
    funding_file: Path | None,
    reports_dir: Path,
    poll_seconds: int,
    duration_hours: float,
    extend_hours_if_no_trades: float,
    min_account_value: float,
    telegram: TelegramConfig | None = None,
) -> TestnetWorkerSummary:
    started_at = pd.Timestamp.now(tz="UTC")
    deadline = started_at + pd.Timedelta(hours=duration_hours)
    extension_used = False
    cycles = 0
    executed_actions = 0
    initial_account_value: float | None = None
    final_account_value = 0.0
    event_log_path = reports_dir / "worker_events.jsonl"
    stop_reason = "time window complete"

    if telegram is not None:
        send_telegram_message(
            telegram,
            format_telegram_worker_event(
                "Worker Started",
                [
                    ("Symbol", symbol),
                    ("Interval", interval),
                    ("Mode", "execute" if execute else "dry-run"),
                    ("Deadline", deadline.isoformat()),
                ],
            ),
        )

    while True:
        cycles += 1
        candles = refresh_candles_cache("hyperliquid", symbol.upper(), interval, days, data_dir)
        funding_rates = read_funding_csv(funding_file) if funding_file is not None and funding_file.exists() else None
        result = sync_best_strategy_to_testnet(
            credentials,
            candles,
            funding_rates=funding_rates,
            execute=execute,
            leverage=leverage,
            slippage=slippage,
        )
        write_testnet_dashboard(result, reports_dir)
        _append_worker_event(event_log_path, result, cycles)

        if initial_account_value is None:
            initial_account_value = result.account_snapshot.account_value
        final_account_value = result.account_snapshot.account_value

        if execute and result.action in {"open_short", "close_short", "close_long"} and result.response is not None:
            executed_actions += 1
            if telegram is not None:
                send_telegram_message(telegram, format_telegram_sync_message(result))

        stop_reason = worker_stop_reason(result.account_snapshot, min_account_value)
        if stop_reason is not None:
            if telegram is not None:
                send_telegram_message(
                    telegram,
                    format_telegram_worker_event(
                        "Worker Stopped",
                        [
                            ("Reason", stop_reason),
                            ("Account", f"{result.account_snapshot.account_value:.2f} USDC"),
                            ("Withdrawable", f"{result.account_snapshot.withdrawable:.2f} USDC"),
                        ],
                    ),
                )
            break

        now = pd.Timestamp.now(tz="UTC")
        deadline, extension_used, status = worker_deadline_update(
            now,
            deadline,
            executed_actions=executed_actions,
            extension_used=extension_used,
            extend_hours_if_no_trades=extend_hours_if_no_trades,
        )
        if status == "extended" and telegram is not None:
            send_telegram_message(
                telegram,
                format_telegram_worker_event(
                    "Worker Extended",
                    [
                        ("Reason", "No trades happened in the first window"),
                        ("New Deadline", deadline.isoformat()),
                    ],
                ),
            )
        elif status == "complete":
            stop_reason = "time window complete"
            break

        if poll_seconds > 0:
            time.sleep(poll_seconds)

    summary = TestnetWorkerSummary(
        started_at=started_at,
        ended_at=pd.Timestamp.now(tz="UTC"),
        initial_account_value=initial_account_value if initial_account_value is not None else final_account_value,
        final_account_value=final_account_value,
        cycles=cycles,
        executed_actions=executed_actions,
        extension_used=extension_used,
        stop_reason=stop_reason,
        event_log_path=str(event_log_path),
    )
    _write_worker_summary(summary, reports_dir / "worker_summary.json")
    if telegram is not None:
        send_telegram_message(telegram, format_telegram_worker_final_summary(summary))
    return summary


def fetch_testnet_account_snapshot(
    info: Info,
    credentials: TestnetCredentials,
    fills_lookback_days: int = 14,
) -> TestnetAccountSnapshot:
    trade_address = credentials.vault_address or credentials.account_address
    user_state = info.user_state(trade_address)
    spot_state = info.spot_user_state(trade_address)
    position = _extract_btc_position(user_state)
    mids = info.all_mids()
    mark_price = float(mids["BTC"])
    open_orders = info.open_orders(trade_address)
    start_time = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=fills_lookback_days)).timestamp() * 1000)
    recent_fills = info.user_fills_by_time(trade_address, start_time)
    account_value = _spot_account_value(spot_state)
    if account_value is None:
        account_value = float(user_state["marginSummary"]["accountValue"])
    withdrawable = _spot_withdrawable(spot_state)
    if withdrawable is None:
        withdrawable = float(user_state["withdrawable"])
    return TestnetAccountSnapshot(
        account_address=credentials.account_address,
        trade_address=trade_address,
        account_value=account_value,
        withdrawable=withdrawable,
        mark_price=mark_price,
        open_orders=open_orders,
        recent_fills=[fill for fill in recent_fills if fill.get("coin") == "BTC"][-12:],
        live_position=position,
    )


def desired_testnet_state(snapshot: PaperSnapshot) -> tuple[str, str, Signal | None]:
    if snapshot.pending_signal is not None:
        if snapshot.pending_signal.side == "short":
            return "short", "the last closed candle generated a fresh short entry for the next candle open", snapshot.pending_signal
        if snapshot.pending_signal.side == "flat":
            return "flat", "the last closed candle generated an exit for the next candle open", snapshot.pending_signal
    if snapshot.open_position is not None and snapshot.open_position.side == "short":
        synthetic_signal = Signal(
            snapshot.last_candle_time,
            snapshot.strategy_name,
            snapshot.params,
            "short",
            "carry_existing_short",
            snapshot.open_position.invalidation_price,
        )
        return "short", "the strategy is already in an active short and wants to stay there", synthetic_signal
    return "flat", "no active short and no fresh short entry signal", None


def worker_stop_reason(snapshot: TestnetAccountSnapshot, min_account_value: float) -> str | None:
    if snapshot.withdrawable <= 0:
        return "withdrawable balance is depleted"
    if snapshot.account_value < min_account_value:
        return f"account value dropped below the safety floor of {min_account_value:.2f} USDC"
    return None


def worker_deadline_update(
    now: pd.Timestamp,
    deadline: pd.Timestamp,
    *,
    executed_actions: int,
    extension_used: bool,
    extend_hours_if_no_trades: float,
) -> tuple[pd.Timestamp, bool, str]:
    if now < deadline:
        return deadline, extension_used, "continue"
    if executed_actions == 0 and not extension_used and extend_hours_if_no_trades > 0:
        return now + pd.Timedelta(hours=extend_hours_if_no_trades), True, "extended"
    return deadline, extension_used, "complete"


def format_testnet_cli_summary(result: TestnetSyncResult) -> str:
    live_position = result.account_snapshot.live_position
    live_text = "flat"
    if live_position is not None:
        live_text = (
            f"{live_position.side} size={live_position.size:.6f} "
            f"entry={live_position.entry_price or 0:.2f}"
        )
    lines = [
        "Hyperliquid testnet sync",
        f"Desired state: {result.desired_state}",
        f"Action: {result.action}",
        f"Reason: {result.action_reason}",
        f"Executed: {result.executed}",
        f"Account value: {result.account_snapshot.account_value:.2f} USDC",
        f"Withdrawable: {result.account_snapshot.withdrawable:.2f} USDC",
        f"Live position before sync: {live_text}",
        f"Current BTC mark: {result.account_snapshot.mark_price:.2f}",
        f"Paper snapshot stance: {_paper_stance(result.paper_snapshot)}",
    ]
    if result.order_size_btc is not None:
        lines.append(f"Order size: {result.order_size_btc:.6f} BTC")
    if result.order_notional_usd is not None:
        lines.append(f"Order notional: {result.order_notional_usd:.2f} USDC")
    if result.response is not None:
        lines.append(f"Exchange response: {json.dumps(result.response)}")
    lines.append("Dashboard: reports/testnet/index.html")
    return "\n".join(lines)


def format_worker_final_summary(summary: TestnetWorkerSummary) -> str:
    pnl = summary.final_account_value - summary.initial_account_value
    pnl_pct = 0.0 if summary.initial_account_value == 0 else (pnl / summary.initial_account_value) * 100.0
    extension = "yes" if summary.extension_used else "no"
    return "\n".join(
        [
            "Hyperliquid worker finished",
            f"started: {summary.started_at.isoformat()}",
            f"ended: {summary.ended_at.isoformat()}",
            f"cycles: {summary.cycles}",
            f"executed actions: {summary.executed_actions}",
            f"extension used: {extension}",
            f"start balance: {summary.initial_account_value:.2f} USDC",
            f"end balance: {summary.final_account_value:.2f} USDC",
            f"PnL: {pnl:+.2f} USDC ({pnl_pct:+.2f}%)",
            f"stop reason: {summary.stop_reason}",
            f"log: {summary.event_log_path}",
        ]
    )


def format_telegram_sync_message(result: TestnetSyncResult) -> str:
    lines = [
        "<b>Hyperliquid Testnet Update</b>",
        "",
        f"• <b>Action</b>: {escape(result.action)}",
        f"• <b>Desired State</b>: {escape(result.desired_state)}",
        f"• <b>Reason</b>: {escape(result.action_reason)}",
        f"• <b>Account</b>: {result.account_snapshot.account_value:.2f} USDC",
        f"• <b>Withdrawable</b>: {result.account_snapshot.withdrawable:.2f} USDC",
        f"• <b>BTC Mark</b>: {result.account_snapshot.mark_price:.2f}",
    ]
    if result.order_size_btc is not None:
        lines.append(f"• <b>Order Size</b>: {result.order_size_btc:.6f} BTC")
    if result.order_notional_usd is not None:
        lines.append(f"• <b>Order Notional</b>: {result.order_notional_usd:.2f} USDC")
    live_position = result.account_snapshot.live_position
    if live_position is not None:
        lines.append(
            f"• <b>Live Position</b>: {escape(live_position.side)} {live_position.size:.6f} BTC"
        )
    lines.append(f"• <b>Time</b>: {result.generated_at.isoformat()}")
    return "\n".join(lines)


def format_telegram_worker_event(title: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"<b>Hyperliquid {escape(title)}</b>", ""]
    for label, value in rows:
        lines.append(f"• <b>{escape(label)}</b>: {escape(value)}")
    return "\n".join(lines)


def format_telegram_worker_final_summary(summary: TestnetWorkerSummary) -> str:
    pnl = summary.final_account_value - summary.initial_account_value
    pnl_pct = 0.0 if summary.initial_account_value == 0 else (pnl / summary.initial_account_value) * 100.0
    return format_telegram_worker_event(
        "Worker Finished",
        [
            ("Cycles", str(summary.cycles)),
            ("Executed Actions", str(summary.executed_actions)),
            ("Extension Used", "yes" if summary.extension_used else "no"),
            ("Start Balance", f"{summary.initial_account_value:.2f} USDC"),
            ("End Balance", f"{summary.final_account_value:.2f} USDC"),
            ("PnL", f"{pnl:+.2f} USDC ({pnl_pct:+.2f}%)"),
            ("Stop Reason", summary.stop_reason),
        ],
    )


def write_testnet_dashboard(result: TestnetSyncResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "snapshot.json"
    payload = {
        "generated_at": result.generated_at.isoformat(),
        "desired_state": result.desired_state,
        "action": result.action,
        "action_reason": result.action_reason,
        "executed": result.executed,
        "order_size_btc": result.order_size_btc,
        "order_notional_usd": result.order_notional_usd,
        "response": result.response,
        "paper": {
            "last_candle_time": result.paper_snapshot.last_candle_time.isoformat(),
            "last_close": result.paper_snapshot.last_close,
            "stance": _paper_stance(result.paper_snapshot),
            "latest_signal_reason": result.paper_snapshot.latest_signal.entry_reason if result.paper_snapshot.latest_signal else None,
        },
        "testnet": {
            "account_address": result.account_snapshot.account_address,
            "trade_address": result.account_snapshot.trade_address,
            "account_value": result.account_snapshot.account_value,
            "withdrawable": result.account_snapshot.withdrawable,
            "mark_price": result.account_snapshot.mark_price,
            "live_position": None if result.account_snapshot.live_position is None else result.account_snapshot.live_position.__dict__,
            "open_orders": result.account_snapshot.open_orders,
            "recent_fills": result.account_snapshot.recent_fills,
        },
    }
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_testnet_html(result, output_dir / "index.html")
    return output_dir


def _build_exchange(credentials: TestnetCredentials) -> Exchange:
    wallet = _wallet_from_secret(credentials.api_secret)
    return Exchange(
        wallet,
        base_url=TESTNET_API_URL,
        account_address=credentials.account_address,
        vault_address=credentials.vault_address,
    )


def _wallet_from_secret(secret: str) -> LocalAccount:
    normalized = secret.strip()
    if not normalized:
        raise ValueError("wallet secret cannot be empty")
    if not normalized.startswith("0x"):
        normalized = f"0x{normalized}"
    return Account.from_key(normalized)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _append_worker_event(path: Path, result: TestnetSyncResult, cycle: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cycle": cycle,
        "generated_at": result.generated_at.isoformat(),
        "desired_state": result.desired_state,
        "action": result.action,
        "action_reason": result.action_reason,
        "executed": result.executed,
        "account_value": result.account_snapshot.account_value,
        "withdrawable": result.account_snapshot.withdrawable,
        "mark_price": result.account_snapshot.mark_price,
        "order_size_btc": result.order_size_btc,
        "order_notional_usd": result.order_notional_usd,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _write_worker_summary(summary: TestnetWorkerSummary, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "started_at": summary.started_at.isoformat(),
                "ended_at": summary.ended_at.isoformat(),
                "initial_account_value": summary.initial_account_value,
                "final_account_value": summary.final_account_value,
                "cycles": summary.cycles,
                "executed_actions": summary.executed_actions,
                "extension_used": summary.extension_used,
                "stop_reason": summary.stop_reason,
                "event_log_path": summary.event_log_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _extract_btc_position(user_state: dict[str, Any]) -> LivePosition | None:
    for raw_position in user_state.get("assetPositions", []):
        position = raw_position.get("position", {})
        if position.get("coin") != "BTC":
            continue
        size = float(position.get("szi", 0.0))
        if abs(size) <= 1e-12:
            continue
        side = "long" if size > 0 else "short"
        leverage_value = position.get("leverage", {}).get("value")
        return LivePosition(
            side=side,
            size=abs(size),
            entry_price=float(position["entryPx"]) if position.get("entryPx") else None,
            position_value=float(position["positionValue"]) if position.get("positionValue") else None,
            unrealized_pnl=float(position["unrealizedPnl"]) if position.get("unrealizedPnl") else None,
            leverage=float(leverage_value) if leverage_value is not None else None,
        )
    return None


def _spot_account_value(spot_state: dict[str, Any]) -> float | None:
    balances = spot_state.get("balances", [])
    for balance in balances:
        if balance.get("coin") == "USDC":
            return float(balance.get("total", 0.0))
    return None


def _spot_withdrawable(spot_state: dict[str, Any]) -> float | None:
    available = spot_state.get("tokenToAvailableAfterMaintenance", [])
    for token, value in available:
        if int(token) == 0:
            return float(value)
    return None


def _paper_stance(snapshot: PaperSnapshot) -> str:
    if snapshot.pending_signal is not None and snapshot.pending_signal.side in {"short", "long"}:
        return f"pending {snapshot.pending_signal.side}"
    if snapshot.pending_signal is not None and snapshot.pending_signal.side == "flat":
        return "pending exit"
    if snapshot.open_position is not None:
        return f"open {snapshot.open_position.side}"
    return "flat"


def _round_size(size: float, info: Info, coin: str) -> float:
    asset = info.coin_to_asset[coin]
    decimals = int(info.asset_to_sz_decimals[asset])
    return round(size, decimals)


def _write_testnet_html(result: TestnetSyncResult, path: Path) -> None:
    paper_stance = _paper_stance(result.paper_snapshot)
    live_position = result.account_snapshot.live_position
    live_card = "<p class='muted'>No BTC position is open on the testnet account.</p>"
    if live_position is not None:
        live_card = f"""
        <div class="stat-grid">
          <div class="stat"><span>Side</span><strong>{live_position.side}</strong></div>
          <div class="stat"><span>Size</span><strong>{live_position.size:.6f} BTC</strong></div>
          <div class="stat"><span>Entry</span><strong>{(live_position.entry_price or 0):.2f}</strong></div>
          <div class="stat"><span>Unrealized</span><strong>{(live_position.unrealized_pnl or 0):.2f}</strong></div>
        </div>
        """
    fill_rows = []
    for fill in reversed(result.account_snapshot.recent_fills[-10:]):
        closed_pnl = float(fill.get("closedPnl", 0.0))
        fill_rows.append(
            f"""
            <tr>
              <td>{pd.to_datetime(fill['time'], unit='ms', utc=True).isoformat()}</td>
              <td>{fill.get('dir', '')}</td>
              <td>{float(fill.get('sz', 0.0)):.6f}</td>
              <td>{float(fill.get('px', 0.0)):.2f}</td>
              <td class="{'good' if closed_pnl >= 0 else 'bad'}">{closed_pnl:.2f}</td>
            </tr>
            """
        )
    fills_html = "<p class='muted'>No recent BTC fills found on the testnet account.</p>"
    if fill_rows:
        fills_html = f"""
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Direction</th>
              <th>Size</th>
              <th>Price</th>
              <th>Closed PnL</th>
            </tr>
          </thead>
          <tbody>{''.join(fill_rows)}</tbody>
        </table>
        """
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hyperliquid Testnet Sync</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #07121f;
        --panel: #101c30;
        --panel-soft: #182842;
        --border: rgba(255,255,255,0.08);
        --text: #f5f7fb;
        --muted: #9db1cd;
        --good: #4dd5a1;
        --bad: #ff7e8f;
        --accent: #7dc1ff;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        background:
          radial-gradient(circle at top left, rgba(125,193,255,0.12), transparent 24%),
          radial-gradient(circle at top right, rgba(77,213,161,0.09), transparent 20%),
          var(--bg);
        color: var(--text);
      }}
      .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
      .hero, .stack {{ display: grid; gap: 18px; }}
      .hero {{ grid-template-columns: 1.4fr 1fr; }}
      .stack {{ margin-top: 18px; }}
      .panel {{
        background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
      }}
      .eyebrow {{ color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
      h1, h2, p {{ margin: 0; }}
      h1 {{ font-size: 34px; line-height: 1.1; margin-top: 8px; }}
      h2 {{ font-size: 20px; margin-bottom: 14px; }}
      .muted {{ color: var(--muted); line-height: 1.5; }}
      .metric-grid, .stat-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
      .metric, .stat {{ background: var(--panel-soft); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
      .metric span, .stat span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
      .metric strong, .stat strong {{ font-size: 20px; line-height: 1.2; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
      th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); }}
      th {{ color: var(--muted); font-weight: 600; }}
      .good {{ color: var(--good); }}
      .bad {{ color: var(--bad); }}
      code {{ color: var(--accent); }}
      @media (max-width: 860px) {{ .hero {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <div class="panel">
          <div class="eyebrow">Hyperliquid Testnet</div>
          <h1>Best Strategy Account Sync</h1>
          <p class="muted" style="margin-top:12px;">
            This page compares the best backtest winner against the live state of the Hyperliquid testnet account and shows the exact action the bot wants to take right now.
          </p>
          <p class="muted" style="margin-top:14px;">Desired state: <code>{result.desired_state}</code> | Action: <code>{result.action}</code></p>
          <p class="muted" style="margin-top:8px;">Reason: {result.action_reason}</p>
          <p class="muted" style="margin-top:8px;">Generated: {result.generated_at.isoformat()}</p>
        </div>
        <div class="panel">
          <h2>Account Snapshot</h2>
          <div class="metric-grid">
            <div class="metric"><span>Account Value</span><strong>${result.account_snapshot.account_value:.2f}</strong></div>
            <div class="metric"><span>Withdrawable</span><strong>${result.account_snapshot.withdrawable:.2f}</strong></div>
            <div class="metric"><span>BTC Mark</span><strong>{result.account_snapshot.mark_price:.2f}</strong></div>
            <div class="metric"><span>Paper Stance</span><strong>{paper_stance}</strong></div>
          </div>
        </div>
      </section>
      <section class="stack">
        <div class="panel">
          <h2>Live Position</h2>
          {live_card}
        </div>
        <div class="panel">
          <h2>Recent Testnet BTC Fills</h2>
          {fills_html}
        </div>
      </section>
    </div>
  </body>
</html>
"""
    path.write_text(html, encoding="utf-8")
