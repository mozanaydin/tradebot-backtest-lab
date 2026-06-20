from __future__ import annotations

import pandas as pd

from tradebot_backtest.engine import Signal
from tradebot_backtest.paper import PaperPosition, PaperSnapshot
from tradebot_backtest.testnet import (
    LivePosition,
    TelegramConfig,
    TestnetAccountSnapshot,
    TestnetCredentials,
    TestnetSyncResult,
    TestnetWorkerSummary,
    desired_testnet_state,
    fetch_testnet_account_snapshot,
    format_testnet_cli_summary,
    format_telegram_sync_message,
    format_telegram_worker_event,
    format_worker_final_summary,
    load_telegram_config_from_env,
    send_telegram_message,
    worker_deadline_update,
    worker_stop_reason,
)


def paper_snapshot(
    pending_signal: Signal | None = None,
    open_position: PaperPosition | None = None,
) -> PaperSnapshot:
    return PaperSnapshot(
        strategy_name="compression_breakout",
        params={"lookback": 48},
        generated_at=pd.Timestamp("2026-06-20T10:00:00Z"),
        candle_count=100,
        last_candle_time=pd.Timestamp("2026-06-20T10:00:00Z"),
        last_close=64000.0,
        cash_equity=1000.0,
        marked_equity=1010.0,
        closed_trade_count=10,
        wins=5,
        losses=5,
        open_position=open_position,
        pending_signal=pending_signal,
        latest_signal=pending_signal,
        trades=[],
    )


def test_desired_testnet_state_prefers_pending_entry() -> None:
    signal = Signal(
        pd.Timestamp("2026-06-20T10:00:00Z"),
        "compression_breakout",
        {"lookback": 48},
        "short",
        "compressed_range_break_low",
        65000.0,
    )

    state, reason, reference = desired_testnet_state(paper_snapshot(pending_signal=signal))

    assert state == "short"
    assert "fresh short entry" in reason
    assert reference == signal


def test_desired_testnet_state_uses_open_short_when_no_pending_signal() -> None:
    position = PaperPosition(
        side="short",
        entry_time=pd.Timestamp("2026-06-19T10:00:00Z"),
        entry_price=65000.0,
        invalidation_price=66000.0,
        notional=300.0,
        units=0.004,
        unrealized_pnl=5.0,
        realized_fees=0.2,
        accumulated_funding=0.0,
    )

    state, reason, reference = desired_testnet_state(paper_snapshot(open_position=position))

    assert state == "short"
    assert "already in an active short" in reason
    assert reference is not None
    assert reference.side == "short"


def test_format_testnet_cli_summary_mentions_dashboard() -> None:
    snapshot = TestnetSyncResult(
        generated_at=pd.Timestamp("2026-06-20T10:05:00Z"),
        paper_snapshot=paper_snapshot(),
        account_snapshot=TestnetAccountSnapshot(
            account_address="0xabc",
            trade_address="0xabc",
            account_value=1250.0,
            withdrawable=900.0,
            mark_price=64000.0,
            open_orders=[],
            recent_fills=[],
            live_position=LivePosition(
                side="short",
                size=0.01,
                entry_price=64500.0,
                position_value=645.0,
                unrealized_pnl=10.0,
                leverage=3.0,
            ),
        ),
        desired_state="flat",
        action="close_short",
        action_reason="exit signal",
        executed=False,
        order_size_btc=None,
        order_notional_usd=None,
        response=None,
    )

    summary = format_testnet_cli_summary(snapshot)

    assert "Hyperliquid testnet sync" in summary
    assert "Dashboard: reports/testnet/index.html" in summary


def test_fetch_testnet_account_snapshot_uses_spot_state_for_unified_balances() -> None:
    class FakeInfo:
        def user_state(self, address: str):
            assert address == "0xabc"
            return {
                "marginSummary": {"accountValue": "0.0"},
                "withdrawable": "0.0",
                "assetPositions": [],
            }

        def spot_user_state(self, address: str):
            assert address == "0xabc"
            return {
                "balances": [
                    {"coin": "USDC", "total": "600.0", "hold": "25.0"},
                ],
                "tokenToAvailableAfterMaintenance": [[0, "575.0"]],
            }

        def all_mids(self):
            return {"BTC": "64000.0"}

        def open_orders(self, address: str):
            assert address == "0xabc"
            return []

        def user_fills_by_time(self, address: str, start_time: int):
            assert address == "0xabc"
            assert start_time > 0
            return []

    snapshot = fetch_testnet_account_snapshot(
        FakeInfo(),
        TestnetCredentials(account_address="0xabc", api_secret="0x123"),
    )

    assert snapshot.account_value == 600.0
    assert snapshot.withdrawable == 575.0


def test_worker_stop_reason_triggers_on_low_balance() -> None:
    snapshot = TestnetAccountSnapshot(
        account_address="0xabc",
        trade_address="0xabc",
        account_value=9.5,
        withdrawable=0.0,
        mark_price=64000.0,
        open_orders=[],
        recent_fills=[],
        live_position=None,
    )

    reason = worker_stop_reason(snapshot, min_account_value=10.0)

    assert "withdrawable balance is depleted" in reason


def test_worker_deadline_update_extends_once_when_no_trades() -> None:
    now = pd.Timestamp("2026-06-22T10:00:00Z")
    deadline = pd.Timestamp("2026-06-22T09:00:00Z")

    new_deadline, extension_used, status = worker_deadline_update(
        now,
        deadline,
        executed_actions=0,
        extension_used=False,
        extend_hours_if_no_trades=48.0,
    )

    assert extension_used is True
    assert status == "extended"
    assert new_deadline == pd.Timestamp("2026-06-24T10:00:00Z")


def test_load_telegram_config_from_env_reads_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "HL_TELEGRAM_BOT_TOKEN=abc123\nHL_TELEGRAM_CHAT_ID=147\n",
        encoding="utf-8",
    )

    config = load_telegram_config_from_env(dotenv_path=dotenv)

    assert config.bot_token == "abc123"
    assert config.chat_id == "147"


def test_send_telegram_message_uses_bot_api() -> None:
    sent: dict[str, object] = {}

    class FakeClient:
        def post(self, url: str, json: dict[str, object], timeout: float):
            sent["url"] = url
            sent["json"] = json
            sent["timeout"] = timeout

            class Response:
                def raise_for_status(self):
                    return None

            return Response()

    send_telegram_message(
        TelegramConfig(bot_token="abc123", chat_id="147"),
        "worker started",
        client=FakeClient(),
    )

    assert sent["url"] == "https://api.telegram.org/botabc123/sendMessage"
    assert sent["json"] == {
        "chat_id": "147",
        "text": "worker started",
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }


def test_send_telegram_message_sets_html_parse_mode() -> None:
    sent: dict[str, object] = {}

    class FakeClient:
        def post(self, url: str, json: dict[str, object], timeout: float):
            sent["json"] = json

            class Response:
                def raise_for_status(self):
                    return None

            return Response()

    send_telegram_message(
        TelegramConfig(bot_token="abc123", chat_id="147"),
        "<b>worker started</b>",
        client=FakeClient(),
    )

    assert sent["json"] == {
        "chat_id": "147",
        "text": "<b>worker started</b>",
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }


def test_format_worker_final_summary_reports_extension_and_pnl() -> None:
    summary = TestnetWorkerSummary(
        started_at=pd.Timestamp("2026-06-20T10:00:00Z"),
        ended_at=pd.Timestamp("2026-06-22T10:00:00Z"),
        initial_account_value=600.0,
        final_account_value=615.0,
        cycles=40,
        executed_actions=2,
        extension_used=True,
        stop_reason="time window complete",
        event_log_path="reports/testnet/worker_events.jsonl",
    )

    text = format_worker_final_summary(summary)

    assert "extension used: yes" in text.lower()
    assert "executed actions: 2" in text.lower()
    assert "+15.00 usdc" in text.lower()


def test_format_telegram_sync_message_is_more_readable() -> None:
    snapshot = TestnetSyncResult(
        generated_at=pd.Timestamp("2026-06-20T10:05:00Z"),
        paper_snapshot=paper_snapshot(),
        account_snapshot=TestnetAccountSnapshot(
            account_address="0xabc",
            trade_address="0xabc",
            account_value=600.0,
            withdrawable=590.0,
            mark_price=64000.0,
            open_orders=[],
            recent_fills=[],
            live_position=LivePosition(
                side="short",
                size=0.01,
                entry_price=64500.0,
                position_value=645.0,
                unrealized_pnl=10.0,
                leverage=3.0,
            ),
        ),
        desired_state="flat",
        action="close_short",
        action_reason="exit signal",
        executed=True,
        order_size_btc=0.01,
        order_notional_usd=640.0,
        response=None,
    )

    message = format_telegram_sync_message(snapshot)

    assert "<b>Hyperliquid Testnet Update</b>" in message
    assert "<b>Action</b>: close_short" in message
    assert "<b>Account</b>: 600.00 USDC" in message


def test_format_telegram_worker_event_is_structured() -> None:
    message = format_telegram_worker_event(
        "Worker Started",
        [
            ("Mode", "execute"),
            ("Deadline", "2026-06-22T10:00:00Z"),
        ],
    )

    assert message.startswith("<b>Hyperliquid Worker Started</b>")
    assert "• <b>Mode</b>: execute" in message
