from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import pandas as pd

Side = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.00045
    slippage_rate: float = 0.0002

    def one_way_cost(self, notional: float) -> float:
        return notional * (self.fee_rate + self.slippage_rate)

    def round_trip_cost(self, notional: float) -> float:
        return self.one_way_cost(notional) * 2


@dataclass(frozen=True)
class BacktestConfig:
    starting_balance: float = 1000.0
    risk_fraction: float = 0.05
    max_leverage: float = 3.0
    cost_model: CostModel = field(default_factory=CostModel)
    include_funding: bool = True


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    strategy_name: str
    params: dict[str, int | float | str]
    side: Side
    entry_reason: str
    invalidation_price: float
    exposure_multiplier: float = 1.0


@dataclass(frozen=True)
class Trade:
    strategy_name: str
    params: dict[str, int | float | str]
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    fees: float
    funding: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict[str, int | float | str]
    trades: list[Trade]
    equity_curve: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve["equity"].iloc[-1])

    @property
    def total_return_pct(self) -> float:
        starting = float(self.equity_curve["equity"].iloc[0]) if not self.equity_curve.empty else 0.0
        if starting == 0:
            return 0.0
        return ((self.final_equity / starting) - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        equity = self.equity_curve["equity"].astype(float)
        peak = equity.cummax()
        drawdown = (equity / peak - 1.0) * 100.0
        return abs(float(drawdown.min()))

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for trade in self.trades if trade.pnl > 0)
        return winners / len(self.trades) * 100.0

    @property
    def exposure_time_pct(self) -> float:
        if self.equity_curve.empty or "exposed" not in self.equity_curve:
            return 0.0
        exposed_by_timestamp = (
            self.equity_curve.assign(exposed=self.equity_curve["exposed"].astype(bool))
            .groupby("timestamp", sort=False)["exposed"]
            .max()
        )
        return float(exposed_by_timestamp.mean() * 100.0)


def calculate_position_notional(equity: float, risk_fraction: float, stop_distance_pct: float, max_leverage: float) -> float:
    if equity <= 0 or risk_fraction <= 0 or stop_distance_pct <= 0 or max_leverage <= 0:
        return 0.0
    risk_notional = equity * risk_fraction / stop_distance_pct
    leverage_cap = equity * max_leverage
    return max(0.0, min(risk_notional, leverage_cap))


def apply_funding(side: Literal["long", "short"], notional: float, funding_rate: float) -> float:
    payment = notional * funding_rate
    return -payment if side == "long" else payment


def run_backtest(
    candles: pd.DataFrame,
    signals: list[Signal],
    config: BacktestConfig,
    funding_rates: pd.DataFrame | None = None,
) -> BacktestResult:
    candles = candles.sort_values("timestamp").reset_index(drop=True)
    signal_by_time = {pd.Timestamp(signal.timestamp): signal for signal in sorted(signals, key=lambda item: item.timestamp)}
    equity = config.starting_balance
    equity_points = [{"timestamp": candles.loc[0, "timestamp"], "equity": equity, "exposed": False}]
    trades: list[Trade] = []
    pending_signal: Signal | None = None
    open_position: dict[str, object] | None = None

    for idx, candle in candles.iterrows():
        timestamp = pd.Timestamp(candle["timestamp"])
        open_price = float(candle["open"])
        close_price = float(candle["close"])

        if pending_signal is not None:
            if open_position is not None and pending_signal.side == "flat":
                equity, trade = _close_position(open_position, pending_signal, timestamp, open_price, equity, config)
                trades.append(trade)
                open_position = None
            elif pending_signal.side in {"long", "short"}:
                if open_position is not None:
                    equity, trade = _close_position(open_position, pending_signal, timestamp, open_price, equity, config)
                    trades.append(trade)
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
                funding = apply_funding(open_position["side"], float(open_position["notional"]), funding_rate)  # type: ignore[arg-type]
                open_position["funding"] = float(open_position["funding"]) + funding
                equity += funding

        current_signal = signal_by_time.get(timestamp)
        if open_position is None:
            if current_signal is not None and current_signal.side in {"long", "short"}:
                pending_signal = current_signal
        else:
            side = open_position["side"]
            entry_signal = open_position["signal"]
            invalidation = float(entry_signal.invalidation_price)  # type: ignore[union-attr]
            invalidated = (side == "long" and close_price <= invalidation) or (side == "short" and close_price >= invalidation)
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

        equity_points.append(
            {
                "timestamp": timestamp,
                "equity": _marked_equity(equity, open_position, close_price),
                "exposed": open_position is not None,
            }
        )

    if open_position is not None and len(candles) > 0:
        last = candles.iloc[-1]
        exit_signal = Signal(pd.Timestamp(last["timestamp"]), open_position["signal"].strategy_name, open_position["signal"].params, "flat", "end_of_data", float(last["close"]))  # type: ignore[union-attr]
        equity, trade = _close_position(open_position, exit_signal, pd.Timestamp(last["timestamp"]), float(last["close"]), equity, config)
        trades.append(trade)
        equity_points.append(
            {
                "timestamp": pd.Timestamp(last["timestamp"]),
                "equity": equity,
                "exposed": False,
            }
        )

    first_signal = signals[0] if signals else Signal(candles.loc[0, "timestamp"], "unknown", {}, "flat", "none", 0.0)
    return BacktestResult(
        strategy_name=first_signal.strategy_name,
        params=first_signal.params,
        trades=trades,
        equity_curve=pd.DataFrame(equity_points),
    )


def buy_and_hold_result(candles: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    candles = candles.sort_values("timestamp").reset_index(drop=True)
    if candles.empty:
        return BacktestResult(
            strategy_name="buy_and_hold",
            params={},
            trades=[],
            equity_curve=pd.DataFrame(columns=["timestamp", "equity", "exposed"]),
        )

    entry_time = pd.Timestamp(candles.loc[0, "timestamp"])
    exit_time = pd.Timestamp(candles.loc[len(candles) - 1, "timestamp"])
    entry_price = float(candles.loc[0, "open"])
    exit_price = float(candles.loc[len(candles) - 1, "close"])
    notional = config.starting_balance
    units = notional / entry_price
    entry_fee = config.cost_model.one_way_cost(notional)
    exit_fee = config.cost_model.one_way_cost(notional)
    fees = entry_fee + exit_fee
    gross = (exit_price - entry_price) * units
    pnl = gross - fees

    marked_curve = pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "equity": (
                config.starting_balance
                - entry_fee
                + (candles["close"].astype(float) - entry_price) * units
            ),
            "exposed": True,
        }
    )
    marked_curve.loc[marked_curve.index[-1], "equity"] -= exit_fee
    equity_curve = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": [entry_time],
                    "equity": [config.starting_balance],
                    "exposed": [False],
                }
            ),
            marked_curve,
        ],
        ignore_index=True,
    )
    trade = Trade(
        strategy_name="buy_and_hold",
        params={},
        side="long",
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        notional=notional,
        pnl=pnl,
        fees=fees,
        funding=0.0,
        return_pct=(pnl / notional) * 100.0,
        exit_reason="end_of_data",
    )
    return BacktestResult(
        strategy_name="buy_and_hold",
        params={},
        trades=[trade],
        equity_curve=equity_curve,
    )


def score_result(result: BacktestResult, minimum_trades: int = 10) -> float:
    if result.trade_count < minimum_trades:
        return float("-inf")
    if result.max_drawdown_pct == 0:
        return result.total_return_pct if result.total_return_pct > 0 else 0.0
    return result.total_return_pct / result.max_drawdown_pct


def _notional_for_signal(
    equity: float,
    invalidation_price: float,
    entry_price: float,
    config: BacktestConfig,
    exposure_multiplier: float = 1.0,
) -> float:
    if (
        entry_price <= 0
        or isinstance(exposure_multiplier, bool)
        or not isinstance(exposure_multiplier, Real)
        or not math.isfinite(exposure_multiplier)
        or not 0 < exposure_multiplier <= 1
    ):
        return 0.0
    stop_distance_pct = abs(entry_price - invalidation_price) / entry_price
    base_notional = calculate_position_notional(equity, config.risk_fraction, stop_distance_pct, config.max_leverage)
    return base_notional * exposure_multiplier


def _close_position(
    position: dict[str, object],
    exit_signal: Signal,
    exit_time: pd.Timestamp,
    exit_price: float,
    equity: float,
    config: BacktestConfig,
) -> tuple[float, Trade]:
    side = position["side"]
    entry_price = float(position["entry_price"])
    notional = float(position["notional"])
    units = notional / entry_price
    gross = (exit_price - entry_price) * units if side == "long" else (entry_price - exit_price) * units
    exit_fee = config.cost_model.one_way_cost(notional)
    fees = float(position["entry_fee"]) + exit_fee
    funding = float(position["funding"])
    closing_cashflow = gross - exit_fee
    equity += closing_cashflow
    net_pnl = gross - fees + funding
    trade = Trade(
        strategy_name=position["signal"].strategy_name,  # type: ignore[union-attr]
        params=position["signal"].params,  # type: ignore[union-attr]
        side=side,  # type: ignore[arg-type]
        entry_time=position["entry_time"],  # type: ignore[arg-type]
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        notional=notional,
        pnl=net_pnl,
        fees=fees,
        funding=funding,
        return_pct=(net_pnl / notional) * 100.0 if notional else 0.0,
        exit_reason=exit_signal.entry_reason,
    )
    return equity, trade


def _funding_rate_at(funding_rates: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if funding_rates.empty:
        return None
    matches = funding_rates[funding_rates["timestamp"] == timestamp]
    if matches.empty:
        return None
    return float(matches.iloc[0]["funding_rate"])


def _marked_equity(equity: float, position: dict[str, object] | None, close_price: float) -> float:
    if position is None:
        return equity
    entry_price = float(position["entry_price"])
    notional = float(position["notional"])
    units = notional / entry_price
    unrealized = (close_price - entry_price) * units
    if position["side"] == "short":
        unrealized = -unrealized
    return equity + unrealized
