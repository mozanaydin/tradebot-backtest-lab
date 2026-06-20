from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_FAPI_URL = "https://fapi.binance.com"
REQUIRED_CANDLE_FIELDS = {"t", "o", "h", "l", "c", "v"}
SUPPORTED_EXCHANGES = {"hyperliquid", "binance"}
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class FundingUnavailable(RuntimeError):
    """Raised when optional funding data cannot be fetched."""


def normalize_candles(raw: list[dict[str, object]]) -> pd.DataFrame:
    if not raw:
        raise ValueError("no candle rows returned")
    missing = REQUIRED_CANDLE_FIELDS - set(raw[0])
    if missing:
        raise ValueError(f"candle rows missing required fields: {sorted(missing)}")
    frame = pd.DataFrame(raw)
    for field in REQUIRED_CANDLE_FIELDS:
        if field not in frame.columns:
            raise ValueError(f"candle rows missing required fields: {field}")
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["t"], unit="ms", utc=True),
            "open": pd.to_numeric(frame["o"], errors="raise").astype(float),
            "high": pd.to_numeric(frame["h"], errors="raise").astype(float),
            "low": pd.to_numeric(frame["l"], errors="raise").astype(float),
            "close": pd.to_numeric(frame["c"], errors="raise").astype(float),
            "volume": pd.to_numeric(frame["v"], errors="raise").astype(float),
        }
    )
    return normalized.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def normalize_binance_klines(raw: list[list[object]]) -> pd.DataFrame:
    if not raw:
        raise ValueError("no candle rows returned")
    frame = pd.DataFrame(raw)
    if frame.shape[1] < 6:
        raise ValueError("binance klines missing required columns")
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame.iloc[:, 0], unit="ms", utc=True),
            "open": pd.to_numeric(frame.iloc[:, 1], errors="raise").astype(float),
            "high": pd.to_numeric(frame.iloc[:, 2], errors="raise").astype(float),
            "low": pd.to_numeric(frame.iloc[:, 3], errors="raise").astype(float),
            "close": pd.to_numeric(frame.iloc[:, 4], errors="raise").astype(float),
            "volume": pd.to_numeric(frame.iloc[:, 5], errors="raise").astype(float),
        }
    )
    return normalized.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def normalize_binance_funding_history(raw: list[dict[str, object]]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
    frame = pd.DataFrame(raw)
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["fundingTime"], unit="ms", utc=True).dt.floor("h"),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="raise"),
            "mark_price": pd.to_numeric(frame["markPrice"], errors="raise"),
        }
    )
    return normalized.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_candles(
    exchange: str,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    exchange = exchange.lower()
    if exchange not in SUPPORTED_EXCHANGES:
        raise ValueError(f"unsupported exchange: {exchange}")
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        if exchange == "hyperliquid":
            response = client.post(
                HYPERLIQUID_INFO_URL,
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": interval,
                        "startTime": int(start.timestamp() * 1000),
                        "endTime": int(end.timestamp() * 1000),
                    },
                },
            )
            response.raise_for_status()
            return normalize_candles(response.json())
        return _fetch_binance_klines(symbol, interval, start, end, client)
    finally:
        if owns_client:
            client.close()


def load_or_fetch_candles(
    exchange: str,
    symbol: str,
    interval: str,
    days: int,
    data_dir: Path,
    data_file: Path | None = None,
) -> pd.DataFrame:
    if data_file is not None:
        return read_candles_csv(data_file)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{exchange.lower()}_{symbol}_{interval}.csv"
    if path.exists():
        return read_candles_csv(path)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=days)
    candles = fetch_candles(exchange, symbol, interval, start, end)
    candles.to_csv(path, index=False)
    return candles


def refresh_candles_cache(
    exchange: str,
    symbol: str,
    interval: str,
    days: int,
    data_dir: Path,
) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=days)
    candles = fetch_candles(exchange, symbol, interval, start, end)
    path = data_dir / f"{exchange.lower()}_{symbol}_{interval}.csv"
    candles.to_csv(path, index=False)
    return candles


def read_candles_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def read_funding_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("h")
    numeric_columns = {"funding_rate", "premium", "mark_price"} & set(frame.columns)
    for column in sorted(numeric_columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding_history(
    exchange: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    exchange = exchange.lower()
    if exchange not in SUPPORTED_EXCHANGES:
        raise ValueError(f"unsupported exchange: {exchange}")
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        if exchange == "hyperliquid":
            return _fetch_hyperliquid_funding(symbol, start, end, client)
        return _fetch_binance_funding(symbol, start, end, client)
    except Exception as exc:  # noqa: BLE001
        raise FundingUnavailable(str(exc)) from exc
    finally:
        if owns_client:
            client.close()


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
) -> pd.DataFrame:
    interval_ms = INTERVAL_TO_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"unsupported binance interval: {interval}")
    rows: list[list[object]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms <= end_ms:
        response = client.get(
            f"{BINANCE_FAPI_URL}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        last_open_time = int(page[-1][0])
        if len(page) < 1500 or last_open_time >= end_ms:
            break
        start_ms = last_open_time + interval_ms
    return normalize_binance_klines(rows)


def _fetch_hyperliquid_funding(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms <= end_ms:
        response = client.post(
            HYPERLIQUID_INFO_URL,
            json={
                "type": "fundingHistory",
                "coin": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        last_time = int(page[-1]["time"])
        if len(page) < 500 or last_time >= end_ms:
            break
        start_ms = last_time + 1
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "premium"])
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["time"], unit="ms", utc=True).dt.floor("h"),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="raise"),
            "premium": pd.to_numeric(frame["premium"], errors="raise"),
        }
    ).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _fetch_binance_funding(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms <= end_ms:
        response = client.get(
            f"{BINANCE_FAPI_URL}/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        last_time = int(page[-1]["fundingTime"])
        if len(page) < 1000 or last_time >= end_ms:
            break
        start_ms = last_time + 1
    return normalize_binance_funding_history(rows)
