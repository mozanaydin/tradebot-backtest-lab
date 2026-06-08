from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

INFO_URL = "https://api.hyperliquid.xyz/info"
REQUIRED_CANDLE_FIELDS = {"t", "o", "h", "l", "c", "v"}


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


def fetch_candles(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.post(
            INFO_URL,
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
    finally:
        if owns_client:
            client.close()


def load_or_fetch_candles(
    symbol: str,
    interval: str,
    days: int,
    data_dir: Path,
    data_file: Path | None = None,
) -> pd.DataFrame:
    if data_file is not None:
        return read_candles_csv(data_file)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"hyperliquid_{symbol}_{interval}.csv"
    if path.exists():
        return read_candles_csv(path)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    candles = fetch_candles(symbol, interval, start, end)
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
    for column in ["funding_rate", "premium"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding_history(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    rows: list[dict[str, object]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    try:
        while start_ms <= end_ms:
            response = client.post(
                INFO_URL,
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
    except Exception as exc:  # noqa: BLE001 - preserve original provider message in a typed optional-data error.
        raise FundingUnavailable(str(exc)) from exc
    finally:
        if owns_client:
            client.close()
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
