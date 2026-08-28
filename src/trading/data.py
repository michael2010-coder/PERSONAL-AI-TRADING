"""Candle sources: a live exchange via ccxt, a CSV cache, or synthetic data.

fetch_ohlcv paginates, because exchanges cap a single call at ~500-1000 bars
and a 90-day 1h request needs about 2160.
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from typing import List, Optional, Sequence

from .strategy import Candle

TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "12h": 43_200_000, "1d": 86_400_000,
}


def timeframe_ms(timeframe: str) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise ValueError("unsupported timeframe {!r}; use one of {}".format(
            timeframe, ", ".join(sorted(TIMEFRAME_MS))
        ))


def bars_per_year(timeframe: str) -> float:
    return 365.0 * 24 * 60 * 60 * 1000 / timeframe_ms(timeframe)


def make_exchange(exchange_id: str, testnet: bool = True, credentials: Optional[dict] = None):
    """Build a ccxt exchange. Imported lazily so the rest of the app runs without it."""
    import ccxt  # noqa: WPS433 (deliberate lazy import)

    if not hasattr(ccxt, exchange_id):
        raise ValueError("ccxt has no exchange {!r}".format(exchange_id))
    # This app is spot-only and long-only: it has no concept of leverage,
    # margin or liquidation. Several exchanges (Bybit among them) default to
    # perpetual swaps in ccxt, where the same symbol means a leveraged
    # contract that can lose far more than the configured stop. Pin it.
    opts = {
        "enableRateLimit": True,
        "timeout": 30_000,
        "options": {"defaultType": "spot"},
    }
    if credentials:
        opts.update({k: v for k, v in credentials.items() if v})
    exchange = getattr(ccxt, exchange_id)(opts)
    if testnet:
        if not exchange.has.get("sandbox", True):
            raise RuntimeError("{} has no sandbox/testnet in ccxt".format(exchange_id))
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    days: float = 90.0,
    limit_per_call: int = 1000,
    now_ms: Optional[int] = None,
) -> List[Candle]:
    step = timeframe_ms(timeframe)
    now = int(time.time() * 1000) if now_ms is None else now_ms
    since = now - int(days * 86_400_000)

    rows: List[list] = []
    cursor = since
    while cursor < now:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + step
        if next_cursor <= cursor:  # exchange refused to advance
            break
        cursor = next_cursor
        if len(batch) < limit_per_call:
            break

    seen = set()
    candles: List[Candle] = []
    for ts, o, h, l, c, v in rows:
        if ts in seen or ts < since:
            continue
        seen.add(ts)
        candles.append(Candle(int(ts), float(o), float(h), float(l), float(c), float(v or 0.0)))
    candles.sort(key=lambda c: c.timestamp)

    # The final bar is still forming; a strategy must not act on a partial bar.
    if candles and candles[-1].timestamp + step > now:
        candles.pop()
    return candles


def save_csv(candles: Sequence[Candle], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])
    return path


def load_csv(path: str) -> List[Candle]:
    out: List[Candle] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Candle(
                int(float(row["timestamp"])), float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]), float(row.get("volume") or 0.0),
            ))
    out.sort(key=lambda c: c.timestamp)
    return out


def synthetic(
    bars: int = 1500,
    start_price: float = 30_000.0,
    timeframe: str = "1h",
    seed: int = 7,
    drift: float = 0.0002,
    volatility: float = 0.01,
    start_ms: int = 1_700_000_000_000,
) -> List[Candle]:
    """Deterministic random-walk candles, for tests and offline demos."""
    rng = random.Random(seed)
    step = timeframe_ms(timeframe)
    price = start_price
    candles: List[Candle] = []
    for i in range(bars):
        ret = drift + rng.gauss(0.0, volatility)
        close = max(price * math.exp(ret), 0.01)
        high = max(price, close) * (1.0 + abs(rng.gauss(0.0, volatility / 3)))
        low = min(price, close) * (1.0 - abs(rng.gauss(0.0, volatility / 3)))
        volume = abs(rng.gauss(1000.0, 300.0)) + 50.0
        candles.append(Candle(start_ms + i * step, price, high, low, close, volume))
        price = close
    return candles
