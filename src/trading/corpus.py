"""Historical corpus: the raw material the analogue engine reasons over.

To find 100,000 situations similar to the one in front of you, you need a
library of millions. This module downloads years of candles across many
symbols and timeframes into data/corpus/, resuming where it left off, so the
download can be interrupted and restarted.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .data import timeframe_ms
from .strategy import Candle

log = logging.getLogger("trading.corpus")


def corpus_dir(root: str) -> str:
    return os.path.join(root, "data", "corpus")


def series_path(directory: str, symbol: str, timeframe: str) -> str:
    return os.path.join(directory, "{}_{}.csv".format(symbol.replace("/", ""), timeframe))


def top_symbols(exchange, quote: str = "USDT", limit: int = 40,
                exclude_stable: bool = True) -> List[str]:
    """The `limit` most-traded spot pairs against `quote`, by 24h volume."""
    tickers = exchange.fetch_tickers()
    markets = exchange.load_markets()
    # Stablecoins and pegged/tokenised assets: their price action is not the
    # thing we are trying to learn from.
    stable = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USD1", "RLUSD",
              "EUR", "GBP", "TRY", "BRL", "ARS", "XAUT", "PAXG"}

    ranked: List[Tuple[float, str]] = []
    for symbol, ticker in tickers.items():
        market = markets.get(symbol) or {}
        if not market.get("spot") or market.get("active") is False:
            continue
        if market.get("quote") != quote:
            continue
        base = market.get("base") or ""
        if exclude_stable and (base in stable or base.endswith("B") and len(base) > 3):
            continue   # the *B suffix is Binance's tokenised-equity naming
        volume = ticker.get("quoteVolume") or 0.0
        ranked.append((float(volume), symbol))
    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked[:limit]]


def fetch_range(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    limit_per_call: int = 1000,
    max_calls: int = 100_000,
) -> Iterable[Candle]:
    """Page forward from `since_ms`, yielding candles as they arrive."""
    step = timeframe_ms(timeframe)
    until = int(time.time() * 1000) if until_ms is None else until_ms
    cursor = since_ms
    calls = 0
    last_ts = None

    while cursor < until and calls < max_calls:
        calls += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit_per_call)
        except Exception as exc:
            log.warning("%s %s: fetch failed at %s (%s); backing off", symbol, timeframe, cursor, exc)
            time.sleep(2.0)
            continue
        if not batch:
            break
        for ts, o, h, l, c, v in batch:
            ts = int(ts)
            if last_ts is not None and ts <= last_ts:
                continue
            if ts >= until:
                continue
            last_ts = ts
            yield Candle(ts, float(o), float(h), float(l), float(c), float(v or 0.0))
        advanced = int(batch[-1][0]) + step
        if advanced <= cursor:
            break
        cursor = advanced
        if len(batch) < limit_per_call:
            break


def last_timestamp(path: str) -> Optional[int]:
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        block = min(4096, end)
        fh.seek(end - block)
        tail = fh.read().decode("utf-8", "ignore").strip().splitlines()
    for line in reversed(tail):
        first = line.split(",")[0]
        try:
            return int(float(first))
        except ValueError:
            continue
    return None


def download_series(
    exchange,
    symbol: str,
    timeframe: str,
    years: float,
    directory: str,
    resume: bool = True,
) -> Dict:
    """Download one symbol/timeframe to CSV. Returns a small stats dict."""
    os.makedirs(directory, exist_ok=True)
    path = series_path(directory, symbol, timeframe)
    step = timeframe_ms(timeframe)
    now = int(time.time() * 1000)
    wanted_since = now - int(years * 365.25 * 86_400_000)

    existing_end = last_timestamp(path) if resume else None
    since = wanted_since if existing_end is None else existing_end + step
    new_rows = 0

    if since >= now - step:
        return {"symbol": symbol, "timeframe": timeframe, "path": path,
                "added": 0, "status": "up-to-date"}

    mode = "a" if (existing_end is not None and os.path.exists(path)) else "w"
    with open(path, mode, newline="") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for candle in fetch_range(exchange, symbol, timeframe, since, now):
            writer.writerow([candle.timestamp, candle.open, candle.high,
                             candle.low, candle.close, candle.volume])
            new_rows += 1
            if new_rows % 20_000 == 0:
                fh.flush()
                log.info("  %s %s: %d rows so far", symbol, timeframe, new_rows)

    return {"symbol": symbol, "timeframe": timeframe, "path": path,
            "added": new_rows, "status": "extended" if mode == "a" else "created"}


def build(
    exchange,
    symbols: List[str],
    timeframes: List[str],
    years: float,
    directory: str,
    resume: bool = True,
) -> Dict:
    total = 0
    series = 0
    for timeframe in timeframes:
        for symbol in symbols:
            stats = download_series(exchange, symbol, timeframe, years, directory, resume)
            total += stats["added"]
            series += 1
            log.info("%-12s %-4s %+8d rows (%s)", symbol, timeframe,
                     stats["added"], stats["status"])
    return {"series": series, "rows_added": total, "directory": directory}


def inventory(directory: str) -> List[Dict]:
    """What is on disk: one row per series, with bar counts."""
    out: List[Dict] = []
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(directory, name)
        with open(path, "rb") as fh:
            bars = max(0, sum(1 for _ in fh) - 1)
        stem = name[:-4]
        symbol, _, timeframe = stem.rpartition("_")
        out.append({"file": name, "symbol": symbol, "timeframe": timeframe,
                    "bars": bars, "path": path,
                    "mb": round(os.path.getsize(path) / 1e6, 1)})
    return out


def total_bars(directory: str) -> int:
    return sum(item["bars"] for item in inventory(directory))
