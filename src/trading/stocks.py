"""US-stock signals -- informational only.

Bamboo and Trove do not offer retail API access, so there is no automated
execution path for stocks here. This module fetches daily candles, runs the
same strategy the crypto side runs, and prints/logs the call. Acting on it is
a manual decision.

Daily candles are tried in this order:

  1. data/stocks/<SYMBOL>.csv, if you dropped a file there (any exported
     daily OHLCV works -- broker, TradingView, Nasdaq).
  2. Alpha Vantage, if ALPHAVANTAGE_API_KEY is set. A free key covers this.
  3. Yahoo's public chart endpoint, cookie + crumb. It is unauthenticated and
     rate-limits hard; when it does, you get a clear message rather than a
     silent HOLD.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .strategy import Candle, Strategy

log = logging.getLogger("trading.stocks")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_daily(
    symbol: str,
    lookback_days: int = 365,
    csv_dir: Optional[str] = None,
    timeout: int = 20,
) -> List[Candle]:
    for name, source in (
        ("local csv", lambda: _from_csv(symbol, csv_dir)),
        ("alpha vantage", lambda: _from_alpha_vantage(symbol, timeout)),
        ("yahoo", lambda: _from_yahoo(symbol, lookback_days, timeout)),
    ):
        candles = source()
        if candles:
            log.info("%s: %d daily candles from %s", symbol, len(candles), name)
            return candles[-max(lookback_days, 60):]
    return []


# -- sources ----------------------------------------------------------------
def _from_csv(symbol: str, csv_dir: Optional[str]) -> List[Candle]:
    if not csv_dir:
        return []
    path = os.path.join(csv_dir, "{}.csv".format(symbol.upper()))
    if not os.path.exists(path):
        return []
    out: List[Candle] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            lower = {k.strip().lower(): v for k, v in row.items() if k}
            date = lower.get("date") or lower.get("timestamp") or lower.get("time")
            try:
                out.append(Candle(
                    _to_ms(date),
                    float(lower["open"]), float(lower["high"]),
                    float(lower["low"]), float(lower.get("close") or lower["adj close"]),
                    float(lower.get("volume") or 0.0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    out.sort(key=lambda c: c.timestamp)
    return out


def _from_alpha_vantage(symbol: str, timeout: int) -> List[Candle]:
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        return []
    import requests

    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_DAILY", "symbol": symbol,
                    "outputsize": "full", "apikey": key},
            timeout=timeout,
        )
        payload = resp.json()
    except Exception as exc:
        log.warning("alpha vantage failed for %s: %s", symbol, exc)
        return []

    series = payload.get("Time Series (Daily)")
    if not series:
        log.warning("alpha vantage returned no series for %s: %s", symbol,
                    payload.get("Note") or payload.get("Information") or payload.get("Error Message"))
        return []
    out: List[Candle] = []
    for date, row in series.items():
        try:
            out.append(Candle(_to_ms(date), float(row["1. open"]), float(row["2. high"]),
                              float(row["3. low"]), float(row["4. close"]),
                              float(row.get("5. volume") or 0.0)))
        except (KeyError, ValueError):
            continue
    out.sort(key=lambda c: c.timestamp)
    return out


def _from_yahoo(symbol: str, lookback_days: int, timeout: int) -> List[Candle]:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": _UA})
    crumb = None
    try:
        session.get("https://finance.yahoo.com/quote/{}/".format(symbol), timeout=timeout)
        got = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=timeout)
        if got.status_code == 200 and got.text and " " not in got.text:
            crumb = got.text.strip()
    except Exception as exc:
        log.debug("yahoo cookie/crumb step failed: %s", exc)

    params = {"range": "{}d".format(max(lookback_days, 40)), "interval": "1d"}
    if crumb:
        params["crumb"] = crumb
    try:
        resp = session.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(symbol),
            params=params, timeout=timeout,
        )
        if resp.status_code == 429:
            log.warning("yahoo is rate-limiting this network (HTTP 429) for %s -- "
                        "set ALPHAVANTAGE_API_KEY or drop a CSV in data/stocks/", symbol)
            return []
        if resp.status_code != 200:
            log.warning("yahoo HTTP %s for %s", resp.status_code, symbol)
            return []
        block = ((resp.json().get("chart") or {}).get("result") or [{}])[0]
    except Exception as exc:
        log.warning("yahoo fetch failed for %s: %s", symbol, exc)
        return []

    stamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    out: List[Candle] = []
    for i, ts in enumerate(stamps):
        try:
            o, h, l, c = (quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i])
        except (KeyError, IndexError):
            continue
        if None in (o, h, l, c):
            continue
        volume = (quote.get("volume") or [None] * len(stamps))[i]
        out.append(Candle(int(ts) * 1000, float(o), float(h), float(l), float(c), float(volume or 0.0)))
    return out


# -- signal -----------------------------------------------------------------
def signal_for(
    symbol: str,
    strategy: Strategy,
    lookback_days: int = 365,
    csv_dir: Optional[str] = None,
) -> Optional[Dict]:
    candles = fetch_daily(symbol, lookback_days, csv_dir)
    if len(candles) < strategy.warmup_bars + 1:
        log.warning("skipping %s: %d daily candles, need %d",
                    symbol, len(candles), strategy.warmup_bars + 1)
        return None
    signal = strategy.evaluate(candles)
    return {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol,
        "action": signal.action,
        "score": round(signal.score, 3),
        "price": round(signal.price, 4),
        "bar_time": datetime.fromtimestamp(signal.timestamp / 1000, timezone.utc).strftime("%Y-%m-%d"),
        "reasons": signal.reasons,
        "note": "informational only -- no broker API, execute manually if you agree",
    }


def append_journal(record: Dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _to_ms(date_str: str) -> int:
    text = (date_str or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return int(float(text))   # already epoch
