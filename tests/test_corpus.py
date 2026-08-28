"""Corpus bookkeeping: resume must not duplicate or skip bars."""
import csv
import os

from trading import corpus
from trading.data import load_csv, save_csv, synthetic


class FakeExchange:
    """Serves candles 1000 at a time, like a real exchange does."""

    def __init__(self, candles, page=1000):
        self.candles = candles
        self.page = page
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        limit = limit or self.page
        rows = [[c.timestamp, c.open, c.high, c.low, c.close, c.volume]
                for c in self.candles if since is None or c.timestamp >= since]
        return rows[:limit]


def test_fetch_range_pages_through_everything_without_duplicates():
    candles = synthetic(bars=2500, timeframe="1h")
    exchange = FakeExchange(candles, page=1000)
    got = list(corpus.fetch_range(exchange, "X/USDT", "1h",
                                  candles[0].timestamp, candles[-1].timestamp + 1))
    stamps = [c.timestamp for c in got]
    assert len(stamps) == len(set(stamps)), "no bar may appear twice"
    assert stamps == sorted(stamps)
    assert len(got) == 2500
    assert exchange.calls >= 3


def test_fetch_range_stops_at_the_until_boundary():
    candles = synthetic(bars=500, timeframe="1h")
    cutoff = candles[200].timestamp
    got = list(corpus.fetch_range(FakeExchange(candles), "X/USDT", "1h",
                                  candles[0].timestamp, cutoff))
    assert got and max(c.timestamp for c in got) < cutoff


def test_last_timestamp_reads_the_final_row(tmp_path):
    candles = synthetic(bars=300, timeframe="1h")
    path = str(tmp_path / "X.csv")
    save_csv(candles, path)
    assert corpus.last_timestamp(path) == candles[-1].timestamp


def test_last_timestamp_of_a_missing_or_empty_file_is_none(tmp_path):
    assert corpus.last_timestamp(str(tmp_path / "nope.csv")) is None
    empty = str(tmp_path / "empty.csv")
    open(empty, "w").close()
    assert corpus.last_timestamp(empty) is None


def test_download_resumes_where_it_stopped_instead_of_starting_over(tmp_path):
    candles = synthetic(bars=1200, timeframe="1h")
    directory = str(tmp_path)
    path = corpus.series_path(directory, "X/USDT", "1h")
    save_csv(candles[:800], path)          # a previous, interrupted run

    exchange = FakeExchange(candles)
    stats = corpus.download_series(exchange, "X/USDT", "1h", years=10,
                                   directory=directory, resume=True)
    assert stats["status"] == "extended"

    rows = load_csv(path)
    stamps = [c.timestamp for c in rows]
    assert len(stamps) == len(set(stamps)), "resume must not duplicate the overlap"
    assert stamps == sorted(stamps)
    assert len(rows) >= 800


def test_inventory_counts_bars_per_series(tmp_path):
    save_csv(synthetic(bars=120, timeframe="1h"),
             corpus.series_path(str(tmp_path), "AAA/USDT", "1h"))
    save_csv(synthetic(bars=60, timeframe="4h"),
             corpus.series_path(str(tmp_path), "BBB/USDT", "4h"))
    items = {item["file"]: item for item in corpus.inventory(str(tmp_path))}
    assert items["AAAUSDT_1h.csv"]["bars"] == 120
    assert items["AAAUSDT_1h.csv"]["timeframe"] == "1h"
    assert items["BBBUSDT_4h.csv"]["bars"] == 60
    assert corpus.total_bars(str(tmp_path)) == 180


def test_an_empty_corpus_directory_reports_nothing_rather_than_failing(tmp_path):
    assert corpus.inventory(str(tmp_path / "missing")) == []
    assert corpus.total_bars(str(tmp_path / "missing")) == 0


def test_exchanges_are_pinned_to_spot_markets():
    """This app has no concept of leverage or liquidation.

    ccxt defaults Bybit (and others) to perpetual swaps, where the same symbol
    is a leveraged contract that can lose far more than the configured stop.
    """
    from trading.data import make_exchange

    for exchange_id in ("binance", "bybit"):
        exchange = make_exchange(exchange_id, testnet=False)
        assert exchange.options.get("defaultType") == "spot", exchange_id
