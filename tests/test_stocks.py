"""The stock side is informational, but its data path still has to be right."""
import csv
import os

from trading.data import synthetic
from trading import stocks
from trading.stocks import fetch_daily, signal_for
from trading.strategy import Strategy


def write_csv(directory, symbol, candles, header=("Date", "Open", "High", "Low", "Close", "Volume")):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "{}.csv".format(symbol))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for c in candles:
            from datetime import datetime, timezone
            date = datetime.fromtimestamp(c.timestamp / 1000, timezone.utc).strftime("%Y-%m-%d")
            w.writerow([date, c.open, c.high, c.low, c.close, c.volume])
    return path


def test_a_local_csv_is_read_in_preference_to_the_network(tmp_path):
    candles = synthetic(bars=300, timeframe="1d", seed=21)
    write_csv(str(tmp_path), "TEST", candles)
    loaded = fetch_daily("TEST", lookback_days=365, csv_dir=str(tmp_path))
    assert len(loaded) == 300
    assert loaded[0].timestamp < loaded[-1].timestamp
    assert loaded[-1].close == candles[-1].close


def test_a_missing_csv_falls_through_to_the_next_source(tmp_path, monkeypatch):
    """No file, no API key, no network: an empty list, not an exception."""
    monkeypatch.setattr(stocks, "_from_alpha_vantage", lambda *a, **k: [])
    monkeypatch.setattr(stocks, "_from_yahoo", lambda *a, **k: [])
    assert fetch_daily("NOSUCHTICKER", 365, csv_dir=str(tmp_path)) == []


def test_sources_are_tried_in_order_and_stop_at_the_first_hit(tmp_path, monkeypatch):
    candles = synthetic(bars=120, timeframe="1d", seed=5)
    write_csv(str(tmp_path), "TEST", candles)
    called = []
    monkeypatch.setattr(stocks, "_from_alpha_vantage",
                        lambda *a, **k: called.append("av") or [])
    monkeypatch.setattr(stocks, "_from_yahoo", lambda *a, **k: called.append("yahoo") or [])
    assert fetch_daily("TEST", 365, csv_dir=str(tmp_path))
    assert called == [], "a local CSV must short-circuit the network sources"


def test_a_signal_from_a_csv_is_explained_and_marked_informational(tmp_path):
    write_csv(str(tmp_path), "TEST", synthetic(bars=300, timeframe="1d", seed=21))
    record = signal_for("TEST", Strategy(), 365, csv_dir=str(tmp_path))
    assert record["symbol"] == "TEST"
    assert record["action"] in ("BUY", "SELL", "HOLD")
    assert record["reasons"]
    assert "informational only" in record["note"]


def test_too_little_history_returns_nothing_rather_than_a_guess(tmp_path):
    write_csv(str(tmp_path), "SHORT", synthetic(bars=10, timeframe="1d"))
    assert signal_for("SHORT", Strategy(), 365, csv_dir=str(tmp_path)) is None
