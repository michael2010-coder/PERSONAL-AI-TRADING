"""Indicator correctness, including the property the backtester depends on."""
import math

import pytest

from trading import indicators as ind


def test_sma_matches_hand_calculation():
    values = [1, 2, 3, 4, 5, 6]
    out = ind.sma(values, 3)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[5] == pytest.approx(5.0)


def test_ema_seeds_with_sma_then_smooths():
    values = [10.0] * 5 + [20.0] * 5
    out = ind.ema(values, 5)
    assert out[3] is None
    assert out[4] == pytest.approx(10.0)          # seed = SMA of first 5
    k = 2 / 6
    expected = (20.0 - 10.0) * k + 10.0
    assert out[5] == pytest.approx(expected)
    assert 10.0 < out[9] < 20.0                    # converging, never overshooting


def test_ema_of_a_constant_series_is_that_constant():
    out = ind.ema([42.0] * 30, 12)
    assert out[-1] == pytest.approx(42.0)


def test_rsi_bounds_and_extremes():
    rising = [float(i) for i in range(1, 40)]
    assert ind.rsi(rising, 14)[-1] == pytest.approx(100.0)
    falling = [float(i) for i in range(40, 1, -1)]
    assert ind.rsi(falling, 14)[-1] == pytest.approx(0.0)
    flat = ind.rsi([5.0] * 40, 14)
    assert flat[-1] == pytest.approx(50.0)


def test_rsi_known_wilder_value():
    # Wilder's own worked example (New Concepts in Technical Trading Systems).
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert ind.rsi(closes, 14)[14] == pytest.approx(70.46, abs=0.05)


def test_macd_histogram_is_line_minus_signal():
    values = [100 + 10 * math.sin(i / 5.0) for i in range(120)]
    line, sig, hist = ind.macd(values)
    for i in range(len(values)):
        if hist[i] is not None:
            assert hist[i] == pytest.approx(line[i] - sig[i])


def test_macd_rejects_fast_slower_than_slow():
    with pytest.raises(ValueError):
        ind.macd([1.0] * 50, fast=30, slow=10)


@pytest.mark.parametrize("period", [5, 14, 26])
def test_indicators_are_causal(period):
    """The value at i must not change when future bars are removed.

    This is what makes the backtester lookahead-free: it computes indicators
    over the full series once, and this proves that is the same as computing
    them fresh at each bar with only the past available.
    """
    values = [100 + 10 * math.sin(i / 7.0) + (i % 5) for i in range(200)]
    full_ema = ind.ema(values, period)
    full_rsi = ind.rsi(values, period)
    _, _, full_hist = ind.macd(values)

    for cut in (60, 100, 150, 199):
        assert ind.ema(values[: cut + 1], period)[cut] == pytest.approx(full_ema[cut])
        assert ind.rsi(values[: cut + 1], period)[cut] == pytest.approx(full_rsi[cut])
        assert ind.macd(values[: cut + 1])[2][cut] == pytest.approx(full_hist[cut])


def test_atr_is_positive_and_tracks_range():
    highs = [10 + i * 0.1 for i in range(60)]
    lows = [9 + i * 0.1 for i in range(60)]
    closes = [9.5 + i * 0.1 for i in range(60)]
    out = ind.atr(highs, lows, closes, 14)
    assert out[13] is None
    assert out[-1] == pytest.approx(1.0, abs=0.2)
