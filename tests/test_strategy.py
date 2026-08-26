import pytest

from trading.data import synthetic
from trading.strategy import BUY, HOLD, SELL, Candle, Strategy, StrategyParams


def flat_then(prices, volume=1000.0, start=1_700_000_000_000, step=3_600_000):
    return [Candle(start + i * step, p, p * 1.001, p * 0.999, p, volume)
            for i, p in enumerate(prices)]


def first_bar_with(candles, action, strategy=None):
    """Index of the first bar the strategy calls `action` on, or None."""
    s = strategy or Strategy()
    for i in range(s.warmup_bars, len(candles)):
        if s.evaluate(candles, i).action == action:
            return i
    return None


def test_holds_until_it_has_enough_history():
    assert Strategy().evaluate(flat_then([100.0] * 10)).action == HOLD


def test_warmup_covers_the_slowest_indicator():
    s = Strategy(StrategyParams(ema_slow=50, macd_slow=26, macd_signal=9, volume_period=20))
    assert s.warmup_bars >= 51


def test_trend_vote_follows_the_ema_pair():
    up = Strategy().evaluate(flat_then([100.0 * (1.005 ** i) for i in range(80)]))
    down = Strategy().evaluate(flat_then([100.0 * (0.995 ** i) for i in range(80)]))
    assert up.votes["ema"] == 1.0
    assert down.votes["ema"] == -1.0


def test_it_refuses_to_chase_a_vertical_move():
    """A straight-line ramp pins RSI at 100, and overbought votes against buying.

    This is the strategy's design, not an accident: trend and momentum both
    say yes, RSI says the move is stretched, and the sum lands short of the
    1.5 threshold. It buys pullbacks in a trend, not blow-off tops.
    """
    signal = Strategy().evaluate(flat_then([100.0 * (1.01 ** i) for i in range(80)]))
    assert signal.votes["ema"] == 1.0
    assert signal.votes["macd"] == 1.0
    assert signal.votes["rsi"] == -1.0
    assert signal.score == pytest.approx(1.0)
    assert signal.action == HOLD


def test_it_finds_both_buys_and_sells_in_ordinary_noisy_data():
    candles = synthetic(bars=600, seed=11, drift=0.0004, volatility=0.012)
    assert first_bar_with(candles, BUY) is not None
    assert first_bar_with(candles, SELL) is not None


def test_thin_volume_suppresses_a_buy():
    candles = synthetic(bars=600, seed=11, drift=0.0004, volatility=0.012)
    i = first_bar_with(candles, BUY)
    assert i is not None
    bar = candles[i]
    candles[i] = Candle(bar.timestamp, bar.open, bar.high, bar.low, bar.close, 1.0)

    blocked = Strategy().evaluate(candles, i)
    assert blocked.action == HOLD
    assert any("suppressed" in r for r in blocked.reasons)

    allowed = Strategy(StrategyParams(require_volume_confirmation=False)).evaluate(candles, i)
    assert allowed.action == BUY


def test_thin_volume_never_suppresses_a_sell():
    """A volume filter that blocked exits would trap a losing position."""
    candles = synthetic(bars=600, seed=11, drift=0.0004, volatility=0.012)
    i = first_bar_with(candles, SELL)
    assert i is not None
    bar = candles[i]
    candles[i] = Candle(bar.timestamp, bar.open, bar.high, bar.low, bar.close, 1.0)
    assert Strategy().evaluate(candles, i).action == SELL


def test_every_signal_shows_its_working():
    signal = Strategy().evaluate(synthetic(bars=200))
    assert any("EMA" in r for r in signal.reasons)
    assert any("RSI" in r for r in signal.reasons)
    assert any("MACD" in r for r in signal.reasons)
    assert any("volume" in r for r in signal.reasons)
    assert signal.score == pytest.approx(sum(signal.votes.values()))


def test_evaluating_an_earlier_bar_ignores_everything_after_it():
    candles = synthetic(bars=300, seed=3)
    at_150 = Strategy().evaluate(candles, 150)
    truncated = Strategy().evaluate(candles[:151], 150)
    assert at_150.action == truncated.action
    assert at_150.score == pytest.approx(truncated.score)
    assert at_150.indicators == pytest.approx(truncated.indicators)


def test_thresholds_move_the_call():
    candles = synthetic(bars=600, seed=11, drift=0.0004, volatility=0.012)
    i = first_bar_with(candles, BUY)
    assert Strategy(StrategyParams(buy_threshold=9.0)).evaluate(candles, i).action == HOLD


def test_unknown_strategy_keys_are_rejected():
    with pytest.raises(ValueError):
        StrategyParams.from_dict({"ema_fastt": 5})


def test_the_cached_and_uncached_paths_agree_exactly():
    """prepare() must be a pure speedup, never a change in behaviour."""
    candles = synthetic(bars=400, seed=23, drift=0.0003, volatility=0.015)
    s = Strategy()
    cache = s.prepare(candles)
    for i in range(s.warmup_bars, len(candles), 7):
        slow = s.evaluate(candles, i)
        fast = s.evaluate(candles, i, cache=cache)
        assert fast.action == slow.action
        assert fast.score == pytest.approx(slow.score)
        assert fast.votes == pytest.approx(slow.votes)
        assert fast.reasons == slow.reasons
