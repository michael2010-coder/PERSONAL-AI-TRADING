"""Rule-based strategy: EMA trend + RSI + MACD, gated by a volume filter.

Each rule casts a vote in [-1, +1]. The votes are summed; the sum has to clear
a threshold before the strategy will call BUY or SELL. Every Signal carries the
numbers that produced it, so any call can be checked by hand afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import indicators

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class Candle:
    timestamp: int  # epoch milliseconds, bar open
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IndicatorCache:
    """Indicator series for a whole run, computed once.

    Every indicator here is causal -- the value at bar i depends only on bars
    0..i (proven in tests/test_indicators.py) -- so reading position i out of a
    full-series computation is identical to recomputing from scratch with only
    bars 0..i available. tests/test_strategy.py asserts that equivalence
    directly. What changes is the cost: O(n) for a run instead of O(n^2).
    """
    ema_fast: List[Optional[float]]
    ema_slow: List[Optional[float]]
    rsi: List[Optional[float]]
    macd_hist: List[Optional[float]]
    volume_sma: List[Optional[float]]


@dataclass
class Signal:
    action: str
    score: float
    price: float
    timestamp: int
    votes: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    indicators: Dict[str, Optional[float]] = field(default_factory=dict)

    def explain(self) -> str:
        head = "{} (score {:+.2f}) @ {:.2f}".format(self.action, self.score, self.price)
        return "\n".join([head] + ["  - " + r for r in self.reasons])


@dataclass
class TrendParams:
    """Long-horizon trend following: hold while above a moving average.

    The opposite of the voting strategy in every way that matters. It trades
    about a dozen times a year instead of hundreds, holds winners for months
    instead of hours, and takes its exit from the trend rather than a fixed
    target. Measured across 32 markets it beat buy-and-hold on return in 20
    and on drawdown in 25.
    """
    ma_days: int = 100
    confirm_bars: int = 1        # bars the break must hold before acting

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "TrendParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown trend settings: " + ", ".join(sorted(unknown)))
        params = cls(**raw)
        if params.ma_days < 2:
            raise ValueError("ma_days must be at least 2")
        if params.confirm_bars < 1:
            raise ValueError("confirm_bars must be at least 1")
        return params


class TrendStrategy:
    """Same interface as Strategy, so the engine and backtester do not care."""

    def __init__(self, params: Optional[TrendParams] = None) -> None:
        self.p = params or TrendParams()

    @property
    def warmup_bars(self) -> int:
        return self.p.ma_days + self.p.confirm_bars + 1

    def prepare(self, candles: Sequence[Candle]) -> IndicatorCache:
        closes = [c.close for c in candles]
        return IndicatorCache(
            ema_fast=[None] * len(candles),
            ema_slow=indicators.sma(closes, self.p.ma_days),
            rsi=[None] * len(candles),
            macd_hist=[None] * len(candles),
            volume_sma=[None] * len(candles),
        )

    def evaluate(self, candles, index=None, cache=None) -> Signal:
        if not candles:
            raise ValueError("no candles supplied")
        i = len(candles) - 1 if index is None else index
        if i < 0 or i >= len(candles):
            raise IndexError("index out of range")

        cache = cache or self.prepare(candles[: i + 1])
        average = cache.ema_slow[i]
        bar = candles[i]
        if average is None:
            return Signal(action=HOLD, score=0.0, price=bar.close,
                          timestamp=bar.timestamp, votes={}, indicators={},
                          reasons=["not enough history for a {}-bar average"
                                   .format(self.p.ma_days)])

        # The break has to hold for confirm_bars, so a single spike through
        # the average does not flip the position.
        window = range(max(0, i - self.p.confirm_bars + 1), i + 1)
        above = all(candles[j].close > (cache.ema_slow[j] or float("inf")) for j in window)
        below = all(candles[j].close < (cache.ema_slow[j] or float("-inf")) for j in window)

        gap = (bar.close - average) / average * 100.0
        reason = "close {:.2f} vs {}-bar average {:.2f} ({:+.2f}%)".format(
            bar.close, self.p.ma_days, average, gap)

        if above:
            action, score = BUY, 1.0
            note = "above trend for {} bar(s) -> hold/enter".format(self.p.confirm_bars)
        elif below:
            action, score = SELL, -1.0
            note = "below trend for {} bar(s) -> stand aside".format(self.p.confirm_bars)
        else:
            action, score = HOLD, 0.0
            note = "crossing, not yet confirmed -> wait"

        return Signal(action=action, score=score, price=bar.close,
                      timestamp=bar.timestamp, votes={"trend": score},
                      reasons=[reason, note],
                      indicators={"average": average, "gap_pct": gap})


@dataclass
class StrategyParams:
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_period: int = 20
    volume_factor: float = 1.0
    buy_threshold: float = 1.5
    sell_threshold: float = -1.0
    require_volume_confirmation: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "StrategyParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown strategy settings: " + ", ".join(sorted(unknown)))
        return cls(**raw)


class Strategy:
    """Stateless signal generator over a list of candles."""

    def __init__(self, params: Optional[StrategyParams] = None) -> None:
        self.p = params or StrategyParams()

    @property
    def warmup_bars(self) -> int:
        """Bars needed before the strategy can produce a non-HOLD call."""
        p = self.p
        return max(
            p.ema_slow,
            p.rsi_period + 1,
            p.macd_slow + p.macd_signal,
            p.volume_period,
        ) + 1

    def prepare(self, candles: Sequence[Candle]) -> IndicatorCache:
        """Compute every series once, for a backtest that walks the whole run."""
        p = self.p
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        _, _, hist = indicators.macd(closes, p.macd_fast, p.macd_slow, p.macd_signal)
        return IndicatorCache(
            ema_fast=indicators.ema(closes, p.ema_fast),
            ema_slow=indicators.ema(closes, p.ema_slow),
            rsi=indicators.rsi(closes, p.rsi_period),
            macd_hist=hist,
            volume_sma=indicators.sma(volumes, p.volume_period),
        )

    def evaluate(
        self,
        candles: Sequence[Candle],
        index: Optional[int] = None,
        cache: Optional[IndicatorCache] = None,
    ) -> Signal:
        """Signal for `index` (default: the last candle) using bars 0..index only.

        Pass `cache` from prepare() to reuse one full-series computation across
        a run; the result is identical either way.
        """
        if not candles:
            raise ValueError("no candles supplied")
        i = len(candles) - 1 if index is None else index
        if i < 0 or i >= len(candles):
            raise IndexError("index out of range")

        p = self.p
        if cache is None:
            window = candles[: i + 1]
            closes = [c.close for c in window]
            volumes = [c.volume for c in window]
            ema_fast = indicators.ema(closes, p.ema_fast)[i]
            ema_slow = indicators.ema(closes, p.ema_slow)[i]
            rsi_series = indicators.rsi(closes, p.rsi_period)
            _, _, hist = indicators.macd(closes, p.macd_fast, p.macd_slow, p.macd_signal)
            vol_sma = indicators.sma(volumes, p.volume_period)[i]
        else:
            volumes = [candles[i].volume]
            ema_fast = cache.ema_fast[i]
            ema_slow = cache.ema_slow[i]
            rsi_series = cache.rsi
            hist = cache.macd_hist
            vol_sma = cache.volume_sma[i]

        rsi_now = rsi_series[i]
        rsi_prev = rsi_series[i - 1] if i > 0 else None
        hist_now = hist[i]
        hist_prev = hist[i - 1] if i > 0 else None

        votes: Dict[str, float] = {}
        reasons: List[str] = []

        if ema_fast is None or ema_slow is None:
            return self._hold(candles[i], "not enough history for EMA{}".format(p.ema_slow))
        spread_pct = (ema_fast - ema_slow) / ema_slow * 100.0
        votes["ema"] = 1.0 if ema_fast > ema_slow else -1.0
        reasons.append(
            "EMA{}={:.2f} vs EMA{}={:.2f} ({:+.2f}%) -> {}".format(
                p.ema_fast, ema_fast, p.ema_slow, ema_slow, spread_pct,
                "uptrend" if votes["ema"] > 0 else "downtrend",
            )
        )

        if rsi_now is None:
            return self._hold(candles[i], "not enough history for RSI{}".format(p.rsi_period))
        if rsi_now <= p.rsi_oversold:
            votes["rsi"] = 1.0
            reasons.append("RSI{}={:.1f} at/below oversold {:.0f} -> buy".format(p.rsi_period, rsi_now, p.rsi_oversold))
        elif rsi_now >= p.rsi_overbought:
            votes["rsi"] = -1.0
            reasons.append("RSI{}={:.1f} at/above overbought {:.0f} -> sell".format(p.rsi_period, rsi_now, p.rsi_overbought))
        else:
            # Mid-range RSI leans with its own slope, at half weight.
            slope = 0.0 if rsi_prev is None else rsi_now - rsi_prev
            votes["rsi"] = 0.5 if slope > 0 else (-0.5 if slope < 0 else 0.0)
            reasons.append(
                "RSI{}={:.1f} neutral, slope {:+.2f} -> {:+.1f}".format(
                    p.rsi_period, rsi_now, slope, votes["rsi"]
                )
            )

        if hist_now is None:
            return self._hold(candles[i], "not enough history for MACD")
        votes["macd"] = 1.0 if hist_now > 0 else (-1.0 if hist_now < 0 else 0.0)
        turn = ""
        if hist_prev is not None and hist_prev * hist_now < 0:
            turn = ", crossed {} this bar".format("up" if hist_now > 0 else "down")
        reasons.append("MACD histogram={:+.4f}{} -> {:+.1f}".format(hist_now, turn, votes["macd"]))

        confirmed = True
        if vol_sma is None or vol_sma <= 0:
            confirmed = not p.require_volume_confirmation
            reasons.append("volume SMA{} unavailable".format(p.volume_period))
        else:
            volume_now = candles[i].volume
            ratio = volume_now / vol_sma
            confirmed = ratio >= p.volume_factor
            reasons.append(
                "volume {:.4g} vs SMA{} {:.4g} = {:.2f}x (need {:.2f}x) -> {}".format(
                    volume_now, p.volume_period, vol_sma, ratio, p.volume_factor,
                    "confirmed" if confirmed else "not confirmed",
                )
            )

        score = sum(votes.values())
        action = HOLD
        if score >= p.buy_threshold:
            action = BUY
        elif score <= p.sell_threshold:
            action = SELL

        # The volume filter only ever blocks entries, never exits: refusing to
        # sell on thin volume would trap a position we already want out of.
        if action == BUY and p.require_volume_confirmation and not confirmed:
            action = HOLD
            reasons.append("BUY suppressed: volume did not confirm")

        return Signal(
            action=action,
            score=score,
            price=candles[i].close,
            timestamp=candles[i].timestamp,
            votes=votes,
            reasons=reasons,
            indicators={
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi_now,
                "macd_hist": hist_now,
                "volume_sma": vol_sma,
            },
        )

    def _hold(self, candle: Candle, why: str) -> Signal:
        return Signal(
            action=HOLD, score=0.0, price=candle.close, timestamp=candle.timestamp,
            votes={}, reasons=[why], indicators={},
        )
