"""Evidence engine: what happened the last 100,000 times this setup appeared?

The rule-based strategy says *what* it wants to do. This says whether history
supports it, and refuses to answer from a small sample.

How it works:

  1. Every bar in the corpus becomes a *situation*: nine scale-free numbers
     describing the market's state at that moment (trend, momentum, stretch,
     volatility, volume, where price sits in its recent range). Scale-free
     matters -- it is what lets a 2019 BTC setup at $8k be comparable to a
     2026 SOL setup at $180.

  2. Every situation is labelled with what actually happened next: entering on
     the following bar's open, did price reach the take-profit before the
     stop-loss, within the horizon? Same pessimistic rule as the backtester --
     a bar that spans both levels counts as a loss.

  3. A query takes the K nearest situations by standardised distance and
     reports the share that won, with a Wilson confidence interval. The gate
     needs the *lower bound* of that interval to clear the threshold, so a
     lucky sample cannot talk its way through.

Labels depend on the stop, target and horizon, so a library is only valid for
the risk settings it was built with; loading checks this.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import indicators
from .strategy import Candle

log = logging.getLogger("trading.analogues")

FEATURES: Tuple[str, ...] = (
    "ema_spread_pct",     # EMA12 vs EMA26, % of the slow EMA -- trend
    "rsi",                # RSI14 / 100 -- stretch
    "macd_hist_pct",      # MACD histogram as % of price -- momentum
    "log_volume_ratio",   # log(volume / its 20-bar average) -- participation
    "return_5_pct",       # short-run drift
    "return_20_pct",      # medium-run drift
    "volatility_20_pct",  # realised volatility of the last 20 log returns
    "range_position",     # 0 at the 20-bar low, 1 at the 20-bar high
    "atr_pct",            # ATR14 as % of price
)

WIN, LOSS, UNRESOLVED = 1, 0, -1


# ---------------------------------------------------------------------------
# Situation features
# ---------------------------------------------------------------------------
def feature_matrix(candles: Sequence[Candle]) -> Tuple[np.ndarray, np.ndarray]:
    """Features for every bar. Returns (features NxF float32, valid mask N bool).

    Column i describes bar i using bars 0..i only -- never later ones.
    """
    n = len(candles)
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    volumes = np.array([c.volume for c in candles], dtype=np.float64)

    out = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)

    ema_fast = _as_array(indicators.ema(closes.tolist(), 12), n)
    ema_slow = _as_array(indicators.ema(closes.tolist(), 26), n)
    rsi = _as_array(indicators.rsi(closes.tolist(), 14), n)
    _, _, hist = indicators.macd(closes.tolist())
    macd_hist = _as_array(hist, n)
    vol_sma = _as_array(indicators.sma(volumes.tolist(), 20), n)
    atr = _as_array(indicators.atr(highs.tolist(), lows.tolist(), closes.tolist(), 14), n)

    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 0] = (ema_fast - ema_slow) / ema_slow * 100.0
        out[:, 1] = rsi / 100.0
        out[:, 2] = macd_hist / closes * 100.0
        out[:, 3] = np.log(np.where(vol_sma > 0, volumes / vol_sma, np.nan))

        out[5:, 4] = (closes[5:] / closes[:-5] - 1.0) * 100.0
        out[20:, 5] = (closes[20:] / closes[:-20] - 1.0) * 100.0

        log_ret = np.full(n, np.nan)
        log_ret[1:] = np.log(closes[1:] / closes[:-1])
        out[:, 6] = _rolling_std(log_ret, 20) * 100.0

        roll_high = _rolling_max(highs, 20)
        roll_low = _rolling_min(lows, 20)
        span = roll_high - roll_low
        out[:, 7] = np.where(span > 0, (closes - roll_low) / span, 0.5)

        out[:, 8] = atr / closes * 100.0

    finite = np.isfinite(out).all(axis=1)
    return out.astype(np.float32), finite


def situation_at(candles: Sequence[Candle], index: Optional[int] = None) -> Optional[np.ndarray]:
    """One situation vector, for the bar at `index` (default: the last)."""
    i = len(candles) - 1 if index is None else index
    features, valid = feature_matrix(candles[: i + 1])
    if not valid[i]:
        return None
    return features[i]


# ---------------------------------------------------------------------------
# Outcome labelling
# ---------------------------------------------------------------------------
def label_outcomes(
    candles: Sequence[Candle],
    stop_loss_pct: float,
    take_profit_pct: float,
    horizon_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What happened after each bar. Returns (outcome, resolve_index, entry_price).

    Entry is the *next* bar's open, so a situation never trades at a price
    that helped describe it. A bar touching both levels resolves as a loss.
    """
    n = len(candles)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    opens = np.array([c.open for c in candles], dtype=np.float64)

    entry = np.full(n, np.nan)
    entry[:-1] = opens[1:]                       # enter at the next bar's open
    tp = entry * (1.0 + take_profit_pct / 100.0)
    sl = entry * (1.0 - stop_loss_pct / 100.0)

    outcome = np.full(n, UNRESOLVED, dtype=np.int8)
    resolve_index = np.full(n, -1, dtype=np.int32)
    resolved = np.zeros(n, dtype=bool)
    resolved[np.isnan(entry)] = True             # no next bar: nothing to enter

    idx = np.arange(n)
    for k in range(1, horizon_bars + 1):
        ahead = idx + k
        usable = (ahead < n) & ~resolved
        if not usable.any():
            break
        j = np.where(usable, ahead, 0)
        hit_sl = usable & (lows[j] <= sl)
        hit_tp = usable & (highs[j] >= tp) & ~hit_sl   # stop wins a tie
        outcome[hit_sl] = LOSS
        outcome[hit_tp] = WIN
        just_resolved = hit_sl | hit_tp
        resolve_index[just_resolved] = k
        resolved |= just_resolved

    # A situation whose window ran off the end of the data is not evidence.
    ran_out = (~resolved) & (idx + horizon_bars >= n)
    outcome[ran_out] = UNRESOLVED
    resolve_index[ran_out] = -1
    return outcome, resolve_index, entry


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
@dataclass
class LibraryParams:
    stop_loss_pct: float
    take_profit_pct: float
    horizon_bars: int
    timeframes: Tuple[str, ...] = ()

    def key(self) -> str:
        return "sl{:g}_tp{:g}_h{}".format(self.stop_loss_pct, self.take_profit_pct, self.horizon_bars)

    def matches(self, other: "LibraryParams") -> bool:
        return (abs(self.stop_loss_pct - other.stop_loss_pct) < 1e-9
                and abs(self.take_profit_pct - other.take_profit_pct) < 1e-9
                and self.horizon_bars == other.horizon_bars)


@dataclass
class Evidence:
    """The answer to 'what happened last time this setup appeared?'"""
    matches: int
    wins: int
    losses: int
    unresolved: int
    success_rate: float
    lower_bound: float
    upper_bound: float
    max_distance: float
    median_distance: float
    mean_forward_bars: float
    passed: bool
    reason: str
    required_matches: int = 0
    required_rate: float = 0.0
    corpus_available: int = 0

    def explain(self) -> str:
        if self.matches == 0:
            return "no comparable situations in the corpus ({})".format(self.reason)
        return (
            "{:,} analogues: {:.1f}% reached target before stop "
            "(95% CI {:.1f}-{:.1f}%), similarity radius {:.2f} -- {}".format(
                self.matches, self.success_rate * 100, self.lower_bound * 100,
                self.upper_bound * 100, self.max_distance, self.reason,
            )
        )


class AnalogueLibrary:
    """A searchable store of labelled historical situations."""

    def __init__(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
        resolve_ts: np.ndarray,
        entry_ts: np.ndarray,
        resolve_bars: np.ndarray,
        series_ids: np.ndarray,
        params: LibraryParams,
        series_names: Optional[List[str]] = None,
    ) -> None:
        order = np.argsort(resolve_ts, kind="stable")
        self.features = np.ascontiguousarray(features[order])
        self.outcomes = outcomes[order]
        self.resolve_ts = resolve_ts[order]
        self.entry_ts = entry_ts[order]
        self.resolve_bars = resolve_bars[order]
        self.series_ids = series_ids[order]
        self.params = params
        self.series_names = series_names or []

        self.centre = np.median(self.features, axis=0).astype(np.float32)
        spread = np.percentile(self.features, 84, axis=0) - np.percentile(self.features, 16, axis=0)
        self.scale = np.where(spread > 1e-9, spread / 2.0, 1.0).astype(np.float32)
        self._standardised = ((self.features - self.centre) / self.scale).astype(np.float32)

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        series,
        params: LibraryParams,
        progress: bool = True,
    ) -> "AnalogueLibrary":
        """`series` is a {name: candles} dict, or an iterable of (name, candles).

        The iterable form matters for a corpus of millions of bars: each series
        is turned into features and thrown away before the next one loads, so
        peak memory stays at one series rather than all of them.
        """
        all_feats: List[np.ndarray] = []
        all_out: List[np.ndarray] = []
        all_res_ts: List[np.ndarray] = []
        all_entry_ts: List[np.ndarray] = []
        all_res_bars: List[np.ndarray] = []
        all_series: List[np.ndarray] = []
        names: List[str] = []

        pairs = sorted(series.items()) if isinstance(series, dict) else series
        for name, candles in pairs:
            if len(candles) < 60 + params.horizon_bars:
                continue
            names.append(name)
            features, valid = feature_matrix(candles)
            outcome, resolve_bars, _ = label_outcomes(
                candles, params.stop_loss_pct, params.take_profit_pct, params.horizon_bars
            )
            timestamps = np.array([c.timestamp for c in candles], dtype=np.int64)

            keep = valid & (outcome != UNRESOLVED)
            if not keep.any():
                continue
            step = int(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0
            resolved_at = timestamps + (resolve_bars.astype(np.int64) + 1) * step

            all_feats.append(features[keep])
            all_out.append(outcome[keep])
            all_res_ts.append(resolved_at[keep])
            all_entry_ts.append(timestamps[keep])
            all_res_bars.append(resolve_bars[keep])
            all_series.append(np.full(int(keep.sum()), len(names) - 1, dtype=np.int16))
            if progress:
                log.info("  %-16s %7d situations kept of %d bars", name, int(keep.sum()), len(candles))

        if not all_feats:
            raise ValueError("no usable situations -- is the corpus empty?")

        return cls(
            features=np.concatenate(all_feats),
            outcomes=np.concatenate(all_out),
            resolve_ts=np.concatenate(all_res_ts),
            entry_ts=np.concatenate(all_entry_ts),
            resolve_bars=np.concatenate(all_res_bars),
            series_ids=np.concatenate(all_series),
            params=params,
            series_names=names,
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features, outcomes=self.outcomes, resolve_ts=self.resolve_ts,
            entry_ts=self.entry_ts, resolve_bars=self.resolve_bars, series_ids=self.series_ids,
            meta=np.array(json.dumps({
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_pct,
                "horizon_bars": self.params.horizon_bars,
                "timeframes": list(self.params.timeframes),
                "series_names": self.series_names,
                "features": list(FEATURES),
            })),
        )
        return path

    @classmethod
    def load(cls, path: str, expect: Optional[LibraryParams] = None) -> "AnalogueLibrary":
        with np.load(path, allow_pickle=False) as blob:
            meta = json.loads(str(blob["meta"]))
            params = LibraryParams(
                stop_loss_pct=meta["stop_loss_pct"],
                take_profit_pct=meta["take_profit_pct"],
                horizon_bars=meta["horizon_bars"],
                timeframes=tuple(meta.get("timeframes", ())),
            )
            if expect is not None and not params.matches(expect):
                raise ValueError(
                    "library was built for {} but the config asks for {}; rebuild it "
                    "with `main.py evidence build`".format(params.key(), expect.key())
                )
            if list(meta.get("features", FEATURES)) != list(FEATURES):
                raise ValueError("library was built with a different feature set; rebuild it")
            return cls(
                features=blob["features"], outcomes=blob["outcomes"],
                resolve_ts=blob["resolve_ts"], entry_ts=blob["entry_ts"],
                resolve_bars=blob["resolve_bars"], series_ids=blob["series_ids"],
                params=params, series_names=meta.get("series_names", []),
            )

    # -- query -----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.features.shape[0])

    def available_before(self, as_of_ms: Optional[int]) -> int:
        """How many situations had already fully resolved by `as_of_ms`."""
        if as_of_ms is None:
            return len(self)
        return int(np.searchsorted(self.resolve_ts, as_of_ms, side="left"))

    def query(
        self,
        situation: np.ndarray,
        min_matches: int,
        min_success_rate: float,
        as_of_ms: Optional[int] = None,
        max_distance: Optional[float] = None,
        confidence_z: float = 1.96,
    ) -> Evidence:
        usable = self.available_before(as_of_ms)
        if usable < min_matches:
            return Evidence(
                matches=0, wins=0, losses=0, unresolved=0, success_rate=0.0,
                lower_bound=0.0, upper_bound=0.0, max_distance=0.0, median_distance=0.0,
                mean_forward_bars=0.0, passed=False,
                reason="only {:,} resolved situations available, needs {:,}".format(usable, min_matches),
                required_matches=min_matches, required_rate=min_success_rate,
                corpus_available=usable,
            )

        pool = self._standardised[:usable]
        point = ((situation.astype(np.float32) - self.centre) / self.scale).astype(np.float32)
        diff = pool - point
        d2 = np.einsum("ij,ij->i", diff, diff)

        k = min_matches
        nearest = np.argpartition(d2, k - 1)[:k]
        distances = np.sqrt(d2[nearest])
        outcomes = self.outcomes[:usable][nearest]

        wins = int((outcomes == WIN).sum())
        losses = int((outcomes == LOSS).sum())
        total = wins + losses
        rate = wins / total if total else 0.0
        low, high = wilson_interval(wins, total, confidence_z)
        radius = float(distances.max())
        median_distance = float(np.median(distances))
        mean_bars = float(self.resolve_bars[:usable][nearest].mean())

        if max_distance is not None and radius > max_distance:
            passed, reason = False, (
                "cohort stretched to radius {:.2f}, past the {:.2f} similarity limit"
                .format(radius, max_distance)
            )
        elif low < min_success_rate:
            passed, reason = False, (
                "{:.1f}% success (95% lower bound {:.1f}%) is below the {:.0f}% required"
                .format(rate * 100, low * 100, min_success_rate * 100)
            )
        else:
            passed, reason = True, (
                "{:.1f}% success across {:,} analogues clears the {:.0f}% bar"
                .format(rate * 100, total, min_success_rate * 100)
            )

        return Evidence(
            matches=k, wins=wins, losses=losses, unresolved=k - total,
            success_rate=rate, lower_bound=low, upper_bound=high,
            max_distance=radius, median_distance=median_distance,
            mean_forward_bars=mean_bars, passed=passed, reason=reason,
            required_matches=min_matches, required_rate=min_success_rate,
            corpus_available=usable,
        )

    def base_rate(self, as_of_ms: Optional[int] = None) -> float:
        """Success rate across the whole corpus -- the number to beat."""
        usable = self.available_before(as_of_ms)
        outcomes = self.outcomes[:usable]
        wins = int((outcomes == WIN).sum())
        total = wins + int((outcomes == LOSS).sum())
        return wins / total if total else 0.0


def wilson_interval(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval -- honest about small samples, tight on large ones."""
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


# -- small helpers ----------------------------------------------------------
def _as_array(series: Sequence[Optional[float]], n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    for i, v in enumerate(series):
        if v is not None:
            out[i] = v
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1:] = np.std(view, axis=1)
    return out


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    out[window - 1:] = np.max(np.lib.stride_tricks.sliding_window_view(values, window), axis=1)
    return out


def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    out[window - 1:] = np.min(np.lib.stride_tricks.sliding_window_view(values, window), axis=1)
    return out


@dataclass
class EvidenceParams:
    enabled: bool = True
    min_similar_situations: int = 100_000
    min_success_rate: float = 0.80
    horizon_bars: int = 48
    max_distance: Optional[float] = None
    confidence_z: float = 1.96
    library: str = "data/corpus/library_1h.npz"

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "EvidenceParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown evidence settings: " + ", ".join(sorted(unknown)))
        params = cls(**raw)
        if params.min_similar_situations < 1:
            raise ValueError("min_similar_situations must be at least 1")
        if not 0 < params.min_success_rate <= 1:
            raise ValueError("min_success_rate must be in (0, 1]")
        if params.horizon_bars < 1:
            raise ValueError("horizon_bars must be at least 1")
        return params


class EvidenceGate:
    """Holds the library and answers one question: does history back this trade?"""

    def __init__(self, library: AnalogueLibrary, params: EvidenceParams) -> None:
        self.library = library
        self.params = params

    @classmethod
    def load(cls, params: EvidenceParams, risk_params, root: str = "") -> "EvidenceGate":
        path = params.library
        if root and not os.path.isabs(path):
            path = os.path.join(root, path)
        expect = LibraryParams(
            stop_loss_pct=risk_params.stop_loss_pct,
            take_profit_pct=risk_params.take_profit_pct,
            horizon_bars=params.horizon_bars,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no analogue library at {}. Build one with:\n"
                "  python scripts/build_corpus.py --years 8\n"
                "  python main.py evidence build".format(path)
            )
        return cls(AnalogueLibrary.load(path, expect), params)

    def check(
        self,
        situation: Optional[np.ndarray],
        as_of_ms: Optional[int] = None,
    ) -> Evidence:
        if situation is None:
            return Evidence(
                matches=0, wins=0, losses=0, unresolved=0, success_rate=0.0,
                lower_bound=0.0, upper_bound=0.0, max_distance=0.0, median_distance=0.0,
                mean_forward_bars=0.0, passed=False,
                reason="not enough history to describe the current situation",
                required_matches=self.params.min_similar_situations,
                required_rate=self.params.min_success_rate,
            )
        return self.library.query(
            situation,
            min_matches=self.params.min_similar_situations,
            min_success_rate=self.params.min_success_rate,
            as_of_ms=as_of_ms,
            max_distance=self.params.max_distance,
            confidence_z=self.params.confidence_z,
        )
