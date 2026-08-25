"""The evidence engine: features, outcome labels, and the gate itself.

The two properties that make this trustworthy are tested here: features at bar
i never depend on bars after i, and a query never sees a situation that had not
already resolved by the time of the query.
"""
import numpy as np
import pytest

from trading.analogues import (LOSS, UNRESOLVED, WIN, AnalogueLibrary, Evidence,
                               EvidenceGate, EvidenceParams, LibraryParams,
                               feature_matrix, label_outcomes, situation_at,
                               wilson_interval)
from trading.data import synthetic
from trading.strategy import Candle


def bars(rows, start=1_700_000_000_000, step=3_600_000):
    """rows: (open, high, low, close) tuples."""
    return [Candle(start + i * step, o, h, l, c, 1000.0)
            for i, (o, h, l, c) in enumerate(rows)]


# -- features ---------------------------------------------------------------
def test_features_are_causal():
    """Deleting the future must not change the present."""
    candles = synthetic(bars=400, seed=31, volatility=0.015)
    full, valid = feature_matrix(candles)
    for cut in (120, 250, 399):
        truncated, tvalid = feature_matrix(candles[: cut + 1])
        assert tvalid[cut] == valid[cut]
        if valid[cut]:
            assert np.allclose(full[cut], truncated[cut], rtol=1e-5, atol=1e-6)


def test_features_are_scale_free():
    """The same shape at $8k and $180 must produce the same situation vector."""
    candles = synthetic(bars=300, seed=12, start_price=8000.0)
    scaled = [Candle(c.timestamp, c.open / 44.0, c.high / 44.0, c.low / 44.0,
                     c.close / 44.0, c.volume) for c in candles]
    a, b = situation_at(candles), situation_at(scaled)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_a_bar_without_enough_history_has_no_situation():
    assert situation_at(synthetic(bars=15)) is None


# -- outcome labelling ------------------------------------------------------
def test_target_before_stop_is_a_win():
    # enter at bar 1's open (100); +4% target = 104, -2% stop = 98
    rows = [(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 105, 100, 104.5)]
    outcome, resolve, entry = label_outcomes(bars(rows), 2.0, 4.0, 10)
    assert entry[0] == pytest.approx(100.0)
    assert outcome[0] == WIN
    assert resolve[0] == 2


def test_stop_before_target_is_a_loss():
    rows = [(100, 100, 100, 100), (100, 101, 97, 98), (98, 106, 97, 105)]
    outcome, _, _ = label_outcomes(bars(rows), 2.0, 4.0, 10)
    assert outcome[0] == LOSS


def test_a_bar_that_spans_both_levels_counts_as_a_loss():
    rows = [(100, 100, 100, 100), (100, 105, 97, 104)]
    outcome, _, _ = label_outcomes(bars(rows), 2.0, 4.0, 10)
    assert outcome[0] == LOSS


def test_neither_level_inside_the_horizon_is_unresolved():
    rows = [(100, 100, 100, 100)] + [(100, 101, 99.5, 100)] * 8
    outcome, _, _ = label_outcomes(bars(rows), 2.0, 4.0, 3)
    assert outcome[0] == UNRESOLVED


def test_a_label_is_only_given_when_the_level_was_actually_touched():
    """No bar may be labelled from a window that ran off the end of the data."""
    candles = synthetic(bars=300, seed=4, volatility=0.02)
    horizon = 48
    outcome, resolve, entry = label_outcomes(candles, 2.0, 4.0, horizon)
    assert outcome[-1] == UNRESOLVED, "the last bar has no next bar to enter on"

    for i, label in enumerate(outcome):
        if label == UNRESOLVED:
            continue
        window = candles[i + 1: i + 1 + horizon]
        assert window, "a labelled situation must have bars after it"
        k = int(resolve[i])
        assert 1 <= k <= horizon
        assert i + k < len(candles)
        if label == WIN:
            assert candles[i + k].high >= entry[i] * 1.04 - 1e-9
        else:
            assert candles[i + k].low <= entry[i] * 0.98 + 1e-9


def test_entry_is_the_next_bars_open_not_this_bars_close():
    rows = [(100, 100, 100, 100), (110, 115, 109, 114)]
    _, _, entry = label_outcomes(bars(rows), 2.0, 4.0, 5)
    assert entry[0] == pytest.approx(110.0)


# -- wilson -----------------------------------------------------------------
def test_wilson_interval_tightens_with_sample_size():
    small_lo, small_hi = wilson_interval(8, 10)
    big_lo, big_hi = wilson_interval(80_000, 100_000)
    assert small_hi - small_lo > big_hi - big_lo
    assert small_lo < 0.8 < small_hi
    assert big_lo < 0.8 < big_hi
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_lower_bound_is_below_the_point_estimate():
    lo, hi = wilson_interval(5_400, 10_000)
    assert lo < 0.54 < hi


# -- library ----------------------------------------------------------------
def rigged_library(n=500, win_share=1.0, params=None):
    """A library where every situation sits at the same point in feature space."""
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, size=(n, 9)).astype(np.float32)
    outcomes = np.where(np.arange(n) < int(n * win_share), WIN, LOSS).astype(np.int8)
    resolve_ts = np.arange(n, dtype=np.int64) * 1000 + 1_000_000
    return AnalogueLibrary(
        features=features, outcomes=outcomes, resolve_ts=resolve_ts,
        entry_ts=resolve_ts - 500, resolve_bars=np.full(n, 5, dtype=np.int32),
        series_ids=np.zeros(n, dtype=np.int16),
        params=params or LibraryParams(2.0, 4.0, 48), series_names=["rigged"],
    )


def test_a_query_needing_more_analogues_than_exist_is_refused():
    library = rigged_library(n=100)
    verdict = library.query(np.zeros(9, dtype=np.float32), min_matches=100_000,
                            min_success_rate=0.8)
    assert not verdict.passed
    assert verdict.matches == 0
    assert "needs 100,000" in verdict.reason


def test_a_cohort_that_all_won_passes():
    library = rigged_library(n=500, win_share=1.0)
    verdict = library.query(np.zeros(9, dtype=np.float32), min_matches=200,
                            min_success_rate=0.8)
    assert verdict.passed
    assert verdict.wins == 200
    assert verdict.success_rate == pytest.approx(1.0)


def test_a_coin_flip_cohort_is_blocked_at_eighty_percent():
    library = rigged_library(n=500, win_share=0.5)
    verdict = library.query(np.zeros(9, dtype=np.float32), min_matches=200,
                            min_success_rate=0.8)
    assert not verdict.passed
    assert "below the 80% required" in verdict.reason


def test_the_gate_uses_the_lower_confidence_bound_not_the_raw_rate():
    """20 of 24 wins is 83%, but the sample is far too small to be sure."""
    library = rigged_library(n=24, win_share=20 / 24)
    verdict = library.query(np.zeros(9, dtype=np.float32), min_matches=24,
                            min_success_rate=0.8)
    assert verdict.success_rate > 0.8
    assert verdict.lower_bound < 0.8
    assert not verdict.passed


def test_a_far_flung_cohort_is_rejected_when_a_distance_cap_is_set():
    library = rigged_library(n=500, win_share=1.0)
    far = np.full(9, 50.0, dtype=np.float32)
    assert library.query(far, 200, 0.8, max_distance=None).passed
    verdict = library.query(far, 200, 0.8, max_distance=2.0)
    assert not verdict.passed
    assert "similarity limit" in verdict.reason


def test_as_of_hides_situations_that_had_not_resolved_yet():
    library = rigged_library(n=500)
    assert library.available_before(None) == 500
    assert library.available_before(1_000_000) == 0
    assert library.available_before(1_100_000) == 100

    verdict = library.query(np.zeros(9, dtype=np.float32), min_matches=200,
                            min_success_rate=0.8, as_of_ms=1_100_000)
    assert not verdict.passed
    assert "only 100 resolved situations available" in verdict.reason


def test_a_library_saved_and_loaded_is_unchanged(tmp_path):
    library = rigged_library(n=300)
    path = str(tmp_path / "lib.npz")
    library.save(path)
    reloaded = AnalogueLibrary.load(path, expect=LibraryParams(2.0, 4.0, 48))
    assert len(reloaded) == 300
    assert np.array_equal(reloaded.outcomes, library.outcomes)
    assert reloaded.params.take_profit_pct == 4.0


def test_loading_a_library_built_for_different_risk_settings_is_refused(tmp_path):
    path = str(tmp_path / "lib.npz")
    rigged_library(n=50).save(path)
    with pytest.raises(ValueError) as err:
        AnalogueLibrary.load(path, expect=LibraryParams(3.0, 4.0, 48))
    assert "rebuild" in str(err.value)


def test_building_from_candles_keeps_only_resolved_situations():
    series = {"TEST 1h": synthetic(bars=800, seed=19, volatility=0.02)}
    library = AnalogueLibrary.build(series, LibraryParams(2.0, 4.0, 48), progress=False)
    assert len(library) > 500
    assert set(np.unique(library.outcomes)).issubset({WIN, LOSS})
    assert 0.0 < library.base_rate() < 1.0


def test_building_accepts_a_stream_of_series():
    def stream():
        yield "A 1h", synthetic(bars=400, seed=1)
        yield "B 1h", synthetic(bars=400, seed=2)

    library = AnalogueLibrary.build(stream(), LibraryParams(2.0, 4.0, 24), progress=False)
    assert len(library.series_names) == 2


# -- the gate in front of the backtester ------------------------------------
def test_the_gate_can_be_switched_off_by_config():
    params = EvidenceParams(enabled=False)
    assert params.enabled is False


def test_unknown_evidence_keys_are_rejected():
    with pytest.raises(ValueError):
        EvidenceParams.from_dict({"min_similar_situation": 100})


def test_evidence_settings_are_validated():
    with pytest.raises(ValueError):
        EvidenceParams.from_dict({"min_success_rate": 1.5})
    with pytest.raises(ValueError):
        EvidenceParams.from_dict({"min_similar_situations": 0})


def test_a_gate_with_no_situation_vector_blocks_rather_than_guesses():
    gate = EvidenceGate(rigged_library(n=500), EvidenceParams(min_similar_situations=200))
    verdict = gate.check(None)
    assert not verdict.passed
    assert "not enough history" in verdict.reason


def test_a_query_can_only_see_situations_that_had_already_resolved():
    """The whole point of as_of: no situation may inform a decision made before
    that situation's own outcome was known."""
    library = AnalogueLibrary.build(
        {"TEST 1h": synthetic(bars=3000, seed=77, volatility=0.02)},
        LibraryParams(2.0, 4.0, 48), progress=False,
    )
    cutoff = int(np.median(library.resolve_ts))
    available = library.available_before(cutoff)
    assert available == int((library.resolve_ts < cutoff).sum())
    assert 0 < available < len(library)
    # Everything the query may draw on resolved strictly before the cutoff.
    assert library.resolve_ts[:available].max() < cutoff
    assert library.resolve_ts[available:].min() >= cutoff
