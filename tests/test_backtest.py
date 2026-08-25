import pytest

from trading.backtest import Backtester
from trading.data import synthetic
from trading.risk import RiskManager, RiskParams
from trading.strategy import BUY, Candle, Strategy, StrategyParams


def make(fee_bps=10.0, slippage_bps=5.0, **risk_overrides):
    params = RiskParams(**{**dict(
        allocated_capital_usdt=1000.0, max_position_size_pct=10.0,
        stop_loss_pct=2.0, take_profit_pct=4.0, max_open_positions=1,
        max_daily_loss_pct=5.0, min_notional_usdt=10.0,
    ), **risk_overrides})
    return Backtester(Strategy(), RiskManager(params), fee_bps, slippage_bps)


def test_it_refuses_a_series_too_short_to_warm_up():
    with pytest.raises(ValueError):
        make().run(synthetic(bars=20))


def test_equity_equals_starting_capital_plus_realised_pnl():
    result = make().run(synthetic(bars=800, seed=5, drift=0.0003, volatility=0.012))
    assert result.ending_equity == pytest.approx(
        result.starting_capital + sum(t.pnl for t in result.trades), abs=1e-6
    )


def test_no_trade_loses_more_than_the_position_was_sized_to_lose():
    """A stop-loss hit must cost roughly 0.2% of capital: 10% of it, stopped at 2%.

    Slippage and gap-downs can push a single loss past that, so the bound is
    the sized loss plus the configured slippage, not the sized loss exactly.
    """
    bt = make()
    result = bt.run(synthetic(bars=1500, seed=9, drift=0.0, volatility=0.02))
    sized_loss = 1000.0 * 0.10 * 0.02          # 2.00 USDT
    for trade in result.trades:
        if trade.exit_reason == "stop_loss":
            assert -trade.pnl <= sized_loss * 1.5, trade


def test_stops_and_targets_land_where_the_risk_manager_put_them():
    bt = make()
    result = bt.run(synthetic(bars=1200, seed=4, volatility=0.015))
    for trade in result.trades:
        if trade.exit_reason == "stop_loss":
            assert trade.exit_price <= trade.entry_price * 0.98 + 1e-9
        elif trade.exit_reason == "take_profit":
            assert trade.exit_price >= trade.entry_price * 1.04 * (1 - 0.0005) - 1e-9


def gapped(candles, gap_pct=0.4):
    """Re-open each bar away from the previous close.

    The synthetic generator makes every open equal the previous close, which
    would hide the difference between filling at the decision bar's close and
    the next bar's open. Forcing a gap makes the two prices distinguishable.
    """
    out = [candles[0]]
    for prev, cur in zip(candles, candles[1:]):
        open_ = prev.close * (1.0 + gap_pct / 100.0)
        out.append(Candle(cur.timestamp, open_, max(cur.high, open_ * 1.001),
                          min(cur.low, open_ * 0.999), cur.close, cur.volume))
    return out


def test_fills_happen_on_the_bar_after_the_signal():
    """The backtester may only trade at a price it has not already used.

    For every entry, the bar it filled on must be the one immediately after
    the bar whose close produced the BUY -- and the fill must be that later
    bar's open, not the close the decision was made from.
    """
    candles = gapped(synthetic(bars=900, seed=6, drift=0.0004, volatility=0.015))
    bt = make(fee_bps=0.0, slippage_bps=0.0)
    actions = {}
    result = bt.run(candles, on_signal=lambda i, s, pos: actions.__setitem__(i, s.action))
    assert result.trades, "test needs at least one trade to prove anything"

    index_of = {c.timestamp: i for i, c in enumerate(candles)}
    for trade in result.trades:
        i = index_of[trade.entry_time]
        assert actions[i - 1] == BUY                                   # decided one bar earlier
        assert trade.entry_price == pytest.approx(candles[i].open)     # filled at the later open
        assert trade.entry_price != pytest.approx(candles[i - 1].close)


def test_a_signal_exit_also_fills_on_the_next_bars_open():
    """Same rule on the way out, with stops set wide enough to let signals exit."""
    candles = gapped(synthetic(bars=900, seed=6, drift=0.0004, volatility=0.015))
    bt = make(fee_bps=0.0, slippage_bps=0.0, stop_loss_pct=25.0, take_profit_pct=50.0)
    result = bt.run(candles)
    index_of = {c.timestamp: i for i, c in enumerate(candles)}
    signal_exits = [t for t in result.trades if t.exit_reason == "signal"]
    assert signal_exits, "test needs at least one signal-driven exit"
    for trade in signal_exits:
        i = index_of[trade.exit_time]
        assert trade.exit_price == pytest.approx(candles[i].open)


def test_a_flat_market_never_trades():
    flat = [Candle(1_700_000_000_000 + i * 3_600_000, 100.0, 100.0, 100.0, 100.0, 1000.0)
            for i in range(300)]
    result = make().run(flat)
    assert result.trades == []
    assert result.ending_equity == pytest.approx(result.starting_capital)


def test_fees_and_slippage_make_a_difference_and_only_ever_cost():
    candles = synthetic(bars=900, seed=6, drift=0.0004, volatility=0.015)
    free = make(fee_bps=0.0, slippage_bps=0.0).run(candles)
    costly = make(fee_bps=25.0, slippage_bps=15.0).run(candles)
    assert sum(t.fees for t in free.trades) == pytest.approx(0.0)
    assert costly.ending_equity < free.ending_equity


def test_it_never_holds_more_than_max_open_positions():
    positions_seen = []
    bt = make()
    bt.run(synthetic(bars=900, seed=8, volatility=0.015),
           on_signal=lambda i, s, pos: positions_seen.append(1 if pos else 0))
    assert max(positions_seen) <= 1


def test_kill_switch_stops_new_entries_for_the_rest_of_the_day():
    """With a hair-trigger daily limit, one loss must end that day's trading."""
    bt = make(max_daily_loss_pct=0.1, stop_loss_pct=1.0)   # limit = 1.00 USDT
    result = bt.run(synthetic(bars=1500, seed=13, drift=-0.0005, volatility=0.02))
    assert result.kill_switch_days, "expected the kill-switch to trip at least once"
    from collections import Counter
    from datetime import datetime, timezone

    def day(ms):
        return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")

    entries_per_killed_day = Counter(
        day(t.entry_time) for t in result.trades if day(t.entry_time) in result.kill_switch_days
    )
    # The trade that trips the switch is allowed; nothing after it that day is.
    assert all(count <= 1 for count in entries_per_killed_day.values())


def test_buy_and_hold_is_reported_for_comparison():
    result = make().run(synthetic(bars=900, seed=2, drift=0.001, volatility=0.01))
    assert result.buy_hold_return_pct > 0
    assert "Buy & hold" in result.summary()


def test_drawdown_is_negative_or_zero_and_summary_renders():
    result = make().run(synthetic(bars=900, seed=17, volatility=0.02))
    assert result.max_drawdown_pct <= 0
    text = result.summary()
    for label in ("Total return", "Max drawdown", "Win rate", "Profit factor"):
        assert label in text


class StubGate:
    """Stands in for the analogue library: answers yes, or no, on demand."""

    def __init__(self, verdict: bool):
        self.verdict = verdict
        self.calls = []

    def check(self, situation, as_of_ms=None):
        from trading.analogues import Evidence
        self.calls.append((situation, as_of_ms))
        return Evidence(
            matches=100_000, wins=80_000, losses=20_000, unresolved=0,
            success_rate=0.8, lower_bound=0.79, upper_bound=0.81,
            max_distance=1.0, median_distance=0.5, mean_forward_bars=12.0,
            passed=self.verdict, reason="stub",
        )


def test_the_evidence_gate_can_stop_every_entry():
    candles = synthetic(bars=900, seed=6, drift=0.0004, volatility=0.015)
    blocked = Backtester(Strategy(), RiskManager(RiskParams(allocated_capital_usdt=1000.0)),
                         evidence=StubGate(False)).run(candles)
    assert blocked.trades == []
    assert blocked.evidence_blocked > 0
    assert blocked.evidence_passed == 0
    assert blocked.ending_equity == pytest.approx(blocked.starting_capital)


def test_an_always_yes_gate_trades_exactly_as_if_ungated():
    candles = synthetic(bars=900, seed=6, drift=0.0004, volatility=0.015)
    ungated = make().run(candles)
    gate = StubGate(True)
    gated = Backtester(Strategy(), RiskManager(RiskParams(
        allocated_capital_usdt=1000.0, max_position_size_pct=10.0, stop_loss_pct=2.0,
        take_profit_pct=4.0, max_open_positions=1, max_daily_loss_pct=5.0,
        min_notional_usdt=10.0)), 10.0, 5.0, evidence=gate).run(candles)
    assert len(gated.trades) == len(ungated.trades)
    assert gated.ending_equity == pytest.approx(ungated.ending_equity)
    assert gated.evidence_passed == len(gate.calls)


def test_the_gate_is_only_asked_about_bars_it_could_have_known_about():
    """Every query must carry the timestamp of the bar being decided."""
    candles = synthetic(bars=600, seed=6, drift=0.0004, volatility=0.015)
    gate = StubGate(True)
    Backtester(Strategy(), RiskManager(RiskParams(allocated_capital_usdt=1000.0)),
               evidence=gate).run(candles)
    stamps = {c.timestamp for c in candles}
    assert gate.calls
    for situation, as_of in gate.calls:
        assert as_of in stamps
