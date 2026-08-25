"""The risk manager is the only thing between a signal and an order.

These tests are the ones worth reading before funding anything: they pin the
numbers that decide how much a single bad trade, or a bad day, can cost.
"""
import pytest

from trading.risk import Position, RiskManager, RiskParams


def mk(**overrides) -> RiskManager:
    params = RiskParams(**{**dict(
        allocated_capital_usdt=1000.0, max_position_size_pct=10.0,
        stop_loss_pct=2.0, take_profit_pct=4.0, max_open_positions=1,
        max_daily_loss_pct=5.0, min_notional_usdt=10.0,
    ), **overrides})
    return RiskManager(params)


def test_position_size_is_a_slice_of_allocated_capital_not_the_balance():
    rm = mk()
    d = rm.review_entry(price=100.0, open_positions=[], free_balance=50_000.0)
    assert d.approved
    assert d.notional == pytest.approx(100.0)   # 10% of 1000, not of 50k
    assert d.qty == pytest.approx(1.0)


def test_a_stop_loss_hit_costs_the_sized_amount_and_no_more():
    rm = mk()
    d = rm.review_entry(price=100.0, open_positions=[])
    # 10% of capital at 2% stop = 0.2% of capital = 2.00 USDT
    assert d.max_loss_usdt == pytest.approx(2.0)
    assert d.max_loss_usdt / rm.p.allocated_capital_usdt * 100 == pytest.approx(0.2)


def test_every_approved_entry_carries_a_stop_and_a_target():
    d = mk().review_entry(price=200.0, open_positions=[])
    assert d.stop_price == pytest.approx(196.0)
    assert d.take_profit_price == pytest.approx(208.0)
    assert d.stop_price < 200.0 < d.take_profit_price


def test_max_open_positions_blocks_the_next_entry():
    rm = mk(max_open_positions=1)
    held = Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0)
    d = rm.review_entry(price=100.0, open_positions=[held])
    assert not d.approved
    assert "max_open_positions" in d.reason


def test_never_doubles_into_a_symbol_already_held():
    rm = mk(max_open_positions=3)
    held = Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0)
    assert not rm.review_entry(price=100.0, open_positions=[held], symbol="BTC/USDT").approved
    assert rm.review_entry(price=100.0, open_positions=[held], symbol="ETH/USDT").approved


def test_daily_loss_kill_switch_blocks_at_the_limit_exactly():
    rm = mk()                       # limit = 5% of 1000 = 50.00
    assert rm.daily_loss_limit() == pytest.approx(50.0)
    assert rm.review_entry(price=100.0, open_positions=[], realised_pnl_today=-49.99).approved
    blocked = rm.review_entry(price=100.0, open_positions=[], realised_pnl_today=-50.0)
    assert not blocked.approved
    assert "kill-switch" in blocked.reason
    assert rm.kill_switch_tripped(-50.0)
    assert not rm.kill_switch_tripped(-49.99)


def test_sizing_never_exceeds_capital_not_already_at_work():
    rm = mk(max_open_positions=5, max_position_size_pct=50.0)
    held = Position("BTC/USDT", "long", 8.0, 100.0, 98.0, 104.0, 0)  # 800 committed
    d = rm.review_entry(price=100.0, open_positions=[held], symbol="ETH/USDT")
    assert d.approved
    assert d.notional == pytest.approx(200.0)   # capped by the 200 of headroom left


def test_sizing_respects_a_thin_wallet():
    rm = mk()
    d = rm.review_entry(price=100.0, open_positions=[], free_balance=40.0)
    assert d.approved
    assert d.notional == pytest.approx(40.0)


def test_dust_orders_are_refused():
    rm = mk(allocated_capital_usdt=50.0, max_position_size_pct=10.0, min_notional_usdt=10.0)
    d = rm.review_entry(price=100.0, open_positions=[])
    assert not d.approved and "minimum" in d.reason


def test_a_stop_is_mandatory_in_config():
    with pytest.raises(ValueError):
        RiskParams(stop_loss_pct=0).validate()
    with pytest.raises(ValueError):
        RiskParams(max_position_size_pct=150).validate()
    with pytest.raises(ValueError):
        RiskParams(allocated_capital_usdt=0).validate()


def test_unknown_risk_keys_are_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError):
        RiskParams.from_dict({"stop_loss_pct": 2.0, "stop_los_pct": 99.0})


def test_when_one_bar_spans_both_levels_the_stop_wins():
    rm = mk()
    pos = Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0)
    assert rm.check_exit(pos, high=105.0, low=97.0) == "stop_loss"
    assert rm.check_exit(pos, high=105.0, low=99.0) == "take_profit"
    assert rm.check_exit(pos, high=103.0, low=99.0) is None


def test_short_side_levels_and_exits_mirror_the_long_side():
    rm = mk()
    levels = rm.exit_levels(100.0, "short")
    assert levels["stop_price"] == pytest.approx(102.0)
    assert levels["take_profit_price"] == pytest.approx(96.0)
    pos = Position("BTC/USDT", "short", 1.0, 100.0, 102.0, 96.0, 0)
    assert rm.check_exit(pos, high=103.0, low=95.0) == "stop_loss"
