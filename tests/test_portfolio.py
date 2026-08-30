"""The anti-greed layer. These tests are the promises the account makes."""
import pytest

from trading.portfolio import Portfolio, PortfolioParams


def make(**overrides):
    return Portfolio(PortfolioParams(**{**dict(
        initial_capital_usdt=400.0, compounding="capped", compounding_cap_pct=50.0,
        profit_lock_step_usdt=50.0, profit_lock_share_pct=50.0,
        max_total_drawdown_pct=20.0,
    ), **overrides}))


def test_it_starts_with_exactly_what_you_gave_it():
    p = make()
    assert p.state.allocated == 400.0
    assert p.state.reserve == 0.0
    assert p.equity == 400.0


def test_a_winning_streak_does_not_make_the_next_position_bigger():
    """Sizing is a share of the trading balance, and the balance is capped."""
    p = make()
    for _ in range(40):
        p.on_trade_closed(10.0)
    # 400 USDT of profit, and the trading balance still cannot pass its ceiling.
    assert p.state.allocated <= p.allocation_ceiling()
    assert p.state.reserve > 0                            # the surplus went out of reach
    assert p.equity == pytest.approx(800.0)               # no profit was lost, only moved
    assert p.state.allocated / p.p.initial_capital_usdt <= 1.5


def test_profit_is_locked_away_in_steps():
    p = make()
    p.on_trade_closed(50.0)          # one full step of profit
    assert p.state.reserve == pytest.approx(25.0)
    assert p.state.allocated == pytest.approx(425.0)
    assert p.equity == pytest.approx(450.0)


def test_locking_only_happens_once_per_step():
    p = make()
    p.on_trade_closed(30.0)
    assert p.state.reserve == 0.0    # not a full step yet
    p.on_trade_closed(25.0)          # now past 50
    assert p.state.reserve == pytest.approx(25.0)
    p.on_trade_closed(5.0)           # still one step
    assert p.state.reserve == pytest.approx(25.0)


def test_losses_come_out_of_the_trading_balance_never_the_reserve():
    p = make()
    p.on_trade_closed(100.0)
    reserve = p.state.reserve
    assert reserve > 0
    p.on_trade_closed(-40.0)
    assert p.state.reserve == pytest.approx(reserve), "the reserve is untouchable"
    assert p.state.allocated < 400.0 + 100.0


def test_compounding_off_keeps_the_trading_balance_flat():
    p = make(compounding="off")
    for _ in range(10):
        p.on_trade_closed(20.0)
    # Profit never raises the trading balance above where it started; it all
    # ends up in the reserve. (It dips slightly below 400 in the moment a lock
    # fires, before the next trade tops it back up.)
    assert p.state.allocated <= 400.0
    assert p.equity == pytest.approx(600.0)
    assert p.state.reserve == pytest.approx(600.0 - p.state.allocated)


def test_full_compounding_has_no_ceiling():
    p = make(compounding="full", profit_lock_share_pct=0.0)
    for _ in range(10):
        p.on_trade_closed(50.0)
    assert p.state.allocated == pytest.approx(900.0)
    assert p.state.reserve == 0.0


def test_a_twenty_percent_drawdown_stops_trading_for_good():
    p = make()
    p.on_trade_closed(100.0)                 # peak equity 500
    assert p.state.peak_equity == pytest.approx(500.0)
    p.on_trade_closed(-101.0)                # equity 399 = -20.2% from peak
    assert p.state.halted
    assert "stopped" in p.state.halt_reason
    assert p.drawdown_pct < -20.0


def test_a_drawdown_just_short_of_the_ceiling_does_not_halt():
    p = make()
    p.on_trade_closed(100.0)
    p.on_trade_closed(-99.0)
    assert not p.state.halted


def test_equity_is_the_trading_balance_plus_the_reserve():
    p = make()
    p.on_trade_closed(120.0)
    assert p.equity == pytest.approx(p.state.allocated + p.state.reserve)
    assert p.equity == pytest.approx(520.0)
    assert p.growth_pct == pytest.approx(30.0)


def test_it_never_locks_the_account_down_to_nothing():
    p = make(initial_capital_usdt=60.0, profit_lock_step_usdt=1.0, profit_lock_share_pct=100.0)
    for _ in range(200):
        p.on_trade_closed(1.0)
    assert p.state.allocated >= 25.0, "there must always be enough left to place an order"


def test_state_survives_a_round_trip():
    p = make()
    p.on_trade_closed(75.0)
    p.on_trade_closed(-20.0)
    revived = Portfolio.from_dict(p.p, p.to_dict())
    assert revived.state.allocated == pytest.approx(p.state.allocated)
    assert revived.state.reserve == pytest.approx(p.state.reserve)
    assert revived.state.trades == p.state.trades


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        PortfolioParams(initial_capital_usdt=0).validate()
    with pytest.raises(ValueError):
        PortfolioParams(compounding="aggressive").validate()
    with pytest.raises(ValueError):
        PortfolioParams(max_total_drawdown_pct=0).validate()
    with pytest.raises(ValueError):
        PortfolioParams.from_dict({"initial_capital": 400})


def test_state_saved_for_a_different_account_size_is_discarded():
    """Resizing 400 -> 49 must not carry the old balance across; that reports
    money the account does not have and a fictional growth figure."""
    big = Portfolio(PortfolioParams(initial_capital_usdt=400.0))
    big.on_trade_closed(12.0)
    saved = big.to_dict()

    small = Portfolio.from_dict(PortfolioParams(initial_capital_usdt=49.0), saved)
    assert small.state.allocated == pytest.approx(49.0)
    assert small.state.trades == 0
    assert small.growth_pct == pytest.approx(0.0)


def test_state_saved_for_the_same_size_is_kept():
    p = Portfolio(PortfolioParams(initial_capital_usdt=49.0))
    p.on_trade_closed(3.0)
    revived = Portfolio.from_dict(PortfolioParams(initial_capital_usdt=49.0), p.to_dict())
    assert revived.state.trades == 1
    assert revived.state.allocated == pytest.approx(52.0)
