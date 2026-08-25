"""When it pauses, and when it starts again. Every pause must end by itself."""
import pytest

from trading.portfolio import Portfolio, PortfolioParams
from trading.supervisor import DAY_MS, HOUR_MS, MINUTE_MS, Supervisor, SupervisorParams

MONDAY = 1_700_000_000_000   # a fixed instant, so day/week rollovers are deterministic


def make(**overrides):
    params = SupervisorParams(**{**dict(
        max_consecutive_losses=3, loss_streak_pause_hours=12.0, weekly_loss_pct=8.0,
        weekly_pause_days=3.0, daily_profit_target_pct=2.0, max_trades_per_day=4,
        stale_data_minutes=30.0, max_error_streak=5, error_streak_pause_minutes=15.0,
        max_volatility_pct=5.0,
    ), **overrides})
    return Supervisor(params, capital=400.0)


def test_a_fresh_supervisor_allows_trading():
    assert make().may_enter(MONDAY).allowed


def test_three_losses_in_a_row_pause_it_for_twelve_hours():
    s = make()
    for _ in range(3):
        s.record_trade(-1.0, MONDAY)
    verdict = s.may_enter(MONDAY)
    assert not verdict.allowed
    assert "3 losses in a row" in verdict.reason
    assert verdict.resume_at_ms == pytest.approx(MONDAY + 12 * HOUR_MS)


def test_a_pause_ends_by_itself():
    s = make()
    for _ in range(3):
        s.record_trade(-1.0, MONDAY)
    assert not s.may_enter(MONDAY).allowed
    later = MONDAY + 13 * HOUR_MS
    assert s.may_enter(later).allowed, "a pause must expire without anyone intervening"
    assert s.state.consecutive_losses == 0, "the streak that caused it must be cleared too"
    assert s.may_enter(later).allowed, "and it must not re-pause on the next check"


def test_a_win_resets_the_losing_streak():
    s = make(max_trades_per_day=99)
    s.record_trade(-1.0, MONDAY)
    s.record_trade(-1.0, MONDAY)
    s.record_trade(+2.0, MONDAY)
    s.record_trade(-1.0, MONDAY)
    assert s.state.consecutive_losses == 1
    assert s.may_enter(MONDAY).allowed


def test_it_stops_for_the_day_once_it_is_up_two_percent():
    s = make()
    s.record_trade(+8.0, MONDAY)          # 2% of 400
    verdict = s.may_enter(MONDAY)
    assert not verdict.allowed
    assert "stopping while ahead" in verdict.reason


def test_a_good_day_does_not_bleed_into_the_next():
    s = make()
    s.record_trade(+8.0, MONDAY)
    assert not s.may_enter(MONDAY).allowed
    assert s.may_enter(MONDAY + DAY_MS).allowed


def test_the_daily_trade_cap_is_enforced():
    s = make(daily_profit_target_pct=0.0)
    for _ in range(4):
        s.record_trade(+0.10, MONDAY)
    verdict = s.may_enter(MONDAY)
    assert not verdict.allowed
    assert "daily maximum" in verdict.reason
    assert s.may_enter(MONDAY + DAY_MS).allowed


def test_a_bad_week_pauses_for_three_days():
    s = make(max_consecutive_losses=99)
    s.record_trade(-33.0, MONDAY)          # past 8% of 400
    verdict = s.may_enter(MONDAY)
    assert not verdict.allowed
    assert "this week" in verdict.reason
    assert verdict.resume_at_ms == pytest.approx(MONDAY + 3 * DAY_MS)


def test_stale_data_stops_it_guessing():
    s = make()
    verdict = s.may_enter(MONDAY, data_age_minutes=45.0)
    assert not verdict.allowed
    assert "stale" in verdict.reason


def test_a_run_of_exchange_errors_pauses_it():
    s = make()
    for _ in range(5):
        s.record_error(MONDAY)
    verdict = s.may_enter(MONDAY)
    assert not verdict.allowed
    assert "errors in a row" in verdict.reason


def test_a_successful_pass_clears_the_error_streak():
    s = make()
    for _ in range(4):
        s.record_error(MONDAY)
    s.record_ok()
    assert s.may_enter(MONDAY).allowed


def test_it_sits_out_a_violent_market():
    s = make()
    assert s.may_enter(MONDAY, volatility_pct=3.0).allowed
    verdict = s.may_enter(MONDAY, volatility_pct=9.0)
    assert not verdict.allowed
    assert "too violent" in verdict.reason


def test_a_halted_portfolio_stops_everything():
    s = make()
    portfolio = Portfolio(PortfolioParams(initial_capital_usdt=400.0,
                                          max_total_drawdown_pct=20.0))
    portfolio.on_trade_closed(100.0)
    portfolio.on_trade_closed(-150.0)
    assert portfolio.state.halted
    verdict = s.may_enter(MONDAY, portfolio=portfolio)
    assert not verdict.allowed
    assert "halted" in verdict.reason


def test_manual_resume_clears_a_pause():
    s = make()
    for _ in range(3):
        s.record_trade(-1.0, MONDAY)
    assert not s.may_enter(MONDAY).allowed
    s.resume_now()
    assert s.may_enter(MONDAY).allowed


def test_a_disabled_supervisor_never_blocks():
    s = make(enabled=False)
    for _ in range(10):
        s.record_trade(-50.0, MONDAY)
    assert s.may_enter(MONDAY, volatility_pct=99.0, data_age_minutes=999.0).allowed


def test_state_survives_a_round_trip():
    s = make()
    s.record_trade(-1.0, MONDAY)
    s.record_trade(-1.0, MONDAY)
    revived = Supervisor.from_dict(s.p, 400.0, s.to_dict())
    assert revived.state.consecutive_losses == 2
    assert revived.state.pnl_today == pytest.approx(-2.0)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        SupervisorParams(max_trades_per_day=0).validate()
    with pytest.raises(ValueError):
        SupervisorParams(loss_streak_pause_hours=-1).validate()
    with pytest.raises(ValueError):
        SupervisorParams.from_dict({"max_consecutive_loss": 3})
