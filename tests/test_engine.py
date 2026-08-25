"""Engine wiring: the loop must not be able to reach the exchange around the
risk manager, and --dry-run must place nothing."""
import os
import time

import pytest

from trading.broker import Broker, Fill
from trading.config import Config, CryptoConfig
from trading.data import synthetic
from trading.engine import TradingEngine
from trading.risk import Position, RiskManager, RiskParams
from trading.state import StateStore
from trading.strategy import BUY, SELL, Strategy, StrategyParams


class FakeBroker(Broker):
    mode = "fake"

    def __init__(self, candles, price, balance=10_000.0):
        self._candles = candles
        self._price = price
        self._balance = balance
        self.orders = []

    def candles(self, symbol, timeframe, days):
        return self._candles

    def price(self, symbol):
        return self._price

    def free_quote_balance(self, symbol):
        return self._balance

    def market_buy(self, symbol, qty):
        self.orders.append(("buy", symbol, qty))
        return Fill(symbol, "buy", qty, self._price, qty * self._price * 0.001, "fake-1")

    def market_sell(self, symbol, qty):
        self.orders.append(("sell", symbol, qty))
        return Fill(symbol, "sell", qty, self._price, qty * self._price * 0.001, "fake-2")


def build(tmp_path, candles, price, dry_run=False, **risk_overrides):
    cfg = Config(crypto=CryptoConfig(symbol="BTC/USDT", timeframe="1h", mode="paper"))
    cfg.risk = RiskParams(**{**dict(
        allocated_capital_usdt=1000.0, max_position_size_pct=10.0, stop_loss_pct=2.0,
        take_profit_pct=4.0, max_open_positions=1, max_daily_loss_pct=5.0,
        min_notional_usdt=10.0,
    ), **risk_overrides})
    broker = FakeBroker(candles, price)
    state = StateStore(os.path.join(str(tmp_path), "state.json"))
    engine = TradingEngine(cfg, broker, Strategy(cfg.strategy), RiskManager(cfg.risk),
                           state, dry_run=dry_run,
                           journal_path=os.path.join(str(tmp_path), "orders.jsonl"))
    return engine, broker, state


def buy_candles():
    candles = synthetic(bars=600, seed=11, drift=0.0004, volatility=0.012)
    s = Strategy()
    for i in range(len(candles) - 1, s.warmup_bars, -1):
        if s.evaluate(candles, i).action == BUY:
            return candles[: i + 1]
    raise AssertionError("no BUY bar in the fixture data")


def test_a_buy_signal_opens_a_sized_position_with_a_stop(tmp_path):
    candles = buy_candles()
    engine, broker, state = build(tmp_path, candles, price=100.0)
    engine.step()

    assert broker.orders and broker.orders[0][0] == "buy"
    qty = broker.orders[0][2]
    assert qty * 100.0 == pytest.approx(100.0)      # 10% of 1000 USDT
    held = state.positions[0]
    assert held.stop_price == pytest.approx(98.0)
    assert held.take_profit_price == pytest.approx(104.0)


def test_dry_run_decides_but_places_nothing(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=100.0, dry_run=True)
    engine.step()
    assert broker.orders == []
    assert state.positions == []


def test_the_kill_switch_blocks_the_next_entry(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=100.0)
    state.record_pnl(-50.0)                          # exactly the 5% daily limit
    engine.step()
    assert broker.orders == []


def test_a_stop_breach_exits_before_anything_else_is_considered(tmp_path):
    candles = buy_candles()
    engine, broker, state = build(tmp_path, candles, price=97.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert broker.orders[0][0] == "sell"
    assert state.positions == []
    assert state.realised_pnl_today < 0


def test_a_target_breach_takes_profit(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=105.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert broker.orders[0][0] == "sell"
    assert state.realised_pnl_today > 0


def test_it_will_not_stack_a_second_position_on_the_same_symbol(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=101.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert all(order[0] != "buy" for order in broker.orders)
    assert len(state.positions) == 1


def test_state_survives_a_restart(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=100.0)
    engine.step()
    reloaded = StateStore(state.path)
    assert len(reloaded.positions) == 1
    assert reloaded.positions[0].symbol == "BTC/USDT"


def test_orders_are_journalled(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=100.0)
    engine.step()
    with open(engine.journal_path) as fh:
        lines = fh.read().strip().splitlines()
    assert lines and '"entry"' in lines[0]


def test_a_stopped_out_position_is_not_immediately_bought_back(tmp_path):
    """Without a cooldown the loop would stop out and re-enter in the same pass,
    at the price it just took a loss at."""
    engine, broker, state = build(tmp_path, buy_candles(), price=97.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert [o[0] for o in broker.orders] == ["sell"]
    assert state.positions == []
    assert engine._cooldown_remaining() > 0

    engine.step()                     # next pass, still inside the cooldown
    assert [o[0] for o in broker.orders] == ["sell"]


def test_the_cooldown_expires(tmp_path):
    engine, broker, state = build(tmp_path, buy_candles(), price=97.0, reentry_cooldown_bars=0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert engine._cooldown_remaining() == 0


def recent(candles, step_ms=3_600_000):
    """Rebase a fixture so its last bar has just closed.

    The supervisor pauses on stale data, which is correct behaviour and would
    otherwise block every one of these tests.
    """
    from trading.strategy import Candle
    end = int(time.time() * 1000) - step_ms
    start = end - (len(candles) - 1) * step_ms
    return [Candle(start + i * step_ms, c.open, c.high, c.low, c.close, c.volume)
            for i, c in enumerate(candles)]


def build_account(tmp_path, candles, price, **overrides):
    """An engine wired to a real portfolio and supervisor."""
    from trading.portfolio import Portfolio, PortfolioParams
    from trading.supervisor import Supervisor, SupervisorParams

    engine, broker, state = build(tmp_path, recent(candles), price, **overrides)
    portfolio = Portfolio(PortfolioParams(initial_capital_usdt=400.0,
                                          profit_lock_step_usdt=50.0,
                                          profit_lock_share_pct=50.0))
    supervisor = Supervisor(SupervisorParams(max_consecutive_losses=2,
                                             daily_profit_target_pct=2.0,
                                             max_trades_per_day=3), capital=400.0)
    engine.portfolio = portfolio
    engine.supervisor = supervisor
    engine.risk.set_allocated(portfolio.state.allocated)
    return engine, broker, state, portfolio, supervisor


def test_position_size_follows_the_portfolio_not_the_config(tmp_path):
    engine, broker, state, portfolio, _ = build_account(tmp_path, buy_candles(), price=100.0)
    engine.step()
    qty = broker.orders[0][2]
    assert qty * 100.0 == pytest.approx(40.0)   # 10% of the 400 portfolio, not the 1000 config


def test_a_closed_trade_is_booked_into_the_portfolio(tmp_path):
    engine, broker, state, portfolio, supervisor = build_account(
        tmp_path, buy_candles(), price=105.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()
    assert portfolio.state.trades == 1
    assert portfolio.state.wins == 1
    assert portfolio.equity > 400.0
    assert supervisor.state.trades_today == 1


def test_the_supervisor_can_stop_the_engine_entering(tmp_path):
    engine, broker, state, portfolio, supervisor = build_account(
        tmp_path, buy_candles(), price=100.0)
    supervisor.record_trade(-1.0, int(time.time() * 1000))
    supervisor.record_trade(-1.0, int(time.time() * 1000))   # hits max_consecutive_losses=2
    engine.step()
    assert broker.orders == [], "a paused supervisor must block the entry"


def test_a_halted_portfolio_blocks_new_entries(tmp_path):
    engine, broker, state, portfolio, _ = build_account(tmp_path, buy_candles(), price=100.0)
    portfolio.state.halted = True
    portfolio.state.halt_reason = "drawdown ceiling"
    engine.step()
    assert broker.orders == []


def test_the_account_survives_a_restart(tmp_path):
    engine, broker, state, portfolio, supervisor = build_account(
        tmp_path, buy_candles(), price=105.0)
    state.add_position(Position("BTC/USDT", "long", 1.0, 100.0, 98.0, 104.0, 0))
    engine.step()

    from trading.portfolio import Portfolio
    from trading.supervisor import Supervisor
    reloaded_state = StateStore(state.path)
    revived = Portfolio.from_dict(portfolio.p, reloaded_state.load_component("portfolio"))
    revived_sup = Supervisor.from_dict(supervisor.p, 400.0,
                                       reloaded_state.load_component("supervisor"))
    assert revived.state.trades == 1
    assert revived.equity == pytest.approx(portfolio.equity)
    assert revived_sup.state.trades_today == 1


def test_stale_market_data_stops_the_engine_trading(tmp_path):
    """The fixture's timestamps are years old; the supervisor must notice."""
    from trading.portfolio import Portfolio, PortfolioParams
    from trading.supervisor import Supervisor, SupervisorParams

    engine, broker, state = build(tmp_path, buy_candles(), price=100.0)
    engine.portfolio = Portfolio(PortfolioParams(initial_capital_usdt=400.0))
    engine.supervisor = Supervisor(SupervisorParams(stale_data_minutes=30.0), capital=400.0)
    engine.step()
    assert broker.orders == []
    assert "stale" in engine.supervisor.state.pause_reason
