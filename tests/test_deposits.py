"""Top-ups: money paid into the wallet joins the trading balance.

The risk in doing this automatically is not the arithmetic, it is mistaking
something else for a deposit. Proceeds from a sale, a balance read mid-order,
fee dust -- each of them makes the wallet number move, and treating any of
them as new capital would size the next position off money that is not there.
"""
import os

import pytest

from trading.broker import Broker, Fill
from trading.config import Config, CryptoConfig
from trading.data import synthetic
from trading.engine import TradingEngine
from trading.portfolio import Portfolio, PortfolioParams
from trading.risk import RiskManager, RiskParams
from trading.state import StateStore
from trading.strategy import Strategy


def make(**overrides):
    return Portfolio(PortfolioParams(**{**dict(
        initial_capital_usdt=49.0, compounding="full", compounding_cap_pct=0,
        profit_lock_step_usdt=10.0, profit_lock_share_pct=50.0,
        max_total_drawdown_pct=55.0, auto_deposit=True, min_deposit_usdt=10.0,
    ), **overrides}))


# -- the accounting ---------------------------------------------------------

def test_a_top_up_is_capital_not_profit():
    p = make()
    p.on_deposit(51.0)
    assert p.state.allocated == pytest.approx(100.0)
    assert p.state.deposited == pytest.approx(51.0)
    assert p.basis == pytest.approx(100.0)
    # The account is bigger and has earned nothing. Growth must say so.
    assert p.growth_pct == pytest.approx(0.0)
    assert p.state.realised_pnl == pytest.approx(0.0)


def test_paying_into_a_losing_account_does_not_erase_the_drawdown():
    """Otherwise the drawdown ceiling switches off exactly when it is needed."""
    p = make()
    p.on_trade_closed(-19.0)
    assert p.drawdown_pct == pytest.approx(-38.78, abs=0.01)
    p.on_deposit(50.0)
    # 99 in, 80 left: still down about a fifth, not magically back to even.
    assert p.equity == pytest.approx(80.0)
    assert p.drawdown_pct == pytest.approx(-19.19, abs=0.01)
    assert p.growth_pct == pytest.approx(-19.19, abs=0.01)


def test_a_deposit_does_not_restart_a_halted_account():
    p = make()
    p.state.halted = True
    p.state.halt_reason = "drawdown ceiling"
    p.on_deposit(100.0)
    assert p.state.halted is True                 # restarting stays a manual act


def test_profit_after_a_top_up_is_measured_from_the_larger_basis():
    p = make()
    p.on_deposit(51.0)
    p.on_trade_closed(10.0)
    assert p.growth_pct == pytest.approx(10.0)    # +10 on 100 in, not on 49


def test_a_capped_ceiling_rises_with_the_deposit():
    p = make(compounding="capped", compounding_cap_pct=50.0)
    before = p.allocation_ceiling()
    p.on_deposit(51.0)
    assert p.allocation_ceiling() == pytest.approx(before * 100.0 / 49.0)


def test_a_deposit_must_be_positive():
    with pytest.raises(ValueError):
        make().on_deposit(0.0)


# -- the detection ----------------------------------------------------------

class WalletBroker(Broker):
    mode = "fake"

    def __init__(self, candles, price, balance, base=0.0):
        self._candles, self._price, self.balance = candles, price, balance
        self.base, self.sold = base, []

    def candles(self, symbol, timeframe, days):
        return self._candles

    def price(self, symbol):
        return self._price

    def free_quote_balance(self, symbol):
        return self.balance

    def free_base_balance(self, symbol):
        return self.base

    def market_buy(self, symbol, qty):
        return Fill(symbol, "buy", qty, self._price, 0.0, "b1")

    def market_sell(self, symbol, qty):
        self.sold.append(qty)
        self.base -= qty
        self.balance += qty * self._price
        return Fill(symbol, "sell", qty, self._price, 0.0, "s1")


def build(tmp_path, balance, portfolio=None, base=0.0, **cfg_overrides):
    cfg = Config(crypto=CryptoConfig(symbol="BTC/USDT", timeframe="1h", mode="paper"))
    cfg.risk = RiskParams(allocated_capital_usdt=1000.0, max_position_size_pct=10.0,
                          stop_loss_pct=2.0, take_profit_pct=4.0, max_open_positions=1,
                          max_daily_loss_pct=5.0, min_notional_usdt=10.0)
    portfolio = portfolio or make(initial_capital_usdt=1000.0)
    broker = WalletBroker(synthetic(bars=300, seed=5), 100.0, balance, base=base)
    state = StateStore(os.path.join(str(tmp_path), "state.json"))
    engine = TradingEngine(cfg, broker, Strategy(cfg.strategy), RiskManager(cfg.risk),
                           state, journal_path=os.path.join(str(tmp_path), "orders.jsonl"),
                           portfolio=portfolio)
    return engine, broker, portfolio


def test_a_top_up_is_only_believed_on_a_second_look(tmp_path):
    """One reading could be a balance still settling after a fill."""
    engine, _, pf = build(tmp_path, balance=1500.0)
    engine._reconcile_deposits()
    assert pf.state.deposited == 0.0              # seen once, not yet banked
    engine._reconcile_deposits()
    assert pf.state.deposited == pytest.approx(500.0)
    assert pf.state.allocated == pytest.approx(1500.0)


def test_only_the_amount_both_polls_agree_on_is_banked(tmp_path):
    engine, broker, pf = build(tmp_path, balance=1500.0)
    engine._reconcile_deposits()
    broker.balance = 1200.0                       # the first read was still settling
    engine._reconcile_deposits()
    assert pf.state.deposited == pytest.approx(200.0)


def test_proceeds_from_a_sale_are_not_a_deposit(tmp_path):
    """The winning trade is already booked into the balance before this runs."""
    engine, broker, pf = build(tmp_path, balance=1000.0)
    pf.on_trade_closed(100.0)                     # +100 realised, 50 locked away
    broker.balance = pf.state.allocated + pf.state.reserve
    engine._reconcile_deposits()
    engine._reconcile_deposits()
    assert pf.state.deposited == 0.0


def test_money_committed_to_an_open_position_is_not_missing_money(tmp_path):
    engine, broker, pf = build(tmp_path, balance=1000.0)
    engine.step()                                 # may open a position
    committed = sum(p.notional() for p in engine.state.positions)
    broker.balance = 1000.0 - committed           # wallet drops by what was spent
    engine._reconcile_deposits()
    engine._reconcile_deposits()
    assert pf.state.deposited == 0.0


def test_fee_dust_is_not_a_deposit(tmp_path):
    engine, _, pf = build(tmp_path, balance=1009.0)   # below the 10 USDT floor
    engine._reconcile_deposits()
    engine._reconcile_deposits()
    assert pf.state.deposited == 0.0


def test_nothing_happens_when_the_feature_is_off(tmp_path):
    pf = make(initial_capital_usdt=1000.0, auto_deposit=False)
    engine, _, pf = build(tmp_path, balance=5000.0, portfolio=pf)
    engine._reconcile_deposits()
    engine._reconcile_deposits()
    assert pf.state.deposited == 0.0


def test_a_top_up_reaches_the_risk_manager(tmp_path):
    """Sizing must use the new balance, or the deposit changes nothing."""
    engine, _, pf = build(tmp_path, balance=2000.0)
    engine._reconcile_deposits()
    engine._reconcile_deposits()
    assert engine.risk.p.allocated_capital_usdt == pytest.approx(pf.state.allocated)


def test_the_step_loop_runs_the_check(tmp_path):
    engine, _, pf = build(tmp_path, balance=1500.0)
    engine.step()
    engine.step()
    assert pf.state.deposited == pytest.approx(500.0)


# -- converting a top-up paid in the traded asset ---------------------------

def convert_pf(**kw):
    return make(initial_capital_usdt=1000.0, auto_convert_base=True, **kw)


def held_position(engine, qty, entry=100.0):
    from trading.risk import Position
    engine.state.add_position(Position(symbol="BTC/USDT", side="long", qty=qty,
                                       entry_price=entry, stop_price=entry * 0.7,
                                       take_profit_price=entry * 100,
                                       opened_at=0))


def test_it_sells_only_what_is_beyond_the_open_position(tmp_path):
    """The position is held in the same asset. Selling into it would close
    part of a live trade to bank a deposit."""
    engine, broker, _ = build(tmp_path, balance=0.0, portfolio=convert_pf(), base=3.0)
    held_position(engine, qty=2.0)                # 2 of the 3 coins are the trade
    engine._convert_base_deposits(100.0)
    engine._convert_base_deposits(100.0)
    assert len(broker.sold) == 1
    assert broker.sold[0] <= 1.0                  # never more than the surplus
    assert broker.base >= 2.0                     # the position survives intact


def test_a_conversion_is_only_made_on_a_second_look(tmp_path):
    engine, broker, _ = build(tmp_path, balance=0.0, portfolio=convert_pf(), base=1.0)
    engine._convert_base_deposits(100.0)
    assert broker.sold == []
    engine._convert_base_deposits(100.0)
    assert len(broker.sold) == 1


def test_nothing_is_sold_when_the_wallet_only_holds_the_position(tmp_path):
    engine, broker, _ = build(tmp_path, balance=0.0, portfolio=convert_pf(), base=2.0)
    held_position(engine, qty=2.0)
    engine._convert_base_deposits(100.0)
    engine._convert_base_deposits(100.0)
    assert broker.sold == []


def test_the_ledger_quantity_leans_away_from_the_position(tmp_path):
    """The ledger records the entry before its fee, so the wallet holds slightly
    less than it says. The surplus must come out understated, not over."""
    engine, broker, _ = build(tmp_path, balance=0.0, portfolio=convert_pf(), base=2.99938)
    held_position(engine, qty=2.0)                # wallet really holds 0.99938 spare
    engine._convert_base_deposits(100.0)
    engine._convert_base_deposits(100.0)
    assert broker.sold[0] < 0.99938
    assert broker.base > 2.0


def test_dust_is_left_alone(tmp_path):
    engine, broker, _ = build(tmp_path, balance=0.0, portfolio=convert_pf(), base=0.05)
    engine._convert_base_deposits(100.0)          # 5 USDT, under the 10 floor
    engine._convert_base_deposits(100.0)
    assert broker.sold == []


def test_conversion_is_off_unless_asked_for(tmp_path):
    engine, broker, _ = build(tmp_path, balance=0.0,
                              portfolio=make(initial_capital_usdt=1000.0), base=5.0)
    engine._convert_base_deposits(100.0)
    engine._convert_base_deposits(100.0)
    assert broker.sold == []


def test_the_proceeds_are_then_banked_as_a_top_up(tmp_path):
    """Sell, then the deposit check picks the USDT up on later polls."""
    pf = convert_pf()
    engine, broker, _ = build(tmp_path, balance=1000.0, portfolio=pf, base=1.0)
    for _ in range(4):
        engine._convert_base_deposits(100.0)
        engine._reconcile_deposits()
    assert broker.sold and pf.state.deposited > 90.0     # ~1 coin at 100, less dust
