"""Bar-by-bar backtester.

It runs the same Strategy and the same RiskManager the live loop runs, over
historical candles. Two rules keep it honest:

  1. A signal computed from bar i's close is filled at bar i+1's open. It can
     never trade at a price it used to make the decision.
  2. Stops and targets are checked against each bar's high/low, and when one
     bar covers both, the stop is assumed to have filled first.

Fees and slippage are charged on every fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .analogues import EvidenceGate, feature_matrix
from .indicators import atr
from .portfolio import Portfolio
from .risk import Position, RiskManager
from .supervisor import Supervisor
from .strategy import BUY, HOLD, SELL, Candle, Signal, Strategy


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: int
    exit_time: int
    exit_reason: str
    fees: float
    pnl: float

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.qty
        return 0.0 if cost == 0 else self.pnl / cost * 100.0


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    bars: int
    start: int
    end: int
    starting_capital: float
    ending_equity: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    kill_switch_days: List[str] = field(default_factory=list)
    signals_seen: Dict[str, int] = field(default_factory=dict)
    evidence_blocked: int = 0
    evidence_passed: int = 0
    supervisor_blocks: int = 0
    pauses: int = 0
    locked_reserve: float = 0.0
    halted_reason: str = ""
    buy_hold_return_pct: float = 0.0
    bars_in_market: int = 0
    deployed_pct: float = 0.0

    # -- metrics ---------------------------------------------------------
    @property
    def total_return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.ending_equity - self.starting_capital) / self.starting_capital * 100.0

    @property
    def time_in_market_pct(self) -> float:
        return 0.0 if not self.bars else self.bars_in_market / self.bars * 100.0

    @property
    def wins(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losses(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate_pct(self) -> float:
        return 0.0 if not self.trades else len(self.wins) / len(self.trades) * 100.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.wins)
        gross_loss = -sum(t.pnl for t in self.losses)
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        peak = float("-inf")
        worst = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = min(worst, (equity - peak) / peak * 100.0)
        return worst

    @property
    def worst_trade_pct(self) -> float:
        return min([t.return_pct for t in self.trades], default=0.0)

    def sharpe(self, bars_per_year: float) -> float:
        rets = []
        for prev, cur in zip(self.equity_curve, self.equity_curve[1:]):
            if prev > 0:
                rets.append((cur - prev) / prev)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        if sd == 0:
            return 0.0
        return mean / sd * math.sqrt(bars_per_year)

    def summary(self, bars_per_year: float = 8760.0) -> str:
        def ts(ms: int) -> str:
            return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")

        lines = [
            "Backtest  {}  {}  {} bars  {} -> {} UTC".format(
                self.symbol, self.timeframe, self.bars, ts(self.start), ts(self.end)
            ),
            "-" * 68,
            "Starting capital     {:>12.2f}".format(self.starting_capital),
            "Ending equity        {:>12.2f}".format(self.ending_equity),
            "Total return         {:>11.2f}%".format(self.total_return_pct),
            "Buy & hold return    {:>11.2f}%".format(self.buy_hold_return_pct),
            "Max drawdown         {:>11.2f}%".format(self.max_drawdown_pct),
            "Time in market       {:>11.1f}%  (only {:.0f}% of capital is deployed per trade)".format(
                self.time_in_market_pct, self.deployed_pct
            ),
            "Sharpe (annualised)  {:>12.2f}".format(self.sharpe(bars_per_year)),
            "-" * 68,
            "Trades               {:>12d}".format(len(self.trades)),
            "Win rate             {:>11.1f}%  ({} win / {} loss)".format(
                self.win_rate_pct, len(self.wins), len(self.losses)
            ),
            "Profit factor        {:>12.2f}".format(self.profit_factor),
            "Worst single trade   {:>11.2f}%".format(self.worst_trade_pct),
            "Total fees paid      {:>12.2f}".format(sum(t.fees for t in self.trades)),
        ]
        if self.signals_seen:
            lines.append("Signals              {}".format(
                "  ".join("{}={}".format(k, v) for k, v in sorted(self.signals_seen.items()))
            ))
        by_reason: Dict[str, int] = {}
        for t in self.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        if by_reason:
            lines.append("Exits                {}".format(
                "  ".join("{}={}".format(k, v) for k, v in sorted(by_reason.items()))
            ))
        if self.supervisor_blocks:
            lines.append("Supervisor           {} entries held back, {} pauses".format(
                self.supervisor_blocks, self.pauses))
        if self.locked_reserve:
            lines.append("Locked reserve       {:>12.2f}  (profit moved out of reach)".format(
                self.locked_reserve))
        if self.halted_reason:
            lines.append("HALTED               {}".format(self.halted_reason))
        if self.evidence_blocked or self.evidence_passed:
            lines.append("Evidence gate        {} entries backed, {} blocked".format(
                self.evidence_passed, self.evidence_blocked
            ))
        if self.kill_switch_days:
            lines.append("Kill-switch tripped  {} day(s): {}".format(
                len(self.kill_switch_days), ", ".join(self.kill_switch_days)
            ))
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        risk: RiskManager,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        evidence: Optional[EvidenceGate] = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk
        self.fee_rate = fee_bps / 10_000.0
        self.slippage = slippage_bps / 10_000.0
        self.evidence = evidence

    def run(
        self,
        candles: Sequence[Candle],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        on_signal=None,
        portfolio: Optional[Portfolio] = None,
        supervisor: Optional[Supervisor] = None,
    ) -> BacktestResult:
        warmup = self.strategy.warmup_bars
        if len(candles) <= warmup + 2:
            raise ValueError(
                "need more than {} candles for this strategy, got {}".format(warmup + 2, len(candles))
            )

        # Situation vectors for the whole run. feature_matrix is causal (proven
        # in tests/test_analogues.py), so computing it once is the same as
        # computing it fresh at every bar -- and 2000x faster.
        situations = None
        situations_valid = None
        if self.evidence is not None:
            situations, situations_valid = feature_matrix(candles)

        # One indicator pass for the whole run instead of one per bar.
        cache = self.strategy.prepare(candles)

        evidence_blocked = 0
        evidence_passed = 0
        supervisor_blocks = 0

        # Volatility for the supervisor, computed once over the whole run.
        volatility = None
        if supervisor is not None:
            atr_series = atr([c.high for c in candles], [c.low for c in candles],
                             [c.close for c in candles], 14)
            volatility = [None if (v is None or c.close <= 0) else v / c.close * 100.0
                          for v, c in zip(atr_series, candles)]

        if portfolio is not None:
            self.risk.set_allocated(portfolio.state.allocated)
        capital = self.risk.p.allocated_capital_usdt
        cash = capital
        position: Optional[Position] = None
        position_entry_fee = 0.0
        pending: Optional[Signal] = None
        trades: List[Trade] = []
        equity_curve: List[float] = []
        signals_seen: Dict[str, int] = {BUY: 0, SELL: 0, HOLD: 0}
        realised_today = 0.0
        cooldown_until = -1   # bar index before which no new entry may open
        bars_in_market = 0
        current_day = _day_of(candles[warmup].timestamp)
        kill_days: List[str] = []

        for i in range(warmup, len(candles)):
            bar = candles[i]
            day = _day_of(bar.timestamp)
            if day != current_day:
                current_day = day
                realised_today = 0.0

            # 1. Fill anything the previous bar decided, at this bar's open.
            if pending is not None:
                allowed = True
                if pending.action == BUY and supervisor is not None:
                    verdict = supervisor.may_enter(
                        bar.timestamp,
                        volatility_pct=volatility[i] if volatility else None,
                        portfolio=portfolio,
                    )
                    allowed = verdict.allowed
                    if not allowed:
                        supervisor_blocks += 1
                if portfolio is not None and portfolio.state.halted:
                    allowed = False

                if pending.action == BUY and position is None and i >= cooldown_until and allowed:
                    decision = self.risk.review_entry(
                        price=bar.open,
                        open_positions=[],
                        realised_pnl_today=realised_today,
                        free_balance=portfolio.state.allocated if portfolio else cash,
                        symbol=symbol,
                    )
                    if decision.approved:
                        fill = bar.open * (1.0 + self.slippage)
                        qty = decision.notional / fill
                        fee = decision.notional * self.fee_rate
                        cash -= decision.notional + fee
                        levels = self.risk.exit_levels(fill, "long")
                        position = Position(
                            symbol=symbol, side="long", qty=qty, entry_price=fill,
                            stop_price=levels["stop_price"],
                            take_profit_price=levels["take_profit_price"],
                            opened_at=bar.timestamp,
                        )
                        position_entry_fee = fee
                    elif decision.reason.startswith("daily loss kill-switch"):
                        if day not in kill_days:
                            kill_days.append(day)
                elif pending.action == SELL and position is not None:
                    fill = bar.open * (1.0 - self.slippage)
                    cash, trade = self._close(position, fill, bar.timestamp, "signal", cash, position_entry_fee)
                    realised_today += trade.pnl
                    trades.append(trade)
                    self._book(trade, bar.timestamp, portfolio, supervisor)
                    position = None
                    cooldown_until = i + 1 + self.risk.p.reentry_cooldown_bars
                pending = None

            # 2. Stops and targets, against this bar's range.
            if position is not None:
                hit = self.risk.check_exit(position, bar.high, bar.low)
                if hit is not None:
                    level = position.stop_price if hit == "stop_loss" else position.take_profit_price
                    # Gaps fill at the open, not at the untouched level.
                    fill = min(level, bar.open) if hit == "stop_loss" else max(level, bar.open)
                    fill *= (1.0 - self.slippage)
                    cash, trade = self._close(position, fill, bar.timestamp, hit, cash, position_entry_fee)
                    realised_today += trade.pnl
                    trades.append(trade)
                    self._book(trade, bar.timestamp, portfolio, supervisor)
                    position = None
                    cooldown_until = i + 1 + self.risk.p.reentry_cooldown_bars
                    if self.risk.kill_switch_tripped(realised_today) and day not in kill_days:
                        kill_days.append(day)

            # 3. Decide from this bar's close; it fills next bar.
            signal = self.strategy.evaluate(candles, i, cache=cache)
            signals_seen[signal.action] = signals_seen.get(signal.action, 0) + 1
            if on_signal is not None:
                on_signal(i, signal, position)
            if signal.action == BUY and self.evidence is not None:
                # Only situations that had already resolved by this bar count,
                # so the gate can never consult its own future.
                vector = situations[i] if situations_valid[i] else None
                verdict = self.evidence.check(vector, as_of_ms=bar.timestamp)
                if verdict.passed:
                    evidence_passed += 1
                else:
                    evidence_blocked += 1
                    signal = Signal(action=HOLD, score=signal.score, price=signal.price,
                                    timestamp=signal.timestamp, votes=signal.votes,
                                    reasons=signal.reasons + ["evidence gate: " + verdict.reason],
                                    indicators=signal.indicators)

            if signal.action in (BUY, SELL):
                pending = signal

            if position is not None:
                bars_in_market += 1
            equity = cash + (position.qty * bar.close if position else 0.0)
            equity_curve.append(equity)

        # Mark any open position out at the last close.
        last = candles[-1]
        if position is not None:
            cash, trade = self._close(
                position, last.close * (1.0 - self.slippage), last.timestamp,
                "end_of_data", cash, position_entry_fee,
            )
            trades.append(trade)
            self._book(trade, last.timestamp, portfolio, supervisor)
            position = None
            equity_curve[-1] = cash

        first_close = candles[warmup].close
        bh = (last.close - first_close) / first_close * 100.0

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            bars=len(candles) - warmup,
            start=candles[warmup].timestamp,
            end=last.timestamp,
            starting_capital=capital,
            ending_equity=cash,
            trades=trades,
            equity_curve=equity_curve,
            kill_switch_days=kill_days,
            signals_seen=signals_seen,
            buy_hold_return_pct=bh,
            bars_in_market=bars_in_market,
            deployed_pct=self.risk.p.max_position_size_pct,
            evidence_blocked=evidence_blocked,
            evidence_passed=evidence_passed,
            supervisor_blocks=supervisor_blocks,
            pauses=supervisor.state.pauses if supervisor else 0,
            locked_reserve=portfolio.state.reserve if portfolio else 0.0,
            halted_reason=portfolio.state.halt_reason if portfolio else "",
        )

    def _book(self, trade: Trade, when: int, portfolio, supervisor) -> None:
        """Tell the portfolio and the supervisor what just happened."""
        if supervisor is not None:
            supervisor.record_trade(trade.pnl, when)
        if portfolio is not None:
            portfolio.on_trade_closed(trade.pnl, when)
            if not portfolio.state.halted:
                self.risk.set_allocated(max(portfolio.state.allocated, 1e-6))

    def _close(self, position: Position, fill: float, when: int, reason: str, cash: float, entry_fee: float):
        proceeds = position.qty * fill
        fee = proceeds * self.fee_rate
        cash += proceeds - fee
        pnl = proceeds - fee - (position.qty * position.entry_price) - entry_fee
        trade = Trade(
            symbol=position.symbol, side=position.side, qty=position.qty,
            entry_price=position.entry_price, exit_price=fill,
            entry_time=position.opened_at, exit_time=when,
            exit_reason=reason, fees=entry_fee + fee, pnl=pnl,
        )
        return cash, trade


def _day_of(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
