"""The trading loop.

One pass = read closed candles, manage the open position, then consider a new
one. Every order goes through RiskManager first; the loop has no path to an
exchange that skips it.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from .analogues import EvidenceGate, situation_at
from .broker import Broker, Fill
from .config import Config
from .risk import Position, RiskManager
from .portfolio import Portfolio
from .state import StateStore
from .strategy import BUY, SELL, Signal, Strategy
from .supervisor import Supervisor

log = logging.getLogger("trading.engine")


class TradingEngine:
    def __init__(
        self,
        config: Config,
        broker: Broker,
        strategy: Strategy,
        risk: RiskManager,
        state: StateStore,
        dry_run: bool = False,
        journal_path: Optional[str] = None,
        evidence: Optional[EvidenceGate] = None,
        portfolio: Optional[Portfolio] = None,
        supervisor: Optional[Supervisor] = None,
    ) -> None:
        self.cfg = config
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.state = state
        self.dry_run = dry_run
        self.journal_path = journal_path
        self.evidence = evidence
        self.portfolio = portfolio
        self.supervisor = supervisor
        self.symbol = config.crypto.symbol
        self.timeframe = config.crypto.timeframe
        if portfolio is not None:
            # Sizing follows the account, not a number frozen in a config file.
            self.risk.set_allocated(portfolio.state.allocated)
            self.state.save_component("portfolio", portfolio.to_dict())

    # -- one pass --------------------------------------------------------
    def step(self) -> Signal:
        history_days = self._history_days()
        candles = self.broker.candles(self.symbol, self.timeframe, history_days)
        if len(candles) < self.strategy.warmup_bars + 1:
            raise RuntimeError(
                "only {} closed candles available, strategy needs {}".format(
                    len(candles), self.strategy.warmup_bars + 1
                )
            )
        signal = self.strategy.evaluate(candles)
        price = self.broker.price(self.symbol)
        self._last_candles = candles

        open_position = self._open_position()
        if open_position is not None:
            self._manage_open(open_position, price, signal)
            open_position = self._open_position()

        if open_position is None and signal.action == BUY and not self._supervisor_allows(candles):
            return signal

        if open_position is None and signal.action == BUY:
            waiting = self._cooldown_remaining()
            if waiting > 0:
                # Re-entering right after a stop-out just pays the spread to
                # take the same losing trade again.
                log.info("BUY held back: %ds of re-entry cooldown left", waiting)
            elif not self._evidence_backs_it(candles):
                pass          # already logged, with the numbers
            else:
                self._consider_entry(signal, price)

        log.info("%s %s | price %.2f | %s", self.symbol, signal.action, price,
                 "; ".join(signal.reasons[:3]))
        return signal

    def run_forever(self, poll_seconds: Optional[int] = None) -> None:
        wait = poll_seconds or self.cfg.crypto.poll_seconds
        log.info("engine start | mode=%s symbol=%s timeframe=%s dry_run=%s poll=%ss",
                 self.broker.mode, self.symbol, self.timeframe, self.dry_run, wait)
        while True:
            try:
                self.step()
                if self.supervisor is not None:
                    self.supervisor.record_ok()
            except KeyboardInterrupt:
                log.info("stopped by user")
                raise
            except Exception as exc:  # a bad poll must not kill a running bot
                from .broker import explain_auth_error
                hint = explain_auth_error(exc, self.cfg.crypto.exchange)
                if hint:
                    log.error("%s", hint)
                else:
                    log.exception("pass failed, retrying next poll: %s", exc)
                if self.supervisor is not None:
                    self.supervisor.record_error(int(time.time() * 1000))
                    self._persist_supervisor()
            time.sleep(wait)

    # -- position management ---------------------------------------------
    def _manage_open(self, position: Position, price: float, signal: Signal) -> None:
        reason = None
        if price <= position.stop_price:
            reason = "stop_loss"
        elif price >= position.take_profit_price:
            reason = "take_profit"
        elif signal.action == SELL:
            reason = "signal"
        if reason is None:
            self._ensure_stop(position)
            log.info("holding %s: %.6f @ %.2f (stop %.2f / target %.2f, unrealised %+.2f)",
                     position.symbol, position.qty, position.entry_price,
                     position.stop_price, position.take_profit_price,
                     position.unrealised(price))
            return
        self._close(position, price, reason)

    def _close(self, position: Position, price: float, reason: str) -> None:
        if self.dry_run:
            log.info("[dry-run] would SELL %.6f %s (%s)", position.qty, position.symbol, reason)
            return
        # Cancel the resting stop first, or the exchange still holds the coins
        # and the sell is rejected for insufficient balance.
        if position.stop_order_id:
            self.broker.cancel(position.symbol, position.stop_order_id)
        fill: Fill = self.broker.market_sell(position.symbol, position.qty)
        pnl = (fill.price - position.entry_price) * fill.qty - fill.fee
        self.state.remove_position(position.symbol)
        self.state.record_pnl(pnl, {
            "at": _now(), "event": "exit", "symbol": position.symbol, "reason": reason,
            "qty": fill.qty, "entry": position.entry_price, "exit": fill.price, "pnl": pnl,
        })
        self._start_cooldown()
        now = int(time.time() * 1000)
        if self.supervisor is not None:
            self.supervisor.record_trade(pnl, now)
            self._persist_supervisor()
        if self.portfolio is not None:
            book = self.portfolio.on_trade_closed(pnl, now)
            self._persist_portfolio()
            if book["locked"]:
                log.info("locked %.2f of profit into the reserve (now %.2f, untouchable)",
                         book["locked"], self.portfolio.state.reserve)
            if book["spilled_to_reserve"]:
                log.info("trading balance hit its compounding ceiling; %.2f spilled to reserve",
                         book["spilled_to_reserve"])
            if self.portfolio.state.halted:
                log.error("PORTFOLIO HALT: %s", self.portfolio.state.halt_reason)
            else:
                self.risk.set_allocated(max(self.portfolio.state.allocated, 1e-6))
        log.info("EXIT %s %.6f @ %.2f (%s) pnl %+.2f | realised today %+.2f",
                 position.symbol, fill.qty, fill.price, reason, pnl, self.state.realised_pnl_today)
        self._journal({"event": "exit", "reason": reason, "symbol": position.symbol,
                       "qty": fill.qty, "price": fill.price, "pnl": pnl, "mode": self.broker.mode})
        if self.risk.kill_switch_tripped(self.state.realised_pnl_today):
            log.warning("DAILY LOSS KILL-SWITCH TRIPPED: %.2f lost today, limit %.2f. "
                        "No new entries until UTC midnight.",
                        -self.state.realised_pnl_today, self.risk.daily_loss_limit())

    def _consider_entry(self, signal: Signal, price: float) -> None:
        try:
            free = self.broker.free_quote_balance(self.symbol)
        except Exception as exc:
            log.warning("could not read balance (%s); sizing from allocated capital only", exc)
            free = None

        decision = self.risk.review_entry(
            price=price,
            open_positions=self.state.positions,
            realised_pnl_today=self.state.realised_pnl_today,
            free_balance=free,
            symbol=self.symbol,
        )
        if not decision.approved:
            log.info("BUY blocked by risk manager: %s", decision.reason)
            self._journal({"event": "blocked", "symbol": self.symbol, "reason": decision.reason,
                           "price": price, "mode": self.broker.mode})
            return

        if self.dry_run:
            log.info("[dry-run] would BUY %.6f %s @ %.2f -- %s",
                     decision.qty, self.symbol, price, decision.reason)
            return

        fill = self.broker.market_buy(self.symbol, decision.qty)
        levels = self.risk.exit_levels(fill.price, "long")
        position = Position(
            symbol=self.symbol, side="long", qty=fill.qty, entry_price=fill.price,
            stop_price=levels["stop_price"], take_profit_price=levels["take_profit_price"],
            opened_at=int(time.time() * 1000),
        )
        # Protection that survives this process dying.
        position.stop_order_id = self.broker.place_stop(
            self.symbol, fill.qty, position.stop_price)
        if position.stop_order_id:
            log.info("resting stop on the exchange at %.2f (order %s)",
                     position.stop_price, position.stop_order_id)
        else:
            log.warning("no exchange-side stop for %s -- the position is only "
                        "protected while this process is running", self.symbol)

        self.state.add_position(position)
        self._persist_portfolio()   # so a restart mid-position sees the account
        self.state.record_pnl(-fill.fee, {
            "at": _now(), "event": "entry", "symbol": self.symbol, "qty": fill.qty,
            "price": fill.price, "fee": fill.fee,
        })
        log.info("ENTRY %s %.6f @ %.2f | stop %.2f target %.2f | max loss %.2f",
                 self.symbol, fill.qty, fill.price, position.stop_price,
                 position.take_profit_price, decision.max_loss_usdt)
        self._journal({"event": "entry", "symbol": self.symbol, "qty": fill.qty,
                       "price": fill.price, "stop": position.stop_price,
                       "target": position.take_profit_price, "score": signal.score,
                       "mode": self.broker.mode})

    # -- helpers ---------------------------------------------------------
    def _supervisor_allows(self, candles) -> bool:
        """Ask the supervisor whether trading should be happening at all now."""
        if self.supervisor is None:
            return True
        now = int(time.time() * 1000)
        verdict = self.supervisor.may_enter(
            now,
            volatility_pct=self._volatility_pct(candles),
            data_age_minutes=self._data_age_minutes(candles),
            portfolio=self.portfolio,
        )
        self._persist_supervisor()
        if verdict.allowed:
            return True
        wait = verdict.wait_text(now)
        log.info("paused: %s%s", verdict.reason, " ({})".format(wait) if wait else "")
        self._journal({"event": "paused", "symbol": self.symbol, "reason": verdict.reason,
                       "resume_at_ms": verdict.resume_at_ms, "mode": self.broker.mode})
        return False

    def _volatility_pct(self, candles) -> Optional[float]:
        from .indicators import atr
        if len(candles) < 20:
            return None
        window = candles[-100:]
        series = atr([c.high for c in window], [c.low for c in window],
                     [c.close for c in window], 14)
        value = series[-1]
        last_close = window[-1].close
        if value is None or last_close <= 0:
            return None
        return value / last_close * 100.0

    def _data_age_minutes(self, candles) -> Optional[float]:
        if not candles:
            return None
        from .data import timeframe_ms
        closed_at = candles[-1].timestamp + timeframe_ms(self.timeframe)
        return max(0.0, (time.time() * 1000 - closed_at) / 60_000.0)

    def _persist_supervisor(self) -> None:
        if self.supervisor is not None:
            self.state.save_component("supervisor", self.supervisor.to_dict())

    def _persist_portfolio(self) -> None:
        if self.portfolio is not None:
            self.state.save_component("portfolio", self.portfolio.to_dict())

    def _evidence_backs_it(self, candles) -> bool:
        """Ask the analogue library before risking anything on a fresh signal."""
        if self.evidence is None or not self.evidence.params.enabled:
            return True
        verdict = self.evidence.check(situation_at(candles), as_of_ms=None)
        if verdict.passed:
            log.info("evidence: %s", verdict.explain())
            return True
        log.info("BUY blocked by the evidence gate: %s", verdict.explain())
        self._journal({"event": "blocked", "symbol": self.symbol, "reason": "evidence",
                       "matches": verdict.matches, "success_rate": verdict.success_rate,
                       "lower_bound": verdict.lower_bound, "detail": verdict.reason,
                       "mode": self.broker.mode})
        return False

    def _ensure_stop(self, position: Position) -> None:
        """Re-rest the stop if it is missing -- it may have been filled,
        cancelled by hand, or lost when the exchange reset."""
        if self.dry_run:
            return
        live = self.broker.open_order_ids(position.symbol)
        if position.stop_order_id and position.stop_order_id in live:
            return
        if position.stop_order_id and not live:
            # Could not read the book; do not stack a second stop on a guess.
            return
        # The stop may be further from the market than the exchange will accept
        # a resting order (Binance caps it at 20% below). Retrying each poll is
        # right -- it becomes placeable as price falls toward it -- but the
        # warning should not repeat every pass.
        now = time.time()
        quiet = now - getattr(self, "_last_stop_warning", 0.0) < 3600
        if quiet:
            logging.getLogger("trading.broker").setLevel(logging.ERROR)
        try:
            new_id = self.broker.place_stop(position.symbol, position.qty, position.stop_price)
        finally:
            logging.getLogger("trading.broker").setLevel(logging.NOTSET)
        if not new_id and not quiet:
            self._last_stop_warning = now
        if new_id:
            log.info("re-rested a missing stop for %s at %.2f (order %s)",
                     position.symbol, position.stop_price, new_id)
            self.state.remove_position(position.symbol)
            position.stop_order_id = new_id
            self.state.add_position(position)

    def _start_cooldown(self) -> None:
        from .data import timeframe_ms
        bars = self.risk.p.reentry_cooldown_bars
        if bars <= 0:
            return
        until = int(time.time() * 1000) + bars * timeframe_ms(self.timeframe)
        self.state.data["cooldown_until_ms"] = until
        self.state.save()

    def _cooldown_remaining(self) -> int:
        until = int(self.state.data.get("cooldown_until_ms", 0))
        return max(0, int((until - time.time() * 1000) / 1000))

    def _open_position(self) -> Optional[Position]:
        for pos in self.state.positions:
            if pos.symbol == self.symbol:
                return pos
        return None

    def _history_days(self) -> float:
        from .data import timeframe_ms
        need_bars = self.strategy.warmup_bars + 50
        return max(1.0, need_bars * timeframe_ms(self.timeframe) / 86_400_000.0)

    def _journal(self, record: dict) -> None:
        if not self.journal_path:
            return
        record.setdefault("at", _now())
        directory = os.path.dirname(os.path.abspath(self.journal_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.journal_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
