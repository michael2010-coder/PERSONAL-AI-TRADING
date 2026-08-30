"""Portfolio management for a small, real account.

This is the anti-greed layer. It has one job: make sure a good run cannot talk
the bot into risking more than it started out willing to lose.

Four rules do the work:

  1. Position size never grows because the last trade won. Sizing is a fixed
     share of *allocated* capital, and allocated capital is not the same thing
     as equity.
  2. Compounding is capped. Allocated capital may grow with profits only up to
     a ceiling; past that, profit accumulates outside the trading balance.
  3. Profits are locked away in steps. Every time realised profit crosses a
     step, a share of it moves to a reserve the bot cannot touch.
  4. A total-drawdown ceiling stops trading outright, and restarting is a
     deliberate manual act rather than something the bot does on its own.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class PortfolioParams:
    initial_capital_usdt: float = 400.0
    compounding: str = "capped"          # "off" | "capped" | "full"
    compounding_cap_pct: float = 50.0    # allocated may grow to +50% of initial, no further
    profit_lock_step_usdt: float = 50.0  # every +50 of realised profit ...
    profit_lock_share_pct: float = 50.0  # ... move half of that step to the reserve
    max_total_drawdown_pct: float = 20.0  # of peak equity: trading stops, permanently
    withdraw_reserve: bool = True        # reserve is never traded

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "PortfolioParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown portfolio settings: " + ", ".join(sorted(unknown)))
        params = cls(**raw)
        params.validate()
        return params

    def validate(self) -> None:
        if self.initial_capital_usdt <= 0:
            raise ValueError("initial_capital_usdt must be positive")
        if self.compounding not in ("off", "capped", "full"):
            raise ValueError("compounding must be off, capped or full")
        if self.compounding_cap_pct < 0:
            raise ValueError("compounding_cap_pct cannot be negative")
        if not 0 <= self.profit_lock_share_pct <= 100:
            raise ValueError("profit_lock_share_pct must be between 0 and 100")
        if self.profit_lock_step_usdt <= 0:
            raise ValueError("profit_lock_step_usdt must be positive")
        if not 0 < self.max_total_drawdown_pct <= 100:
            raise ValueError("max_total_drawdown_pct must be in (0, 100]")


@dataclass
class PortfolioState:
    allocated: float = 0.0        # capital the bot may trade with
    reserve: float = 0.0          # locked profit, never traded
    peak_equity: float = 0.0
    realised_pnl: float = 0.0
    locked_steps: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    started_at: Optional[int] = None
    initial_capital: float = 0.0   # what the account was sized for when saved

    def to_dict(self) -> Dict:
        return asdict(self)


class Portfolio:
    def __init__(self, params: PortfolioParams, state: Optional[PortfolioState] = None) -> None:
        params.validate()
        self.p = params
        if state is None:
            state = PortfolioState(
                allocated=params.initial_capital_usdt,
                peak_equity=params.initial_capital_usdt,
                initial_capital=params.initial_capital_usdt,
            )
        self.state = state

    # -- views -----------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.state.allocated + self.state.reserve

    @property
    def growth_pct(self) -> float:
        start = self.p.initial_capital_usdt
        return (self.equity - start) / start * 100.0

    @property
    def drawdown_pct(self) -> float:
        peak = self.state.peak_equity
        if peak <= 0:
            return 0.0
        return (self.equity - peak) / peak * 100.0

    @property
    def win_rate_pct(self) -> float:
        total = self.state.wins + self.state.losses
        return 0.0 if not total else self.state.wins / total * 100.0

    # -- the ceiling on trading capital ----------------------------------
    def allocation_ceiling(self) -> float:
        if self.p.compounding == "off":
            return self.p.initial_capital_usdt
        if self.p.compounding == "full":
            return float("inf")
        return self.p.initial_capital_usdt * (1.0 + self.p.compounding_cap_pct / 100.0)

    # -- events ----------------------------------------------------------
    def on_trade_closed(self, pnl: float, when_ms: Optional[int] = None) -> Dict:
        """Book a closed trade and apply the compounding and profit-lock rules."""
        state = self.state
        if state.started_at is None:
            state.started_at = when_ms

        state.allocated += pnl
        state.realised_pnl += pnl
        state.trades += 1
        if pnl > 0:
            state.wins += 1
        else:
            state.losses += 1

        locked_now = self._lock_profit()
        spilled = self._apply_ceiling()

        state.peak_equity = max(state.peak_equity, self.equity)
        if self.drawdown_pct <= -self.p.max_total_drawdown_pct:
            state.halted = True
            state.halt_reason = (
                "total drawdown {:.1f}% breached the {:.1f}% ceiling -- trading stopped, "
                "restart is a manual decision".format(self.drawdown_pct, self.p.max_total_drawdown_pct)
            )

        return {"pnl": pnl, "locked": locked_now, "spilled_to_reserve": spilled,
                "allocated": state.allocated, "reserve": state.reserve,
                "equity": self.equity, "halted": state.halted}

    def _lock_profit(self) -> float:
        """Move a share of each completed profit step out of reach."""
        if self.p.profit_lock_share_pct <= 0:
            return 0.0
        step = self.p.profit_lock_step_usdt
        steps_earned = int(max(0.0, self.state.realised_pnl) // step)
        new_steps = steps_earned - self.state.locked_steps
        if new_steps <= 0:
            return 0.0
        amount = new_steps * step * self.p.profit_lock_share_pct / 100.0
        amount = min(amount, max(0.0, self.state.allocated - self._floor()))
        if amount <= 0:
            return 0.0
        self.state.allocated -= amount
        self.state.reserve += amount
        self.state.locked_steps = steps_earned
        return amount

    def _apply_ceiling(self) -> float:
        """Profit above the compounding ceiling spills into the reserve."""
        ceiling = self.allocation_ceiling()
        if self.state.allocated <= ceiling:
            return 0.0
        spill = self.state.allocated - ceiling
        self.state.allocated = ceiling
        self.state.reserve += spill
        return spill

    def _floor(self) -> float:
        """Never lock so much that the account can no longer place an order."""
        return min(self.p.initial_capital_usdt, 25.0)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict:
        return self.state.to_dict()

    @classmethod
    def from_dict(cls, params: PortfolioParams, raw: Optional[Dict]) -> "Portfolio":
        """Restore saved state -- unless it was saved for a different account size.

        Resizing the account (say 400 -> 49) leaves a state file describing the
        old one. Carrying it over reports a balance the account does not have
        and a growth figure that is pure fiction, so start fresh instead.
        """
        if not raw:
            return cls(params)
        known = {f for f in PortfolioState.__dataclass_fields__}  # type: ignore[attr-defined]
        state = PortfolioState(**{k: v for k, v in raw.items() if k in known})
        saved_for = state.initial_capital or 0.0
        if abs(saved_for - params.initial_capital_usdt) > 1e-9:
            import logging
            logging.getLogger("trading.portfolio").warning(
                "saved account was sized for %.2f, config now says %.2f -- "
                "starting the ledger fresh (%d old trade(s) stay in logs/orders.jsonl)",
                saved_for, params.initial_capital_usdt, state.trades)
            return cls(params)
        return cls(params, state)

    def summary(self) -> str:
        s = self.state
        lines = [
            "Portfolio",
            "  started with       {:>10.2f}".format(self.p.initial_capital_usdt),
            "  trading balance    {:>10.2f}".format(s.allocated),
            "  locked reserve     {:>10.2f}   (never traded)".format(s.reserve),
            "  equity             {:>10.2f}   ({:+.2f}%)".format(self.equity, self.growth_pct),
            "  peak equity        {:>10.2f}".format(s.peak_equity),
            "  drawdown from peak {:>9.2f}%   (stops at -{:.0f}%)".format(
                self.drawdown_pct, self.p.max_total_drawdown_pct),
            "  realised P&L       {:>10.2f}".format(s.realised_pnl),
            "  trades             {:>10d}   ({} win / {} loss, {:.0f}%)".format(
                s.trades, s.wins, s.losses, self.win_rate_pct),
        ]
        if s.halted:
            lines.append("  HALTED: {}".format(s.halt_reason))
        return "\n".join(lines)
