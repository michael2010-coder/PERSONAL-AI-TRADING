"""Risk manager. Every order -- backtest, testnet or live -- goes through here.

Nothing in this module talks to an exchange or reads the clock beyond the
timestamp it is handed. That is deliberate: it makes the limits testable, and
the same object enforces the same numbers in a backtest and in live trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RiskParams:
    allocated_capital_usdt: float = 100.0
    max_position_size_pct: float = 10.0   # of allocated capital, per position
    stop_loss_pct: float = 2.0            # from entry
    take_profit_pct: float = 4.0          # from entry
    max_open_positions: int = 1
    max_daily_loss_pct: float = 5.0       # of allocated capital; trips kill-switch
    min_notional_usdt: float = 10.0       # exchange minimum order size
    max_notional_pct_of_balance: float = 100.0  # never risk more than the wallet holds
    reentry_cooldown_bars: int = 3        # bars to sit out after closing a position

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "RiskParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown risk settings: " + ", ".join(sorted(unknown)))
        params = cls(**raw)
        params.validate()
        return params

    def validate(self) -> None:
        if self.allocated_capital_usdt <= 0:
            raise ValueError("allocated_capital_usdt must be positive")
        if not 0 < self.max_position_size_pct <= 100:
            raise ValueError("max_position_size_pct must be in (0, 100]")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive -- every position needs a stop")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")
        if self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be positive")
        if self.reentry_cooldown_bars < 0:
            raise ValueError("reentry_cooldown_bars cannot be negative")


@dataclass
class Position:
    symbol: str
    side: str          # "long" or "short"
    qty: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    opened_at: int     # epoch ms
    stop_order_id: Optional[str] = None   # resting stop on the exchange

    def notional(self, price: Optional[float] = None) -> float:
        return self.qty * (self.entry_price if price is None else price)

    def unrealised(self, price: float) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return (price - self.entry_price) * self.qty * direction


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    qty: float = 0.0
    notional: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    max_loss_usdt: float = 0.0


class RiskManager:
    def __init__(self, params: RiskParams) -> None:
        params.validate()
        self.p = params

    def set_allocated(self, amount: float) -> None:
        """Point sizing at the portfolio's current trading balance.

        Called by the engine after every closed trade, so position size tracks
        the money actually available -- and shrinks after losses instead of
        holding steady while the account gets smaller.
        """
        if amount <= 0:
            raise ValueError("allocated capital must stay positive")
        self.p.allocated_capital_usdt = amount

    # -- exit levels -----------------------------------------------------
    def exit_levels(self, entry_price: float, side: str = "long") -> Dict[str, float]:
        if entry_price <= 0:
            raise ValueError("entry price must be positive")
        sl = self.p.stop_loss_pct / 100.0
        tp = self.p.take_profit_pct / 100.0
        if side == "long":
            return {
                "stop_price": entry_price * (1.0 - sl),
                "take_profit_price": entry_price * (1.0 + tp),
            }
        return {
            "stop_price": entry_price * (1.0 + sl),
            "take_profit_price": entry_price * (1.0 - tp),
        }

    # -- entry gate ------------------------------------------------------
    def review_entry(
        self,
        price: float,
        open_positions: List[Position],
        realised_pnl_today: float = 0.0,
        free_balance: Optional[float] = None,
        side: str = "long",
        symbol: Optional[str] = None,
    ) -> RiskDecision:
        p = self.p
        if price <= 0:
            return RiskDecision(False, "price must be positive")

        loss_limit = p.allocated_capital_usdt * p.max_daily_loss_pct / 100.0
        if realised_pnl_today <= -loss_limit:
            return RiskDecision(
                False,
                "daily loss kill-switch: {:.2f} lost today, limit is {:.2f}".format(
                    -realised_pnl_today, loss_limit
                ),
            )

        if len(open_positions) >= p.max_open_positions:
            return RiskDecision(
                False,
                "max_open_positions reached ({}/{})".format(len(open_positions), p.max_open_positions),
            )

        if symbol is not None and any(pos.symbol == symbol for pos in open_positions):
            return RiskDecision(False, "already holding a position in {}".format(symbol))

        notional = p.allocated_capital_usdt * p.max_position_size_pct / 100.0

        # Never commit more than the allocated capital that is not already at work.
        committed = sum(pos.notional() for pos in open_positions)
        headroom = p.allocated_capital_usdt - committed
        if notional > headroom:
            notional = headroom

        if free_balance is not None:
            cap = free_balance * p.max_notional_pct_of_balance / 100.0
            if notional > cap:
                notional = cap

        if notional < p.min_notional_usdt:
            return RiskDecision(
                False,
                "sized order {:.2f} USDT is below the {:.2f} USDT minimum".format(
                    notional, p.min_notional_usdt
                ),
            )

        levels = self.exit_levels(price, side)
        qty = notional / price
        max_loss = abs(price - levels["stop_price"]) * qty

        return RiskDecision(
            approved=True,
            reason="approved: {:.2f} USDT ({:.6f} units), risking {:.2f} USDT to the stop".format(
                notional, qty, max_loss
            ),
            qty=qty,
            notional=notional,
            stop_price=levels["stop_price"],
            take_profit_price=levels["take_profit_price"],
            max_loss_usdt=max_loss,
        )

    # -- kill-switch -----------------------------------------------------
    def daily_loss_limit(self) -> float:
        return self.p.allocated_capital_usdt * self.p.max_daily_loss_pct / 100.0

    def kill_switch_tripped(self, realised_pnl_today: float) -> bool:
        return realised_pnl_today <= -self.daily_loss_limit()

    # -- exit check ------------------------------------------------------
    def check_exit(self, position: Position, high: float, low: float) -> Optional[str]:
        """Did this bar hit the stop or the target?

        When a single bar spans both levels we assume the stop filled first.
        That is the pessimistic reading, and the only honest one without
        tick data -- it stops a backtest from flattering itself.
        """
        if position.side == "long":
            if low <= position.stop_price:
                return "stop_loss"
            if high >= position.take_profit_price:
                return "take_profit"
            return None
        if high >= position.stop_price:
            return "stop_loss"
        if low <= position.take_profit_price:
            return "take_profit"
        return None
