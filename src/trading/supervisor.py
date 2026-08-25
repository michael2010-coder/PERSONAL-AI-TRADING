"""The pause/resume brain.

The trading loop knows how to trade. The supervisor decides whether it should
be trading at all right now. It pauses on losing streaks, on a bad day, on a
bad week, when data goes stale, when the exchange starts erroring, and when
the market turns unusually violent. It also pauses *after a good day* -- the
rule that stops a winning streak from turning into a giving-back streak.

Every pause carries an explicit resume time, so a paused bot is never a stuck
bot. Only two things need a human: a portfolio-level drawdown halt, and going
live in the first place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


@dataclass
class SupervisorParams:
    enabled: bool = True
    max_consecutive_losses: int = 3
    loss_streak_pause_hours: float = 12.0
    weekly_loss_pct: float = 8.0          # of starting capital, then pause
    weekly_pause_days: float = 3.0
    daily_profit_target_pct: float = 2.0  # hit it and stop for the day
    max_trades_per_day: int = 4
    stale_data_minutes: float = 30.0
    error_streak_pause_minutes: float = 15.0
    max_error_streak: int = 5
    max_volatility_pct: float = 5.0       # ATR as % of price; above this, sit out
    require_flat_to_pause: bool = False   # pauses block entries; exits always run

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "SupervisorParams":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError("unknown supervisor settings: " + ", ".join(sorted(unknown)))
        params = cls(**raw)
        params.validate()
        return params

    def validate(self) -> None:
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be at least 1")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        for name in ("loss_streak_pause_hours", "weekly_pause_days",
                     "stale_data_minutes", "error_streak_pause_minutes"):
            if getattr(self, name) < 0:
                raise ValueError("{} cannot be negative".format(name))


@dataclass
class SupervisorState:
    day: str = ""
    week: str = ""
    consecutive_losses: int = 0
    trades_today: int = 0
    pnl_today: float = 0.0
    pnl_week: float = 0.0
    error_streak: int = 0
    paused_until_ms: int = 0
    pause_reason: str = ""
    pauses: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Verdict:
    allowed: bool
    reason: str
    resume_at_ms: int = 0

    def wait_text(self, now_ms: int) -> str:
        if not self.resume_at_ms or self.resume_at_ms <= now_ms:
            return ""
        minutes = (self.resume_at_ms - now_ms) / MINUTE_MS
        if minutes < 90:
            return "resumes in {:.0f} min".format(minutes)
        return "resumes in {:.1f} h".format(minutes / 60)


class Supervisor:
    def __init__(self, params: SupervisorParams, capital: float,
                 state: Optional[SupervisorState] = None) -> None:
        params.validate()
        self.p = params
        self.capital = capital
        self.state = state or SupervisorState()

    # -- rollovers -------------------------------------------------------
    def roll(self, now_ms: int) -> None:
        day = _day(now_ms)
        week = _week(now_ms)
        if self.state.day != day:
            self.state.day = day
            self.state.trades_today = 0
            self.state.pnl_today = 0.0
        if self.state.week != week:
            self.state.week = week
            self.state.pnl_week = 0.0

    # -- the question the engine asks ------------------------------------
    def may_enter(
        self,
        now_ms: int,
        volatility_pct: Optional[float] = None,
        data_age_minutes: Optional[float] = None,
        portfolio=None,
    ) -> Verdict:
        if not self.p.enabled:
            return Verdict(True, "supervisor disabled")
        self.roll(now_ms)

        if portfolio is not None and portfolio.state.halted:
            return Verdict(False, "portfolio halted: " + portfolio.state.halt_reason)

        if self.state.paused_until_ms > now_ms:
            return Verdict(False, self.state.pause_reason, self.state.paused_until_ms)

        if self.state.paused_until_ms:
            # The pause has run its course. Clear what caused it, or the very
            # next check would re-trigger on the same stale counters and the
            # bot would never trade again.
            self.state.paused_until_ms = 0
            self.state.consecutive_losses = 0
            self.state.error_streak = 0

        if data_age_minutes is not None and data_age_minutes > self.p.stale_data_minutes:
            return self._pause(now_ms, self.p.stale_data_minutes * MINUTE_MS,
                               "market data is {:.0f} min stale".format(data_age_minutes))

        if self.state.error_streak >= self.p.max_error_streak:
            return self._pause(now_ms, self.p.error_streak_pause_minutes * MINUTE_MS,
                               "{} exchange errors in a row".format(self.state.error_streak))

        if volatility_pct is not None and volatility_pct > self.p.max_volatility_pct:
            return Verdict(False, "market too violent right now: ATR {:.2f}% of price, "
                                  "limit {:.2f}%".format(volatility_pct, self.p.max_volatility_pct))

        if self.state.trades_today >= self.p.max_trades_per_day:
            return Verdict(False, "{} trades today is the daily maximum".format(
                self.state.trades_today), _next_midnight(now_ms))

        target = self.capital * self.p.daily_profit_target_pct / 100.0
        if self.p.daily_profit_target_pct > 0 and self.state.pnl_today >= target:
            return Verdict(False, "up {:.2f} today, at or past the {:.2f} target -- "
                                  "stopping while ahead".format(self.state.pnl_today, target),
                           _next_midnight(now_ms))

        weekly_limit = self.capital * self.p.weekly_loss_pct / 100.0
        if self.state.pnl_week <= -weekly_limit:
            return self._pause(now_ms, self.p.weekly_pause_days * DAY_MS,
                               "down {:.2f} this week, past the {:.2f} weekly limit".format(
                                   -self.state.pnl_week, weekly_limit))

        if self.state.consecutive_losses >= self.p.max_consecutive_losses:
            return self._pause(now_ms, self.p.loss_streak_pause_hours * HOUR_MS,
                               "{} losses in a row".format(self.state.consecutive_losses))

        return Verdict(True, "clear to trade")

    # -- events ----------------------------------------------------------
    def record_trade(self, pnl: float, now_ms: int) -> None:
        self.roll(now_ms)
        self.state.trades_today += 1
        self.state.pnl_today += pnl
        self.state.pnl_week += pnl
        if pnl > 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

    def record_error(self, now_ms: int) -> None:
        self.state.error_streak += 1

    def record_ok(self) -> None:
        self.state.error_streak = 0

    def resume_now(self, note: str = "manually resumed") -> None:
        self.state.paused_until_ms = 0
        self.state.pause_reason = note
        self.state.consecutive_losses = 0
        self.state.error_streak = 0

    # -- helpers ---------------------------------------------------------
    def _pause(self, now_ms: int, duration_ms: float, reason: str) -> Verdict:
        until = int(now_ms + duration_ms)
        self.state.paused_until_ms = until
        self.state.pause_reason = reason
        self.state.pauses += 1
        return Verdict(False, reason, until)

    def to_dict(self) -> Dict:
        return self.state.to_dict()

    @classmethod
    def from_dict(cls, params: SupervisorParams, capital: float,
                  raw: Optional[Dict]) -> "Supervisor":
        if not raw:
            return cls(params, capital)
        known = {f for f in SupervisorState.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(params, capital, SupervisorState(**{k: v for k, v in raw.items() if k in known}))

    def summary(self, now_ms: int) -> str:
        s = self.state
        lines = ["Supervisor"]
        if s.paused_until_ms > now_ms:
            lines.append("  PAUSED: {} ({})".format(
                s.pause_reason, Verdict(False, "", s.paused_until_ms).wait_text(now_ms)))
        else:
            lines.append("  running")
        lines += [
            "  trades today       {:>4d} / {}".format(s.trades_today, self.p.max_trades_per_day),
            "  P&L today          {:>7.2f}   (stops for the day at +{:.2f})".format(
                s.pnl_today, self.capital * self.p.daily_profit_target_pct / 100.0),
            "  P&L this week      {:>7.2f}   (pauses at -{:.2f})".format(
                s.pnl_week, self.capital * self.p.weekly_loss_pct / 100.0),
            "  losing streak      {:>4d} / {}".format(s.consecutive_losses, self.p.max_consecutive_losses),
            "  pauses so far      {:>4d}".format(s.pauses),
        ]
        return "\n".join(lines)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _week(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, timezone.utc)
    year, week, _ = dt.isocalendar()
    return "{}-W{:02d}".format(year, week)


def _next_midnight(ms: int) -> int:
    dt = datetime.fromtimestamp(ms / 1000, timezone.utc)
    nxt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(nxt.timestamp() * 1000)
