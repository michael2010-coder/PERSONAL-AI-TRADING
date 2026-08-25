"""Small JSON store for open positions and today's realised PnL.

The trading loop is restartable: it reloads whatever it held before, so a
crash mid-session does not lose track of an open position or reset the
daily-loss counter to zero.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .risk import Position


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict = {"positions": [], "day": _today(), "realised_pnl_today": 0.0,
                           "history": [], "portfolio": {}, "supervisor": {}}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.data = json.load(fh)
        else:
            # Leave a record from the first run onwards, so `status` can tell
            # "never started" apart from "started and did nothing".
            self.save()
        if self.data.get("day") != _today():
            self.data["day"] = _today()
            self.data["realised_pnl_today"] = 0.0
            self.save()

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2)
        os.replace(tmp, self.path)

    # -- positions -------------------------------------------------------
    @property
    def positions(self) -> List[Position]:
        return [Position(**p) for p in self.data.get("positions", [])]

    def add_position(self, position: Position) -> None:
        self.data.setdefault("positions", []).append(asdict(position))
        self.save()

    def remove_position(self, symbol: str) -> Optional[Position]:
        kept, removed = [], None
        for raw in self.data.get("positions", []):
            if removed is None and raw["symbol"] == symbol:
                removed = Position(**raw)
            else:
                kept.append(raw)
        self.data["positions"] = kept
        self.save()
        return removed

    # -- daily pnl -------------------------------------------------------
    @property
    def realised_pnl_today(self) -> float:
        self.roll_day()
        return float(self.data.get("realised_pnl_today", 0.0))

    def record_pnl(self, pnl: float, note: Optional[Dict] = None) -> None:
        self.roll_day()
        self.data["realised_pnl_today"] = float(self.data.get("realised_pnl_today", 0.0)) + pnl
        if note:
            self.data.setdefault("history", []).append(note)
            self.data["history"] = self.data["history"][-500:]
        self.save()

    # -- portfolio and supervisor ----------------------------------------
    def save_component(self, name: str, payload: Dict) -> None:
        self.data[name] = payload
        self.save()

    def load_component(self, name: str) -> Dict:
        return self.data.get(name) or {}

    def roll_day(self) -> None:
        if self.data.get("day") != _today():
            self.data["day"] = _today()
            self.data["realised_pnl_today"] = 0.0
            self.save()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
