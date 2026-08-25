"""Configuration: config.yaml for behaviour, .env for secrets.

API keys never live in config.yaml -- it names an environment variable, and
the value is read from the process environment or .env at load time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from .analogues import EvidenceParams
from .portfolio import PortfolioParams
from .risk import RiskParams
from .strategy import StrategyParams
from .supervisor import SupervisorParams

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv(path: Optional[str] = None) -> None:
    """Minimal .env loader: KEY=value lines, # comments, optional quotes."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


@dataclass
class CryptoConfig:
    exchange: str = "binance"
    mode: str = "testnet"          # "testnet" | "live" | "paper"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    poll_seconds: int = 60
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def credentials(self) -> Dict[str, str]:
        return {
            "apiKey": os.environ.get(self.api_key_env, ""),
            "secret": os.environ.get(self.api_secret_env, ""),
        }

    def has_credentials(self) -> bool:
        creds = self.credentials()
        return bool(creds["apiKey"] and creds["secret"])


@dataclass
class StocksConfig:
    enabled: bool = True
    symbols: list = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    interval: str = "1d"
    lookback_days: int = 365


@dataclass
class Config:
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    stocks: StocksConfig = field(default_factory=StocksConfig)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    evidence: EvidenceParams = field(default_factory=EvidenceParams)
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)
    supervisor: SupervisorParams = field(default_factory=SupervisorParams)
    path: str = ""

    @property
    def is_live(self) -> bool:
        return self.crypto.mode == "live"


def _section(raw: Dict[str, Any], key: str, cls):
    data = raw.get(key) or {}
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        raise ValueError("unknown {} settings in config: {}".format(key, ", ".join(sorted(unknown))))
    return cls(**data)


def load_config(path: Optional[str] = None) -> Config:
    path = path or os.path.join(ROOT, "config.yaml")
    load_dotenv()
    if not os.path.exists(path):
        raise FileNotFoundError("config file not found: {}".format(path))
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config(
        crypto=_section(raw, "crypto", CryptoConfig),
        stocks=_section(raw, "stocks", StocksConfig),
        strategy=StrategyParams.from_dict(raw.get("strategy")),
        risk=RiskParams.from_dict(raw.get("risk")),
        evidence=EvidenceParams.from_dict(raw.get("evidence")),
        portfolio=PortfolioParams.from_dict(raw.get("portfolio")),
        supervisor=SupervisorParams.from_dict(raw.get("supervisor")),
        path=path,
    )
    if cfg.crypto.mode not in ("testnet", "live", "paper"):
        raise ValueError("crypto.mode must be testnet, live or paper (got {!r})".format(cfg.crypto.mode))
    return cfg
