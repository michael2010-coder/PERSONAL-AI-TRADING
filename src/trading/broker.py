"""Order placement. One interface, three backings: paper, testnet, live.

Testnet and live are the same code path through ccxt -- the only difference is
which endpoint the exchange object points at. That is on purpose: what you
prove out on testnet is the code that runs with real money.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .data import fetch_ohlcv, make_exchange
from .strategy import Candle


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    order_id: str = ""
    raw: Optional[Dict] = None


class Broker:
    mode = "abstract"

    def candles(self, symbol: str, timeframe: str, days: float) -> List[Candle]:
        raise NotImplementedError

    def price(self, symbol: str) -> float:
        raise NotImplementedError

    def free_quote_balance(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    def market_buy(self, symbol: str, qty: float) -> Fill:
        raise NotImplementedError

    def market_sell(self, symbol: str, qty: float) -> Fill:
        raise NotImplementedError


class CcxtBroker(Broker):
    """Testnet or live, depending on `testnet`."""

    def __init__(self, exchange_id: str, testnet: bool, credentials: Optional[Dict] = None,
                 fee_bps: float = 10.0) -> None:
        self.mode = "testnet" if testnet else "live"
        self.exchange = make_exchange(exchange_id, testnet=testnet, credentials=credentials)
        self.fee_rate = fee_bps / 10_000.0

    def candles(self, symbol: str, timeframe: str, days: float) -> List[Candle]:
        return fetch_ohlcv(self.exchange, symbol, timeframe, days)

    def price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        last = ticker.get("last") or ticker.get("close")
        if last is None:
            raise RuntimeError("no last price for {}".format(symbol))
        return float(last)

    def free_quote_balance(self, symbol: str) -> Optional[float]:
        quote = symbol.split("/")[-1].split(":")[0]
        balance = self.exchange.fetch_balance()
        free = balance.get("free", {}).get(quote)
        return None if free is None else float(free)

    def market_buy(self, symbol: str, qty: float) -> Fill:
        return self._order(symbol, "buy", qty)

    def market_sell(self, symbol: str, qty: float) -> Fill:
        return self._order(symbol, "sell", qty)

    def _order(self, symbol: str, side: str, qty: float) -> Fill:
        amount = float(self.exchange.amount_to_precision(symbol, qty))
        order = self.exchange.create_order(symbol, "market", side, amount)
        price = order.get("average") or order.get("price") or self.price(symbol)
        filled = float(order.get("filled") or amount)
        fee_obj = order.get("fee") or {}
        fee = float(fee_obj.get("cost") or filled * float(price) * self.fee_rate)
        return Fill(symbol=symbol, side=side, qty=filled, price=float(price),
                    fee=fee, order_id=str(order.get("id", "")), raw=order)


class PaperBroker(Broker):
    """Real market data, imaginary money. No API key needed."""

    mode = "paper"

    def __init__(self, exchange_id: str = "binance", fee_bps: float = 10.0,
                 slippage_bps: float = 5.0, starting_balance: float = 10_000.0) -> None:
        self.exchange = make_exchange(exchange_id, testnet=False)
        self.fee_rate = fee_bps / 10_000.0
        self.slippage = slippage_bps / 10_000.0
        self.balance = starting_balance
        self._order_seq = 0

    def candles(self, symbol: str, timeframe: str, days: float) -> List[Candle]:
        return fetch_ohlcv(self.exchange, symbol, timeframe, days)

    def price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker.get("last") or ticker.get("close"))

    def free_quote_balance(self, symbol: str) -> Optional[float]:
        return self.balance

    def market_buy(self, symbol: str, qty: float) -> Fill:
        price = self.price(symbol) * (1.0 + self.slippage)
        fee = qty * price * self.fee_rate
        self.balance -= qty * price + fee
        return self._fill(symbol, "buy", qty, price, fee)

    def market_sell(self, symbol: str, qty: float) -> Fill:
        price = self.price(symbol) * (1.0 - self.slippage)
        fee = qty * price * self.fee_rate
        self.balance += qty * price - fee
        return self._fill(symbol, "sell", qty, price, fee)

    def _fill(self, symbol: str, side: str, qty: float, price: float, fee: float) -> Fill:
        self._order_seq += 1
        return Fill(symbol=symbol, side=side, qty=qty, price=price, fee=fee,
                    order_id="paper-{}".format(self._order_seq))
