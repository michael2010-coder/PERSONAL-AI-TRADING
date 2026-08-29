"""Order placement. One interface, three backings: paper, testnet, live.

Testnet and live are the same code path through ccxt -- the only difference is
which endpoint the exchange object points at. That is on purpose: what you
prove out on testnet is the code that runs with real money.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional

from .data import fetch_ohlcv, make_exchange
from .strategy import Candle

log = logging.getLogger("trading.broker")


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

    def balances(self) -> Dict[str, Dict[str, float]]:
        """Every asset with a non-zero balance: {asset: {free, used, total}}."""
        raise NotImplementedError

    def place_stop(self, symbol: str, qty: float, stop_price: float) -> Optional[str]:
        """Rest a stop-loss on the exchange. Returns an order id, or None.

        The bot's own stop only fires when it polls, so a crash or a sleeping
        laptop leaves a position unprotected. This one lives on the exchange
        and works whether or not the bot is running.
        """
        return None

    def cancel(self, symbol: str, order_id: str) -> bool:
        return False

    def open_order_ids(self, symbol: str) -> List[str]:
        return []

    def quote_prices(self, assets, quote: str = "USDT") -> Dict[str, float]:
        """Price of each asset in `quote`, in ONE request.

        Pricing a wallet one ticker at a time is a request per asset; a
        pre-funded testnet account has enough of them to stall for minutes.
        """
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

    def balances(self) -> Dict[str, Dict[str, float]]:
        raw = self.exchange.fetch_balance()
        out: Dict[str, Dict[str, float]] = {}
        for asset, total in (raw.get("total") or {}).items():
            if not total:
                continue
            out[asset] = {
                "free": float((raw.get("free") or {}).get(asset) or 0.0),
                "used": float((raw.get("used") or {}).get(asset) or 0.0),
                "total": float(total),
            }
        return out

    def price_of(self, asset: str, quote: str = "USDT") -> Optional[float]:
        """What one unit of `asset` is worth in `quote`, or None if untradeable."""
        if asset == quote:
            return 1.0
        try:
            return self.price("{}/{}".format(asset, quote))
        except Exception:
            return None

    def quote_prices(self, assets, quote: str = "USDT") -> Dict[str, float]:
        out: Dict[str, float] = {quote: 1.0}
        wanted = [a for a in assets if a != quote]
        if not wanted:
            return out
        try:
            self.exchange.load_markets()
        except Exception:
            pass
        # Ask for everything rather than naming the symbols: a pre-funded
        # testnet account holds hundreds of assets, and listing them all in the
        # query string overruns the exchange's URL limits.
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as exc:
            log.warning("could not price the wallet (%s)", exc)
            return out
        for asset in wanted:
            ticker = tickers.get("{}/{}".format(asset, quote)) or {}
            last = ticker.get("last") or ticker.get("close")
            if last:
                out[asset] = float(last)
        return out

    def place_stop(self, symbol: str, qty: float, stop_price: float) -> Optional[str]:
        amount = float(self.exchange.amount_to_precision(symbol, qty))
        trigger = float(self.exchange.price_to_precision(symbol, stop_price))
        # A limit a little under the trigger: a pure limit at the trigger can
        # be jumped in a fast drop, and Binance spot needs a limit price.
        limit = float(self.exchange.price_to_precision(symbol, stop_price * 0.995))
        try:
            order = self.exchange.create_order(
                symbol, "STOP_LOSS_LIMIT", "sell", amount, limit,
                {"stopPrice": trigger, "timeInForce": "GTC"},
            )
            return str(order.get("id") or "")
        except Exception as exc:
            log.warning("could not rest a stop on the exchange for %s: %s", symbol, exc)
            return None

    def cancel(self, symbol: str, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            log.warning("could not cancel order %s on %s: %s", order_id, symbol, exc)
            return False

    def open_order_ids(self, symbol: str) -> List[str]:
        try:
            return [str(o["id"]) for o in self.exchange.fetch_open_orders(symbol)]
        except Exception as exc:
            log.warning("could not list open orders for %s: %s", symbol, exc)
            return []

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

    def balances(self) -> Dict[str, Dict[str, float]]:
        return {"USDT": {"free": self.balance, "used": 0.0, "total": self.balance}}

    def price_of(self, asset: str, quote: str = "USDT") -> Optional[float]:
        if asset == quote:
            return 1.0
        try:
            return self.price("{}/{}".format(asset, quote))
        except Exception:
            return None

    def quote_prices(self, assets, quote: str = "USDT") -> Dict[str, float]:
        return {quote: 1.0}

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
