"""Reading the exchange account back after a deposit.

The bot never holds funds; these tests cover the one thing it can do about
money, which is tell you whether what you deposited is actually usable.
"""
import pytest

import main as cli


class StubBroker:
    def __init__(self, holdings, prices=None):
        self._holdings = holdings
        self._prices = prices or {"BTC": 80_000.0, "USDT": 1.0}

    def balances(self):
        return self._holdings

    def price_of(self, asset, quote="USDT"):
        return self._prices.get(asset)

    def quote_prices(self, assets, quote="USDT"):
        self.priced_in_one_call = True
        return {a: self._prices[a] for a in assets if a in self._prices}


def run(monkeypatch, capsys, holdings, mode="live", capital=400.0, prices=None):
    cfg = cli.load_config(cli.os.path.join(cli.ROOT, "config.yaml"))
    cfg.portfolio.initial_capital_usdt = capital
    monkeypatch.setattr(cli, "CcxtBroker", lambda *a, **k: StubBroker(holdings, prices))
    monkeypatch.setattr(cfg.crypto.__class__, "has_credentials", lambda self: True)

    class Args:
        pass

    args = Args()
    args.mode = mode
    args.top = 12
    code = cli.cmd_balance(args, cfg)
    return code, capsys.readouterr().out


def holding(free, total=None):
    return {"free": free, "used": 0.0, "total": total if total is not None else free}


def test_paper_mode_has_nothing_to_check(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, {}, mode="paper")
    assert code == 0
    assert "imaginary money" in out


def test_an_empty_account_is_reported_as_empty(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, {})
    assert code == 1
    assert "nothing has been deposited" in out


def test_enough_usdt_is_reported_as_funded(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, {"USDT": holding(500.0)}, capital=400.0)
    assert code == 0
    assert "Funded" in out
    assert "500.00" in out


def test_too_little_usdt_is_refused_with_the_shortfall(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, {"USDT": holding(120.0)}, capital=400.0)
    assert code == 1
    assert "NOT ready" in out
    assert "120.00" in out


def test_a_btc_only_deposit_is_valued_but_flagged_as_unusable(monkeypatch, capsys):
    """The strategy buys BTC with USDT. Holding BTC means nothing to buy with."""
    code, out = run(monkeypatch, capsys, {"BTC": holding(0.01)}, capital=400.0)
    assert code == 1
    assert "NOT ready" in out
    assert "800.00" in out           # 0.01 BTC valued at 80,000
    assert "Convert" in out


def test_an_asset_with_no_market_is_listed_without_a_value(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, {"WEIRDCOIN": holding(5.0)},
                    prices={"USDT": 1.0})
    assert "WEIRDCOIN" in out
    assert "n/a" in out


def test_the_wallet_is_priced_in_a_single_request(monkeypatch, capsys):
    """One ticker request per asset stalls for minutes on a funded testnet
    account. The wallet must be priced in one call."""
    holdings = {a: holding(1.0) for a in ("BTC", "ETH", "BNB", "LTC", "TRX", "USDT")}
    prices = {"BTC": 80_000.0, "ETH": 2_400.0, "BNB": 600.0,
              "LTC": 90.0, "TRX": 0.3, "USDT": 1.0}
    broker = StubBroker(holdings, prices)
    cfg = cli.load_config(cli.os.path.join(cli.ROOT, "config.yaml"))
    monkeypatch.setattr(cli, "CcxtBroker", lambda *a, **k: broker)
    monkeypatch.setattr(cfg.crypto.__class__, "has_credentials", lambda self: True)

    class Args:
        mode = "live"
        top = 12

    cli.cmd_balance(Args(), cfg)
    assert getattr(broker, "priced_in_one_call", False)
    out = capsys.readouterr().out
    assert "80,000.00" in out and "2,400.00" in out
