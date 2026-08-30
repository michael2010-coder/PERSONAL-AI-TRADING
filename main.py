#!/usr/bin/env python3
"""AI trading app -- command line entry point.

    python main.py backtest --symbol BTC/USDT --days 90
    python main.py signal --symbol BTC/USDT
    python main.py trade --dry-run
    python main.py stocks
    python main.py check

Live trading needs BOTH crypto.mode: "live" in config.yaml AND the
--i-understand-this-is-live flag. Neither alone will do it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from trading import corpus as corpus_mod                 # noqa: E402
from trading import data as data_mod                      # noqa: E402
from trading.analogues import (AnalogueLibrary, EvidenceGate,  # noqa: E402
                               LibraryParams, label_outcomes, situation_at)
from trading.backtest import Backtester                   # noqa: E402
from trading.broker import CcxtBroker, PaperBroker        # noqa: E402
from trading.config import ROOT, load_config              # noqa: E402
from trading.engine import TradingEngine                  # noqa: E402
from trading.logging_setup import setup_logging           # noqa: E402
from trading.portfolio import Portfolio                   # noqa: E402
from trading.risk import RiskManager                      # noqa: E402
from trading.state import StateStore                      # noqa: E402
from trading.stocks import append_journal, signal_for     # noqa: E402
from trading.strategy import Strategy                     # noqa: E402
from trading.supervisor import Supervisor                 # noqa: E402

log = logging.getLogger("trading.cli")
LIVE_FLAG = "--i-understand-this-is-live"


def build(cfg):
    from trading.config import make_strategy
    return make_strategy(cfg), RiskManager(cfg.risk)


def load_gate(cfg, required: bool):
    """Load the evidence gate, or explain clearly why it is not available."""
    if not cfg.evidence.enabled and not required:
        return None
    try:
        gate = EvidenceGate.load(cfg.evidence, cfg.risk, ROOT)
    except (FileNotFoundError, ValueError) as exc:
        if required:
            log.error("%s", exc)
            raise SystemExit(2)
        log.warning("evidence gate unavailable, running without it: %s",
                    str(exc).splitlines()[0])
        return None
    log.info("evidence gate: %s situations, needs %s analogues at %.0f%% success",
             "{:,}".format(len(gate.library)),
             "{:,}".format(cfg.evidence.min_similar_situations),
             cfg.evidence.min_success_rate * 100)
    return gate


def load_account(cfg, state):
    """Portfolio and supervisor, restored from the state file if present."""
    portfolio = Portfolio.from_dict(cfg.portfolio, state.load_component("portfolio"))
    supervisor = Supervisor.from_dict(cfg.supervisor, cfg.portfolio.initial_capital_usdt,
                                      state.load_component("supervisor"))
    return portfolio, supervisor


def corpus_series(directory, timeframes=None):
    """Yield (name, candles) one series at a time -- never all at once."""
    for item in corpus_mod.inventory(directory):
        if timeframes and item["timeframe"] not in timeframes:
            continue
        candles = data_mod.load_csv(item["path"])
        if candles:
            yield "{} {}".format(item["symbol"], item["timeframe"]), candles


# -- commands ---------------------------------------------------------------
def cmd_backtest(args, cfg) -> int:
    strategy, risk = build(cfg)
    symbol = args.symbol or cfg.crypto.symbol
    timeframe = args.timeframe or cfg.crypto.timeframe

    if args.csv:
        candles = data_mod.load_csv(args.csv)
        source = args.csv
    elif args.synthetic:
        candles = data_mod.synthetic(bars=args.synthetic, timeframe=timeframe, seed=args.seed)
        source = "synthetic ({} bars, seed {})".format(args.synthetic, args.seed)
    else:
        exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
        log.info("fetching %s %s for the last %s days from %s ...",
                 symbol, timeframe, args.days, cfg.crypto.exchange)
        candles = data_mod.fetch_ohlcv(exchange, symbol, timeframe, args.days)
        source = "{} live history".format(cfg.crypto.exchange)
        if args.save:
            data_mod.save_csv(candles, args.save)
            log.info("saved %d candles to %s", len(candles), args.save)

    if not candles:
        log.error("no candles: nothing to backtest")
        return 1
    print("Data source: {}  ({} candles)".format(source, len(candles)))

    if args.min_success is not None:
        cfg.evidence.min_success_rate = args.min_success
    if args.min_analogues is not None:
        cfg.evidence.min_similar_situations = args.min_analogues
    gate = load_gate(cfg, required=False) if args.evidence else None
    tester = Backtester(strategy, risk, cfg.crypto.fee_bps, cfg.crypto.slippage_bps,
                        evidence=gate)
    result = tester.run(candles, symbol=symbol, timeframe=timeframe)
    print()
    print(result.summary(bars_per_year=data_mod.bars_per_year(timeframe)))

    if args.trades and result.trades:
        print("\nTrades")
        print("-" * 68)
        for t in result.trades:
            print("  {}  {:.6f} @ {:.2f} -> {:.2f}  {:<12} {:+.2f} ({:+.2f}%)".format(
                _fmt(t.entry_time), t.qty, t.entry_price,
                t.exit_price, t.exit_reason, t.pnl, t.return_pct))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "symbol": symbol, "timeframe": timeframe, "bars": result.bars,
                "total_return_pct": result.total_return_pct,
                "buy_hold_return_pct": result.buy_hold_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate_pct": result.win_rate_pct,
                "profit_factor": None if result.profit_factor == float("inf") else result.profit_factor,
                "trades": [t.__dict__ for t in result.trades],
            }, fh, indent=2)
        print("\nWrote {}".format(args.json))
    return 0


def cmd_signal(args, cfg) -> int:
    strategy, _ = build(cfg)
    symbol = args.symbol or cfg.crypto.symbol
    timeframe = args.timeframe or cfg.crypto.timeframe
    exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
    days = max(2.0, (strategy.warmup_bars + 20) * data_mod.timeframe_ms(timeframe) / 86_400_000.0)
    candles = data_mod.fetch_ohlcv(exchange, symbol, timeframe, days)
    if len(candles) < strategy.warmup_bars + 1:
        log.error("only %d closed candles, need %d", len(candles), strategy.warmup_bars + 1)
        return 1
    signal = strategy.evaluate(candles)
    print("{}  {}  last closed bar {}".format(symbol, timeframe, _fmt(signal.timestamp)))
    print(signal.explain())
    return 0


def cmd_trade(args, cfg) -> int:
    strategy, risk = build(cfg)
    mode = cfg.crypto.mode

    if mode == "live":
        if not args.i_understand_this_is_live:
            log.error("config.yaml says mode: live but %s was not passed. Refusing to trade.", LIVE_FLAG)
            return 2
        if not cfg.crypto.has_credentials():
            log.error("live mode needs %s and %s in .env", cfg.crypto.api_key_env, cfg.crypto.api_secret_env)
            return 2
        blocker = _validation_blocks_live(cfg)
        if blocker and not args.skip_validation:
            log.error("refusing to trade live: %s", blocker)
            log.error("run `python main.py validate`, or pass --skip-validation "
                      "if you have decided to override this deliberately.")
            return 2
        broker = CcxtBroker(cfg.crypto.exchange, testnet=False,
                            credentials=cfg.crypto.credentials(), fee_bps=cfg.crypto.fee_bps)
        log.warning("*** LIVE MODE: real money on %s. Capital at risk: %.2f USDT, "
                    "max %.2f per position, daily loss limit %.2f ***",
                    cfg.crypto.exchange, cfg.risk.allocated_capital_usdt,
                    cfg.risk.allocated_capital_usdt * cfg.risk.max_position_size_pct / 100,
                    risk.daily_loss_limit())
    elif mode == "testnet":
        if args.i_understand_this_is_live:
            log.warning("%s passed, but config.yaml still says mode: testnet. Staying on testnet.", LIVE_FLAG)
        if not cfg.crypto.has_credentials():
            if not args.dry_run:
                log.error("testnet mode needs testnet keys in .env (%s, %s). "
                          "Use `mode: paper` to run with no keys at all.",
                          cfg.crypto.api_key_env, cfg.crypto.api_secret_env)
                return 2
            # --dry-run places no orders, so the whole decision path can be
            # exercised against testnet's public data before any key exists.
            log.warning("no testnet keys yet -- dry run against public data only")
        broker = CcxtBroker(cfg.crypto.exchange, testnet=True,
                            credentials=cfg.crypto.credentials(), fee_bps=cfg.crypto.fee_bps)
    else:
        broker = PaperBroker(cfg.crypto.exchange, cfg.crypto.fee_bps, cfg.crypto.slippage_bps)

    # Credentials preflight. Testnet and live use different keys under the
    # same variable names, so the common mistake is running one with the
    # other's keys. Fail with a sentence instead of a traceback.
    if isinstance(broker, CcxtBroker):
        try:
            broker.balances()
        except Exception as exc:
            if "api-key" in str(exc).lower() or "authentication" in type(exc).__name__.lower():
                log.error("%s rejected the API keys in .env for %s mode: %s",
                          cfg.crypto.exchange, mode, str(exc)[:120])
                log.error("Testnet keys (testnet.binance.vision) and live keys "
                          "(binance.com) are different. Check which are in .env.")
                return 2
            log.warning("could not read the account up front (%s); continuing", str(exc)[:120])

    if args.symbol:
        cfg.crypto.symbol = args.symbol
    # One state file per symbol, so several instances can run side by side
    # without overwriting each other's positions.
    default_state = "state.{}.{}.json".format(
        broker.mode, cfg.crypto.symbol.replace("/", "").lower())
    state = StateStore(args.state or os.path.join(ROOT, default_state))
    gate = load_gate(cfg, required=cfg.evidence.enabled and not args.no_evidence)
    portfolio, supervisor = load_account(cfg, state)

    if portfolio.state.halted:
        log.error("PORTFOLIO HALTED: %s", portfolio.state.halt_reason)
        log.error("Restarting is deliberate: review what happened, then clear "
                  "'portfolio' in %s to start again.", state.path)
        return 3

    print(portfolio.summary())
    print()
    print(supervisor.summary(int(time.time() * 1000)))
    print()

    engine = TradingEngine(cfg, broker, strategy, risk, state,
                           dry_run=args.dry_run,
                           journal_path=os.path.join(ROOT, "logs", "orders.jsonl"),
                           evidence=None if args.no_evidence else gate,
                           portfolio=portfolio, supervisor=supervisor)
    if args.once:
        engine.step()
        return 0
    try:
        engine.run_forever(args.poll)
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_balance(args, cfg) -> int:
    """What is actually in the exchange account, and can the bot use it?

    The bot never custodies funds -- the money sits in your own exchange
    account and the API key may trade it, nothing more. This reads that
    account back so a deposit can be confirmed before anything is risked.
    """
    mode = args.mode or cfg.crypto.mode
    quote = cfg.crypto.symbol.split("/")[-1]

    if mode == "paper":
        print("Paper mode holds imaginary money -- nothing to check.")
        print("Set crypto.mode to testnet or live to read a real balance.")
        return 0
    if not cfg.crypto.has_credentials():
        log.error("no API keys in .env (%s, %s)", cfg.crypto.api_key_env, cfg.crypto.api_secret_env)
        return 2

    broker = CcxtBroker(cfg.crypto.exchange, testnet=(mode == "testnet"),
                        credentials=cfg.crypto.credentials(), fee_bps=cfg.crypto.fee_bps)
    try:
        holdings = broker.balances()
    except Exception as exc:
        log.error("could not read the account: %s", exc)
        return 1

    print("{} account ({} mode)".format(cfg.crypto.exchange, mode))
    print("=" * 60)
    if not holdings:
        print("  empty -- nothing has been deposited yet")
        return 1

    prices = broker.quote_prices(list(holdings), quote)   # one request, not one per asset

    # Value everything, then show only what matters. A pre-funded testnet
    # account holds hundreds of assets and printing them all is noise.
    valued = []
    total_value = quote_free = other_value = 0.0
    for asset, amounts in holdings.items():
        price = prices.get(asset)
        value = None if price is None else amounts["total"] * price
        if value is not None:
            total_value += value
            if asset == quote:
                quote_free = amounts["free"]
            else:
                other_value += value
        valued.append((value if value is not None else -1.0, asset, amounts, value))
    valued.sort(reverse=True)

    dust_cutoff = max(1.0, total_value * 0.001)
    # Anything we could not price is shown rather than hidden: an unknown
    # holding might be the largest thing in the account.
    unpriceable = [v for v in valued if v[3] is None]
    material = [v for v in valued if v[3] is not None
                and (v[3] >= dust_cutoff or v[1] == quote)]
    shown = (material + unpriceable)[:args.top]
    hidden = len(valued) - len(shown)

    print("  {:<8} {:>16} {:>16}".format("asset", "free", "value in " + quote))
    print("  " + "-" * 42)
    for _, asset, amounts, value in shown:
        print("  {:<8} {:>16.8f} {:>16}".format(
            asset, amounts["free"], "n/a" if value is None else "{:,.2f}".format(value)))
    if hidden > 0:
        print("  {:<8} {:>16} {:>16}".format(
            "+{} more".format(hidden), "", "below {:,.2f}".format(dust_cutoff)))
    print("  " + "-" * 42)
    print("  {:<8} {:>16} {:>16,.2f}".format("total", "", total_value))

    print()
    needed = cfg.portfolio.initial_capital_usdt
    print("The bot is configured to trade {:,.2f} {}.".format(needed, quote))
    if quote_free >= needed:
        print("Funded: {:,.2f} {} free, which covers it.".format(quote_free, quote))
        return 0

    print("NOT ready: only {:,.2f} {} is free.".format(quote_free, quote))
    if other_value > 0:
        print()
        print("You are holding {:,.2f} {} of value in other assets. The strategy buys"
              .format(other_value, quote))
        print("{} with {}, so it needs {} to spend -- holding {} instead means it has"
              .format(cfg.crypto.symbol.split("/")[0], quote, quote,
                      cfg.crypto.symbol.split("/")[0]))
        print("nothing to buy with. Convert what you want it to trade into {}.".format(quote))
        print()
        print("On Binance: Wallet -> Convert, {} -> {}. No trading fee on Convert.".format(
            [a for a in holdings if a != quote][:1][0] if len(holdings) > 1 else "BTC", quote))
    print()
    print("Or lower portfolio.initial_capital_usdt in config.yaml to what you actually have.")
    return 1


def cmd_serve(args, cfg) -> int:
    """Run the bot with a local control panel in front of it."""
    from trading.webui import Controller, serve

    strategy, risk = build(cfg)
    mode = cfg.crypto.mode

    if mode == "live":
        if not args.i_understand_this_is_live:
            log.error("config says mode: live but %s was not passed. Refusing.", LIVE_FLAG)
            return 2
        blocker = _validation_blocks_live(cfg)
        if blocker and not args.skip_validation:
            log.error("refusing to serve a live bot: %s", blocker)
            return 2
        broker = CcxtBroker(cfg.crypto.exchange, testnet=False,
                            credentials=cfg.crypto.credentials(), fee_bps=cfg.crypto.fee_bps)
    elif mode == "testnet":
        broker = CcxtBroker(cfg.crypto.exchange, testnet=True,
                            credentials=cfg.crypto.credentials(), fee_bps=cfg.crypto.fee_bps)
    else:
        broker = PaperBroker(cfg.crypto.exchange, cfg.crypto.fee_bps, cfg.crypto.slippage_bps)

    if args.symbol:
        cfg.crypto.symbol = args.symbol
    state = StateStore(args.state or os.path.join(
        ROOT, "state.{}.{}.json".format(broker.mode, cfg.crypto.symbol.replace("/", "").lower())))
    portfolio, supervisor = load_account(cfg, state)
    gate = load_gate(cfg, required=False)

    engine = TradingEngine(cfg, broker, strategy, risk, state,
                           dry_run=args.dry_run,
                           journal_path=os.path.join(ROOT, "logs", "orders.jsonl"),
                           evidence=gate, portfolio=portfolio, supervisor=supervisor)
    controller = Controller(engine, cfg.crypto.poll_seconds,
                            label="{} {}".format(broker.mode, cfg.crypto.symbol))

    if args.host not in ("127.0.0.1", "localhost"):
        log.warning("*** binding to %s exposes start/stop controls beyond this "
                    "machine. Only do this behind a firewall or a VPN. ***", args.host)

    server, token = serve(controller, args.host, args.port, args.token)
    url = "http://{}:{}/?token={}".format(
        "localhost" if args.host == "127.0.0.1" else args.host, args.port, token)

    print()
    print("Control panel: {}".format(url))
    print("  mode: {}   symbol: {}   dry-run: {}".format(
        broker.mode, cfg.crypto.symbol, args.dry_run))
    print("  The panel is not the bot -- press Start on the page, or use --autostart.")
    print("  Keep this token private; it can start and stop live trading.")
    print()

    if args.autostart:
        controller.start()
        print("Trading loop started automatically.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping ...")
        controller.stop()
        server.shutdown()
    return 0


def cmd_status(args, cfg) -> int:
    mode = args.mode or cfg.crypto.mode
    if args.state:
        paths = [args.state]
    else:
        import glob
        paths = sorted(glob.glob(os.path.join(ROOT, "state.{}*.json".format(mode))))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("Nothing has run in {} mode yet.".format(mode))
        return 1
    if len(paths) > 1:
        print("{} instances running in {} mode".format(len(paths), mode))
        print("=" * 60)
    for index, path in enumerate(paths):
        if index:
            print("\n" + "-" * 60 + "\n")
        print("[{}]".format(os.path.basename(path)))
        _print_one_state(path, cfg, args)
    return 0


def _print_one_state(path, cfg, args) -> None:
    state = StateStore(path)
    portfolio, supervisor = load_account(cfg, state)
    now = int(time.time() * 1000)

    print(portfolio.summary())
    print()
    print(supervisor.summary(now))
    print()
    positions = state.positions
    if positions:
        print("Open positions")
        for pos in positions:
            print("  {} {:.6f} @ {:.2f}  stop {:.2f}  target {:.2f}".format(
                pos.symbol, pos.qty, pos.entry_price, pos.stop_price, pos.take_profit_price))
    else:
        print("Open positions       none")

    history = state.data.get("history", [])[-args.history:]
    if history:
        print()
        print("Recent activity")
        for row in history:
            if row.get("event") == "exit":
                print("  {}  exit {:<12} pnl {:+.2f}".format(
                    row.get("at", "")[:16], row.get("reason", ""), row.get("pnl", 0.0)))
            else:
                print("  {}  {} {:.6f} @ {:.2f}".format(
                    row.get("at", "")[:16], row.get("event", ""),
                    row.get("qty", 0.0), row.get("price", 0.0)))
    return 0


def cmd_simulate(args, cfg) -> int:
    """Run the whole stack -- gate, risk, supervisor, portfolio -- over history."""
    strategy, risk = build(cfg)
    symbol = args.symbol or cfg.crypto.symbol
    timeframe = args.timeframe or cfg.crypto.timeframe

    if args.csv:
        candles = data_mod.load_csv(args.csv)
    else:
        exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
        log.info("fetching %s %s for %.0f days ...", symbol, timeframe, args.days)
        candles = data_mod.fetch_ohlcv(exchange, symbol, timeframe, args.days)
    if not candles:
        log.error("no candles to simulate")
        return 1

    if args.min_success is not None:
        cfg.evidence.min_success_rate = args.min_success
    if args.min_analogues is not None:
        cfg.evidence.min_similar_situations = args.min_analogues
    gate = load_gate(cfg, required=False) if not args.no_evidence else None

    portfolio = Portfolio(cfg.portfolio)
    supervisor = Supervisor(cfg.supervisor, cfg.portfolio.initial_capital_usdt)
    risk.set_allocated(cfg.portfolio.initial_capital_usdt)

    tester = Backtester(strategy, risk, cfg.crypto.fee_bps, cfg.crypto.slippage_bps,
                        evidence=gate)
    result = tester.run(candles, symbol=symbol, timeframe=timeframe,
                        portfolio=portfolio, supervisor=supervisor)

    print()
    print(result.summary(bars_per_year=data_mod.bars_per_year(timeframe)))
    print()
    print(portfolio.summary())
    print()
    days = (candles[-1].timestamp - candles[0].timestamp) / 86_400_000.0
    print("Over {:.0f} days: {:.2f} -> {:.2f} ({:+.2f}%)".format(
        days, cfg.portfolio.initial_capital_usdt, portfolio.equity, portfolio.growth_pct))
    if days > 0:
        annualised = ((portfolio.equity / cfg.portfolio.initial_capital_usdt)
                      ** (365.0 / days) - 1.0) * 100.0
        print("Annualised at this rate: {:+.1f}%".format(annualised))
    return 0


VALIDATION_FILE = "data/validation.json"


def _validation_path() -> str:
    return os.path.join(ROOT, VALIDATION_FILE)


def _window_candles(cfg, symbol, timeframe, days):
    """Prefer the local corpus; fall back to the exchange."""
    path = corpus_mod.series_path(corpus_mod.corpus_dir(ROOT), symbol, timeframe)
    if os.path.exists(path):
        candles = data_mod.load_csv(path)
        cutoff = candles[-1].timestamp - int(days * 86_400_000)
        sliced = [c for c in candles if c.timestamp >= cutoff]
        if len(sliced) > 200:
            return sliced, "corpus"
    exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
    return data_mod.fetch_ohlcv(exchange, symbol, timeframe, days), "exchange"


def cmd_validate(args, cfg) -> int:
    """Would this configuration have made money? Several windows, one verdict.

    Nothing here is optional flattery: a window that loses counts against the
    verdict, and the verdict is what live mode checks before it will run.
    """
    symbols = args.symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    timeframe = args.timeframe or cfg.crypto.timeframe
    gate = None if args.no_evidence else load_gate(cfg, required=False)

    print("Validating the current settings over {} windows of {:.0f} days".format(
        len(symbols), args.days))
    print("=" * 72)
    print("  {:<12} {:>10} {:>10} {:>9} {:>8} {:>9}".format(
        "market", "start", "end", "return", "trades", "max DD"))
    print("  " + "-" * 62)

    rows = []
    for symbol in symbols:
        candles, source = _window_candles(cfg, symbol, timeframe, args.days)
        strategy, risk = build(cfg)
        if not candles or len(candles) < strategy.warmup_bars + 50:
            print("  {:<12} no usable data".format(symbol))
            continue
        portfolio = Portfolio(cfg.portfolio)
        supervisor = Supervisor(cfg.supervisor, cfg.portfolio.initial_capital_usdt)
        risk.set_allocated(cfg.portfolio.initial_capital_usdt)
        result = Backtester(strategy, risk, cfg.crypto.fee_bps, cfg.crypto.slippage_bps,
                            evidence=gate).run(candles, symbol=symbol, timeframe=timeframe,
                                               portfolio=portfolio, supervisor=supervisor)
        row = {
            "symbol": symbol, "source": source, "bars": result.bars,
            "start": result.start, "end": result.end,
            "return_pct": portfolio.growth_pct, "trades": len(result.trades),
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate_pct": result.win_rate_pct,
            "buy_hold_pct": result.buy_hold_return_pct,
        }
        rows.append(row)
        print("  {:<12} {:>10} {:>10} {:>8.2f}% {:>8d} {:>8.2f}%".format(
            symbol, _fmt(result.start)[:10], _fmt(result.end)[:10],
            row["return_pct"], row["trades"], row["max_drawdown_pct"]))

    if not rows:
        log.error("no windows could be evaluated")
        return 1

    winners = [r for r in rows if r["return_pct"] > 0]
    traded = [r for r in rows if r["trades"] > 0]
    average = sum(r["return_pct"] for r in rows) / len(rows)
    worst_dd = min(r["max_drawdown_pct"] for r in rows)
    total_trades = sum(r["trades"] for r in rows)

    print()
    print("  windows profitable   {}/{}".format(len(winners), len(rows)))
    print("  average return       {:+.2f}%".format(average))
    print("  worst drawdown       {:.2f}%".format(worst_dd))
    print("  trades taken         {}".format(total_trades))

    reasons = []
    if not traded:
        reasons.append("it never traded, so there is nothing to validate")
    if average <= 0:
        reasons.append("average return across windows is {:+.2f}%".format(average))
    if len(winners) < len(rows) * args.min_win_share:
        reasons.append("only {}/{} windows made money".format(len(winners), len(rows)))
    if worst_dd < -cfg.portfolio.max_total_drawdown_pct:
        reasons.append("a window drew down {:.1f}%, past the {:.0f}% ceiling".format(
            worst_dd, cfg.portfolio.max_total_drawdown_pct))

    passed = not reasons
    verdict = {
        "at": _now_iso(), "passed": passed, "reasons": reasons, "windows": rows,
        "average_return_pct": average, "worst_drawdown_pct": worst_dd,
        "total_trades": total_trades, "timeframe": timeframe,
        "config": _validated_shape(cfg, gate is not None),
    }
    os.makedirs(os.path.dirname(_validation_path()), exist_ok=True)
    with open(_validation_path(), "w") as fh:
        json.dump(verdict, fh, indent=2)

    print()
    if passed:
        print("PASS -- these settings made money across the windows tested.")
        print("That is a necessary condition for going live, not a promise.")
    else:
        print("FAIL -- do not fund this configuration:")
        for reason in reasons:
            print("  - {}".format(reason))
    print()
    print("Written to {}".format(_validation_path()))
    return 0 if passed else 1


def _validated_shape(cfg, evidence_enabled: bool) -> dict:
    """Everything that would change the result if it changed.

    A validation is only evidence about the settings it actually ran. Recording
    the stop and target alone once let a PASS earned on the 4h configuration
    unlock live trading for the 1h one, which loses money.
    """
    return {
        "timeframe": cfg.crypto.timeframe,
        "stop_loss_pct": cfg.risk.stop_loss_pct,
        "take_profit_pct": cfg.risk.take_profit_pct,
        "max_position_size_pct": cfg.risk.max_position_size_pct,
        "buy_threshold": cfg.strategy.buy_threshold,
        "sell_threshold": cfg.strategy.sell_threshold,
        "fee_bps": cfg.crypto.fee_bps,
        "slippage_bps": cfg.crypto.slippage_bps,
        "evidence_enabled": evidence_enabled,
        "min_similar_situations": cfg.evidence.min_similar_situations,
        "min_success_rate": cfg.evidence.min_success_rate,
        "evidence_library": cfg.evidence.library,
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validation_blocks_live(cfg, max_age_days: float = 30.0):
    """Live mode needs a recent PASS on disk. Returns a reason, or None."""
    path = _validation_path()
    if not os.path.exists(path):
        return ("no validation on record -- run `python main.py validate` first")
    try:
        with open(path) as fh:
            verdict = json.load(fh)
    except (ValueError, OSError) as exc:
        return "validation file unreadable ({})".format(exc)
    if not verdict.get("passed"):
        return "the last validation FAILED: " + "; ".join(verdict.get("reasons", []))
    from datetime import datetime, timezone
    try:
        when = datetime.fromisoformat(verdict["at"])
    except (KeyError, ValueError):
        return "validation has no usable timestamp"
    age_days = (datetime.now(timezone.utc) - when).total_seconds() / 86_400.0
    if age_days > max_age_days:
        return "the last validation is {:.0f} days old (limit {:.0f})".format(age_days, max_age_days)
    stored = verdict.get("config") or {}
    current = _validated_shape(cfg, verdict.get("config", {}).get("evidence_enabled", True))
    drifted = []
    for key, value in current.items():
        if key == "evidence_enabled":
            continue
        was = stored.get(key)
        if was is None:
            drifted.append("{} was not recorded".format(key))
        elif isinstance(value, float) and isinstance(was, (int, float)):
            if abs(float(was) - value) > 1e-9:
                drifted.append("{}: validated {}, now {}".format(key, was, value))
        elif was != value:
            drifted.append("{}: validated {!r}, now {!r}".format(key, was, value))
    if drifted:
        return ("the settings changed since the last validation ({}) -- re-run it"
                .format("; ".join(drifted[:3])))
    return None


def cmd_publish(args, cfg) -> int:
    """Write the snapshot the web dashboard reads.

    Only what is safe to serve publicly: settings, the recorded validation
    verdict, and portfolio totals. No keys, no order history, no addresses.
    """
    import glob

    # Account details are OFF by default. The page is served from a host that
    # cannot be locked without a paid plan, so the safe assumption is that
    # anything published is public. Settings and validation are already in the
    # README; balances and open positions are not.
    include_account = bool(args.include_account)

    out = {
        "generated_at": _now_iso(),
        "account_included": include_account,
        "mode": cfg.crypto.mode,
        "exchange": cfg.crypto.exchange,
        "timeframe": cfg.crypto.timeframe,
        "settings": {
            "capital": cfg.portfolio.initial_capital_usdt,
            "position_pct": cfg.risk.max_position_size_pct,
            "stop_pct": cfg.risk.stop_loss_pct,
            "target_pct": cfg.risk.take_profit_pct,
            "daily_loss_pct": cfg.risk.max_daily_loss_pct,
            "max_drawdown_pct": cfg.portfolio.max_total_drawdown_pct,
            "analogues": cfg.evidence.min_similar_situations,
            "min_success_rate": cfg.evidence.min_success_rate,
        },
        "instances": [],
        "validation": None,
    }

    path = _validation_path()
    if os.path.exists(path):
        with open(path) as fh:
            v = json.load(fh)
        out["validation"] = {
            "at": v.get("at"), "passed": v.get("passed"),
            "average_return_pct": v.get("average_return_pct"),
            "worst_drawdown_pct": v.get("worst_drawdown_pct"),
            "total_trades": v.get("total_trades"),
            "windows": [{"symbol": w["symbol"], "return_pct": w["return_pct"],
                         "trades": w["trades"]} for w in v.get("windows", [])],
            "reasons": v.get("reasons", []),
        }

    for state_path in (sorted(glob.glob(os.path.join(ROOT, "state.*.json")))
                       if include_account else []):
        try:
            state = StateStore(state_path)
        except (ValueError, OSError):
            continue
        portfolio, supervisor = load_account(cfg, state)
        name = os.path.basename(state_path)[len("state."):-len(".json")]
        out["instances"].append({
            "name": name,
            "equity": round(portfolio.equity, 2),
            "allocated": round(portfolio.state.allocated, 2),
            "reserve": round(portfolio.state.reserve, 2),
            "growth_pct": round(portfolio.growth_pct, 2),
            "trades": portfolio.state.trades,
            "wins": portfolio.state.wins,
            "losses": portfolio.state.losses,
            "halted": portfolio.state.halted,
            "paused_until_ms": supervisor.state.paused_until_ms,
            "pause_reason": supervisor.state.pause_reason,
            "open_positions": [
                {"symbol": p.symbol, "qty": p.qty, "entry": p.entry_price,
                 "stop": p.stop_price, "target": p.take_profit_price}
                for p in state.positions
            ],
        })

    target = args.out or os.path.join(ROOT, "web", "status.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as fh:
        json.dump(out, fh, indent=2)

    # This file carries balances and open positions. Vercel can be locked
    # behind a login; a public git repository cannot, so refuse to leave it
    # tracked rather than relying on anyone noticing.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", target],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if tracked:
        log.error("%s is tracked by git. It holds balances and open positions, "
                  "and this repository is published. Run:\n"
                  "  git rm --cached %s && echo web/status.json >> .gitignore",
                  os.path.basename(target), target)
        return 1
    print("Wrote {} ({}, validation {})".format(
        target,
        "{} instance(s)".format(len(out["instances"])) if include_account
        else "no account data -- safe to publish",
        "PASS" if (out["validation"] or {}).get("passed") else "FAIL/none"))
    if include_account:
        print("WARNING: this snapshot contains balances and open positions. "
              "Do not deploy it to a publicly reachable host.")
    return 0


def cmd_plan(args, cfg) -> int:
    p = cfg.portfolio
    r = cfg.risk
    s = cfg.supervisor
    per_trade = p.initial_capital_usdt * r.max_position_size_pct / 100.0
    worst_trade = per_trade * r.stop_loss_pct / 100.0
    print("The plan for {:.0f} USDT over a year".format(p.initial_capital_usdt))
    print("=" * 68)
    print()
    print("What it risks")
    print("  per trade            {:>8.2f} USDT position, {:.2f} USDT to the stop".format(
        per_trade, worst_trade))
    print("  worst day            {:>8.2f} USDT, then it stops until UTC midnight".format(
        p.initial_capital_usdt * r.max_daily_loss_pct / 100.0))
    print("  worst week           {:>8.2f} USDT, then it pauses {:.0f} days".format(
        p.initial_capital_usdt * s.weekly_loss_pct / 100.0, s.weekly_pause_days))
    print("  worst case ever      {:>8.2f} USDT ({:.0f}% of peak), then it stops for good".format(
        p.initial_capital_usdt * p.max_total_drawdown_pct / 100.0, p.max_total_drawdown_pct))
    print()
    print("What stops it getting greedy")
    print("  position size is a fixed {:.0f}% of the trading balance -- a winning".format(
        r.max_position_size_pct))
    print("  streak never buys bigger, and a losing streak never doubles down")
    print("  compounding: {} (ceiling {:.2f} USDT)".format(
        p.compounding, p.initial_capital_usdt * (1 + p.compounding_cap_pct / 100.0)))
    print("  every +{:.0f} USDT realised moves {:.0f}% into a reserve it cannot trade".format(
        p.profit_lock_step_usdt, p.profit_lock_share_pct))
    print("  up {:.0f}% on the day and it stops trading until tomorrow".format(
        s.daily_profit_target_pct))
    print("  at most {} trades a day".format(s.max_trades_per_day))
    print()
    print("When it pauses, and for how long")
    print("  {} losses in a row      -> {:.0f} h".format(
        s.max_consecutive_losses, s.loss_streak_pause_hours))
    print("  daily loss limit hit    -> until UTC midnight")
    print("  weekly loss limit hit   -> {:.0f} days".format(s.weekly_pause_days))
    print("  data older than {:.0f} min  -> until fresh data arrives".format(s.stale_data_minutes))
    print("  {} exchange errors       -> {:.0f} min".format(
        s.max_error_streak, s.error_streak_pause_minutes))
    print("  ATR above {:.1f}% of price -> sits out until it calms".format(s.max_volatility_pct))
    print("  total drawdown {:.0f}%      -> stops for good, needs you to restart it".format(
        p.max_total_drawdown_pct))
    print()
    print("What has to be true before it trades at all")
    print("  a BUY from the rules, AND")
    print("  {:,} similar historical situations found, AND".format(
        cfg.evidence.min_similar_situations))
    print("  at least {:.0f}% of them reached target before stop".format(
        cfg.evidence.min_success_rate * 100))
    print("  (95% confidence lower bound, not the raw rate)")
    print()
    print("Run `python main.py evidence check` to see what that gate says today,")
    print("and `python main.py simulate --days 365` for what a year would have")
    print("looked like on real data with these exact settings.")
    return 0


def cmd_stocks(args, cfg) -> int:
    if not cfg.stocks.enabled and not args.symbols:
        log.info("stocks disabled in config.yaml")
        return 0
    strategy, risk = build(cfg)
    symbols = args.symbols or cfg.stocks.symbols
    journal = os.path.join(ROOT, "logs", "stock_signals.jsonl")
    any_ok = False
    for symbol in symbols:
        record = signal_for(symbol, strategy, cfg.stocks.lookback_days,
                            csv_dir=args.csv_dir or os.path.join(ROOT, "data", "stocks"))
        if record is None:
            print("{:<6} no usable data".format(symbol))
            continue
        any_ok = True
        notional = cfg.risk.allocated_capital_usdt * cfg.risk.max_position_size_pct / 100.0
        print("\n{:<6} {:<5} score {:+.2f}  close {:.2f}  ({})".format(
            record["symbol"], record["action"], record["score"], record["price"], record["bar_time"]))
        for reason in record["reasons"]:
            print("        - {}".format(reason))
        if record["action"] == "BUY":
            levels = risk.exit_levels(record["price"], "long")
            print("        sizing (manual): ~{:.2f} of capital = {:.4f} shares, "
                  "stop {:.2f}, target {:.2f}".format(
                      notional, notional / record["price"],
                      levels["stop_price"], levels["take_profit_price"]))
        append_journal(record, journal)
    if any_ok:
        print("\nInformational only -- no US broker API here, so nothing was ordered.")
        print("Logged to {}".format(journal))
    return 0 if any_ok else 1


def cmd_corpus(args, cfg) -> int:
    directory = corpus_mod.corpus_dir(ROOT)
    items = corpus_mod.inventory(directory)
    if not items:
        print("No corpus yet. Build one with:")
        print("  .venv/bin/python scripts/build_corpus.py --years 8")
        return 1
    by_tf = {}
    for item in items:
        by_tf.setdefault(item["timeframe"], [0, 0])
        by_tf[item["timeframe"]][0] += 1
        by_tf[item["timeframe"]][1] += item["bars"]
    print("Corpus at {}".format(directory))
    print("-" * 68)
    for timeframe, (series, bars) in sorted(by_tf.items()):
        print("  {:<5} {:>3} series  {:>12,} bars".format(timeframe, series, bars))
    total = sum(item["bars"] for item in items)
    size = sum(item["mb"] for item in items)
    print("-" * 68)
    print("  total {:>3} series  {:>12,} bars  ({:.0f} MB on disk)".format(len(items), total, size))
    if args.verbose:
        print()
        for item in sorted(items, key=lambda x: -x["bars"]):
            print("  {:<14} {:<4} {:>9,} bars".format(item["symbol"], item["timeframe"], item["bars"]))
    return 0


def cmd_evidence(args, cfg) -> int:
    directory = corpus_mod.corpus_dir(ROOT)
    library_path = cfg.evidence.library
    if not os.path.isabs(library_path):
        library_path = os.path.join(ROOT, library_path)

    if args.action == "build":
        # Default to the timeframe you actually trade. A 15m bar and a 1h bar
        # do not mean the same thing to a 48-bar horizon, so mixing them in one
        # library would make "similar" mean two different things per row.
        timeframes = (args.timeframes.split(",") if args.timeframes
                      else [cfg.crypto.timeframe])
        params = LibraryParams(
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            horizon_bars=cfg.evidence.horizon_bars,
            timeframes=tuple(timeframes or ()),
        )
        log.info("labelling situations: stop -%.2f%%, target +%.2f%%, horizon %d bars",
                 params.stop_loss_pct, params.take_profit_pct, params.horizon_bars)
        if len(timeframes) == 1 and "{}".format(timeframes[0]) not in os.path.basename(library_path):
            log.warning("library path %s does not mention the %s timeframe it is being "
                        "built from -- keep them in step", library_path, timeframes[0])
        library = AnalogueLibrary.build(corpus_series(directory, timeframes), params)
        library.save(library_path)
        print()
        print("Timeframe: {}".format(", ".join(timeframes)))
        base = library.base_rate()
        print()
        print("Built {:,} labelled situations from {} series".format(
            len(library), len(library.series_names)))
        print("Saved to {} ({:.0f} MB)".format(library_path, os.path.getsize(library_path) / 1e6))
        print()
        print("Corpus base rate: {:.1f}% of all situations reached +{:.1f}% before -{:.1f}%".format(
            base * 100, params.take_profit_pct, params.stop_loss_pct))
        print("That is the number any setup has to beat to be worth anything.")
        return 0

    if args.action == "check":
        gate = load_gate(cfg, required=True)
        symbol = args.symbol or cfg.crypto.symbol
        timeframe = args.timeframe or cfg.crypto.timeframe
        strategy, risk = build(cfg)
        exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
        days = max(2.0, (strategy.warmup_bars + 30) * data_mod.timeframe_ms(timeframe) / 86_400_000.0)
        candles = data_mod.fetch_ohlcv(exchange, symbol, timeframe, days)
        signal = strategy.evaluate(candles)
        vector = situation_at(candles)
        verdict = gate.check(vector, as_of_ms=None)

        print("{}  {}  bar {}".format(symbol, timeframe, _fmt(signal.timestamp)))
        print("=" * 68)
        print("Rules say        {} (score {:+.2f}) @ {:.2f}".format(
            signal.action, signal.score, signal.price))
        print()
        print("Situation now")
        if vector is not None:
            from trading.analogues import FEATURES
            for name, value in zip(FEATURES, vector):
                print("  {:<20} {:>10.3f}".format(name, value))
        print()
        print("History")
        print("  library holds        {:>12,} resolved situations".format(len(gate.library)))
        print("  analogues consulted  {:>12,}".format(verdict.matches))
        print("  reached target first {:>12,}".format(verdict.wins))
        print("  stopped out first    {:>12,}".format(verdict.losses))
        print("  success rate         {:>11.1f}%   (95% CI {:.1f}-{:.1f}%)".format(
            verdict.success_rate * 100, verdict.lower_bound * 100, verdict.upper_bound * 100))
        print("  corpus base rate     {:>11.1f}%   (all situations, not just similar ones)".format(
            gate.library.base_rate() * 100))
        print("  similarity radius    {:>12.2f}   (median {:.2f})".format(
            verdict.max_distance, verdict.median_distance))
        print("  typical time to res. {:>12.1f} bars".format(verdict.mean_forward_bars))
        print()
        print("Cohort size vs. what it tells you")
        print("  {:>10} {:>9} {:>16} {:>9}".format("analogues", "success", "95% CI", "radius"))
        print("  " + "-" * 48)
        required = cfg.evidence.min_similar_situations
        ladder = sorted({k for k in (100, 1_000, 10_000, 100_000, required)
                         if k <= len(gate.library) and k <= max(required, 100_000)})
        for k in ladder:
            rung = gate.library.query(vector, min_matches=k, min_success_rate=0.0)
            mark = "  <- required" if k == required else ""
            print("  {:>10,} {:>8.1f}% {:>7.1f}-{:>5.1f}% {:>9.2f}{}".format(
                k, rung.success_rate * 100, rung.lower_bound * 100,
                rung.upper_bound * 100, rung.max_distance, mark))
        print("  {:>10} {:>8.1f}% {:>21}".format(
            "all", gate.library.base_rate() * 100, "(the base rate)"))
        print()
        print("A bigger cohort is a tighter confidence interval bought with looser")
        print("similarity. Where the two lines cross is the real question.")
        print()
        print("Verdict: {} -- {}".format("PASS" if verdict.passed else "BLOCKED", verdict.reason))
        if signal.action == "BUY" and not verdict.passed:
            print("The rules wanted to buy. The evidence gate stops it.")
        return 0

    if args.action == "sweep":
        print("How the stop/target pair moves the success rate, and what it costs.")
        print("Sampled from the corpus: {}".format(args.timeframes or "all timeframes"))
        print()
        timeframes = args.timeframes.split(",") if args.timeframes else None
        loaded = []
        for name, candles in corpus_series(directory, timeframes):
            loaded.append((name, candles))
            if len(loaded) >= args.series:
                break
        if not loaded:
            log.error("no corpus series found -- build the corpus first")
            return 1
        bars = sum(len(c) for _, c in loaded)
        print("{} series, {:,} bars".format(len(loaded), bars))
        print()
        print("  {:>6} {:>7} {:>6} {:>10} {:>9} {:>12}".format(
            "stop", "target", "R:R", "situations", "success", "expectancy"))
        print("  " + "-" * 56)
        for sl, tp in _sweep_grid(cfg):
            wins = losses = 0
            for _, candles in loaded:
                outcome, _, _ = label_outcomes(candles, sl, tp, cfg.evidence.horizon_bars)
                wins += int((outcome == 1).sum())
                losses += int((outcome == 0).sum())
            total = wins + losses
            if not total:
                continue
            rate = wins / total
            expectancy = rate * tp - (1 - rate) * sl        # % of position, per trade
            flag = "  <- clears 80%" if rate >= 0.80 else ""
            print("  {:>5.1f}% {:>6.1f}% {:>6.2f} {:>10,} {:>8.1f}% {:>11.3f}%{}".format(
                sl, tp, tp / sl, total, rate * 100, expectancy, flag))
        print()
        print("Expectancy is the average % move per trade, before fees.")
        print("A high success rate bought by moving the target closer is not an edge.")
        return 0

    log.error("unknown evidence action %r", args.action)
    return 2


def _sweep_grid(cfg):
    """Target sizes from a tenth of the stop up to three times it.

    The point of the wide range is to show where an 80% success rate becomes
    reachable, and what it costs in expectancy to get there.
    """
    stop = cfg.risk.stop_loss_pct
    ratios = (0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    return [(stop, round(stop * r, 4)) for r in ratios]


def cmd_check(args, cfg) -> int:
    strategy, risk = build(cfg)
    print("config          {}".format(cfg.path))
    print("mode            {}".format(cfg.crypto.mode))
    print("exchange        {}  symbol {}  timeframe {}".format(
        cfg.crypto.exchange, cfg.crypto.symbol, cfg.crypto.timeframe))
    print("credentials     {} / {}: {}".format(
        cfg.crypto.api_key_env, cfg.crypto.api_secret_env,
        "present" if cfg.crypto.has_credentials() else "MISSING"))
    print("warmup bars     {}".format(strategy.warmup_bars))
    print("allocated       {:.2f} USDT".format(cfg.risk.allocated_capital_usdt))
    print("per position    {:.2f} USDT ({}%)".format(
        cfg.risk.allocated_capital_usdt * cfg.risk.max_position_size_pct / 100,
        cfg.risk.max_position_size_pct))
    print("stop / target   -{}% / +{}%".format(cfg.risk.stop_loss_pct, cfg.risk.take_profit_pct))
    print("worst case/trade{:>7.2f} USDT".format(
        cfg.risk.allocated_capital_usdt * cfg.risk.max_position_size_pct / 100
        * cfg.risk.stop_loss_pct / 100))
    print("daily kill-switch at -{:.2f} USDT".format(risk.daily_loss_limit()))

    try:
        exchange = data_mod.make_exchange(cfg.crypto.exchange, testnet=False)
        ticker = exchange.fetch_ticker(cfg.crypto.symbol)
        print("market data     OK -- {} last {:.2f}".format(cfg.crypto.symbol, float(ticker["last"])))
    except Exception as exc:
        print("market data     UNREACHABLE: {}".format(exc))
        return 1
    return 0


def _fmt(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


# -- wiring -----------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="run the strategy over historical candles")
    b.add_argument("--symbol")
    b.add_argument("--timeframe")
    b.add_argument("--days", type=float, default=90.0)
    b.add_argument("--csv", help="backtest a saved CSV instead of fetching")
    b.add_argument("--save", help="save fetched candles to this CSV")
    b.add_argument("--synthetic", type=int, metavar="BARS", help="use generated candles (offline)")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--trades", action="store_true", help="print every trade")
    b.add_argument("--json", help="write results to this JSON file")
    b.add_argument("--evidence", action="store_true",
                   help="require the analogue gate to back every entry")
    b.add_argument("--min-success", type=float,
                   help="override evidence.min_success_rate for this run, e.g. 0.4")
    b.add_argument("--min-analogues", type=int,
                   help="override evidence.min_similar_situations for this run")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signal", help="print the current call for a symbol")
    s.add_argument("--symbol")
    s.add_argument("--timeframe")
    s.set_defaults(func=cmd_signal)

    t = sub.add_parser("trade", help="run the trading loop")
    t.add_argument("--once", action="store_true", help="single pass, then exit")
    t.add_argument("--dry-run", action="store_true", help="decide and log, place no orders")
    t.add_argument("--poll", type=int, help="seconds between passes")
    t.add_argument("--state", help="path to the state file")
    t.add_argument("--symbol", help="override the symbol; gets its own state file")
    t.add_argument("--skip-validation", action="store_true",
                   help="go live without a passing validation on record")
    t.add_argument("--no-evidence", action="store_true",
                   help="trade on the rules alone, without the analogue gate")
    t.add_argument(LIVE_FLAG, action="store_true",
                   help="required, with config mode: live, to trade real money")
    t.set_defaults(func=cmd_trade)

    bal = sub.add_parser("balance", help="what is in the exchange account, and is it usable")
    bal.add_argument("--mode", help="paper, testnet or live (default: config)")
    bal.add_argument("--top", type=int, default=12, help="how many holdings to list")
    bal.set_defaults(func=cmd_balance)

    sv = sub.add_parser("serve", help="run the bot with a local web control panel")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--token", help="fixed access token (generated if omitted)")
    sv.add_argument("--symbol")
    sv.add_argument("--state")
    sv.add_argument("--dry-run", action="store_true")
    sv.add_argument("--autostart", action="store_true",
                    help="begin trading immediately instead of waiting for Start")
    sv.add_argument("--skip-validation", action="store_true")
    sv.add_argument(LIVE_FLAG, action="store_true")
    sv.set_defaults(func=cmd_serve)

    st = sub.add_parser("status", help="portfolio, pauses and open positions")
    st.add_argument("--mode", help="which state file to read (paper/testnet/live)")
    st.add_argument("--state", help="explicit path to a state file")
    st.add_argument("--history", type=int, default=10)
    st.set_defaults(func=cmd_status)

    sm = sub.add_parser("simulate", help="a year of the full stack on real history")
    sm.add_argument("--symbol")
    sm.add_argument("--timeframe")
    sm.add_argument("--days", type=float, default=365.0)
    sm.add_argument("--csv")
    sm.add_argument("--min-success", type=float)
    sm.add_argument("--min-analogues", type=int)
    sm.add_argument("--no-evidence", action="store_true")
    sm.set_defaults(func=cmd_simulate)

    v = sub.add_parser("validate", help="would these settings have made money? "
                                        "live mode requires a pass")
    v.add_argument("symbols", nargs="*")
    v.add_argument("--timeframe")
    v.add_argument("--days", type=float, default=365.0)
    v.add_argument("--min-win-share", type=float, default=0.6,
                   help="share of windows that must be profitable (default 0.6)")
    v.add_argument("--no-evidence", action="store_true")
    v.set_defaults(func=cmd_validate)

    pub = sub.add_parser("publish", help="write the snapshot the web dashboard reads")
    pub.add_argument("--out", help="path for status.json")
    pub.add_argument("--include-account", action="store_true",
                     help="include balances and open positions (local viewing only)")
    pub.set_defaults(func=cmd_publish)

    pl = sub.add_parser("plan", help="what it will and will not do with your money")
    pl.set_defaults(func=cmd_plan)

    k = sub.add_parser("stocks", help="print informational US-stock signals")
    k.add_argument("symbols", nargs="*", help="override config.yaml symbols")
    k.add_argument("--csv-dir", help="directory of <SYMBOL>.csv daily OHLCV files")
    k.set_defaults(func=cmd_stocks)

    e = sub.add_parser("evidence", help="the historical-analogue gate")
    e.add_argument("action", choices=["build", "check", "sweep"])
    e.add_argument("--symbol")
    e.add_argument("--timeframe")
    e.add_argument("--timeframes", help="restrict to these corpus timeframes, e.g. 1h,4h")
    e.add_argument("--series", type=int, default=12, help="series to sample in a sweep")
    e.set_defaults(func=cmd_evidence)

    cp = sub.add_parser("corpus", help="what historical data is on disk")
    cp.add_argument("--verbose", action="store_true")
    cp.set_defaults(func=cmd_corpus)

    c = sub.add_parser("check", help="show effective settings and test connectivity")
    c.set_defaults(func=cmd_check)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, os.path.join(ROOT, "logs", "trading.log"))
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error("config error: %s", exc)
        return 2
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
