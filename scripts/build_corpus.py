#!/usr/bin/env python3
"""Download the historical corpus. Safe to interrupt and re-run -- it resumes.

    .venv/bin/python scripts/build_corpus.py --years 8
"""
import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trading import corpus                       # noqa: E402
from trading.data import make_exchange           # noqa: E402
from trading.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="binance")
    p.add_argument("--years", type=float, default=8.0)
    p.add_argument("--symbols", type=int, default=45, help="how many top pairs by volume")
    p.add_argument("--timeframes", default="1h,4h")
    p.add_argument("--fine-timeframe", default="15m")
    p.add_argument("--fine-symbols", type=int, default=12,
                   help="top N symbols also downloaded at the fine timeframe")
    args = p.parse_args()

    setup_logging("INFO", os.path.join(ROOT, "logs", "corpus.log"))
    log = logging.getLogger("trading.corpus")

    directory = corpus.corpus_dir(ROOT)
    exchange = make_exchange(args.exchange, testnet=False)
    symbols = corpus.top_symbols(exchange, limit=args.symbols)
    log.info("corpus target: %d symbols x [%s] over %.1f years, plus %s for the top %d",
             len(symbols), args.timeframes, args.years, args.fine_timeframe, args.fine_symbols)

    corpus.build(exchange, symbols, args.timeframes.split(","), args.years, directory)
    if args.fine_symbols:
        corpus.build(exchange, symbols[:args.fine_symbols], [args.fine_timeframe],
                     args.years, directory)

    bars = corpus.total_bars(directory)
    log.info("corpus now holds %s bars across %d series",
             "{:,}".format(bars), len(corpus.inventory(directory)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
