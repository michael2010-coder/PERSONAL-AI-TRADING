#!/usr/bin/env python3
"""Tell me on Telegram when the bot closes a trade, halts, or pauses.

    python3 scripts/notify_exit.py            # once; run it from cron

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the same .env as the
exchange keys, so no secret is ever written into this repository.

It notifies on a *change*, not on a state, and remembers what it last sent.
A trend position is held for weeks: something that messaged every run would
be ignored within a day, and an alert that gets ignored is not an alert.

Silence means the position is still open and the bot is healthy. That is
only true because halts and pauses are reported here as well -- a watcher
that reports good news only cannot be trusted to be quiet.
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def send(token, chat, text):
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": "true"}).encode()
    req = urllib.request.Request(
        "https://api.telegram.org/bot{}/sendMessage".format(token), data=data)
    with urllib.request.urlopen(req, timeout=20) as fh:
        return json.load(fh).get("ok", False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", default=os.path.join(ROOT, "state.live.btcusdt.json"))
    p.add_argument("--seen", default=os.path.join(ROOT, "logs", "notified.json"))
    p.add_argument("--env", default=os.path.join(ROOT, ".env"))
    p.add_argument("--test", action="store_true", help="send a message and exit")
    args = p.parse_args()

    e = env(args.env)
    token = e.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = e.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("no TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in {}".format(args.env))
        return 2

    if args.test:
        ok = send(token, chat, "✅ <b>Lissafi bot</b>\nNotifications are wired up. "
                               "You will hear from me when a trade closes.")
        print("sent" if ok else "failed")
        return 0 if ok else 1

    with open(args.state) as fh:
        st = json.load(fh)
    pf = st.get("portfolio", {})
    sup = st.get("supervisor", {})
    now = {"trades": pf.get("trades", 0), "halted": bool(pf.get("halted")),
           "paused": bool(sup.get("paused_until_ms"))}

    seen = {}
    if os.path.exists(args.seen):
        try:
            with open(args.seen) as fh:
                seen = json.load(fh)
        except ValueError:
            seen = {}

    msgs = []
    if now["trades"] > seen.get("trades", now["trades"]):
        hist = [h for h in st.get("history", []) if h.get("event") == "exit"]
        last = hist[-1] if hist else {}
        pnl = last.get("pnl", 0.0)
        msgs.append(
            "{} <b>Trade closed</b>\n"
            "{} {:.6f} @ {:.2f}\n"
            "Result: <b>{:+.2f} USDT</b>\n"
            "Reason: {}\n\n"
            "Trading balance: {:.2f}\nLocked reserve: {:.2f}\n"
            "Record: {} trades, {} win / {} loss\n\n"
            "The bot is flat. New capital added now is put to work at the next entry."
            .format("\U0001F7E2" if pnl >= 0 else "\U0001F534",
                    last.get("symbol", "BTC/USDT"), last.get("qty", 0.0),
                    last.get("price", 0.0), pnl, last.get("reason", "signal"),
                    pf.get("allocated", 0.0), pf.get("reserve", 0.0),
                    pf.get("trades", 0), pf.get("wins", 0), pf.get("losses", 0)))
    if now["halted"] and not seen.get("halted"):
        msgs.append("⛔ <b>Trading halted</b>\n{}\nRestarting is a manual decision."
                    .format(pf.get("halt_reason", "")))
    if now["paused"] and not seen.get("paused"):
        msgs.append("⏸ <b>Trading paused</b>\n{}".format(sup.get("pause_reason", "")))

    for m in msgs:
        send(token, chat, m)
        print("sent:", m.splitlines()[0])

    os.makedirs(os.path.dirname(args.seen), exist_ok=True)
    with open(args.seen, "w") as fh:
        json.dump(now, fh)
    if not msgs:
        print("nothing to report (trades={}, halted={}, paused={})".format(
            now["trades"], now["halted"], now["paused"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
