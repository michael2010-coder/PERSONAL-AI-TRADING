#!/usr/bin/env python3
"""A live view of the running bot, served on this machine only.

    export BOT_HOST=<server> BOT_KEY=~/path/to/key.pem
    python3 scripts/live_dashboard.py            # then open http://localhost:8788

It reads three things and combines them:

  * the bot's state file on the server, over ssh, with `cat` -- read only, so
    it can never race the trading process for the file it owns;
  * the current price from Binance's public endpoint;
  * the daily candles, to recompute the bot's own entry rule independently.

That last one matters. The position in the state file only changes when the bot
trades, which for a daily trend strategy is a dozen times a year. Recomputing
the rule here means the page can say when the trend has broken and the position
on screen is about to be stale, instead of quietly showing an old holding.

Nothing here can place, cancel or modify an order.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

API = "https://api.binance.com/api/v3"
STATE = {"data": None, "error": None, "at": 0}
LOCK = threading.Lock()


def _get(url):
    with urllib.request.urlopen(url, timeout=15) as fh:
        return json.load(fh)


def read_remote_state(args):
    """`cat` the state file over ssh. Read only, and never sudo."""
    out = subprocess.run(
        ["ssh", "-i", args.key, "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
         "{}@{}".format(args.user, args.host), "cat " + args.state],
        capture_output=True, text=True, timeout=40)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or "ssh failed").strip().splitlines()[-1])
    return json.loads(out.stdout)


def service_health(args):
    """Whether systemd still has the bot running, and since when."""
    out = subprocess.run(
        ["ssh", "-i", args.key, "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
         "{}@{}".format(args.user, args.host),
         "systemctl is-active {0}; systemctl show {0} "
         "-p ActiveEnterTimestamp -p NRestarts --value".format(args.service)],
        capture_output=True, text=True, timeout=40)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    return {"active": lines[0] if lines else "unknown",
            "since": lines[1] if len(lines) > 1 else "",
            "restarts": lines[2] if len(lines) > 2 else ""}


def collect(args):
    state = read_remote_state(args)
    price = float(_get("{}/ticker/price?symbol={}".format(API, args.pair))["price"])

    # Only closed candles, matching what the bot sees on its own poll.
    now_ms = time.time() * 1000
    candles = [c for c in _get("{}/klines?symbol={}&interval=1d&limit={}".format(
        API, args.pair, args.ma + 2)) if c[6] < now_ms]
    closes = [float(c[4]) for c in candles]
    ma = sum(closes[-args.ma:]) / args.ma
    last_close = closes[-1]

    pos = (state.get("positions") or [None])[0]
    pf = state.get("portfolio", {})
    sup = state.get("supervisor", {})

    out = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
        "price": price,
        "last_close": last_close,
        "ma": ma,
        "ma_days": args.ma,
        "above_trend": last_close > ma,
        "trend_gap_pct": (last_close - ma) / ma * 100,
        "position": None,
        "closed_trades": pf.get("trades", 0),
        "wins": pf.get("wins", 0),
        "losses": pf.get("losses", 0),
        "realised": pf.get("realised_pnl", 0.0),
        "initial": pf.get("initial_capital", 0.0),
        "halted": pf.get("halted", False),
        "halt_reason": pf.get("halt_reason", ""),
        "paused": bool(sup.get("paused_until_ms")),
        "pause_reason": sup.get("pause_reason", ""),
        "health": service_health(args),
    }

    if pos:
        cost = pos["qty"] * pos["entry_price"]
        value = pos["qty"] * price
        out["position"] = {
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "entry": pos["entry_price"],
            "stop": pos["stop_price"],
            "cost": cost,
            "value": value,
            "pnl": value - cost,
            "pnl_pct": (value - cost) / cost * 100 if cost else 0.0,
            "opened_at": pos.get("opened_at"),
            "held_days": (now_ms - pos.get("opened_at", now_ms)) / 86400000,
            # A stop the exchange refused sits in this process only. If the
            # process dies, nothing on the exchange protects the position.
            "stop_resting": bool(pos.get("stop_order_id")),
            "stop_distance_pct": (price - pos["stop_price"]) / price * 100,
        }
        out["equity"] = pf.get("allocated", 0.0) + (value - cost)
    else:
        out["equity"] = pf.get("allocated", 0.0)
    return out


def poller(args):
    while True:
        try:
            data = collect(args)
            with LOCK:
                STATE.update(data=data, error=None, at=time.time())
        except Exception as exc:                       # noqa: BLE001
            with LOCK:
                STATE.update(error="{}: {}".format(type(exc).__name__, exc),
                             at=time.time())
        time.sleep(args.interval)


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Live trading</title><style>
:root{--bg:#faf9f7;--panel:#fff;--line:#e6e2dc;--ink:#1a1a1a;--muted:#6b6660;
--ok:#1f6f4a;--warn:#9a5b1e;--bad:#a33228;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1c1c1a;--line:#2e2d2a;
--ink:#f0eee9;--muted:#9b958c;--ok:#5bbd8b;--warn:#d99a4e;--bad:#e0705f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:36px 20px 70px}
h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0 0 26px;font-size:.9rem}
h2{font-size:.82rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
margin:30px 0 12px;font-weight:600}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;gap:14px;padding:7px 0;border-bottom:1px dashed var(--line)}
.row:last-child{border-bottom:0}.row .k{color:var(--muted)}
.row .v{font-family:var(--mono);text-align:right}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.stat .n{font-family:var(--mono);font-size:1.4rem;display:block;letter-spacing:-.02em}
.stat .l{color:var(--muted);font-size:.8rem}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.pill{display:inline-block;font-size:.72rem;font-weight:600;padding:3px 9px;
border-radius:999px;border:1px solid currentColor}
.note{border-left:3px solid var(--warn);padding:2px 0 2px 14px;color:var(--muted);
margin:14px 0 0;font-size:.88rem}
footer{margin-top:36px;color:var(--muted);font-size:.82rem}
</style></head><body><div class=wrap>
<h1>Live trading</h1>
<p class=sub id=sub>connecting&hellip;</p>
<div id=main></div>
<footer id=foot></footer></div>
<script>
const n=(x,d=2)=>Number(x).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const sg=x=>(x>=0?"+":"")+n(x);
async function tick(){
  let d;
  try{ d = await (await fetch("data.json?"+Date.now())).json(); }
  catch(e){ document.getElementById("sub").textContent="dashboard unreachable"; return; }
  const f=document.getElementById("foot");
  if(d.error){
    document.getElementById("main").innerHTML=
      `<div class="card"><span class="bad">Could not read the bot: `+d.error+`</span></div>`;
    document.getElementById("sub").textContent="error";
    return;
  }
  const s=d.data; if(!s){ return; }
  const p=s.position;
  const live=s.health.active==="active";
  document.getElementById("sub").innerHTML=
    `<span class="pill ${live?'ok':'bad'}">${live?'RUNNING':s.health.active.toUpperCase()}</span>`+
    `<span style="margin-left:10px">${s.health.restarts||0} restarts &middot; since ${s.health.since||'?'}</span>`;
  let html=`<div class="grid">
    <div class="stat"><span class="n">${n(s.price)}</span><span class="l">BTC/USDT now</span></div>
    <div class="stat"><span class="n">${n(s.equity)}</span><span class="l">equity USDT</span></div>
    <div class="stat"><span class="n ${p&&p.pnl>=0?'ok':(p?'bad':'')}">${p?sg(p.pnl):'--'}</span><span class="l">unrealised USDT</span></div>
    <div class="stat"><span class="n">${s.closed_trades}</span><span class="l">closed trades</span></div></div>`;
  html+=`<h2>Signal</h2><div class="card">
    <div class="row"><span class="k">Last daily close</span><span class="v">${n(s.last_close)}</span></div>
    <div class="row"><span class="k">${s.ma_days}-day average</span><span class="v">${n(s.ma)}</span></div>
    <div class="row"><span class="k">Rule</span><span class="v ${s.above_trend?'ok':'bad'}">
      ${s.above_trend?'above trend &rarr; hold':'BELOW TREND &rarr; exit due'}</span></div>
    <div class="row"><span class="k">Distance</span><span class="v ${s.above_trend?'ok':'bad'}">${sg(s.trend_gap_pct)}%</span></div>
    ${s.above_trend?'':'<p class="note">The close has fallen below the average. The bot exits on its next poll; the position below is about to change.</p>'}
  </div>`;
  if(p){
    html+=`<h2>Open position</h2><div class="card">
      <div class="row"><span class="k">${p.symbol}</span><span class="v">${p.qty} @ ${n(p.entry)}</span></div>
      <div class="row"><span class="k">Held</span><span class="v">${n(p.held_days,1)} days</span></div>
      <div class="row"><span class="k">Cost / value</span><span class="v">${n(p.cost)} &rarr; ${n(p.value)}</span></div>
      <div class="row"><span class="k">Unrealised</span><span class="v ${p.pnl>=0?'ok':'bad'}">${sg(p.pnl)} USDT (${sg(p.pnl_pct)}%)</span></div>
      <div class="row"><span class="k">Stop</span><span class="v">${n(p.stop)} &middot; ${n(p.stop_distance_pct,1)}% below</span></div>
      <div class="row"><span class="k">Stop resting on exchange</span><span class="v ${p.stop_resting?'ok':'warn'}">${p.stop_resting?'yes':'no'}</span></div>
      ${p.stop_resting?'':'<p class="note">The exchange will not rest a stop this far out, so it is enforced by the bot process only. If that process stops, nothing on the exchange protects this position.</p>'}
    </div>`;
  } else {
    html+=`<h2>Open position</h2><div class="card"><span class="k">flat &mdash; no position open</span></div>`;
  }
  if(s.halted||s.paused){
    html+=`<div class="card"><span class="bad">${s.halted?'HALTED: '+s.halt_reason:'PAUSED: '+s.pause_reason}</span></div>`;
  }
  document.getElementById("main").innerHTML=html;
  f.textContent="Bot state and price read "+s.at+". Read-only; this page cannot trade.";
}
tick(); setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):                                  # noqa: N802
        if self.path.startswith("/data.json"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self._send(body, "application/json")
        elif self.path == "/" or self.path.startswith("/?"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                         # quiet
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # No server details are hard-coded. This repository is public, and the
    # address of the box holding live API keys is not something to publish.
    # Set them in the environment (or pass the flags):
    #   export BOT_HOST=... BOT_USER=... BOT_KEY=~/path/to/key.pem
    p.add_argument("--host", default=os.environ.get("BOT_HOST"),
                   help="server running the bot [$BOT_HOST]")
    p.add_argument("--user", default=os.environ.get("BOT_USER", "ubuntu"),
                   help="ssh user [$BOT_USER]")
    p.add_argument("--key", default=os.environ.get("BOT_KEY"),
                   help="ssh private key [$BOT_KEY]")
    p.add_argument("--state", default=os.environ.get(
        "BOT_STATE", "/opt/personal-ai-trading/state.live.btcusdt.json"))
    p.add_argument("--service", default="ai-trading-bot")
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--ma", type=int, default=100, help="must match trend.ma_days")
    p.add_argument("--interval", type=int, default=20, help="seconds between polls")
    p.add_argument("--port", type=int, default=8788)
    args = p.parse_args()
    missing = [n for n in ("host", "key") if not getattr(args, n)]
    if missing:
        p.error("no {} given. Set {} or pass {}.".format(
            " or ".join(missing),
            " and ".join("$BOT_" + m.upper() for m in missing),
            " and ".join("--" + m for m in missing)))
    args.key = os.path.expanduser(args.key)

    threading.Thread(target=poller, args=(args,), daemon=True).start()
    # 127.0.0.1 only: the page carries balances and an open position.
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print("Live dashboard: http://localhost:{}".format(args.port))
    print("  reading {}@{} every {}s -- read only, it cannot trade".format(
        args.user, args.host, args.interval))
    print("  Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
