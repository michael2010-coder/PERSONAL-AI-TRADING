"""A local control panel for the bot.

Runs beside the trading loop, not on a public host. It holds the API keys in
the same process that already has them, and it never listens on a public
interface unless explicitly told to. Access needs a token printed at startup.

Why not on the public site: a page anyone can open must not be able to move
money. This one is reachable only from the machine it runs on (or through an
SSH tunnel), which is what makes exposing start/stop controls reasonable.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("trading.webui")


class Controller:
    """Owns the engine and the thread that drives it."""

    def __init__(self, engine, poll_seconds: int, label: str = "") -> None:
        self.engine = engine
        self.poll_seconds = poll_seconds
        self.label = label
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_error = ""
        self.last_pass_at = 0.0
        self.passes = 0
        self.last_signal = ""
        self._wallet = {"free": None, "total": None, "at": 0.0}

    # -- lifecycle -------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        with self._lock:
            if self.running:
                return "already running"
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="trading-loop")
            self._thread.start()
            return "started"

    def stop(self) -> str:
        with self._lock:
            if not self.running:
                return "already stopped"
            self._stop.set()
            return "stopping after the current pass"

    def _loop(self) -> None:
        log.info("control panel started the trading loop (%s)", self.label)
        while not self._stop.is_set():
            try:
                signal = self.engine.step()
                self.last_signal = "{} {} @ {:.2f}".format(
                    signal.action, self.engine.symbol, signal.price)
                self.last_error = ""
                if self.engine.supervisor is not None:
                    self.engine.supervisor.record_ok()
            except Exception as exc:                      # keep the loop alive
                self.last_error = str(exc)[:200]
                log.exception("pass failed: %s", exc)
                if self.engine.supervisor is not None:
                    self.engine.supervisor.record_error(int(time.time() * 1000))
            self.passes += 1
            self.last_pass_at = time.time()
            self._stop.wait(self.poll_seconds)
        log.info("trading loop stopped")

    def _wallet_snapshot(self, quote: str, max_age_s: float = 30.0) -> dict:
        """The real exchange balance, cached so the page does not hammer the API.

        The portfolio below is the bot's own ledger of what it is allowed to
        trade. This is the money that is actually in the account.
        """
        if time.time() - self._wallet["at"] < max_age_s:
            return self._wallet
        try:
            balances = self.engine.broker.balances()
            held = balances.get(quote) or {}
            self._wallet = {"free": round(float(held.get("free") or 0.0), 2),
                            "total": round(float(held.get("total") or 0.0), 2),
                            "at": time.time()}
        except Exception as exc:
            log.debug("wallet read failed: %s", exc)
            self._wallet = {"free": None, "total": None, "at": time.time()}
        return self._wallet

    # -- what the page shows ---------------------------------------------
    def snapshot(self) -> dict:
        engine = self.engine
        portfolio = engine.portfolio
        supervisor = engine.supervisor
        now_ms = int(time.time() * 1000)

        positions = [
            {"symbol": p.symbol, "qty": p.qty, "entry": p.entry_price,
             "stop": p.stop_price, "target": p.take_profit_price,
             "protected": bool(p.stop_order_id)}
            for p in engine.state.positions
        ]
        paused_for = ""
        if supervisor is not None and supervisor.state.paused_until_ms > now_ms:
            mins = (supervisor.state.paused_until_ms - now_ms) / 60000.0
            paused_for = "{} ({:.0f} min left)".format(supervisor.state.pause_reason, mins)

        quote = engine.symbol.split("/")[-1]
        wallet = self._wallet_snapshot(quote)

        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "running": self.running,
            "quote": quote,
            "wallet_free": wallet["free"],
            "wallet_total": wallet["total"],
            "mode": engine.broker.mode,
            "symbol": engine.symbol,
            "timeframe": engine.timeframe,
            "dry_run": engine.dry_run,
            "passes": self.passes,
            "last_pass_ago_s": (int(time.time() - self.last_pass_at)
                                if self.last_pass_at else None),
            "last_signal": self.last_signal,
            "last_error": self.last_error,
            "paused": paused_for,
            "equity": round(portfolio.equity, 2) if portfolio else None,
            "allocated": round(portfolio.state.allocated, 2) if portfolio else None,
            "reserve": round(portfolio.state.reserve, 2) if portfolio else None,
            "growth_pct": round(portfolio.growth_pct, 2) if portfolio else None,
            "trades": portfolio.state.trades if portfolio else 0,
            "wins": portfolio.state.wins if portfolio else 0,
            "losses": portfolio.state.losses if portfolio else 0,
            "halted": portfolio.state.halted if portfolio else False,
            "halt_reason": portfolio.state.halt_reason if portfolio else "",
            "trades_today": supervisor.state.trades_today if supervisor else 0,
            "pnl_today": round(supervisor.state.pnl_today, 2) if supervisor else 0.0,
            "positions": positions,
        }


def make_handler(controller: Controller, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "trading-panel"

        def log_message(self, fmt, *args):      # quieter than the default
            log.debug("%s - %s", self.address_string(), fmt % args)

        # -- auth --------------------------------------------------------
        def _authorised(self, query) -> bool:
            supplied = (self.headers.get("X-Token")
                        or (query.get("token") or [""])[0])
            return secrets.compare_digest(str(supplied), token)

        def _send(self, code, body, content_type="application/json"):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        # -- routes ------------------------------------------------------
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not self._authorised(query):
                self._send(401, json.dumps({"error": "bad or missing token"}))
                return
            if parsed.path == "/":
                self._send(200, PAGE, "text/html")
            elif parsed.path == "/api/status":
                self._send(200, json.dumps(controller.snapshot()))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not self._authorised(query):
                self._send(401, json.dumps({"error": "bad or missing token"}))
                return
            if parsed.path == "/api/start":
                result = controller.start()
            elif parsed.path == "/api/stop":
                result = controller.stop()
            else:
                self._send(404, json.dumps({"error": "not found"}))
                return
            log.info("control panel: %s", result)
            self._send(200, json.dumps({"result": result,
                                        "running": controller.running}))

    return Handler


def serve(controller: Controller, host: str, port: int,
          token: Optional[str] = None) -> tuple:
    token = token or secrets.token_urlsafe(24)
    server = ThreadingHTTPServer((host, port), make_handler(controller, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="control-panel")
    thread.start()
    return server, token


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading control</title><style>
:root{--bg:#faf9f7;--panel:#fff;--line:#e6e2dc;--ink:#1a1a1a;--muted:#6b6660;
--ok:#1f6f4a;--bad:#a33228;--warn:#9a5b1e;--mono:ui-monospace,Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1c1c1a;--line:#2e2d2a;
--ink:#f0eee9;--muted:#9b958c;--ok:#5bbd8b;--bad:#e0705f;--warn:#d99a4e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:640px;margin:0 auto;padding:32px 18px 60px}
h1{font-size:1.3rem;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 22px;font-size:.9rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px dashed var(--line)}
.row:last-child{border-bottom:0}.row .k{color:var(--muted)}.row .v{font-family:var(--mono);text-align:right}
button{font:inherit;font-weight:600;padding:10px 20px;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);cursor:pointer;margin-right:8px}
button.go{border-color:var(--ok);color:var(--ok)}button.halt{border-color:var(--bad);color:var(--bad)}
button:disabled{opacity:.4;cursor:default}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
.on{background:var(--ok)}.off{background:var(--muted)}
.tag{font-size:.72rem;font-weight:700;letter-spacing:.06em;padding:2px 8px;border-radius:999px;border:1px solid currentColor}
.live{color:var(--bad)}.test{color:var(--ok)}
.err{color:var(--bad);font-family:var(--mono);font-size:.85rem;margin-top:8px}
</style></head><body><div class="wrap">
<h1>Trading control <span id="tag"></span></h1>
<p class="sub" id="sub">connecting…</p>
<div class="card"><div id="ctl">
  <button class="go" id="start">Start</button>
  <button class="halt" id="stop">Stop</button>
  <span id="state"></span></div><div class="err" id="err"></div></div>
<div class="card" id="acct"></div>
<div class="card" id="pos"></div>
</div><script>
const t = new URLSearchParams(location.search).get("token") || "";
const q = s => document.querySelector(s);
const money = n => n == null ? "—" : Number(n).toFixed(2);
async function api(path, method){
  const r = await fetch(path + "?token=" + encodeURIComponent(t), {method: method||"GET"});
  if(!r.ok) throw new Error((await r.json()).error || r.status);
  return r.json();
}
async function refresh(){
  try{
    const d = await api("/api/status");
    q("#tag").innerHTML = `<span class="tag ${d.mode==='live'?'live':'test'}">${d.mode.toUpperCase()}</span>`;
    q("#sub").textContent = `${d.symbol} · ${d.timeframe} · ${d.passes} passes`
      + (d.last_pass_ago_s!=null ? ` · last ${d.last_pass_ago_s}s ago` : "");
    q("#state").innerHTML = `<span class="dot ${d.running?'on':'off'}"></span>`
      + (d.running ? "running" : "stopped") + (d.dry_run ? " (dry run)" : "");
    q("#start").disabled = d.running; q("#stop").disabled = !d.running;
    q("#err").textContent = d.halted ? "HALTED: " + d.halt_reason
      : (d.paused ? "Paused: " + d.paused : (d.last_error ? "Last error: " + d.last_error : ""));
    q("#acct").innerHTML = `
      <div class="row"><span class="k">In your ${d.quote} wallet</span><span class="v">${
        d.wallet_free==null ? "unreadable" : money(d.wallet_free) + " free"}</span></div>
      <div class="row"><span class="k">Equity (bot ledger)</span><span class="v">${money(d.equity)}</span></div>
      <div class="row"><span class="k">Trading balance</span><span class="v">${money(d.allocated)}</span></div>
      <div class="row"><span class="k">Locked reserve</span><span class="v">${money(d.reserve)}</span></div>
      <div class="row"><span class="k">Growth</span><span class="v">${d.growth_pct==null?"—":d.growth_pct+"%"}</span></div>
      <div class="row"><span class="k">Trades</span><span class="v">${d.trades} (${d.wins}W / ${d.losses}L)</span></div>
      <div class="row"><span class="k">Today</span><span class="v">${d.trades_today} trades, ${money(d.pnl_today)}</span></div>
      <div class="row"><span class="k">Last signal</span><span class="v">${d.last_signal||"—"}</span></div>`;
    q("#pos").innerHTML = d.positions.length
      ? d.positions.map(p=>`<div class="row"><span class="k">${p.symbol}</span>
          <span class="v">${p.qty.toFixed(6)} @ ${money(p.entry)} · stop ${money(p.stop)}
          ${p.protected?"✓ on exchange":"⚠ bot-only"}</span></div>`).join("")
      : `<div class="row"><span class="k">Open positions</span><span class="v">none</span></div>`;
  }catch(e){ q("#sub").textContent = "error: " + e.message; }
}
q("#start").onclick = async()=>{ await api("/api/start","POST"); refresh(); };
q("#stop").onclick  = async()=>{ await api("/api/stop","POST");  refresh(); };
refresh(); setInterval(refresh, 4000);
</script></body></html>"""
