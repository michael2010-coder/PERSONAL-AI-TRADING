# AI Trading App

A rule-based crypto trading bot with a hard risk layer, plus informational
signals for US stocks. Every decision traces back to a number you can check;
every order passes the risk manager before it reaches an exchange.

```
python main.py check                                  # settings + connectivity
python main.py backtest --symbol BTC/USDT --days 90   # real historical candles
python main.py signal  --symbol BTC/USDT              # what it would do right now
python main.py evidence check --symbol BTC/USDT       # what history says about it
python main.py trade --once --dry-run                 # one pass, places nothing
python main.py stocks AAPL NVDA                       # informational only
```

Before an entry is allowed, the bot asks a second question: *the last 100,000
times the market looked like this, how often did it work?* If the answer is
below the bar you set, it does not trade. See
[The evidence gate](#the-evidence-gate).

## Setup

```bash
cd ~/ai-trading-app
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then fill in keys, if you have them
.venv/bin/python -m pytest -q # 150 tests
```

`paper` mode needs no keys at all -- it reads real prices and trades imaginary
money. Use it first.

## What's in here

| Piece | File | What it does |
| --- | --- | --- |
| Strategy | [src/trading/strategy.py](src/trading/strategy.py) | EMA trend + RSI + MACD, each casting a vote, gated by a volume filter |
| Risk manager | [src/trading/risk.py](src/trading/risk.py) | Position sizing, stop/target, max open positions, daily-loss kill-switch, re-entry cooldown |
| Backtester | [src/trading/backtest.py](src/trading/backtest.py) | Same strategy, same risk manager, bar by bar over history |
| Trading loop | [src/trading/engine.py](src/trading/engine.py) | Paper / testnet / live, via ccxt |
| Evidence gate | [src/trading/analogues.py](src/trading/analogues.py) | Finds the N most similar historical situations and reports what happened next |
| Corpus | [src/trading/corpus.py](src/trading/corpus.py) | Downloads years of candles across dozens of pairs |
| Stock signals | [src/trading/stocks.py](src/trading/stocks.py) | Daily signals you act on yourself |

### How a call is made

Three rules vote in [-1, +1]:

- **EMA 12 vs 26** -- +1 in an uptrend, -1 in a downtrend.
- **RSI 14** -- +1 oversold (<=30), -1 overbought (>=70), otherwise +/-0.5 with its slope.
- **MACD histogram** -- +1 above zero, -1 below.

BUY needs the sum at or above `buy_threshold` (1.5) **and** volume at or above
its 20-bar average. SELL needs the sum at or below `sell_threshold` (-1.0);
the volume filter never blocks an exit, because a filter that can trap you in
a losing position is not a safety feature.

One consequence worth knowing: in a vertical move RSI pins at 100 and votes
*against* buying, so the sum lands at 1.0 and the bot sits it out. It is built
to buy pullbacks inside a trend, not blow-off tops. That behaviour is pinned by
a test.

## The evidence gate

The rules say *what* to do. The gate decides whether history backs it.

### How it works

1. **A corpus.** `scripts/build_corpus.py` downloads years of candles across
   the most-traded USDT pairs. Millions of bars, resumable, on disk in
   `data/corpus/`.

2. **Every bar becomes a situation** -- nine scale-free numbers describing the
   market's state at that moment: trend (EMA spread), stretch (RSI), momentum
   (MACD histogram), participation (volume vs its average), short and medium
   drift, realised volatility, position within the 20-bar range, and ATR.
   Scale-free is the point: it is what makes a 2019 BTC setup at $8k
   comparable to a 2026 SOL setup at $180.

3. **Every situation is labelled with what happened next.** Entering at the
   *next* bar's open, did price reach the take-profit before the stop-loss,
   within the horizon? A bar that spans both levels counts as a loss -- the
   same pessimistic rule the backtester uses.

4. **A query takes the K nearest situations** by standardised distance and
   reports how many won, with a Wilson confidence interval. The gate requires
   the interval's **lower bound** to clear your threshold, so a lucky sample
   cannot talk its way in.

```bash
.venv/bin/python scripts/build_corpus.py --years 8   # download (resumable)
.venv/bin/python main.py corpus                      # what is on disk
.venv/bin/python main.py evidence build              # label + index it
.venv/bin/python main.py evidence check --symbol BTC/USDT
.venv/bin/python main.py evidence sweep              # stop/target vs success rate
```

Labels depend on the stop, target and horizon, so a library is only valid for
the risk settings it was built with -- change `stop_loss_pct` and loading it
refuses until you rebuild. Build one library per timeframe you trade; a 15m bar
and a 1h bar do not mean the same thing to a 48-bar horizon.

### Configuration

```yaml
evidence:
  enabled: true
  min_similar_situations: 100000  # nearest analogues to consult
  min_success_rate: 0.80          # the share of them that must have worked
  horizon_bars: 48                # bars allowed to reach target before stop
  max_distance: null              # optional cap on how far a match may be
  confidence_z: 1.96              # gate on the 95% lower bound, not the raw rate
```

The gate blocks **entries only**. Exits, stops, take-profits and the
kill-switch are never gated -- an evidence engine that could keep you in a
losing position would be a liability, not a safeguard.

### No lookahead

A backtest query passes the timestamp of the bar being decided, and the library
only exposes situations that had **fully resolved** before it. A trade in
July 2026 cannot be justified by an analogue that was still open in July 2026.
`test_a_query_can_only_see_situations_that_had_already_resolved` and
`test_features_are_causal` pin this.

## Your autopilot: 400 USDT, one year

```bash
.venv/bin/python main.py plan                # what it will and will not do
.venv/bin/python main.py validate            # would these settings have made money?
.venv/bin/python main.py simulate --days 365 # a year, on real history, with your 400
.venv/bin/python main.py trade               # run it (paper by default)
.venv/bin/python main.py status              # portfolio, pauses, open positions
```

### The stop lives on the exchange

An earlier version only checked the stop when the bot polled. That has two
failure modes, both seen on testnet: price can move past the stop *between*
polls (one overnight exit filled 3.5% below its trigger, losing 10x what the
position was sized to lose), and a position is completely unprotected whenever
the process is not running -- a crash, a deploy, a sleeping laptop.

Every entry now rests a stop-loss order on the exchange itself. Verified by
killing the bot with a position open: the stop stayed live on Binance. If the
resting stop goes missing it is put back on the next poll, and it is cancelled
before any exit the bot initiates itself.

### What it risks, at 400 USDT

| | |
| --- | --- |
| Per trade | 40.00 position, **0.80** to the stop |
| Worst day | -20.00, then it stops until UTC midnight |
| Worst week | -32.00, then it pauses 3 days |
| Worst case ever | -80.00 (-20% of peak), then it stops for good and waits for you |

### How it refuses to get greedy

- **Position size never grows because the last trade won.** It is a fixed 10%
  of the trading balance. A losing streak never doubles down either.
- **Compounding is capped.** The trading balance may grow from 400 to 600 and
  no further; profit beyond that spills into a reserve the bot cannot trade.
- **Profit is locked away in steps.** Every +50 realised, 25 moves to the
  reserve permanently.
- **A good day ends the day.** Up 2% and it stops trading until tomorrow.
- **Four trades a day, maximum.** Churn is a cost, not a strategy.

### When it pauses, and when it starts again

| Trigger | Pause |
| --- | --- |
| 3 losses in a row | 12 hours |
| Daily loss limit (-5% of capital) | until UTC midnight |
| Weekly loss limit (-8%) | 3 days |
| Market data older than 30 min | until fresh data arrives |
| 5 exchange errors in a row | 15 minutes |
| ATR above 5% of price | sits out until it calms |
| Total drawdown -20% from peak | **stops for good** -- restarting is your decision |

Every pause carries a resume time, so a paused bot is never a stuck bot. Only
the drawdown halt needs a human. Pauses block **entries only** -- stops,
targets and exits always run, because a safety feature that can trap you in a
losing position is not a safety feature.

### Four locks before real money moves

1. `crypto.mode: "live"` in config.yaml
2. `--i-understand-this-is-live` on the command line
3. A **passing** `main.py validate` on record, no more than 30 days old, for
   the settings you are actually running
4. The evidence gate agreeing, trade by trade

Any one of them says no, and nothing is ordered.

## What the evidence actually says

This is the part to read before funding anything.

You asked for entries backed by 100,000+ similar historical situations with
80%+ accuracy. I built exactly that, ran it on **1,521,868 labelled situations
from 45 Binance pairs (1.69M hourly bars, 8 years)**, and here is what came
back.

### At the shipped 2% stop / 4% target

The gate blocks everything. Over the last year on BTC/USDT: **782 BUY signals,
782 blocked, 0 trades.** The reason is not a bug:

| Cohort | Success | 95% CI | Similarity radius |
| --- | --- | --- | --- |
| 100 nearest | 30.0% | 21.9-39.6% | 0.45 |
| 1,000 | 30.7% | 27.9-33.6% | 0.63 |
| 10,000 | 31.2% | 30.3-32.1% | 0.94 |
| **100,000** | **31.3%** | 31.0-31.6% | 1.58 |
| the whole corpus | 31.4% | -- | -- |

Similar situations do no better than random ones. That is the finding, and no
amount of extra data moves it -- the 100-nearest cohort is no better than the
base rate either.

### The same test on every timeframe in the corpus

Libraries built per timeframe (mixing them would make "similar" mean two
different things), all at a 2% stop / 4% target, 48-bar horizon:

| Timeframe | Situations | Base rate | 100k-analogue cohort | Verdict |
| --- | --- | --- | --- | --- |
| 15m | 1,230,848 | 26.1% | 25.5% (25.3-25.8) | BLOCKED |
| 1h | 1,521,868 | 31.4% | 31.3% (31.0-31.6) | BLOCKED |
| 4h | 414,187 | 32.1% | **34.0%** (33.7-34.3) | BLOCKED |

4h is the one place the engine finds anything at all: the cohort beats its base
rate by ~2 points, and the confidence interval does not overlap it, so the lift
is real rather than noise. It is also far too small to matter. At 34% success
on a 2:1 target the expectancy is 0.34x4 - 0.66x2 = **+0.04% per trade gross**,
and a round trip costs about 0.30% in fees and slippage. The edge is real,
measurable, and an order of magnitude too small to pay for the trading.

### 80% accuracy is purchasable, and it loses money

Across 935,820 situations, moving the target closer to the entry buys a high
success rate:

| Stop | Target | R:R | Situations | Success | Expectancy |
| --- | --- | --- | --- | --- | --- |
| 2.0% | 0.2% | 0.10 | 918,663 | **86.5%** | -0.096% |
| 2.0% | 0.3% | 0.15 | 908,213 | **83.5%** | -0.080% |
| 2.0% | 1.0% | 0.50 | 834,319 | 65.3% | -0.042% |
| 2.0% | 4.0% | 2.00 | 681,980 | 29.8% | -0.213% |

Every row is negative, and the **80%+ rows are the worst ones**. So I built the
configuration that satisfies your requirement literally -- 2% stop, 0.3%
target -- and ran the year:

```
Evidence gate    781 entries backed, 1 blocked   (100,000 analogues, 85.5% success)
Trades           286
Win rate         80.1%   (229 win / 57 loss)
400.00 -> 368.25 (-7.94%)
```

**80.1% accuracy. Down 7.9%.** 229 wins at +0.3% could not pay for 57 losses at
-2%, and fees took 21.98 on top. A win rate is not an edge; expectancy is.

Requiring the cohort to beat its own base rate instead of an absolute 80%
(88% required against an 82.4% base) let **2 trades through in a year**. At 90%,
none.

### The strategy itself, over the last year

Rules + risk manager + supervisor, evidence gate off, 400 USDT:

| Market | Return | Trades | Win rate | Max DD | Buy & hold |
| --- | --- | --- | --- | --- | --- |
| BTC/USDT | -3.74% | 124 | 29.0% | -4.72% | -27.74% |
| ETH/USDT | -5.09% | 139 | -- | -6.11% | -- |
| SOL/USDT | -3.96% | 157 | -- | -5.12% | -- |
| BNB/USDT | -2.21% | 138 | -- | -5.32% | -- |
| XRP/USDT | -6.98% | 149 | -- | -7.27% | -- |

0 of 5 windows profitable, average -4.39%. `main.py validate` returns **FAIL**,
which is why live mode is currently locked.

### What this means for the 400 USDT

The machinery all works and is heavily tested. The risk controls did their job
in every run -- worst single trade -2.25%, worst drawdown -7.27% against a -20%
ceiling, supervisor pausing 22 times over the year. What is missing is an edge:
this strategy loses roughly 4% a year in fees and small losses, and the
analogue engine says it cannot find one either.

So the honest position is:

- **Run it in `paper` mode.** It costs nothing and it will tell you the truth.
- **Do not fund it live while `validate` returns FAIL.** The app enforces this;
  please do not use `--skip-validation` to get around it.
- **The work that would change this is in the strategy**, not the plumbing.
  Better features for the analogue engine, a different market regime, a
  different instrument, or a genuinely different signal. The backtester,
  library and validator are the tools for testing that -- they are built, and
  they are honest.

A note on the arithmetic of 400 USDT: at 10% per position and a 2% stop, one
trade risks 0.80. That is deliberately careful, and it also means no
realistic edge turns 400 into a large number in a year. A *good* real strategy
might return 10-30% annually. Anything promising more is selling something.

## What the risk manager guarantees

With the shipped `config.yaml` (400 USDT allocated, 10% per position, 2% stop):

| Limit | Setting | Effect |
| --- | --- | --- |
| Position size | `max_position_size_pct: 10` | 40.00 USDT per trade, never the whole balance |
| Stop loss | `stop_loss_pct: 2` | A stopped-out trade costs ~0.80 USDT = 0.2% of capital |
| Take profit | `take_profit_pct: 4` | Set at entry, alongside the stop |
| Concurrency | `max_open_positions: 1` | No stacking, and never twice into one symbol |
| Daily loss | `max_daily_loss_pct: 5` | At -20.00 USDT realised, no new entries until UTC midnight |
| Re-entry | `reentry_cooldown_bars: 3` | After any exit, it sits out 3 bars |

Sizing is a slice of the portfolio's trading balance, not of your exchange
balance -- a 50,000 USDT wallet still trades 40.00 USDT a position, and the
size follows the account down after losses. It is also capped by
the capital not already at work and by what the wallet actually holds.

Those numbers are asserted in [tests/test_risk.py](tests/test_risk.py). Read
that file before you fund anything; it is short, and it is the contract.

## Backtest results (real Binance data)

Run on this machine on 2026-08-24, shipped settings, fees 10bps + slippage 5bps:

| Market | Timeframe | Window | Strategy | Buy & hold | Max DD | Trades | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC/USDT | 1h | 90 days | **-0.46%** | +2.12% | -1.46% | 31 | 35.5% |
| BTC/USDT | 4h | 180 days | **-0.00%** | +14.02% | -1.80% | 32 | 34.4% |
| BTC/USDT | 1d | 730 days | **+1.09%** | +22.75% | -1.04% | 33 | 42.4% |
| ETH/USDT | 1h | 90 days | **-0.48%** | +17.28% | -1.53% | 40 | 30.0% |
| SOL/USDT | 1h | 90 days | **+0.17%** | +11.18% | -1.47% | 39 | 30.8% |

Read that honestly: **the risk layer works and the strategy does not beat
holding.** Drawdowns stayed inside 1.8% and no single trade lost more than
2.25% of the position, exactly as sized -- but returns hover around zero while
buying and holding made money in every one of these windows. Two things soften
the comparison slightly: only 10% of capital is deployed per trade, and the bot
is in the market roughly 45% of the time, so it is a small position compared
against a full one. It does not rescue the result.

The strategy needs work before it earns real money. The place to do that work
is the backtester, over multiple windows, including bad ones.

Reproduce any row:

```bash
.venv/bin/python main.py backtest --symbol ETH/USDT --timeframe 1h --days 90 --trades
.venv/bin/python main.py backtest --csv data/BTCUSDT_1h_90d.csv --json out.json
```

## Going live

Live trading is behind two locks that have to be opened separately:

1. `crypto.mode: "live"` in `config.yaml`, and
2. `--i-understand-this-is-live` on the command line.

Either one alone refuses to trade and says why. There is no third way to get
there, and a config with `mode: testnet` ignores the flag entirely.

Before you use either:

- **Backtest across several windows**, not one flattering one. A bull quarter
  will make almost anything look clever. Include a period that fell.
- **Run on testnet long enough to see a losing streak.** Testnet keys come from
  [testnet.binance.vision](https://testnet.binance.vision) (Binance) or
  [testnet.bybit.com](https://testnet.bybit.com) (Bybit) and are separate from
  live keys. What matters is watching it handle five losses in a row, not five
  minutes of it working.
- **Create trade-only API keys. Never enable withdrawals.** Then the worst case
  from a leak or a bug is a bad trade, not an emptied account. IP-whitelist the
  key if your exchange supports it.
- **Re-check the risk numbers against money you would be okay losing.**
  `allocated_capital_usdt` is the entire budget the bot may touch. Set it to an
  amount whose total loss would be annoying, not damaging.

The bot is not a place to keep your funds. Keep the trading balance small and
the rest elsewhere.

### Running it continuously

A laptop that sleeps is not a host, and this will not run on Vercel, Netlify or
any serverless platform -- it is a long-running process that holds a position
and a state file between polls, and a function that times out in seconds cannot
do that. It wants a small always-on box.

**Where to host it**

| | |
| --- | --- |
| Recommended | Hetzner CX22, Falkenstein or Helsinki -- about EUR 3.79/mo |
| Simpler alternative | DigitalOcean $6 droplet, Frankfurt or Amsterdam |
| Free, with caveats | Oracle Cloud Always Free ARM instance, Frankfurt |

**Not the United States.** Binance geo-blocks US IP addresses and answers with
HTTP 451, so a US-region box fails every API call however well the bot is
configured. Frankfurt, Helsinki, Amsterdam, Singapore and Tokyo all work. The
same goes for anywhere you do not control the egress region -- which is the
second, independent reason serverless platforms are unsuitable here.

The first thing to run on a new box is `main.py check`: it fetches a live
ticker, so a geo-block or a firewall problem shows up immediately rather than
at the first trade.

Sizing: 1 vCPU and 1 GB RAM is plenty. Disk depends on whether you build the
corpus there (~500 MB) or copy just the library the bot reads (54 MB).

```bash
# on a fresh Debian/Ubuntu box
git clone https://github.com/michael2010-coder/PERSONAL-AI-TRADING.git
cd PERSONAL-AI-TRADING
sudo bash deploy/install.sh
```

That installs Python, creates a `trader` system user, builds the virtualenv,
runs the tests, and installs [deploy/ai-trading-bot.service](deploy/ai-trading-bot.service).
It deliberately does **not** start trading. The script prints the next steps:
build the corpus, run `check` and `plan`, get a passing `validate`, then

```bash
sudo systemctl enable --now ai-trading-bot
journalctl -u ai-trading-bot -f
```

The service restarts on crash with a 30-second backoff, and gives up after 5
failures in 10 minutes rather than hammering the exchange. It stops with SIGINT
and 60 seconds of grace, so a deploy never kills the process mid-order.

Building the corpus on the VPS takes about 30 minutes and 230 MB. Copying the
one file the bot actually reads is faster:

```bash
rsync -avz data/corpus/library_1h.npz root@<host>:/opt/personal-ai-trading/data/corpus/
```

State lives in `state.<mode>.json`, so a restart picks up an open position and
the day's realised PnL instead of losing track of both. Orders are journalled
to `logs/orders.jsonl`, and `main.py status` reads all of it back.

Both `data/` and `logs/` are gitignored: the corpus and libraries are hundreds
of megabytes and fully reproducible, and `state.*.json` holds your live
balances. Nothing about your account is in the repository.

## Can it go live?

The short version: **one configuration now passes validation, and it still is
not worth funding.** Both halves of that matter, so here is the work.

### Where the money was actually going

Re-running the same year with fees and slippage set to zero:

| | Return |
| --- | --- |
| With real costs | -3.74% |
| With zero costs | **-0.29%** |

**Costs were 92% of the loss.** The strategy was not picking badly -- gross it
was flat, a coin flip. It lost because it turned over 12x the account per year
at ~0.3% a round trip. That reframed the work: stop hunting for a big edge,
stop paying 3.4% a year in friction.

### Cutting the friction

Trading a slower timeframe and assuming limit rather than market orders took
the same strategy from -3.74% to **-0.18%**. Across a 48-configuration sweep
(2 timeframes x 3 signal thresholds x 4 stop/target pairs x 2 cost models, 5
markets, one year each) **not one configuration was profitable.** Cost work
alone gets you to break-even, never past it.

### Finding the one thing that was not noise

4h was the only place the analogue engine ever measured a real lift (34.0%
against a 32.1% base rate, confidence interval clear of it). Gating entries on
*beating the base rate* -- rather than an absolute 80% -- is the version of
your requirement that has any chance, so that is what was tested.

Parameters were chosen on **2019-2025** data and then run on data never used to
choose them:

| Window | Return |
| --- | --- |
| Train: 5 majors, 6 years to 1 year ago | +0.64% |
| Test: same 5 majors, last year | +0.38% |
| Test: **6 symbols never used in selection**, last year | +1.04% |
| Test: those unseen symbols, the older era | +0.22% |

The sign held everywhere, including on symbols the parameters had never seen.
That is the strongest result in this repository.

`main.py validate` on [config.4h.yaml](config.4h.yaml), with honest taker costs
(10bps fee, 5bps slippage -- what the bot actually pays placing market orders),
across 11 markets:

```
windows profitable   8/11
average return       +0.24%
worst drawdown       -2.52%
trades taken         263

PASS -- these settings made money across the windows tested.
```

### Why you still should not fund it

**+0.24% a year on 400 USDT is 96 cents.**

A VPS to run it costs $48-72 a year. Hosting is **50 to 75 times** the expected
profit. Running this live is a guaranteed loss that has nothing to do with the
strategy being wrong -- the edge is simply smaller than the electricity bill.

And 0.24% is inside the noise of things this backtest does not model: exchange
downtime, partial fills, the spread widening when you actually need it, a
listing being delisted mid-position. An edge needs to be big enough to survive
the modelling error, and this one is not.

Scaling does not rescue it either, because the edge is a percentage: 4,000 USDT
returns about 9.60 a year, 40,000 returns about 96. You would need roughly 20x
this edge before the hosting cost stops dominating.

### What would change the answer

An edge of 5-10% a year, not 0.24%. That means better features for the analogue
engine (higher-timeframe context, regime, time of day, cross-asset correlation),
or a genuinely different signal -- not more parameter tuning on these nine
features, which is now thoroughly explored and lands at zero.

Everything needed to test that is built: a 4.3M-bar corpus, a labelled library
with no-lookahead queries, a backtester that agrees with the live engine, and a
validator with held-out windows that will tell you the truth. Use them on a new
idea rather than re-tuning this one.

### If you want to run it anyway

Do it on **testnet**, which costs nothing and proves the whole pipeline.
[docs/TESTNET.md](docs/TESTNET.md) is the runbook, including what to try to
break and how to know when the run is finished:

```bash
# keys from testnet.binance.vision, then
.venv/bin/python main.py --config config.testnet.yaml balance --mode testnet
.venv/bin/python main.py --config config.testnet.yaml trade --once --dry-run
.venv/bin/python main.py --config config.testnet.yaml trade --symbol BTC/USDT
```

`config.4h.yaml` is the validated configuration. The default `config.yaml`
remains the 1h/80% setup, which is documented above and loses money -- it is
kept because it is the honest record of what was asked for and what happened.

## Funding it with BTC

**The bot never holds your money.** There is no deposit address here, no
wallet, no custody. Your funds sit in your own Binance account, and the bot
holds an API key that may place trades and nothing else. That is the whole
security model, and it is why there is no deposit system to build: "funding the
bot" means funding your own exchange account.

### The one thing that trips people up

The strategy buys BTC **with** USDT. If you deposit BTC and stop there, the bot
has nothing to spend -- it is already holding the asset it wants to buy. So:

```
BTC from your wallet  ->  Binance BTC address  ->  Convert to USDT  ->  bot trades BTC/USDT
```

`main.py balance` checks exactly this and tells you which step you are on.

### Step by step

1. **Deposit BTC.** Binance -> Wallet -> Deposit -> BTC. Copy the address, and
   **match the network to the one you are sending from**. Native BTC sends to a
   `bc1...`/`3...` address; BEP20 BTCB is a different chain with a different
   address. Sending on the wrong network loses the coins permanently, and no
   support ticket brings them back. Send a small test amount first.

2. **Convert to USDT.** Wallet -> Convert -> BTC to USDT. Convert charges no
   trading fee and has no order book to get wrong. Convert only what you intend
   the bot to trade; keeping the rest as BTC is a position you are choosing to
   hold, not something the bot manages.

3. **Create a trade-only API key.** Binance -> API Management -> Create API.
   - Enable **Spot Trading**.
   - Leave **Withdrawals disabled**. This is the single most important setting
     on this page. With it off, the worst case from a leaked key or a bug in
     this code is a bad trade, not an emptied account.
   - Restrict access to your VPS IP address.

4. **Put the key on the box**, in `/opt/personal-ai-trading/.env`:

   ```
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   ```

   `chmod 600 .env`. It is gitignored; keep it that way.

5. **Confirm the bot can see the money:**

   ```bash
   .venv/bin/python main.py balance --mode live
   ```

   ```
   binance account (live mode)
     asset              free      value in USDT
     USDT             420.00             420.00
     total                               420.00

   The bot is configured to trade 400.00 USDT.
   Funded: 420.00 USDT free, which covers it.
   ```

   If you skipped the conversion it says so, values the BTC you are holding,
   and points you at Convert.

### How much to send

`portfolio.initial_capital_usdt` (400 by default) is the **entire** budget the
bot may ever touch. Send that plus a little slack for fees. Do not treat the
account as a savings account with a bot attached -- deposit what you have
decided you can lose, and keep the rest in your own wallet.

Test on **testnet first**, where the money is fake: keys from
[testnet.binance.vision](https://testnet.binance.vision), `crypto.mode: testnet`,
and the same `balance` command works.

### Paying for the VPS in crypto

Hetzner and DigitalOcean both want a card. If you would rather pay in BTC:

- **BitLaunch** -- funded with BTC, deploys DigitalOcean/Vultr/Linode servers,
  so you still pick a Frankfurt or Amsterdam region.
- **1984 Hosting** (Iceland) and **Njalla** both take BTC directly.

Prices and payment policies at these change; check before committing. And note
that paper mode runs on your own machine for free, which is where this should
live until `main.py validate` stops returning FAIL.

## US stocks

Informational only. Bamboo and Trove have no retail API, so nothing here can
place a stock order -- it prints the call and the sizing that the same risk
rules would suggest, and you decide.

Daily candles come from, in order: `data/stocks/<SYMBOL>.csv` if present,
Alpha Vantage if `ALPHAVANTAGE_API_KEY` is set, then Yahoo's public endpoint.
**Yahoo is currently rate-limiting this network (HTTP 429) and Stooq has put up
a bot wall**, so on this machine, today, use a free Alpha Vantage key or export
daily OHLCV to `data/stocks/AAPL.csv`. Any export with Date/Open/High/Low/Close
columns is read.

## Tests

```bash
.venv/bin/python -m pytest -q       # 150 tests, no network
```

The ones that matter:

- `test_indicators_are_causal` -- an indicator's value at bar *i* is unchanged
  when the bars after it are deleted. This is what lets the backtester compute
  the full series at once and still be lookahead-free.
- `test_fills_happen_on_the_bar_after_the_signal` -- a decision made from bar
  *i*'s close fills at bar *i+1*'s open, proven on data where those two prices
  differ.
- `test_no_trade_loses_more_than_the_position_was_sized_to_lose`
- `test_daily_loss_kill_switch_blocks_at_the_limit_exactly`
- `test_live_mode_without_the_flag_refuses_to_trade`
- `test_a_stopped_out_position_is_not_immediately_bought_back`
- `test_features_are_causal` and `test_features_are_scale_free` -- a situation
  vector depends only on the past, and describes shape rather than price level.
- `test_a_query_can_only_see_situations_that_had_already_resolved` -- the
  evidence gate cannot consult its own future.
- `test_the_gate_uses_the_lower_confidence_bound_not_the_raw_rate` -- 20 wins
  out of 24 is 83%, and still does not pass an 80% gate.
- `test_a_bar_that_spans_both_levels_counts_as_a_loss` -- labelling is
  pessimistic, so the base rate is not flattered.
- `test_a_winning_streak_does_not_make_the_next_position_bigger` -- the
  anti-greed promise, in one assertion.
- `test_a_pause_ends_by_itself` -- and clears what caused it, so a paused bot
  is never a stuck bot.
- `test_live_is_refused_when_validation_fails_even_with_the_flag_and_keys`

## Configuration

Everything behavioural is in [config.yaml](config.yaml); secrets are named
there but read from `.env`. An unknown or misspelled key is an error at load
time rather than a silently ignored setting -- `stop_los_pct: 2.0` will refuse
to start instead of trading with no stop.
