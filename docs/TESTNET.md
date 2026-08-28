# Testnet run

Real market data, real exchange mechanics, fake money. The point of this run is
**not** to find out whether the strategy is profitable -- a year of data would
be needed for that, and we already have it. The point is to find out whether
the software works against a real exchange, which 160 unit tests cannot tell
you.

## What we are trying to break

Integration is where this class of program fails, and none of it is exercised
by the test suite:

- order amount precision and minimum-notional rules
- an exchange error mid-order, or a key that stops working
- the process being killed while holding a position
- the supervisor and portfolio surviving a restart with real numbers in them

## Step 1: get testnet keys (the only step needing a browser)

1. Go to **https://testnet.binance.vision** and sign in with GitHub.
2. **Generate HMAC_SHA256 Key.** Copy both the key and the secret -- the secret
   is shown once.
3. The account is pre-funded with fake USDT and BTC. Nothing to deposit.

Testnet is periodically reset by Binance, which wipes keys and balances. If
things stop working after a few weeks, generate a new key rather than debugging.

## Step 2: put them on disk

```bash
cd ~/ai-trading-app
cp .env.example .env          # if you have not already
chmod 600 .env
```

Then edit `.env`:

```
BINANCE_API_KEY=<the testnet key>
BINANCE_API_SECRET=<the testnet secret>
```

These are testnet credentials. They control no real money, but keep the habit:
`.env` is gitignored and should stay that way.

## Step 3: confirm it can see the account

```bash
.venv/bin/python main.py --config config.testnet.yaml balance --mode testnet
```

Expect a table of fake balances. If it errors, the key is wrong or testnet has
been reset.

## Step 4: exercise the decision path, place nothing

```bash
.venv/bin/python main.py --config config.testnet.yaml trade --once --dry-run
```

This works even with no keys at all. It fetches candles from testnet, runs the
strategy, asks the evidence gate, sizes the position, and logs what it *would*
do.

## Step 5: let it run

One symbol trades roughly twice a month on these settings, so run several to
see anything in a week. Each gets its own state file automatically:

```bash
.venv/bin/python main.py --config config.testnet.yaml trade --symbol BTC/USDT &
.venv/bin/python main.py --config config.testnet.yaml trade --symbol ETH/USDT &
.venv/bin/python main.py --config config.testnet.yaml trade --symbol SOL/USDT &
```

Check on them any time:

```bash
.venv/bin/python main.py status --mode testnet     # every instance at once
tail -f logs/trading.log
cat logs/orders.jsonl | tail -5
```

On a VPS use the systemd unit instead, one per symbol.

## Definition of done

Tick these off, then stop and decide. Do not judge profitability from this.

- [ ] `balance` reads the testnet account
- [ ] `trade --once --dry-run` produces a decision with its reasoning
- [ ] **one complete entry -> exit cycle**, with the exit booked into the
      portfolio and the reason (stop / target / signal) in `logs/orders.jsonl`
- [ ] **kill the process while a position is open, restart it**, and confirm
      `status` still shows the position and the same realised P&L
- [ ] **force an error** -- revoke the key, or run with `--symbol NOPE/USDT` --
      and confirm the supervisor pauses instead of the process dying

The fourth is the one most likely to find a real bug, and the one that matters
most if this ever runs unattended with real money.

## What this run cannot tell you

Whether the strategy makes money. Testnet's order book is synthetic, fills are
optimistic, and two trades a month is not a sample. The measured answer to that
question is in the README: +0.24% a year, which is 96 cents on 400 USDT.
