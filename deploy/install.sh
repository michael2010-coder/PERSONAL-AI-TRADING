#!/usr/bin/env bash
# Set up the trading bot on a fresh Debian/Ubuntu box.
#
#   curl -fsSL https://raw.githubusercontent.com/michael2010-coder/PERSONAL-AI-TRADING/main/deploy/install.sh | bash
# or, from a clone:
#   sudo bash deploy/install.sh
#
# Idempotent: safe to re-run. It never starts live trading -- the service runs
# whatever mode config.yaml says, and that ships as paper.
set -euo pipefail

REPO="${REPO:-https://github.com/michael2010-coder/PERSONAL-AI-TRADING.git}"
DEST="${DEST:-/opt/personal-ai-trading}"
USER_NAME="${USER_NAME:-trader}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "Run this with sudo: sudo bash deploy/install.sh" >&2
  exit 1
fi

say "Checking this host can reach the exchange"
if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://api.binance.com/api/v3/ping || echo 000)
  case "$code" in
    200) echo "  api.binance.com reachable (HTTP 200)" ;;
    451) echo "  HTTP 451: Binance blocks this region. Move to a non-US host" >&2
         echo "  (Frankfurt, Helsinki, Amsterdam, Singapore, Tokyo all work)." >&2
         exit 1 ;;
    *)   echo "  WARNING: api.binance.com returned HTTP $code -- check firewall/region" >&2 ;;
  esac
fi

say "Installing system packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git ca-certificates

say "Creating the ${USER_NAME} user"
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$USER_NAME"
fi

say "Fetching the code into ${DEST}"
if [[ -d "$DEST/.git" ]]; then
  sudo -u "$USER_NAME" git -C "$DEST" pull --ff-only
else
  mkdir -p "$DEST"
  chown "$USER_NAME:$USER_NAME" "$DEST"
  sudo -u "$USER_NAME" git clone --depth 1 "$REPO" "$DEST"
fi

say "Building the virtualenv"
sudo -u "$USER_NAME" python3 -m venv "$DEST/.venv"
sudo -u "$USER_NAME" "$DEST/.venv/bin/python" -m pip install -q --upgrade pip
sudo -u "$USER_NAME" "$DEST/.venv/bin/python" -m pip install -q -r "$DEST/requirements.txt"

say "Running the test suite"
sudo -u "$USER_NAME" "$DEST/.venv/bin/python" -m pytest -q "$DEST"

if [[ ! -f "$DEST/.env" ]]; then
  say "Creating .env from the template"
  sudo -u "$USER_NAME" cp "$DEST/.env.example" "$DEST/.env"
  chmod 600 "$DEST/.env"
  chown "$USER_NAME:$USER_NAME" "$DEST/.env"
  echo "Put TRADE-ONLY API keys in $DEST/.env. Never enable withdrawals."
fi

say "Installing the systemd service"
cp "$DEST/deploy/ai-trading-bot.service" /etc/systemd/system/
sed -i "s#/opt/personal-ai-trading#${DEST}#g; s#User=trader#User=${USER_NAME}#; s#Group=trader#Group=${USER_NAME}#" \
  /etc/systemd/system/ai-trading-bot.service
systemctl daemon-reload

cat <<NEXT

Installed. It is NOT running yet, and it is not configured to trade real money.

Next, as a deliberate sequence:

  1. Build the evidence corpus (about 30 min, ~230 MB):
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/scripts/build_corpus.py --years 8
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/main.py evidence build

     Or copy the library from a machine that already built it (much faster):
       rsync -avz data/corpus/library_1h.npz root@<host>:${DEST}/data/corpus/

  2. Check it can see the market and the settings are what you expect:
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/main.py check
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/main.py plan

  3. Prove the settings are worth running. This must PASS before live:
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/main.py validate

  4. Start it (paper mode, no keys needed, costs nothing):
       systemctl enable --now ai-trading-bot
       journalctl -u ai-trading-bot -f

  5. Check on it any time:
       sudo -u ${USER_NAME} ${DEST}/.venv/bin/python ${DEST}/main.py status --mode paper

Going live needs all four locks: config.yaml mode: live, the
--i-understand-this-is-live flag in the service's ExecStart, a passing
validate on record, and the evidence gate agreeing trade by trade.

NEXT
