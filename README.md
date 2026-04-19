# Zerodha Algorithmic Trading Bot

An end-to-end Indian equity trading bot built on **Zerodha Kite Connect**.  
Features: live & paper trading, VWAP/Volume strategy, risk manager, SQLite trade log,  
Flask web dashboard, Telegram command bot, and a full backtest + optimizer engine.

---

## Architecture

```
main.py                   ← entry point (CLI, lifecycle)
config.py                 ← all settings via .env / env vars (pydantic-settings)
brokers/
  zerodha_broker.py       ← Kite Connect: orders, quotes, WebSocket ticks
strategies/
  vwap_volume.py          ← VWAP + EMA9 + RSI + ATR + Volume strategy
risk/
  risk_manager.py         ← position sizing, trailing SL, daily loss cap
orders/
  order_manager.py        ← signal → order → position lifecycle
data/
  feed.py                 ← WebSocket tick feed + candle builder
db/
  trade_logger.py         ← SQLite ORM (trades, daily summaries, bot logs)
dashboard/
  app.py                  ← Flask REST API + single-page dark UI
notifications/
  telegram_bot.py         ← async Telegram bot (alerts + remote commands)
backtest/
  engine.py               ← vectorised backtest engine + optimizer
backtest_run.py           ← CLI: run backtests and optimizations
scripts/
  checklist.py            ← pre-live safety checker
  check_requirements.py   ← dependency verifier
```

---

## Quick Start

### 1 — Clone and set up environment

```bash
cd c:\Project\trade-bot          # or wherever you placed it

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / Mac

pip install -r requirements.txt
```

### 2 — Configure credentials

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # Linux / Mac
```

Edit `.env`:

```ini
# Zerodha Kite Connect
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token   # regenerate daily

# Trading mode — ALWAYS start with true
PAPER_TRADE=true

# Capital and risk
TRADING_CAPITAL=100000
RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS_INR=10000

# Telegram (optional — leave as DUMMY to disable)
TELEGRAM_BOT_TOKEN=DUMMY_TELEGRAM_TOKEN
TELEGRAM_CHAT_ID=000000000
```

### 3 — Generate / refresh the Zerodha access token

Zerodha access tokens expire daily at 06:00 IST.

```bash
python generate_token.py
# Opens browser → log in → paste the request_token → token saved to .env
```

### 4 — Run pre-flight checks

```bash
python scripts/checklist.py
```

All items must be **PASS** or **WARN** before going live.  
Fix any **FAIL** items first.

### 5 — Test broker connection

```bash
python test_connection.py
```

### 6 — Paper trading (always start here)

```bash
# Windows
run_bot.bat --paper --symbol NIFTY

# Linux / Mac
./run_bot.sh --paper --symbol NIFTY
```

Open the dashboard: **http://localhost:5000**

### 7 — Live trading (after thorough paper testing)

```bash
run_bot.bat --symbol NIFTY      # will prompt "are you sure?"
```

---

## CLI Reference

```
python main.py [OPTIONS]

  --paper          Force paper trading (overrides .env)
  --symbol TEXT    Trading symbol  [default: NIFTY]
  --no-dashboard   Skip the web dashboard
  --log-level TEXT Override log level (DEBUG/INFO/WARNING/ERROR)
  --help           Show this message and exit
```

```
python backtest_run.py [OPTIONS]

  --strategy TEXT  Strategy name  [default: vwap_volume]
  --symbol TEXT    Symbol         [default: NIFTY]
  --from TEXT      Start date YYYY-MM-DD
  --to TEXT        End date   YYYY-MM-DD
  --broker         Use live data from Zerodha (auto-sets --from/--to)
  --days INT       Days of history when --broker is used  [default: 90]
  --optimize       Run parameter grid search
  --help           Show this message and exit
```

---

## Dashboard

The dashboard starts automatically on **http://localhost:5000**.

| Tab | Contents |
|-----|----------|
| Overview | Bot status, daily P&L, open positions |
| Trades | Full trade history table |
| Charts | Equity curve, win/loss doughnut, daily P&L bars |
| Config | Live-edit strategy parameters (no restart needed) |

### Dashboard API (for scripting / monitoring)

```
GET  /api/status          bot status + risk metrics
GET  /api/positions       open positions list
GET  /api/trades          trade history
GET  /api/daily-pnl       last 30 days P&L
GET  /api/equity-curve    cumulative equity series
GET  /api/logs            recent bot log entries
GET  /api/config          current strategy config
POST /api/config          update config (JSON body)
POST /api/control         {"action": "pause"|"resume"|"squareoff"}
```

---

## Telegram Bot (optional)

1. Create a bot via [@BotFather](https://t.me/botfather) → get token
2. Start a chat with your bot → get your chat ID
3. Set in `.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=1234567890:AAH...
   TELEGRAM_CHAT_ID=987654321
   ```

Commands available from your phone:

| Command | Action |
|---------|--------|
| `/start` | Welcome + help |
| `/status` | Open positions + bot state |
| `/today` | Today's trade summary |
| `/pause` | Pause new trade entries |
| `/resume` | Resume trading |
| `/squareoff` | Emergency close all positions |

---

## Backtesting

```bash
# 90 days of live data from Zerodha
python backtest_run.py --broker --days 90

# From CSV
python backtest_run.py --from 2024-01-01 --to 2024-12-31

# Parameter optimization (grid search)
python backtest_run.py --broker --days 180 --optimize
```

Optimization results are printed ranked by Sharpe ratio.  
Results flagged `OVERFIT` (< 30 trades) should be ignored.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Skip slow optimizer test
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Pre-Live Checklist

Run `python scripts/checklist.py` before every live session.  
It checks:

1. Python version ≥ 3.9 (3.11+ recommended)
2. All required packages installed
3. `.env` file exists
4. Zerodha credentials are not DUMMY values
5. `PAPER_TRADE` flag (warns if on, expected for first run)
6. Capital and risk settings are sane
7. `logs/` directory is writable
8. SQLite DB initialises cleanly
9. Broker connect / disconnect succeeds
10. Telegram token format valid (warns if not configured)
11. No stale PID lock from a previous crashed instance
12. Unit tests pass

---

## Daily Workflow

```
06:30  python generate_token.py          # refresh access token
07:00  python scripts/checklist.py       # pre-flight
09:00  run_bot.bat --symbol NIFTY        # start bot
15:30  Ctrl+C or /squareoff on Telegram  # stop bot (square-off runs automatically at 15:15)
```

---

## Strategy: VWAP + Volume

Entry conditions (LONG):
- Price above VWAP
- Price above EMA-9
- RSI between 45 and 70
- Volume ≥ 1.5× 20-period volume MA
- Within trading hours (09:20 – 14:30 IST)

Stop-loss: ATR-based (`atr_multiplier × ATR`)  
Target: `risk_reward_ratio × risk`  
Trailing SL: activates after `trail_trigger_atr × ATR` profit

Configure via `.env` or the dashboard Config tab — changes apply to the **next** trade without restart.

---

## Project Layout

```
c:\Project\trade-bot\
├── .env                    ← your secrets (git-ignored)
├── .env.example            ← template
├── main.py
├── config.py
├── requirements.txt
├── run_bot.bat             ← Windows start script
├── run_bot.sh              ← Linux/Mac start script
├── generate_token.py
├── test_connection.py
├── backtest_run.py
├── brokers/
├── strategies/
├── risk/
├── orders/
├── data/
├── db/
├── dashboard/
├── notifications/
├── backtest/
├── scripts/
│   ├── checklist.py
│   └── check_requirements.py
├── tests/
│   └── test_integration.py
└── logs/                   ← created automatically
```

---

## Deployment Notes

- **Token refresh**: Zerodha tokens expire at 06:00 IST. Re-run `generate_token.py` each morning before starting the bot.
- **Second laptop**: Copy the entire project folder and `.env` file. Run `pip install -r requirements.txt` once. The same `run_bot.bat` works.
- **Duplicate instance guard**: The bot uses a file lock (`logs/trading_bot.pid.lock`) to prevent two instances running simultaneously. If the bot crashes, delete the lock file manually or run `scripts/checklist.py` — it will warn you.
- **Log rotation**: Daily log files in `logs/`, retained 30 days by default (`LOG_RETENTION_DAYS` in `.env`).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `403 TokenException` on startup | Re-run `generate_token.py` — token expired |
| Bot starts but no trades | Check `PAPER_TRADE=true`, trading hours, and RSI/volume filters in logs |
| Dashboard blank | Visit http://localhost:5000 — may take 10s to warm up; check `logs/` for Flask errors |
| Telegram bot not responding | Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`; run checklist |
| `Bot is already running` error | Delete `logs/trading_bot.pid.lock` and restart |
| Optimizer very slow | Use `--days 30` for quick tests; the engine uses O(n) precomputed indicators |
