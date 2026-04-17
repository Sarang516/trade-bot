# Algorithmic Trading Bot — Phase 1

## Quick Start

```bash
# 1. Clone / download project
cd trading_bot

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 5. Verify broker connection
python test_connection.py

# 6. Run in paper trading mode (ALWAYS start here)
python main.py --paper --symbol NIFTY

# 7. Open dashboard
open http://localhost:5000
```

## Project Structure
```
trading_bot/
├── main.py              ← Entry point (run this)
├── config.py            ← All settings
├── requirements.txt     ← pip dependencies
├── .env.example         ← Copy to .env and fill keys
├── brokers/             ← Zerodha + ICICI adapters
├── strategies/          ← VWAP+Volume and future strategies
├── risk/                ← Stop loss, trailing SL, position sizing
├── data/                ← Live tick feed + candle builder
├── orders/              ← Order placement + paper trading
├── db/                  ← SQLite trade history
├── notifications/       ← Telegram alerts
└── dashboard/           ← Flask web dashboard
```

## Phases
- Phase 1 (this): Project skeleton ✅
- Phase 2: Broker API connections
- Phase 3: Live data feed
- Phase 4: Technical indicators
- Phase 5: Strategy logic
- Phase 6: Risk engine
- Phase 7: Order engine + paper trading
- Phase 8: Backtesting
- Phase 9: Dashboard + alerts
- Phase 10: Go-live checklist
