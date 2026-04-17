"""
test_connection.py — Verify broker API keys before running the bot.

Run: python test_connection.py
"""
from config import settings
from brokers import get_broker
from brokers.base_broker import Exchange

def main():
    print(f"\nTesting {settings.broker.upper()} connection...")
    print(f"Mode: {'PAPER' if settings.paper_trade else 'LIVE'}\n")

    broker = get_broker(settings)
    try:
        broker.connect()
        print("✓ Connection successful")

        margins = broker.get_margins()
        print(f"✓ Available cash: ₹{margins.available_cash:,.2f}")

        ltp = broker.get_ltp("NIFTY 50", Exchange.NSE)
        print(f"✓ Nifty LTP: {ltp}")

        print("\nAll checks passed — bot is ready to run!\n")
    except Exception as exc:
        print(f"✗ Connection failed: {exc}")
        raise

if __name__ == "__main__":
    main()
