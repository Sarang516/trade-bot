"""
brokers/__init__.py — Broker factory.

Usage:
    from brokers import get_broker
    broker = get_broker(settings)
    broker.connect()
"""

from __future__ import annotations

from brokers.base_broker import BaseBroker


def get_broker(settings) -> BaseBroker:
    """Return the correct broker instance based on settings.broker."""
    name = settings.broker.lower()

    if name == "zerodha":
        from brokers.zerodha import ZerodhaBroker
        return ZerodhaBroker(settings)

    # ── ICICI Breeze (disabled — uncomment when ready to use ICICI) ──────────
    # Steps to re-enable:
    #   1. Uncomment the block below.
    #   2. Uncomment ICICI credential fields in config.py.
    #   3. Change broker Literal type in config.py to include "icici".
    #   4. Set BROKER=icici and ICICI_* credentials in .env.
    # if name == "icici":
    #     from brokers.icici import ICICIBroker
    #     return ICICIBroker(settings)

    raise ValueError(
        f"Unknown broker '{name}'. Set BROKER=zerodha in .env"
    )


__all__ = ["get_broker", "BaseBroker"]
