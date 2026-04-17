"""
strategies/__init__.py — Strategy factory.

Usage:
    from strategies import get_strategy
    strategy = get_strategy("vwap_volume", symbol="NIFTY", settings=settings)
"""

from __future__ import annotations

from strategies.base_strategy import BaseStrategy


def get_strategy(name: str, symbol: str, settings) -> BaseStrategy:
    """Return the strategy instance for the given strategy name."""
    name = name.lower()

    if name == "vwap_volume":
        from strategies.vwap_volume import VWAPVolumeStrategy
        return VWAPVolumeStrategy(symbol=symbol, settings=settings)

    raise ValueError(
        f"Unknown strategy '{name}'. "
        "Available strategies: vwap_volume"
    )


__all__ = ["get_strategy", "BaseStrategy"]
