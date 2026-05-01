"""
scripts/agent_manual.py - Run the trading agent manually to see today's decision.

Usage:
    python scripts/agent_manual.py --symbol RELIANCE
    python scripts/agent_manual.py --symbol RELIANCE --apply
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import click
from loguru import logger
logger.remove()  # suppress loguru noise


@click.command()
@click.option("--symbol", default="RELIANCE", show_default=True, help="Symbol to analyze")
@click.option("--apply", is_flag=True, default=False, help="Apply the agent's decision to strategy config")
def main(symbol: str, apply: bool):
    """Run the trading agent and display today's decision."""

    from config import settings
    from brokers import get_broker
    from agents.trading_agent import TradingAgent

    print("\n" + "=" * 80)
    print(f"  Claude Trading Agent — Analysis for {symbol.upper()}")
    print("=" * 80 + "\n")

    # Initialize
    broker = get_broker(settings)
    broker.connect()
    agent = TradingAgent(broker, settings)

    # Analyze
    decision = agent.analyze_and_decide(symbol)
    broker.disconnect()

    # Display
    print(f"Regime          : {decision.regime}")
    print(f"Confidence      : {decision.confidence_score:.0f}%")
    print(f"Should Trade    : {'YES' if decision.should_trade else 'NO'}")
    print(f"Trading Mode    : {decision.trading_mode}")
    print(f"Position Size   : {decision.position_size_multiplier:.1f}x default")
    print(f"Trailing SL     : {'Enabled' if decision.trailing_sl_enabled else 'Disabled'}")
    print(f"Max Open Pos    : {decision.max_open_positions}")
    print()

    print("Recommended Parameters:")
    for k, v in decision.parameters.items():
        print(f"  {k}: {v}")
    print()

    print("Reasoning:")
    print(f"  {decision.reasoning}")
    print()

    # Apply if requested
    if apply:
        from strategies import get_strategy
        strategy = get_strategy(settings.strategy, symbol=symbol, settings=settings)

        # Apply parameters
        for k, v in decision.parameters.items():
            if hasattr(strategy.cfg, k):
                setattr(strategy.cfg, k, v)

        print(f"[OK] Applied {len(decision.parameters)} parameters to strategy config.")
        print(f"[OK] Position size multiplier: {decision.position_size_multiplier}x")
        print()

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
