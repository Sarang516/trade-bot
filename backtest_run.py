"""
backtest_run.py — Command-line entry point for running backtests.

Usage examples
--------------
    # Backtest using broker historical data (needs active Zerodha session)
    python backtest_run.py --from 2024-01-01 --to 2024-03-31

    # Backtest from CSV file (no broker needed)
    python backtest_run.py --csv data/nifty_5min.csv

    # Custom symbol and interval
    python backtest_run.py --symbol BANKNIFTY --interval 15 --from 2024-01-01 --to 2024-03-31

    # Export trade log
    python backtest_run.py --csv data/nifty_5min.csv --export trades.csv
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import click
from loguru import logger
from rich.console import Console

console = Console()


@click.command()
@click.option("--symbol",   default="NIFTY",       show_default=True, help="Trading symbol")
@click.option("--strategy", default="vwap_volume", show_default=True, help="Strategy name")
@click.option("--interval", default=5,             show_default=True, type=int, help="Candle interval (minutes)")
@click.option("--from",  "from_date", default=None, help="Start date YYYY-MM-DD (broker mode)")
@click.option("--to",    "to_date",   default=None, help="End date YYYY-MM-DD (broker mode)")
@click.option("--csv",   default=None, help="Path to CSV file (offline mode — no broker needed)")
@click.option("--export", default=None, help="Export trade log to this CSV path")
@click.option("--equity-csv", default=None, help="Export equity curve to this CSV path")
@click.option("--commission", default=40.0, show_default=True, type=float,
              help="Round-trip commission per trade in ₹")
@click.option("--slippage", default=0.05, show_default=True, type=float,
              help="One-way slippage in % (e.g. 0.05 = 0.05%%)")
def main(
    symbol: str,
    strategy: str,
    interval: int,
    from_date: str | None,
    to_date: str | None,
    csv: str | None,
    export: str | None,
    equity_csv: str | None,
    commission: float,
    slippage: float,
) -> None:
    """Run a backtest for the VWAP+Volume strategy."""

    # Configure logging — suppress chatty loguru output during backtest
    logger.remove()
    logger.add(sys.stderr, level="WARNING",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    from config import settings
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(
        strategy_name=strategy,
        symbol=symbol,
        settings=settings,
        commission_inr=commission,
        slippage_pct=slippage / 100.0,
    )

    # ── CSV mode ─────────────────────────────────────────────────────────────
    if csv:
        console.print(f"[cyan]Loading data from:[/cyan] {csv}")
        result = engine.run_from_csv(csv, interval_minutes=interval)

    # ── Broker mode ──────────────────────────────────────────────────────────
    elif from_date and to_date:
        try:
            fd = datetime.strptime(from_date, "%Y-%m-%d").date()
            td = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]Date format must be YYYY-MM-DD[/red]")
            sys.exit(1)

        console.print(f"[cyan]Connecting to broker to fetch data for {symbol} ...[/cyan]")

        from brokers import get_broker
        broker = get_broker(settings)
        try:
            broker.connect()
        except Exception as exc:
            console.print(f"[red]Broker connection failed: {exc}[/red]")
            console.print("[yellow]Tip: generate_token.py refreshes your access token.[/yellow]")
            sys.exit(1)

        result = engine.run_from_broker(broker, fd, td, interval_minutes=interval)
        broker.disconnect()

    else:
        console.print(
            "[red]Provide either --csv <file> or both --from <date> --to <date>[/red]"
        )
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────────
    result.print_summary()

    if export:
        result.to_csv(export)
        console.print(f"[green]Trade log saved → {export}[/green]")

    if equity_csv:
        result.to_equity_csv(equity_csv)
        console.print(f"[green]Equity curve saved → {equity_csv}[/green]")


if __name__ == "__main__":
    main()
