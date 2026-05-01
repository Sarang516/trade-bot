"""
scripts/batch_backtest.py - Run multiple backtests systematically to populate registry.

This script runs a grid of backtests across different symbols, timeframes, and periods
to build up a rich registry of "what works where". The agent uses this data to make
better parameter decisions.

Usage:
    python scripts/batch_backtest.py --quick       # 3 quick tests for validation
    python scripts/batch_backtest.py --full        # Full grid (may take hours)
    python scripts/batch_backtest.py --15min       # Just 15-min candle tests
    python scripts/batch_backtest.py --symbol BANKNIFTY --from 2024-01-01 --to 2025-01-01
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

console = Console()

# Test configurations: (symbol, interval_minutes, from_date, to_date, label)
QUICK_TESTS = [
    ("RELIANCE", 15, "2024-06-01", "2025-06-01", "RELIANCE 15min Bear"),
    ("BANKNIFTY", 15, "2024-06-01", "2025-06-01", "BANKNIFTY 15min Bear"),
    ("FINNIFTY", 15, "2024-06-01", "2025-06-01", "FINNIFTY 15min Bear"),
]

FULL_TESTS = [
    # 15-min candles on recent bear market (2024-2025)
    ("RELIANCE", 15, "2024-06-01", "2025-06-01", "RELIANCE 15min Bear"),
    ("BANKNIFTY", 15, "2024-06-01", "2025-06-01", "BANKNIFTY 15min Bear"),
    ("FINNIFTY", 15, "2024-06-01", "2025-06-01", "FINNIFTY 15min Bear"),

    # 15-min candles on older bull market (2022)
    ("RELIANCE", 15, "2022-01-01", "2022-12-31", "RELIANCE 15min Bull2022"),
    ("BANKNIFTY", 15, "2022-01-01", "2022-12-31", "BANKNIFTY 15min Bull2022"),

    # 5-min for comparison (if we want to see the difference)
    ("RELIANCE", 5, "2024-06-01", "2025-06-01", "RELIANCE 5min Bear"),
]

TIMEFRAME_TESTS_15MIN = [
    ("RELIANCE", 15, "2024-06-01", "2025-06-01", "RELIANCE 15min"),
    ("BANKNIFTY", 15, "2024-06-01", "2025-06-01", "BANKNIFTY 15min"),
    ("FINNIFTY", 15, "2024-06-01", "2025-06-01", "FINNIFTY 15min"),
]


def run_backtest(symbol: str, interval: int, from_date: str, to_date: str, label: str) -> tuple[bool, dict]:
    """Run a single backtest and return (success, result_dict)."""
    cmd = [
        "python",
        "backtest_run.py",
        "--symbol", symbol,
        "--interval", str(interval),
        "--from", from_date,
        "--to", to_date,
    ]

    console.print(f"\n[cyan]Running: {label}[/cyan]")
    console.print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=False,
            text=True,
            timeout=600,  # 10 min timeout per test
        )

        if result.returncode == 0:
            console.print(f"[green][OK][/green] {label}")
            return True, {"label": label, "symbol": symbol, "interval": interval}
        else:
            console.print(f"[red][FAILED][/red] {label} (exit code {result.returncode})")
            return False, {"label": label, "error": f"exit code {result.returncode}"}
    except subprocess.TimeoutExpired:
        console.print(f"[red][TIMEOUT][/red] {label}")
        return False, {"label": label, "error": "timeout"}
    except Exception as exc:
        console.print(f"[red][ERROR][/red] {label}: {exc}")
        return False, {"label": label, "error": str(exc)}


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all results."""
    console.print("\n" + "=" * 80)
    console.print("BATCH BACKTEST SUMMARY")
    console.print("=" * 80 + "\n")

    table = Table(title="Backtest Results")
    table.add_column("Label", style="cyan")
    table.add_column("Symbol", style="magenta")
    table.add_column("Interval", justify="right")
    table.add_column("Status", style="green")

    passed = 0
    failed = 0

    for r in results:
        status = "OK" if r.get("success") else "FAILED"
        status_color = "green" if r.get("success") else "red"
        table.add_row(
            r.get("label", "?"),
            r.get("symbol", "?"),
            str(r.get("interval", "?")),
            f"[{status_color}]{status}[/{status_color}]",
        )
        if r.get("success"):
            passed += 1
        else:
            failed += 1

    console.print(table)
    console.print(f"\nTotal: {passed} passed, {failed} failed out of {passed + failed} tests")
    console.print("\nNext steps:")
    console.print("  1. Check parameter registry: python scripts/regime_report.py")
    console.print("  2. Run agent analysis: python scripts/agent_manual.py --symbol RELIANCE")
    console.print("  3. Start bot with enriched data: python main.py")


@click.command()
@click.option("--quick", is_flag=True, help="Run quick validation tests only")
@click.option("--full", is_flag=True, help="Run full test grid")
@click.option("--15min", is_flag=True, help="Run only 15-min candle tests")
@click.option("--symbol", default=None, help="Override: test only this symbol")
@click.option("--from", "from_date", default=None, help="Override: from date (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="Override: to date (YYYY-MM-DD)")
def main(quick: bool, full: bool, symbol: str | None, from_date: str | None, to_date: str | None, **kwargs):
    """Run batch backtests to populate the parameter registry."""

    console.print(Panel(
        "Batch Backtest Runner\n"
        "Systematically tests different symbols/timeframes to populate registry",
        title="[bold cyan]BATCH BACKTEST[/bold cyan]",
        border_style="cyan",
    ))

    # Determine test set
    if quick:
        tests = QUICK_TESTS
        console.print("[yellow]Mode: QUICK (3 tests)[/yellow]")
    elif kwargs.get("15min"):
        tests = TIMEFRAME_TESTS_15MIN
        console.print("[yellow]Mode: 15-MIN ONLY[/yellow]")
    elif full:
        tests = FULL_TESTS
        console.print("[yellow]Mode: FULL GRID[/yellow]")
    else:
        tests = QUICK_TESTS
        console.print("[yellow]Mode: QUICK (default, use --full for all tests)[/yellow]")

    # Apply CLI overrides
    if symbol or from_date or to_date:
        if not (symbol and from_date and to_date):
            console.print("[red]ERROR: Must provide all of --symbol, --from, --to together[/red]")
            sys.exit(1)
        tests = [(symbol, 15, from_date, to_date, f"{symbol} {from_date} to {to_date}")]
        console.print(f"[yellow]Mode: CUSTOM ({symbol})[/yellow]")

    console.print(f"Tests to run: {len(tests)}")
    for sym, intv, f, t, label in tests:
        console.print(f"  - {label}")

    # Run tests
    console.print("\n" + "=" * 80)
    results = []
    start_time = datetime.now()

    for i, (sym, intv, from_d, to_d, label) in enumerate(tests, 1):
        console.print(f"\n[bold]Test {i}/{len(tests)}[/bold]")
        success, result = run_backtest(sym, intv, from_d, to_d, label)
        results.append({
            "label": label,
            "symbol": sym,
            "interval": intv,
            "success": success,
            **result,
        })

    elapsed = datetime.now() - start_time
    console.print(f"\n[cyan]Total time: {elapsed}[/cyan]")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
