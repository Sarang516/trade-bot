"""
scripts/regime_report.py - Show parameter performance by market regime.

Usage:
    python scripts/regime_report.py                  # all symbols
    python scripts/regime_report.py --symbol RELIANCE
    python scripts/regime_report.py --regime BEAR
    python scripts/regime_report.py --detect RELIANCE  # detect today's regime live
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import click
from loguru import logger
logger.remove()   # suppress loguru noise in CLI


@click.command()
@click.option("--symbol",  default=None, help="Filter by symbol (e.g. RELIANCE)")
@click.option("--regime",  default=None, help="Filter by regime (BULL/BEAR/RANGE/VOLATILE)")
@click.option("--detect",  default=None, metavar="SYMBOL",
              help="Detect today's regime for SYMBOL using live broker data")
@click.option("--runs",    is_flag=True, default=False,
              help="Show individual run history (not just best params)")
@click.option("--limit",   default=20, show_default=True, type=int,
              help="Max rows to show for --runs")
def main(symbol, regime, detect, runs, limit):
    """Parameter performance tracker - shows what worked in each market regime."""

    from data.parameter_registry import ParameterRegistry
    reg = ParameterRegistry()

    # ── Live regime detection ────────────────────────────────────────────────
    if detect:
        _detect_live(detect)
        print()

    # ── Individual run history ───────────────────────────────────────────────
    if runs:
        data = reg.get_runs(symbol=symbol, regime=regime, limit=limit)
        if not data:
            print("No runs found.")
            return

        print()
        print("=" * 100)
        print("  Run History")
        print("=" * 100)
        print(f"  {'#':>4} {'Date':<12} {'Symbol':<12} {'Regime':<10} {'Src':<9} "
              f"{'Int':>4} {'Trades':>7} {'WR%':>6} {'Net P&L':>12} {'Sharpe':>8}")
        print(f"  {'-'*4} {'-'*12} {'-'*12} {'-'*10} {'-'*9} {'-'*4} "
              f"{'-'*7} {'-'*6} {'-'*12} {'-'*8}")
        for r in data:
            pnl_str = f"Rs.{r['net_pnl']:+,.0f}"
            print(
                f"  {r['id']:>4} {r['run_date']:<12} {r['symbol']:<12} "
                f"{r['regime']:<10} {r['source']:<9} {r['interval_min']:>4} "
                f"{r['total_trades']:>7} {r['win_rate_pct']:>5.1f}% "
                f"{pnl_str:>12} {r['sharpe_ratio']:>8.3f}"
            )
        print()
        return

    # ── Regime summary (default) ─────────────────────────────────────────────
    reg.print_report(symbol=symbol)

    # Filter by regime if requested
    if regime:
        summaries = [
            s for s in reg.get_regime_summary()
            if s["regime"] == regime.upper()
            and (symbol is None or s["symbol"] == symbol.upper())
        ]
        if summaries:
            print(f"Best parameters for {regime.upper()} markets:")
            for s in summaries:
                print(f"\n  {s['symbol']} / {s['regime']} (Sharpe {s['best_sharpe']:.3f}, "
                      f"{s['run_count']} runs):")
                for k, v in s["best_params"].items():
                    print(f"    {k}: {v}")


def _detect_live(symbol: str) -> None:
    """Detect and print the current market regime for a symbol."""
    try:
        from config import settings
        from brokers import get_broker
        from strategies.regime_detector import RegimeDetector, describe_regime

        print(f"\nDetecting market regime for {symbol}...")
        broker = get_broker(settings)
        broker.connect()
        detector = RegimeDetector(broker, settings)
        regime = detector.detect(symbol)
        broker.disconnect()

        print(f"\n  Symbol : {symbol}")
        print(f"  Regime : {regime.value}")
        print(f"  Meaning: {describe_regime(regime)}")

        # Show best known params for this regime
        from data.parameter_registry import ParameterRegistry
        reg = ParameterRegistry()
        best = reg.best_params(symbol, regime.value)
        if best:
            print(f"\n  Best known parameters for {symbol} in {regime.value} market:")
            for k, v in best.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  No parameter history yet for {symbol} / {regime.value}.")
            print("  Run backtests to build up the registry.")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
