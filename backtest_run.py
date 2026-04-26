"""
backtest_run.py - Command-line entry point for running backtests.

Usage examples
--------------
    # Backtest using broker historical data (needs active Zerodha session)
    python backtest_run.py --from 2024-01-01 --to 2024-03-31

    # Backtest from CSV file (no broker needed)
    python backtest_run.py --csv data/nifty_5min.csv

    # Custom symbol and interval
    python backtest_run.py --symbol BANKNIFTY --interval 15 --from 2024-01-01 --to 2024-03-31

    # Export trade log and equity curve
    python backtest_run.py --csv data/nifty_5min.csv --export trades.csv --equity-csv equity.csv

    # Save equity curve chart
    python backtest_run.py --csv data/nifty_5min.csv --chart output/

    # Run parameter grid optimization (top 5 by Sharpe)
    python backtest_run.py --csv data/nifty_5min.csv --optimize

    # Generate sample NIFTY data (no broker needed) and exit
    python backtest_run.py --generate-sample data/sample_nifty.csv
    python backtest_run.py --generate-sample data/sample_nifty.csv --days 180
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
@click.option("--symbol",    default="NIFTY",       show_default=True, help="Trading symbol")
@click.option("--strategy",  default="vwap_volume", show_default=True, help="Strategy name")
@click.option("--interval",  default=5,             show_default=True, type=int,
              help="Candle interval (minutes)")
@click.option("--from",  "from_date", default=None, help="Start date YYYY-MM-DD (broker mode)")
@click.option("--to",    "to_date",   default=None, help="End date YYYY-MM-DD (broker mode)")
@click.option("--csv",   default=None, help="Path to CSV file (offline mode - no broker needed)")
@click.option("--export", default=None, help="Export trade log to this CSV path")
@click.option("--equity-csv", default=None, help="Export equity curve to this CSV path")
@click.option("--commission", default=40.0, show_default=True, type=float,
              help="Round-trip commission per trade in Rs.")
@click.option("--slippage", default=0.05, show_default=True, type=float,
              help="One-way slippage in %% (e.g. 0.05 = 0.05%%)")
@click.option("--chart", default=None, metavar="DIR",
              help="Save equity curve chart to this directory (requires matplotlib)")
@click.option("--optimize", is_flag=True, default=False,
              help="Run parameter grid optimization and show top 5 by Sharpe ratio")
@click.option("--top-n", default=5, show_default=True, type=int,
              help="Number of top results to show when using --optimize")
@click.option("--broker", is_flag=True, default=False,
              help="Fetch data from broker using --days lookback (no --from/--to needed)")
@click.option("--generate-sample", default=None, metavar="PATH",
              help="Generate synthetic NIFTY CSV data and exit (no broker needed)")
@click.option("--days", default=90, show_default=True, type=int,
              help="Lookback days for --broker mode, or calendar days for --generate-sample")
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
    chart: str | None,
    optimize: bool,
    top_n: int,
    broker: bool,
    generate_sample: str | None,
    days: int,
) -> None:
    """Run a backtest, optimization, or data generation for the VWAP+Volume strategy."""

    # Configure logging - suppress chatty output during backtest
    logger.remove()
    logger.add(sys.stderr, level="WARNING",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # -- Generate sample data and exit -----------------------------------------
    if generate_sample:
        from backtest.engine import generate_sample_data
        console.print(f"[cyan]Generating {days}-day synthetic NIFTY data...[/cyan]")
        df = generate_sample_data(
            n_days=days,
            interval_minutes=interval,
            output_csv=generate_sample,
        )
        console.print(
            f"[green]Sample data saved -> {generate_sample}[/green] "
            f"([white]{len(df):,} candles[/white])"
        )
        return

    from config import settings
    from backtest.engine import BacktestEngine

    # -- --broker shorthand: auto-compute from/to from --days -----------------
    if broker and not from_date and not to_date:
        from datetime import timedelta
        td = date.today()
        fd = td - timedelta(days=days)
        from_date = fd.strftime("%Y-%m-%d")
        to_date   = td.strftime("%Y-%m-%d")

    # -- Load data -------------------------------------------------------------
    df = _load_data(symbol, csv, from_date, to_date, interval, settings)

    # -- Optimization mode -----------------------------------------------------
    if optimize:
        from backtest.optimizer import run_grid_optimization, print_optimization_report, DEFAULT_PARAM_GRID

        console.print(
            f"\n[cyan]Running grid optimization | {symbol} | {strategy} | "
            f"{interval}-min | {len(DEFAULT_PARAM_GRID)} params[/cyan]"
        )
        total_combos = 1
        for v in DEFAULT_PARAM_GRID.values():
            total_combos *= len(v)
        console.print(f"[dim]Total combinations: {total_combos} -> showing top {top_n}[/dim]\n")

        results = run_grid_optimization(
            symbol=symbol,
            strategy_name=strategy,
            settings=settings,
            df=df,
            interval_minutes=interval,
            param_grid=DEFAULT_PARAM_GRID,
            commission_inr=commission,
            slippage_pct=slippage / 100.0,
            top_n=top_n,
            show_progress=True,
        )
        print_optimization_report(results)
        return

    # -- Detect market regime (uses broker if available, else skips) -----------
    detected_regime = "UNKNOWN"
    if from_date and to_date and not csv:
        try:
            from brokers import get_broker as _get_broker
            from strategies.regime_detector import RegimeDetector
            _rb = _get_broker(settings)
            _rb.connect()
            _det = RegimeDetector(_rb, settings)
            detected_regime = _det.detect(symbol).value
            _rb.disconnect()
            console.print(f"[cyan]Market regime detected: [bold]{detected_regime}[/bold][/cyan]")
        except Exception as _e:
            console.print(f"[yellow]Regime detection skipped: {_e}[/yellow]")

    # -- Standard backtest -----------------------------------------------------
    engine = BacktestEngine(
        strategy_name=strategy,
        symbol=symbol,
        settings=settings,
        commission_inr=commission,
        slippage_pct=slippage / 100.0,
    )
    result = engine.run_from_dataframe(df, interval_minutes=interval)

    result.print_summary()

    # -- Log result to parameter registry -------------------------------------
    try:
        from data.parameter_registry import ParameterRegistry
        from strategies.vwap_volume import VWAPVolumeConfig
        _cfg = VWAPVolumeConfig()
        _params = {
            "volume_surge_multiplier": _cfg.volume_surge_multiplier,
            "volume_ma_period":        _cfg.volume_ma_period,
            "rsi_long_min":            _cfg.rsi_long_min,
            "rsi_long_max":            _cfg.rsi_long_max,
            "rsi_short_min":           _cfg.rsi_short_min,
            "rsi_short_max":           _cfg.rsi_short_max,
            "ema_period":              _cfg.ema_period,
            "ema_trend_period":        _cfg.ema_trend_period,
            "sl_atr_multiplier":       _cfg.sl_atr_multiplier,
            "rr_ratio":                _cfg.rr_ratio,
            "vwap_exit_candles":       _cfg.vwap_exit_candles,
            "interval_minutes":        interval,
        }
        _reg = ParameterRegistry()
        _reg.log_run(
            symbol=symbol,
            regime=detected_regime,
            interval_minutes=interval,
            params=_params,
            result={
                "total_trades":     result.total_trades,
                "win_rate_pct":     result.win_rate_pct,
                "net_pnl":          result.net_pnl,
                "profit_factor":    result.profit_factor,
                "sharpe_ratio":     result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "cagr_pct":         result.cagr_pct,
            },
            source="backtest",
            period_from=from_date,
            period_to=to_date,
        )
        console.print("[dim]Result saved to parameter registry.[/dim]")
    except Exception as _e:
        console.print(f"[yellow]Registry log skipped: {_e}[/yellow]")

    if export:
        result.to_csv(export)
        console.print(f"[green]Trade log saved -> {export}[/green]")

    if equity_csv:
        result.to_equity_csv(equity_csv)
        console.print(f"[green]Equity curve saved -> {equity_csv}[/green]")

    if chart:
        path = result.plot_charts(chart)
        if path:
            console.print(f"[green]Chart saved -> {path}[/green]")
        else:
            console.print(
                "[yellow]Chart skipped - install matplotlib:[/yellow] "
                "[dim]pip install matplotlib[/dim]"
            )


# -- Data loading helper --------------------------------------------------------

def _load_data(symbol, csv, from_date, to_date, interval, settings):
    """Load DataFrame from CSV or broker. Exits on error."""
    if csv:
        import pandas as _pd
        from pathlib import Path as _P
        console.print(f"[cyan]Loading data from:[/cyan] {csv}")
        path = _P(csv)
        if not path.exists():
            console.print(f"[red]File not found: {csv}[/red]")
            sys.exit(1)
        df = _pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        if dt_col:
            df[dt_col] = _pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        return df

    elif from_date and to_date:
        try:
            fd = datetime.strptime(from_date, "%Y-%m-%d").date()
            td = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]Date format must be YYYY-MM-DD[/red]")
            sys.exit(1)

        total_days = (td - fd).days
        n_chunks = max(1, (total_days + 59) // 60)
        console.print(
            f"[cyan]Connecting to broker to fetch {symbol} data "
            f"({from_date} to {to_date}, ~{total_days} days"
            + (f", {n_chunks} chunks" if n_chunks > 1 else "")
            + ")...[/cyan]"
        )
        from brokers import get_broker
        broker = get_broker(settings)
        try:
            broker.connect()
        except Exception as exc:
            console.print(f"[red]Broker connection failed: {exc}[/red]")
            console.print("[yellow]Tip: generate_token.py refreshes your access token.[/yellow]")
            sys.exit(1)

        from data.feed import HistoricalData
        from brokers.base_broker import Exchange
        hist = HistoricalData(broker)
        df = hist.fetch(
            symbol=symbol,
            exchange=Exchange.NSE,
            from_date=datetime.combine(fd, datetime.min.time()),
            to_date=datetime.combine(td, datetime.max.time()),
            interval_minutes=interval,
        )
        broker.disconnect()
        if df.empty:
            console.print(f"[red]No data returned for {symbol} {from_date} to {to_date}[/red]")
            console.print(
                "[yellow]This usually means the access token expired or the symbol is wrong.\n"
                "Run: python generate_token.py   to refresh your token.[/yellow]"
            )
            sys.exit(1)
        console.print(f"[green]Fetched {len(df):,} candles ({from_date} to {to_date})[/green]")
        return df

    else:
        console.print(
            "[red]Provide either --csv <file> or both --from <date> --to <date>[/red]\n"
            "[dim]Or use --generate-sample <path> to create sample data.[/dim]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
