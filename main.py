"""
main.py - Trading Bot entry point.

Run:
    python main.py              # Start bot (reads .env)
    python main.py --paper      # Force paper trading mode
    python main.py --help       # Show all options
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from config import settings

console = Console()


def _configure_logging() -> None:
    """Set up Loguru with file rotation and console output."""
    logger.remove()
    log_file = settings.log_dir / "bot_{time:YYYY-MM-DD}.log"
    logger.add(
        log_file,
        level=settings.log_level,
        rotation="00:00",
        retention=f"{settings.log_retention_days} days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
        enqueue=True,
    )
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )


_pid_lock = None  # module-level so GC never releases the lock while bot runs


def _acquire_pid_lock() -> None:
    """Prevent duplicate bot instances using a PID file lock."""
    global _pid_lock
    from filelock import FileLock, Timeout

    pid_file = Path(settings.log_dir) / "trading_bot.pid"
    _pid_lock = FileLock(str(pid_file) + ".lock", timeout=1)
    try:
        _pid_lock.acquire()
        pid_file.write_text(str(threading.get_ident()))
    except Timeout:
        console.print("[red]Bot is already running! Exiting.[/red]")
        sys.exit(1)


def _run_dashboard() -> None:
    """Start the Flask dashboard in a background thread."""
    try:
        from dashboard.app import create_app
        app = create_app()
        app.run(
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            debug=False,
            use_reloader=False,
        )
    except Exception as exc:
        logger.warning("Dashboard failed to start: {}", exc)


def _warmup_strategy(broker, strategy, symbol: str, s) -> None:
    """Load historical candles into the strategy before live trading begins."""
    from brokers.base_broker import Exchange
    from data.feed import HistoricalData

    logger.info("Warming up strategy with historical candles...")
    try:
        hist = HistoricalData(broker)
        candles = hist.warmup_candles(
            symbol=symbol,
            exchange=Exchange.NSE,
            interval_minutes=s.candle_interval,
            n_candles=strategy._warmup_candles + 50,
        )
        for candle in candles:
            strategy.on_candle(candle)
        logger.info("Warmup complete - {} candles loaded", len(candles))
    except Exception as exc:
        logger.warning("Warmup failed (continuing without historical data): {}", exc)


def _trading_loop(broker, strategy, order_manager, risk_manager, symbol: str, s,
                  bot_status: dict, trade_logger=None, notifier=None) -> None:
    """Main trading loop - runs during market hours."""
    from data.feed import TickDataFeed, MarketHours

    market = MarketHours()

    feed = TickDataFeed(
        broker=broker,
        symbols=[symbol],
        max_ticks=1000,
        settings=s,
    )

    def on_candle(candle) -> None:
        try:
            strategy.on_candle(candle)
            if bot_status.get("status") == "PAUSED":
                return
            signal = strategy.generate_signal()
            if signal:
                order_manager.process_signal(signal)
        except Exception as exc:
            logger.error("on_candle error (candle skipped): {}", exc)

    def on_tick(tick: dict) -> None:
        try:
            strategy.on_tick(tick)
        except Exception as exc:
            logger.warning("on_tick error: {}", exc)

    feed.subscribe_tick(on_tick)
    feed.subscribe_candle(on_candle, interval_minutes=s.candle_interval)
    feed.start()

    logger.info("Trading loop started | Strategy: {} | Symbol: {}", s.strategy, symbol)

    _last_market_open_day: datetime | None = None

    while True:
        now = datetime.now()

        if not market.is_market_open(now):
            wait = market.minutes_to_open(now)
            if wait and wait > 0:
                logger.info("Market closed. Next open in {:.0f} min", wait)
            time.sleep(60)
            continue

        # Reset candle builders at market open (once per day)
        today = now.date()
        if _last_market_open_day != today:
            _last_market_open_day = today
            feed.reset_candles()
            strategy.on_market_open()
            logger.info("Market open - candles reset for {}", today)

        # Periodic reconciliation every 5 minutes
        if now.second < 5 and now.minute % 5 == 0:
            try:
                order_manager.sync_with_broker()
                risk_manager.update_daily_pnl()
            except Exception as exc:
                logger.error("Reconciliation error: {}", exc)

        # Square off check
        if now.time() >= s.squareoff:
            logger.info("Square-off time reached - closing all positions")
            try:
                order_manager.square_off_all()
                strategy.on_market_close()
                if trade_logger and notifier:
                    try:
                        trade_logger.compute_daily_summary()
                        summary = trade_logger.get_today_summary()
                        notifier.notify_daily_summary(summary)
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Square-off failed: {}", exc)
            time.sleep(900)

        time.sleep(0.5)


@click.command()
@click.option("--paper", is_flag=True, help="Force paper trading mode")
@click.option("--symbol", default="NIFTY", show_default=True, help="Trading symbol")
@click.option("--no-dashboard", is_flag=True, help="Skip the web dashboard")
@click.option("--log-level", default=None, help="Override log level")
def main(paper: bool, symbol: str, no_dashboard: bool, log_level: str | None) -> None:
    """Algorithmic Trading Bot - starts the live or paper trading session."""

    # Apply CLI overrides
    if paper:
        import os
        os.environ["PAPER_TRADE"] = "true"
    if log_level:
        import os
        os.environ["LOG_LEVEL"] = log_level.upper()

    # Reload settings after env overrides
    from importlib import reload
    import config as cfg_module
    reload(cfg_module)
    from config import settings as s

    _configure_logging()
    _acquire_pid_lock()

    # Print startup banner
    mode_color = "yellow" if s.paper_trade else "red"
    mode_label = "PAPER TRADING" if s.paper_trade else "*** LIVE TRADING ***"
    console.print(
        Panel(
            s.describe(),
            title=f"[{mode_color}]{mode_label}[/{mode_color}]",
            border_style=mode_color,
        )
    )

    if not s.paper_trade:
        console.print(
            "[red bold]WARNING: LIVE TRADING MODE - real money will be used![/red bold]"
        )
        if not click.confirm("Are you sure you want to continue?"):
            sys.exit(0)

    # -- Initialise components -----------------------------------------
    logger.info("Initialising components...")

    from brokers import get_broker
    from strategies import get_strategy
    from risk.risk_manager import RiskManager
    from orders.order_manager import OrderManager
    from db.trade_logger import TradeLogger

    # -- Telegram notifier ----------------------------------------------------
    # Active when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set in .env.
    # Falls back to console logging when tokens are DUMMY values.
    from notifications.telegram_bot import TelegramNotifier
    _tg_active = (
        s.telegram_bot_token not in ("DUMMY_TELEGRAM_TOKEN", "")
        and s.telegram_chat_id not in ("000000000", "")
    )
    notifier = TelegramNotifier(settings=s)

    broker = get_broker(s)
    strategy = get_strategy(s.strategy, symbol=symbol, settings=s)

    # -- Regime-aware parameter loading (the "agent" decision at startup) ------
    # Detects today's market regime and loads the best-known parameters for it
    # from the parameter registry.  On the first run the registry is empty and
    # defaults are used; after a few backtests it auto-selects the best config.
    _current_regime = "UNKNOWN"
    try:
        from strategies.regime_detector import RegimeDetector, describe_regime
        from data.parameter_registry import ParameterRegistry
        _det = RegimeDetector(broker, s)
        # broker not yet connected — connect briefly just for regime fetch
        broker.connect()
        _current_regime = _det.detect(symbol).value
        broker.disconnect()

        logger.info("Market regime: {} - {}", _current_regime, describe_regime(
            __import__("strategies.regime_detector", fromlist=["Regime"]).Regime(_current_regime)
        ))
        console.print(f"[cyan]Market regime: [bold]{_current_regime}[/bold][/cyan]")

        _reg = ParameterRegistry()
        _best = _reg.best_params(symbol, _current_regime)
        if _best and hasattr(strategy, "cfg"):
            applied = []
            for k, v in _best.items():
                if hasattr(strategy.cfg, k):
                    setattr(strategy.cfg, k, type(getattr(strategy.cfg, k))(v))
                    applied.append(f"{k}={v}")
            if applied:
                logger.info("Auto-loaded regime params for {}/{}: {}", symbol, _current_regime, applied)
                console.print(f"[green]Regime params applied ({len(applied)} settings)[/green]")
            else:
                logger.info("No matching regime params found — using defaults")
        else:
            logger.info("No registry entry for {}/{} — using default parameters", symbol, _current_regime)
            console.print(f"[yellow]No regime params yet for {symbol}/{_current_regime} - run backtests to build history[/yellow]")
    except Exception as _exc:
        logger.warning("Regime detection failed (using defaults): {}", _exc)

    risk_manager = RiskManager(settings=s)
    trade_logger = TradeLogger()
    order_manager = OrderManager(
        broker=broker,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
        notifier=notifier,
        settings=s,
        strategy=strategy,
    )

    # -- Shared bot state (dashboard + Telegram both read/write this) ----------
    _bot_status = {"status": "RUNNING"}

    # Wire context into dashboard
    from dashboard.app import set_context as _dash_set
    _dash_set(
        order_manager=order_manager,
        risk_manager=risk_manager,
        strategy=strategy,
        trade_logger=trade_logger,
        symbol=symbol,
        bot_status=_bot_status,
    )

    # Wire context into Telegram bot and start it
    notifier.set_context(
        order_manager=order_manager,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
        bot_status=_bot_status,
    )
    if _tg_active:
        notifier.start()
        logger.info("Telegram bot started")
        trade_logger.log_bot_event("INFO", "Telegram bot started")
    else:
        logger.info("Telegram disabled - set TELEGRAM_BOT_TOKEN in .env to enable")

    # -- Connect broker ------------------------------------------------
    logger.info("Connecting to {} broker...", s.broker)
    try:
        broker.connect()
    except Exception as exc:
        logger.critical("Broker connection failed: {}", exc)
        console.print(f"[red]Broker connection failed: {exc}[/red]")
        sys.exit(1)

    # -- Historical warmup - pre-load candles before live trading ------
    _warmup_strategy(broker, strategy, symbol, s)

    # -- Dashboard (background thread) ---------------------------------
    if not no_dashboard:
        dash_thread = threading.Thread(target=_run_dashboard, daemon=True)
        dash_thread.start()
        logger.info(
            "Dashboard running at http://{}:{}", s.dashboard_host, s.dashboard_port
        )

    # -- Graceful shutdown ---------------------------------------------
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received - squaring off and disconnecting...")
        try:
            order_manager.square_off_all()
        except Exception:
            pass
        try:
            broker.disconnect()
        except Exception:
            pass
        logger.info("Bot stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # -- Start trading loop --------------------------------------------
    try:
        _trading_loop(broker, strategy, order_manager, risk_manager, symbol, s,
                      bot_status=_bot_status, trade_logger=trade_logger, notifier=notifier)
    except Exception as exc:
        logger.critical("Fatal error in trading loop: {}", exc)
        # -- Telegram crash alert (notifier is a stub until Telegram is enabled) -
        try:
            notifier.notify_error(f"BOT CRASHED: {exc}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
