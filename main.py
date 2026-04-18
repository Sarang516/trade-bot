"""
main.py — Trading Bot entry point.

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


def _acquire_pid_lock() -> None:
    """Prevent duplicate bot instances using a PID file lock."""
    from filelock import FileLock, Timeout

    pid_file = Path(settings.log_dir) / "trading_bot.pid"
    lock = FileLock(str(pid_file) + ".lock", timeout=1)
    try:
        lock.acquire()
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


def _trading_loop(broker, strategy, order_manager, risk_manager) -> None:
    """Main trading loop — runs during market hours."""
    from data.feed import CandleBuilder, MarketHours

    market = MarketHours()
    candle_builder = CandleBuilder(interval_minutes=settings.candle_interval)

    def on_tick(tick: dict) -> None:
        """Called on every live price tick."""
        candle = candle_builder.process_tick(tick)
        if candle:
            # A candle just closed — run strategy
            strategy.on_candle(candle)
            signal = strategy.generate_signal()
            if signal:
                order_manager.process_signal(signal)

    broker.register_tick_callback(on_tick)

    logger.info("Trading loop started | Strategy: {}", settings.strategy)

    while True:
        now = datetime.now()

        if not market.is_market_open(now):
            wait = market.minutes_to_open(now)
            if wait and wait > 0:
                logger.info("Market closed. Next open in {} min", int(wait))
            time.sleep(60)
            continue

        # Market is open — ensure ticks are flowing
        if not broker.is_connected():
            logger.warning("Broker disconnected — reconnecting...")
            try:
                broker.connect()
            except Exception as exc:
                logger.error("Reconnect failed: {}", exc)
                time.sleep(10)

        # Periodic reconciliation every 5 minutes
        if now.second < 5 and now.minute % 5 == 0:
            try:
                order_manager.sync_with_broker()
                risk_manager.update_daily_pnl()
            except Exception as exc:
                logger.error("Reconciliation error: {}", exc)

        # Square off check
        if now.time() >= settings.squareoff:
            logger.info("Square-off time reached — closing all positions")
            try:
                order_manager.square_off_all()
            except Exception as exc:
                logger.error("Square-off failed: {}", exc)
            # Wait until market close to restart loop
            time.sleep(900)

        time.sleep(0.5)


@click.command()
@click.option("--paper", is_flag=True, help="Force paper trading mode")
@click.option("--symbol", default="NIFTY", show_default=True, help="Trading symbol")
@click.option("--no-dashboard", is_flag=True, help="Skip the web dashboard")
@click.option("--log-level", default=None, help="Override log level")
def main(paper: bool, symbol: str, no_dashboard: bool, log_level: str | None) -> None:
    """Algorithmic Trading Bot — starts the live or paper trading session."""

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
            "[red bold]WARNING: LIVE TRADING MODE — real money will be used![/red bold]"
        )
        if not click.confirm("Are you sure you want to continue?"):
            sys.exit(0)

    # ── Initialise components ─────────────────────────────────────────
    logger.info("Initialising components...")

    from brokers import get_broker
    from strategies import get_strategy
    from risk.risk_manager import RiskManager
    from orders.order_manager import OrderManager
    from db.trade_logger import TradeLogger

    # ── Telegram notifier (disabled — uncomment when ready) ───────────────────
    # Steps to re-enable:
    #   1. Uncomment the two lines below.
    #   2. Remove the _NullNotifier block below them.
    #   3. Uncomment telegram_bot_token / telegram_chat_id in config.py.
    #   4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
    # from notifications.telegram_bot import TelegramNotifier
    # notifier = TelegramNotifier(settings=s)
    # ─────────────────────────────────────────────────────────────────────────

    class _NullNotifier:
        """Stub notifier — logs to console until Telegram is activated."""
        def send(self, message: str) -> None:
            logger.info("[Notifier] {}", message)

    notifier = _NullNotifier()

    broker = get_broker(s)
    strategy = get_strategy(s.strategy, symbol=symbol, settings=s)
    risk_manager = RiskManager(settings=s)
    trade_logger = TradeLogger()
    order_manager = OrderManager(
        broker=broker,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
        notifier=notifier,
        settings=s,
    )

    # ── Connect broker ────────────────────────────────────────────────
    logger.info("Connecting to {} broker...", s.broker)
    try:
        broker.connect()
    except Exception as exc:
        logger.critical("Broker connection failed: {}", exc)
        console.print(f"[red]Broker connection failed: {exc}[/red]")
        sys.exit(1)

    broker.subscribe_ticks([symbol])
    logger.info("Subscribed to live ticks for {}", symbol)

    # ── Dashboard (background thread) ─────────────────────────────────
    if not no_dashboard:
        dash_thread = threading.Thread(target=_run_dashboard, daemon=True)
        dash_thread.start()
        logger.info(
            "Dashboard running at http://{}:{}", s.dashboard_host, s.dashboard_port
        )

    # ── Graceful shutdown ─────────────────────────────────────────────
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received — squaring off and disconnecting...")
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

    # ── Start trading loop ────────────────────────────────────────────
    try:
        _trading_loop(broker, strategy, order_manager, risk_manager)
    except Exception as exc:
        logger.critical("Fatal error in trading loop: {}", exc)
        # ── Telegram crash alert (notifier is a stub until Telegram is enabled) ─
        try:
            notifier.send(f"BOT CRASHED: {exc}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
