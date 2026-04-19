"""
notifications/telegram_bot.py — Telegram alert + command bot.

Activation
----------
1. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.
2. Uncomment telegram_bot_token / telegram_chat_id in config.py.
3. In main.py the bot is already wired — it will start automatically.

Alert methods (called by OrderManager / main loop)
---------------------------------------------------
    notifier.send(plain_text)           # raw send — already called by OrderManager
    notifier.notify_trade_entry(...)
    notifier.notify_trade_exit(...)
    notifier.notify_daily_summary(...)
    notifier.notify_error(...)
    notifier.notify_daily_loss_cap(...)

Bot commands
------------
    /start      welcome + help
    /status     open positions
    /today      today's trade summary
    /pause      pause new trade entries
    /resume     resume trading
    /squareoff  emergency close all positions
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from loguru import logger


class TelegramNotifier:
    """
    Full async Telegram bot.

    Runs python-telegram-bot's Application in a dedicated background thread
    with its own event loop so it doesn't block the trading loop.

    The send() / notify_*() methods are synchronous — safe to call from
    any thread via asyncio.run_coroutine_threadsafe().
    """

    def __init__(self, settings) -> None:
        self._token   = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app     = None
        self._ctx: dict = {}   # injected by set_context()

    def set_context(self, **kwargs) -> None:
        """Inject order_manager, risk_manager, trade_logger, bot_status."""
        self._ctx.update(kwargs)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the bot in a daemon thread. Non-blocking."""
        t = threading.Thread(target=self._run_forever, daemon=True, name="TelegramBot")
        t.start()
        logger.info("Telegram bot starting in background thread")

    def _run_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_polling())
        except Exception as exc:
            logger.error("Telegram bot crashed: {}", exc)

    async def _start_polling(self) -> None:
        from telegram.ext import Application, CommandHandler
        self._app = Application.builder().token(self._token).build()

        from telegram.ext import CommandHandler
        cmds = [
            ("start",     self._cmd_start),
            ("status",    self._cmd_status),
            ("today",     self._cmd_today),
            ("pause",     self._cmd_pause),
            ("resume",    self._cmd_resume),
            ("squareoff", self._cmd_squareoff),
        ]
        for name, handler in cmds:
            self._app.add_handler(CommandHandler(name, handler))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot polling started")
        await asyncio.Event().wait()   # run forever

    # ── Send (sync, thread-safe) ──────────────────────────────────────────────

    def send(self, message: str) -> None:
        """Send a plain-text message. Safe to call from any thread."""
        if self._loop is None or self._app is None:
            logger.info("[Telegram not ready] {}", message)
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._app.bot.send_message(
                chat_id=self._chat_id, text=message, parse_mode="HTML"
            ),
            self._loop,
        )
        try:
            fut.result(timeout=8)
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)

    # ── Structured alert helpers ──────────────────────────────────────────────

    def notify_trade_entry(
        self,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        target: float,
        quantity: int,
        reason: str = "",
        is_paper: bool = True,
    ) -> None:
        mode = "📄 PAPER" if is_paper else "💰 LIVE"
        icon = "🟢 BUY" if direction == "LONG" else "🔴 SELL"
        msg = (
            f"{mode} | {icon} <b>{symbol}</b>\n"
            f"Entry  : ₹{price:.2f}\n"
            f"SL     : ₹{sl:.2f}  (risk ₹{abs(price-sl)*quantity:.0f})\n"
            f"Target : ₹{target:.2f}\n"
            f"Qty    : {quantity}\n"
            f"Reason : {reason or '—'}"
        )
        self.send(msg)

    def notify_trade_exit(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        pnl: float,
        reason: str,
        is_paper: bool = True,
    ) -> None:
        mode  = "📄" if is_paper else "💰"
        icon  = "✅" if pnl >= 0 else "❌"
        sign  = "+" if pnl >= 0 else ""
        msg = (
            f"{mode} {icon} EXIT <b>{symbol}</b>\n"
            f"Price  : ₹{exit_price:.2f}\n"
            f"P&amp;L   : <b>{sign}₹{pnl:.2f}</b>\n"
            f"Reason : {reason}"
        )
        self.send(msg)

    def notify_daily_loss_cap(self, realised_pnl: float) -> None:
        self.send(
            f"🛑 <b>DAILY LOSS CAP HIT</b>\n"
            f"Realised P&amp;L today: ₹{realised_pnl:.2f}\n"
            f"No more trades for today."
        )

    def notify_daily_summary(self, summary: dict) -> None:
        pnl  = summary.get("net_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        icon = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{icon} <b>End-of-Day Summary</b>\n"
            f"Date    : {summary.get('date', '—')}\n"
            f"Trades  : {summary.get('total_trades', 0)}  "
            f"({summary.get('winning_trades', 0)}W / {summary.get('losing_trades', 0)}L)\n"
            f"Win Rate: {summary.get('win_rate', 0)}%\n"
            f"Net P&amp;L: <b>{sign}₹{pnl:,.2f}</b>"
        )
        self.send(msg)

    def notify_error(self, error: str) -> None:
        self.send(f"⚠️ <b>BOT ERROR</b>\n{error}")

    # ── Command handlers (async) ──────────────────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        await update.message.reply_text(
            "📈 <b>Trading Bot</b> is active!\n\n"
            "/status   — open positions\n"
            "/today    — today's summary\n"
            "/pause    — pause new entries\n"
            "/resume   — resume trading\n"
            "/squareoff — close ALL positions 🚨",
            parse_mode="HTML",
        )

    async def _cmd_status(self, update, context) -> None:
        risk = self._ctx.get("risk_manager")
        if risk is None:
            await update.message.reply_text("Bot is not running.")
            return

        positions = risk.get_open_positions()
        bot_st    = self._ctx.get("bot_status", {}).get("status", "UNKNOWN")

        if not positions:
            msg = f"Bot: <b>{bot_st}</b>\nNo open positions."
        else:
            lines = [f"Bot: <b>{bot_st}</b>\n<b>Open Positions:</b>"]
            for p in positions:
                icon = "🟢" if p.direction == "LONG" else "🔴"
                lines.append(
                    f"{icon} {p.direction} <b>{p.symbol}</b> "
                    f"@ ₹{p.entry_price:.0f} | SL ₹{p.current_sl:.0f} | T ₹{p.target_price:.0f}"
                )
            msg = "\n".join(lines)
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_today(self, update, context) -> None:
        tl = self._ctx.get("trade_logger")
        if tl is None:
            await update.message.reply_text("No trade data available.")
            return
        s    = tl.get_today_summary()
        pnl  = s.get("net_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        icon = "📈" if pnl >= 0 else "📉"
        await update.message.reply_text(
            f"{icon} <b>Today ({s.get('date', '—')})</b>\n"
            f"Trades  : {s.get('total_trades', 0)}\n"
            f"W / L   : {s.get('winning_trades', 0)} / {s.get('losing_trades', 0)}\n"
            f"Win Rate: {s.get('win_rate', 0)}%\n"
            f"Net P&amp;L: <b>{sign}₹{pnl:,.2f}</b>",
            parse_mode="HTML",
        )

    async def _cmd_pause(self, update, context) -> None:
        bs = self._ctx.get("bot_status")
        if bs is not None:
            bs["status"] = "PAUSED"
            tl = self._ctx.get("trade_logger")
            if tl:
                tl.log_bot_event("INFO", "Trading paused via Telegram /pause")
        await update.message.reply_text("⏸ <b>Trading paused.</b> No new entries until /resume.", parse_mode="HTML")

    async def _cmd_resume(self, update, context) -> None:
        bs = self._ctx.get("bot_status")
        if bs is not None:
            bs["status"] = "RUNNING"
            tl = self._ctx.get("trade_logger")
            if tl:
                tl.log_bot_event("INFO", "Trading resumed via Telegram /resume")
        await update.message.reply_text("▶ <b>Trading resumed.</b>", parse_mode="HTML")

    async def _cmd_squareoff(self, update, context) -> None:
        om = self._ctx.get("order_manager")
        if om is None:
            await update.message.reply_text("Order manager not available — bot may not be running.")
            return
        await update.message.reply_text("🚨 Squaring off all open positions…")
        try:
            om.square_off_all()
            tl = self._ctx.get("trade_logger")
            if tl:
                tl.log_bot_event("WARNING", "Emergency square-off via Telegram /squareoff")
            await update.message.reply_text("✅ All positions closed.")
        except Exception as exc:
            await update.message.reply_text(f"❌ Error: {exc}")
