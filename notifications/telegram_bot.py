"""
notifications/telegram_bot.py — Telegram alert integration.

CURRENTLY DISABLED — the full implementation is commented out below.
The active class (_NullNotifier equivalent via TelegramNotifier stub) just
logs messages to the console so the rest of the bot works without a token.

To activate Telegram alerts:
  1. Uncomment the "Real implementation" block below.
  2. Delete or comment out the stub class at the bottom.
  3. Uncomment telegram_bot_token / telegram_chat_id in config.py.
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.
  5. Uncomment the TelegramNotifier import in main.py.
"""

from loguru import logger


# ── Real Telegram implementation (disabled — uncomment when ready) ────────────
#
# from telegram import Bot
# from telegram.error import TelegramError
#
# class TelegramNotifier:
#     """Sends alerts to a Telegram chat via the Bot API."""
#
#     def __init__(self, settings) -> None:
#         self._token = settings.telegram_bot_token
#         self._chat_id = settings.telegram_chat_id
#         self._bot = Bot(token=self._token)
#         logger.info("Telegram notifier ready (chat_id={})", self._chat_id)
#
#     def send(self, message: str) -> None:
#         """Send a plain-text message to the configured chat."""
#         import asyncio
#         try:
#             asyncio.run(
#                 self._bot.send_message(chat_id=self._chat_id, text=message)
#             )
#             logger.debug("Telegram alert sent: {}", message)
#         except TelegramError as exc:
#             logger.warning("Telegram alert failed: {}", exc)
#
# ─────────────────────────────────────────────────────────────────────────────


class TelegramNotifier:
    """
    Stub notifier — logs to console instead of sending Telegram messages.
    Replace this class with the real implementation above when ready.
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    def send(self, message: str) -> None:
        logger.info("[Telegram disabled] {}", message)
