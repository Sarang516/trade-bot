"""notifications/telegram_bot.py — Built in Phase 9."""
from loguru import logger
class TelegramNotifier:
    def __init__(self, settings): self._settings = settings
    def send(self, message: str):
        logger.info("[Telegram stub] {}", message)
