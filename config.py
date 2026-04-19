"""
config.py — Centralised configuration for the trading bot.

All settings are loaded from environment variables / .env file.
Import `settings` anywhere in the project:

    from config import settings
    print(settings.broker)
"""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    All bot configuration.  Values come from (in priority order):
    1. Actual environment variables
    2. .env file in the project root
    3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Broker ────────────────────────────────────────────────────────
    # To enable ICICI later: change Literal["zerodha"] → Literal["zerodha", "icici"]
    # and uncomment the ICICI credential fields below.
    broker: Literal["zerodha"] = Field(
        default="zerodha",
        description="Active broker: zerodha (icici support available — see comments below)",
    )

    # ── Zerodha Kite Connect ──────────────────────────────────────────
    zerodha_api_key: str = Field(default="DUMMY_ZERODHA_KEY")
    zerodha_api_secret: str = Field(default="DUMMY_ZERODHA_SECRET")
    zerodha_access_token: str = Field(default="DUMMY_ACCESS_TOKEN")

    # ── ICICI Breeze (disabled — uncomment all three lines when ready) ────────
    # To activate: also uncomment the ICICI block in brokers/__init__.py
    # and add ICICI credentials to your .env file.
    # icici_api_key: str = Field(default="DUMMY_ICICI_KEY")
    # icici_api_secret: str = Field(default="DUMMY_ICICI_SECRET")
    # icici_session_token: str = Field(default="DUMMY_ICICI_SESSION")

    # ── Telegram ──────────────────────────────────────────────────────────────
    # Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.
    # Leave as DUMMY values to disable — bot will log alerts to console instead.
    telegram_bot_token: str = Field(default="DUMMY_TELEGRAM_TOKEN")
    telegram_chat_id:   str = Field(default="000000000")

    # ── Trading Mode ──────────────────────────────────────────────────
    paper_trade: bool = Field(
        default=True,
        description="True → simulate orders, no real money at risk",
    )

    # ── Capital & Risk ────────────────────────────────────────────────
    trading_capital: float = Field(default=100_000.0, ge=1_000)
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    max_risk_per_trade_inr: float = Field(default=5_000.0, ge=100)
    max_daily_loss_inr: float = Field(default=10_000.0, ge=500)
    max_daily_profit_inr: float = Field(default=15_000.0, ge=500)
    max_open_positions: int = Field(default=3, ge=1, le=10)
    max_trades_per_day: int = Field(default=5, ge=1, le=20)

    # ── Strategy ──────────────────────────────────────────────────────
    strategy: str = Field(default="vwap_volume")
    candle_interval: Literal[1, 3, 5, 15, 30, 60] = Field(default=5)

    @field_validator("candle_interval", mode="before")
    @classmethod
    def _coerce_interval(cls, v):
        return int(v)
    trade_start_time: str = Field(default="09:20")
    trade_end_time: str = Field(default="14:30")
    squareoff_time: str = Field(default="15:15")

    # ── Dashboard ─────────────────────────────────────────────────────
    dashboard_host: str = Field(default="127.0.0.1")
    dashboard_port: int = Field(default=5000, ge=1024, le=65535)

    # ── Logging ───────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_dir: Path = Field(default=ROOT_DIR / "logs")
    log_retention_days: int = Field(default=30, ge=1)

    # ── Derived helpers ───────────────────────────────────────────────
    @property
    def trade_start(self) -> time:
        h, m = self.trade_start_time.split(":")
        return time(int(h), int(m))

    @property
    def trade_end(self) -> time:
        h, m = self.trade_end_time.split(":")
        return time(int(h), int(m))

    @property
    def squareoff(self) -> time:
        h, m = self.squareoff_time.split(":")
        return time(int(h), int(m))

    @property
    def is_paper(self) -> bool:
        return self.paper_trade

    @field_validator("log_dir", mode="before")
    @classmethod
    def ensure_log_dir(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def effective_risk_per_trade(self) -> float:
        """Return the smaller of % risk and hard INR cap."""
        pct_risk = self.trading_capital * self.risk_per_trade_pct / 100
        return min(pct_risk, self.max_risk_per_trade_inr)

    def describe(self) -> str:
        """Human-readable config summary for startup logs."""
        mode = "PAPER TRADING" if self.paper_trade else "*** LIVE TRADING ***"
        return (
            f"\n{'='*55}\n"
            f"  Trading Bot Configuration\n"
            f"{'='*55}\n"
            f"  Mode       : {mode}\n"
            f"  Broker     : {self.broker.upper()}\n"
            f"  Strategy   : {self.strategy}\n"
            f"  Capital    : ₹{self.trading_capital:,.0f}\n"
            f"  Max risk/trade: ₹{self.max_risk_per_trade_inr:,.0f}\n"
            f"  Daily loss cap: ₹{self.max_daily_loss_inr:,.0f}\n"
            f"  Trade hours: {self.trade_start_time} – {self.trade_end_time} IST\n"
            f"  Square-off : {self.squareoff_time} IST\n"
            f"  Log level  : {self.log_level}\n"
            f"{'='*55}\n"
        )


# Singleton — import this everywhere
settings = Settings()
