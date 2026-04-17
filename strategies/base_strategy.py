"""
strategies/base_strategy.py — Abstract base class for all trading strategies.

Every strategy (VWAP+Volume, Supertrend, RSI, etc.) inherits from BaseStrategy
and implements the abstract methods.  The bot's main loop only ever calls
this interface — strategies are hot-swappable.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd


# ── Signal Enumerations ───────────────────────────────────────────────────────

class SignalDirection(str, Enum):
    LONG = "LONG"       # Go long (buy stock / buy call option)
    SHORT = "SHORT"     # Go short (sell stock / buy put option)
    FLAT = "FLAT"       # No trade / close existing position
    HOLD = "HOLD"       # Already in trade, do nothing


class ExitReason(str, Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TRAILING_SL_HIT = "TRAILING_SL_HIT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    TIME_SQUAREOFF = "TIME_SQUAREOFF"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MANUAL = "MANUAL"


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Candle:
    """A single OHLCV candle."""

    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    interval_minutes: int = 5

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        return {
            "datetime": self.datetime.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Signal:
    """
    A trading signal emitted by a strategy.

    The order manager acts on this signal to place / modify / close trades.
    """

    direction: SignalDirection
    symbol: str
    exchange: str = "NSE"

    # Price levels
    entry_price: float = 0.0
    stop_loss: float = 0.0      # Absolute price (not %)
    target: float = 0.0         # Absolute price

    # For F&O: which option to trade
    option_type: Optional[str] = None       # "CE" | "PE" | None
    strike_price: Optional[float] = None
    expiry: Optional[str] = None            # "YYYY-MM-DD"

    # Metadata
    confidence: int = 0         # 0–100 score
    reason: str = ""            # Human-readable explanation
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""

    # Exit context (only set when direction == FLAT)
    exit_reason: Optional[ExitReason] = None

    def risk_reward_ratio(self) -> float:
        """Return the R:R of this signal (0 if undefined)."""
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.target <= 0:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def is_valid(self) -> bool:
        """Basic sanity check on the signal."""
        if self.direction == SignalDirection.FLAT:
            return True
        if self.direction == SignalDirection.HOLD:
            return True
        if self.entry_price <= 0:
            return False
        if self.stop_loss <= 0 or self.target <= 0:
            return False
        # SL should be on the correct side of entry
        if self.direction == SignalDirection.LONG:
            return self.stop_loss < self.entry_price < self.target
        if self.direction == SignalDirection.SHORT:
            return self.target < self.entry_price < self.stop_loss
        return False

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "option_type": self.option_type,
            "strike_price": self.strike_price,
            "confidence": self.confidence,
            "reason": self.reason,
            "rr_ratio": self.risk_reward_ratio(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StrategyState:
    """Mutable runtime state carried across candles."""

    in_trade: bool = False
    trade_direction: Optional[SignalDirection] = None
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    current_sl: float = 0.0
    current_target: float = 0.0
    candles_in_trade: int = 0
    last_signal: Optional[Signal] = None
    indicators: dict[str, Any] = field(default_factory=dict)


# ── Abstract Strategy ─────────────────────────────────────────────────────────

class BaseStrategy(abc.ABC):
    """
    Abstract base for all trading strategies.

    Lifecycle per candle (called by the bot's main loop):
        1. on_tick(tick)      → called on every price tick (optional)
        2. on_candle(candle)  → called when a candle closes  ← main logic
        3. generate_signal()  → returns the current Signal
    """

    def __init__(self, symbol: str, settings) -> None:
        self.symbol = symbol
        self._settings = settings
        self.name: str = self.__class__.__name__
        self.state = StrategyState()
        self._candle_history: list[Candle] = []
        self._df: pd.DataFrame = pd.DataFrame()   # Indicator DataFrame
        self._warmup_candles: int = 50            # Min candles before trading

    # ── Abstract methods ──────────────────────────────────────────────

    @abc.abstractmethod
    def on_candle(self, candle: Candle) -> None:
        """
        Process a completed candle.  Update indicators and internal state.
        Called by the bot main loop every time a candle closes.
        """

    @abc.abstractmethod
    def generate_signal(self) -> Signal:
        """
        Evaluate current indicator state and return a Signal.
        Called immediately after on_candle().
        """

    @abc.abstractmethod
    def on_trade_entry(self, signal: Signal, filled_price: float) -> None:
        """
        Notify the strategy that a trade was entered.
        Use to update internal state (entry price, SL, etc.).
        """

    @abc.abstractmethod
    def on_trade_exit(self, exit_reason: ExitReason, exit_price: float) -> None:
        """
        Notify the strategy that an open trade was closed.
        Use to reset state and record trade outcome.
        """

    # ── Optional overrides ────────────────────────────────────────────

    def on_tick(self, tick: dict) -> None:
        """
        Called on every incoming price tick (high frequency).
        Only override if you need intra-candle logic (e.g., tick-based SL).
        Default: no-op.
        """

    def on_market_open(self) -> None:
        """Called once at 9:15 AM IST on each trading day."""

    def on_market_close(self) -> None:
        """Called once at 3:30 PM IST on each trading day."""

    # ── Concrete helpers (available to all strategies) ─────────────────

    def is_warmed_up(self) -> bool:
        """Return True once enough candles exist to compute indicators."""
        return len(self._candle_history) >= self._warmup_candles

    def add_candle(self, candle: Candle) -> None:
        """Append candle and rebuild the indicator DataFrame."""
        self._candle_history.append(candle)
        self._rebuild_df()

    def _rebuild_df(self) -> None:
        """Convert candle history to a pandas DataFrame."""
        if not self._candle_history:
            return
        self._df = pd.DataFrame(
            [c.to_dict() for c in self._candle_history]
        )
        self._df["datetime"] = pd.to_datetime(self._df["datetime"])
        self._df.set_index("datetime", inplace=True)

    def last_candle(self) -> Optional[Candle]:
        return self._candle_history[-1] if self._candle_history else None

    def prev_candle(self, n: int = 1) -> Optional[Candle]:
        """Return the n-th candle from the end (1 = second-to-last)."""
        idx = -(n + 1)
        if abs(idx) <= len(self._candle_history):
            return self._candle_history[idx]
        return None

    @property
    def df(self) -> pd.DataFrame:
        """Read-only view of the candle + indicator DataFrame."""
        return self._df.copy()

    def reset_state(self) -> None:
        """Reset trade state (call after a trade is closed)."""
        self.state = StrategyState()

    def describe(self) -> str:
        """Short human-readable description of the strategy."""
        return (
            f"{self.name} | "
            f"Symbol: {self.symbol} | "
            f"In trade: {self.state.in_trade} | "
            f"Candles: {len(self._candle_history)}"
        )

    def __repr__(self) -> str:
        return f"<{self.name} symbol={self.symbol} warmed_up={self.is_warmed_up()}>"


# ── Strategy Exceptions ───────────────────────────────────────────────────────

class StrategyError(Exception):
    """Base exception for strategy errors."""


class InsufficientDataError(StrategyError):
    """Raised when not enough candles exist to compute a signal."""


class InvalidSignalError(StrategyError):
    """Raised when generate_signal() produces a logically invalid Signal."""
