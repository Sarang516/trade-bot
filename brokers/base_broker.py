"""
brokers/base_broker.py — Abstract base class that every broker must implement.

All broker integrations (Zerodha, ICICI, etc.) inherit from BaseBroker
and implement every abstract method.  The rest of the bot only ever
talks to this interface — swapping brokers requires zero changes elsewhere.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import pandas as pd


# ── Enumerations ──────────────────────────────────────────────────────────────

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # Stop-loss with limit price
    SL_M = "SL-M"      # Stop-loss market


class ProductType(str, Enum):
    MIS = "MIS"         # Intraday (auto squares-off at EOD)
    NRML = "NRML"       # Overnight / F&O carry
    CNC = "CNC"         # Delivery (equity only)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"       # Placed, waiting broker ACK
    OPEN = "OPEN"             # Live on exchange
    COMPLETE = "COMPLETE"     # Fully filled
    CANCELLED = "CANCELLED"   # Cancelled by user / system
    REJECTED = "REJECTED"     # Rejected by broker / exchange


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"   # NSE F&O
    BFO = "BFO"   # BSE F&O
    MCX = "MCX"


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Order:
    """Represents a single order (placed or to be placed)."""

    order_id: str = ""
    symbol: str = ""
    exchange: Exchange = Exchange.NSE
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.MIS
    quantity: int = 0
    price: float = 0.0          # Limit price (0 for MARKET)
    trigger_price: float = 0.0  # For SL / SL-M orders
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    rejection_reason: str = ""
    tag: str = "trading_bot"    # Identifier tag visible in broker console

    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETE

    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "side": self.side.value,
            "type": self.order_type.value,
            "product": self.product.value,
            "quantity": self.quantity,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "status": self.status.value,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Quote:
    """Live market quote for a symbol."""

    symbol: str
    exchange: Exchange
    ltp: float                  # Last traded price
    bid: float = 0.0
    ask: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0          # Previous day close
    volume: int = 0
    oi: int = 0                 # Open Interest (F&O)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """An open position held in the account."""

    symbol: str
    exchange: Exchange
    product: ProductType
    quantity: int               # +ve = long, -ve = short
    average_price: float
    ltp: float = 0.0
    pnl: float = 0.0
    day_pnl: float = 0.0


@dataclass
class AccountMargins:
    """Available funds / margins."""

    available_cash: float
    used_margin: float
    total_balance: float
    currency: str = "INR"


# ── Abstract Broker ───────────────────────────────────────────────────────────

class BaseBroker(abc.ABC):
    """
    Abstract interface every broker adapter must implement.

    Usage:
        broker = ZerodhaBroker(settings)
        broker.connect()
        quote = broker.get_quote("RELIANCE", Exchange.NSE)
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._connected: bool = False
        self._tick_callbacks: list[Callable[[dict], None]] = []

    # ── Connection ────────────────────────────────────────────────────

    @abc.abstractmethod
    def connect(self) -> None:
        """Authenticate and establish session with the broker."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Clean up session / WebSocket connections."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if session is active and valid."""

    # ── Market Data ───────────────────────────────────────────────────

    @abc.abstractmethod
    def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Fetch a full market quote for the given symbol."""

    @abc.abstractmethod
    def get_ltp(self, symbol: str, exchange: Exchange = Exchange.NSE) -> float:
        """Return only the last traded price (faster than full quote)."""

    @abc.abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        exchange: Exchange,
        from_date: datetime,
        to_date: datetime,
        interval: str,          # "minute", "3minute", "5minute", "day" etc.
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for backtesting / indicator warmup.

        Returns a DataFrame with columns:
            datetime, open, high, low, close, volume
        Sorted ascending by datetime.
        """

    # ── Order Management ──────────────────────────────────────────────

    @abc.abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Submit an order to the broker.

        Returns:
            order_id (str) assigned by the broker.
        Raises:
            BrokerOrderError on rejection.
        """

    @abc.abstractmethod
    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None,
    ) -> bool:
        """Modify a pending/open order.  Returns True on success."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending/open order.  Returns True on success."""

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Fetch current status of an order from the broker."""

    @abc.abstractmethod
    def get_all_orders(self) -> list[Order]:
        """Return all orders placed in the current session."""

    # ── Positions & Funds ─────────────────────────────────────────────

    @abc.abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions."""

    @abc.abstractmethod
    def get_margins(self) -> AccountMargins:
        """Return available funds and margin usage."""

    # ── WebSocket Streaming ───────────────────────────────────────────

    @abc.abstractmethod
    def subscribe_ticks(self, symbols: list[str], exchange: Exchange = Exchange.NSE) -> None:
        """Subscribe to live tick stream for a list of symbols."""

    @abc.abstractmethod
    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe from live tick stream."""

    def register_tick_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Register a function to be called on every incoming tick.

        Tick dict keys: symbol, ltp, volume, oi, timestamp
        """
        self._tick_callbacks.append(callback)

    def _dispatch_tick(self, tick: dict) -> None:
        """Internal: call all registered tick callbacks."""
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as exc:  # noqa: BLE001
                # Never let a callback crash the feed
                import logging
                logging.getLogger(__name__).error(
                    "Tick callback %s raised: %s", cb.__name__, exc
                )

    # ── Instrument Lookup ─────────────────────────────────────────────

    @abc.abstractmethod
    def get_instrument_token(self, symbol: str, exchange: Exchange) -> int:
        """Return the broker's internal numeric token for the symbol."""

    @abc.abstractmethod
    def search_instruments(self, query: str, exchange: Exchange) -> list[dict]:
        """Search for instruments matching a partial name / symbol."""

    # ── Convenience helpers (concrete, no override needed) ────────────

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        exchange: Exchange = Exchange.NSE,
        product: ProductType = ProductType.MIS,
    ) -> str:
        order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=OrderType.MARKET,
            product=product,
            quantity=quantity,
        )
        return self.place_order(order)

    def place_sl_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trigger_price: float,
        limit_price: float,
        exchange: Exchange = Exchange.NSE,
        product: ProductType = ProductType.MIS,
    ) -> str:
        order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=OrderType.SL,
            product=product,
            quantity=quantity,
            price=limit_price,
            trigger_price=trigger_price,
        )
        return self.place_order(order)

    def square_off_position(self, position: Position) -> str:
        """Close an open position at market price."""
        close_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        return self.place_market_order(
            symbol=position.symbol,
            side=close_side,
            quantity=abs(position.quantity),
            exchange=position.exchange,
            product=position.product,
        )

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"<{self.__class__.__name__} [{status}]>"


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class BrokerError(Exception):
    """Base exception for all broker errors."""


class BrokerConnectionError(BrokerError):
    """Raised when connection / authentication fails."""


class BrokerOrderError(BrokerError):
    """Raised when an order is rejected or fails."""

    def __init__(self, message: str, order: Optional[Order] = None) -> None:
        super().__init__(message)
        self.order = order


class BrokerDataError(BrokerError):
    """Raised when market data fetch fails."""


class SessionExpiredError(BrokerConnectionError):
    """Raised when the broker session / access token has expired."""
