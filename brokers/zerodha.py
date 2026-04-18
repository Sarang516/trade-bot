"""
brokers/zerodha.py — Zerodha Kite Connect broker adapter.

Implements every method in BaseBroker using the official kiteconnect SDK.
Install: pip install kiteconnect

Documentation: https://kite.trade/docs/connect/v3/
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from brokers.base_broker import (
    AccountMargins,
    BaseBroker,
    BrokerConnectionError,
    BrokerDataError,
    BrokerOrderError,
    Exchange,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Quote,
    SessionExpiredError,
)


# ── Mapping helpers ───────────────────────────────────────────────────────────

_EXCHANGE_MAP = {
    Exchange.NSE: "NSE",
    Exchange.BSE: "BSE",
    Exchange.NFO: "NFO",
    Exchange.BFO: "BFO",
    Exchange.MCX: "MCX",
}

_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}

_PRODUCT_MAP = {
    ProductType.MIS: "MIS",
    ProductType.NRML: "NRML",
    ProductType.CNC: "CNC",
}

_STATUS_MAP = {
    "COMPLETE": OrderStatus.COMPLETE,
    "OPEN": OrderStatus.OPEN,
    "PENDING": OrderStatus.PENDING,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.OPEN,
}


class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite Connect broker adapter.

    Handles:
    - REST API for orders / quotes / history
    - WebSocket KiteTicker for live tick streaming
    - Auto-reconnect with exponential backoff
    """

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._kite = None           # KiteConnect instance
        self._ticker = None         # KiteTicker instance
        self._subscribed_tokens: list[int] = []
        self._token_symbol_map: dict[int, str] = {}
        self._ws_lock = threading.Lock()
        self._instruments_cache: dict[str, int] = {}  # "EXCHANGE:SYMBOL" -> token
        self._last_ping: float = 0.0                  # monotonic time of last API ping

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate with Zerodha using api_key + access_token."""
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError as exc:
            raise BrokerConnectionError(
                "kiteconnect not installed. Run: pip install kiteconnect"
            ) from exc

        try:
            self._kite = KiteConnect(api_key=self._settings.zerodha_api_key)
            self._kite.set_access_token(self._settings.zerodha_access_token)

            # Validate by fetching profile
            profile = self._kite.profile()
            self._connected = True
            logger.info(
                "Zerodha connected | User: {} ({})",
                profile.get("user_name"),
                profile.get("user_id"),
            )
        except Exception as exc:
            self._connected = False
            if "TokenException" in type(exc).__name__:
                raise SessionExpiredError(
                    "Access token expired. Generate a new one at kite.trade"
                ) from exc
            raise BrokerConnectionError(f"Zerodha login failed: {exc}") from exc

    def disconnect(self) -> None:
        """Stop WebSocket and invalidate the session."""
        if self._ticker:
            try:
                self._ticker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._ticker = None
        self._connected = False
        logger.info("Zerodha disconnected")

    def is_connected(self) -> bool:
        if not self._connected or self._kite is None:
            return False
        # Rate-limit the liveness check to once per 60 s — calling profile() on every
        # 0.5 s trading-loop tick would exhaust Zerodha's API quota within minutes.
        now = time.monotonic()
        if now - self._last_ping < 60.0:
            return self._connected
        try:
            self._kite.profile()
            self._last_ping = now
            return True
        except Exception:  # noqa: BLE001
            self._connected = False
            return False

    # ── Market Data ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        ex = _EXCHANGE_MAP[exchange]
        key = f"{ex}:{symbol}"
        try:
            data = self._kite.quote([key])[key]
            return Quote(
                symbol=symbol,
                exchange=exchange,
                ltp=data["last_price"],
                bid=data.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                ask=data.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                open=data.get("ohlc", {}).get("open", 0),
                high=data.get("ohlc", {}).get("high", 0),
                low=data.get("ohlc", {}).get("low", 0),
                close=data.get("ohlc", {}).get("close", 0),
                volume=data.get("volume", 0),
                oi=data.get("oi", 0),
                timestamp=datetime.now(),
            )
        except Exception as exc:
            raise BrokerDataError(f"get_quote failed for {symbol}: {exc}") from exc

    def get_ltp(self, symbol: str, exchange: Exchange = Exchange.NSE) -> float:
        ex = _EXCHANGE_MAP[exchange]
        key = f"{ex}:{symbol}"
        try:
            return self._kite.ltp([key])[key]["last_price"]
        except Exception as exc:
            raise BrokerDataError(f"get_ltp failed for {symbol}: {exc}") from exc

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_historical_data(
        self,
        symbol: str,
        exchange: Exchange,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        token = self.get_instrument_token(symbol, exchange)
        try:
            records = self._kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
            )
            df = pd.DataFrame(records)
            if df.empty:
                return df
            df.rename(columns={"date": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            raise BrokerDataError(
                f"get_historical_data failed for {symbol}: {exc}"
            ) from exc

    # ── Order Management ──────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_order(self, order: Order) -> str:
        params = {
            "tradingsymbol": order.symbol,
            "exchange": _EXCHANGE_MAP[order.exchange],
            "transaction_type": order.side.value,
            "order_type": _ORDER_TYPE_MAP[order.order_type],
            "product": _PRODUCT_MAP[order.product],
            "quantity": order.quantity,
            "tag": order.tag,
        }
        if order.order_type == OrderType.LIMIT:
            params["price"] = order.price
        if order.order_type in (OrderType.SL, OrderType.SL_M):
            params["trigger_price"] = order.trigger_price
        if order.order_type == OrderType.SL:
            params["price"] = order.price

        try:
            order_id = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR, **params
            )
            logger.info(
                "Order placed | {} {} {} @ {} | ID: {}",
                order.side.value,
                order.symbol,
                order.quantity,
                order.price or "MARKET",
                order_id,
            )
            return str(order_id)
        except Exception as exc:
            raise BrokerOrderError(
                f"Order placement failed for {order.symbol}: {exc}", order
            ) from exc

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None,
    ) -> bool:
        params: dict = {"order_id": order_id, "variety": self._kite.VARIETY_REGULAR}
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if quantity is not None:
            params["quantity"] = quantity
        try:
            self._kite.modify_order(**params)
            logger.info("Order modified | ID: {}", order_id)
            return True
        except Exception as exc:
            logger.error("modify_order failed | ID: {} | {}", order_id, exc)
            return False

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(
                variety=self._kite.VARIETY_REGULAR, order_id=order_id
            )
            logger.info("Order cancelled | ID: {}", order_id)
            return True
        except Exception as exc:
            logger.error("cancel_order failed | ID: {} | {}", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> Order:
        try:
            history = self._kite.order_history(order_id=order_id)
            latest = history[-1]
            return self._parse_order(latest)
        except Exception as exc:
            raise BrokerDataError(f"get_order_status failed: {exc}") from exc

    def get_all_orders(self) -> list[Order]:
        try:
            raw_orders = self._kite.orders()
            return [self._parse_order(o) for o in raw_orders]
        except Exception as exc:
            raise BrokerDataError(f"get_all_orders failed: {exc}") from exc

    # ── Positions & Funds ─────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        try:
            raw = self._kite.positions()
            positions = []
            for p in raw.get("day", []):
                if p["quantity"] == 0:
                    continue
                positions.append(
                    Position(
                        symbol=p["tradingsymbol"],
                        exchange=Exchange(p["exchange"]),
                        product=ProductType(p["product"]),
                        quantity=p["quantity"],
                        average_price=p["average_price"],
                        ltp=p.get("last_price", 0),
                        pnl=p.get("pnl", 0),
                        day_pnl=p.get("day_pnl", 0),
                    )
                )
            return positions
        except Exception as exc:
            raise BrokerDataError(f"get_positions failed: {exc}") from exc

    def get_margins(self) -> AccountMargins:
        try:
            margins = self._kite.margins(segment="equity")
            return AccountMargins(
                available_cash=margins.get("available", {}).get("cash", 0),
                used_margin=margins.get("utilised", {}).get("exposure", 0),
                total_balance=margins.get("net", 0),
            )
        except Exception as exc:
            raise BrokerDataError(f"get_margins failed: {exc}") from exc

    # ── WebSocket Streaming ───────────────────────────────────────────

    def subscribe_ticks(
        self, symbols: list[str], exchange: Exchange = Exchange.NSE
    ) -> None:
        try:
            from kiteconnect import KiteTicker  # type: ignore
        except ImportError as exc:
            raise BrokerConnectionError("kiteconnect not installed") from exc

        tokens = [self.get_instrument_token(s, exchange) for s in symbols]
        for tok, sym in zip(tokens, symbols):
            self._token_symbol_map[tok] = sym
        self._subscribed_tokens.extend(tokens)

        def _on_ticks(_ws, ticks):
            for t in ticks:
                sym = self._token_symbol_map.get(t["instrument_token"], "UNKNOWN")
                tick = {
                    "symbol": sym,
                    "ltp": t.get("last_price", 0),
                    "volume": t.get("volume", 0),
                    "oi": t.get("oi", 0),
                    "timestamp": datetime.now(),
                }
                self._dispatch_tick(tick)

        def _on_connect(ws, _response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info("WebSocket connected | Subscribed to {} tokens", len(tokens))

        def _on_close(_ws, code, reason):
            logger.warning("WebSocket closed: {} {}", code, reason)

        def _on_error(_ws, code, reason):
            logger.error("WebSocket error: {} {}", code, reason)

        self._ticker = KiteTicker(
            self._settings.zerodha_api_key,
            self._settings.zerodha_access_token,
        )
        self._ticker.on_ticks = _on_ticks
        self._ticker.on_connect = _on_connect
        self._ticker.on_close = _on_close
        self._ticker.on_error = _on_error
        self._ticker.connect(threaded=True)

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        if self._ticker:
            tokens = [
                tok
                for tok, sym in self._token_symbol_map.items()
                if sym in symbols
            ]
            self._ticker.unsubscribe(tokens)

    # ── Instrument Lookup ─────────────────────────────────────────────

    def get_instrument_token(self, symbol: str, exchange: Exchange) -> int:
        cache_key = f"{exchange.value}:{symbol}"
        if cache_key in self._instruments_cache:
            return self._instruments_cache[cache_key]

        # Cache miss — download full instrument list once per exchange and
        # populate cache for all symbols so subsequent calls are instant.
        ex = _EXCHANGE_MAP[exchange]
        try:
            instruments = self._kite.instruments(exchange=ex)
            for inst in instruments:
                key = f"{exchange.value}:{inst['tradingsymbol']}"
                self._instruments_cache[key] = inst["instrument_token"]
            if cache_key not in self._instruments_cache:
                raise BrokerDataError(f"Symbol '{symbol}' not found on {ex}")
            return self._instruments_cache[cache_key]
        except BrokerDataError:
            raise
        except Exception as exc:
            raise BrokerDataError(
                f"get_instrument_token failed for {symbol}: {exc}"
            ) from exc

    def search_instruments(self, query: str, exchange: Exchange) -> list[dict]:
        ex = _EXCHANGE_MAP[exchange]
        try:
            instruments = self._kite.instruments(exchange=ex)
            query_lower = query.lower()
            return [
                i
                for i in instruments
                if query_lower in i["tradingsymbol"].lower()
                or query_lower in i.get("name", "").lower()
            ]
        except Exception as exc:
            raise BrokerDataError(f"search_instruments failed: {exc}") from exc

    # ── Private helpers ───────────────────────────────────────────────

    def _parse_order(self, raw: dict) -> Order:
        return Order(
            order_id=str(raw.get("order_id", "")),
            symbol=raw.get("tradingsymbol", ""),
            exchange=Exchange(raw.get("exchange", "NSE")),
            side=OrderSide(raw.get("transaction_type", "BUY")),
            order_type=OrderType(raw.get("order_type", "MARKET")),
            product=ProductType(raw.get("product", "MIS")),
            quantity=raw.get("quantity", 0),
            price=raw.get("price", 0),
            trigger_price=raw.get("trigger_price", 0),
            status=_STATUS_MAP.get(raw.get("status", ""), OrderStatus.PENDING),
            filled_price=raw.get("average_price", 0),
            filled_quantity=raw.get("filled_quantity", 0),
            rejection_reason=raw.get("status_message", ""),
        )
