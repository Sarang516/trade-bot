"""
brokers/icici.py — ICICI Direct Breeze API broker adapter.

STATUS: DISABLED — fully implemented and ready, but not wired in.
        Activate by following the steps in brokers/__init__.py comments.

Implements every method in BaseBroker using the breeze-connect SDK.
Install: pip install breeze-connect

Documentation: https://api.icicidirect.com/apiuser/
"""

from __future__ import annotations

import threading
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
}

_ORDER_TYPE_MAP = {
    OrderType.MARKET: "Market",
    OrderType.LIMIT: "Limit",
    OrderType.SL: "StopLoss",
    OrderType.SL_M: "StopLossMarket",
}

_PRODUCT_MAP = {
    ProductType.MIS: "Intraday",
    ProductType.NRML: "Margin",
    ProductType.CNC: "Cash",
}

_STATUS_MAP = {
    "Executed": OrderStatus.COMPLETE,
    "Ordered": OrderStatus.OPEN,
    "Pending": OrderStatus.PENDING,
    "Cancelled": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
    "Modified": OrderStatus.OPEN,
}


class ICICIBroker(BaseBroker):
    """
    ICICI Direct Breeze API broker adapter.

    Implements the same interface as ZerodhaBroker so the rest of the
    codebase doesn't need to change when switching brokers.
    """

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._breeze = None
        self._subscribed_symbols: list[str] = []

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from breeze_connect import BreezeConnect  # type: ignore
        except ImportError as exc:
            raise BrokerConnectionError(
                "breeze-connect not installed. Run: pip install breeze-connect"
            ) from exc

        try:
            self._breeze = BreezeConnect(api_key=self._settings.icici_api_key)
            self._breeze.generate_session(
                api_secret=self._settings.icici_api_secret,
                session_token=self._settings.icici_session_token,
            )
            self._connected = True
            logger.info("ICICI Breeze connected")
        except Exception as exc:
            self._connected = False
            if "SessionError" in str(exc) or "Token" in str(exc):
                raise SessionExpiredError(
                    "ICICI session token expired. Generate a new one."
                ) from exc
            raise BrokerConnectionError(f"ICICI Breeze login failed: {exc}") from exc

    def disconnect(self) -> None:
        if self._breeze:
            try:
                self._breeze.ws_disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
        logger.info("ICICI Breeze disconnected")

    def is_connected(self) -> bool:
        if not self._connected or self._breeze is None:
            return False
        try:
            self._breeze.get_customer_details()
            return True
        except Exception:  # noqa: BLE001
            self._connected = False
            return False

    # ── Market Data ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        ex = _EXCHANGE_MAP.get(exchange, "NSE")
        try:
            resp = self._breeze.get_quotes(
                stock_code=symbol,
                exchange_code=ex,
                product_type="Cash",
                right="others",
                strike_price="0",
            )
            data = resp.get("Success", [{}])[0]
            return Quote(
                symbol=symbol,
                exchange=exchange,
                ltp=float(data.get("ltp", 0)),
                open=float(data.get("open", 0)),
                high=float(data.get("high", 0)),
                low=float(data.get("low", 0)),
                close=float(data.get("last_close", 0)),
                volume=int(data.get("total_quantity_traded", 0)),
                oi=int(data.get("open_interest", 0)),
                timestamp=datetime.now(),
            )
        except Exception as exc:
            raise BrokerDataError(f"get_quote failed for {symbol}: {exc}") from exc

    def get_ltp(self, symbol: str, exchange: Exchange = Exchange.NSE) -> float:
        return self.get_quote(symbol, exchange).ltp

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_historical_data(
        self,
        symbol: str,
        exchange: Exchange,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        # Breeze uses different interval labels
        interval_map = {
            "minute": "1minute",
            "3minute": "3minute",
            "5minute": "5minute",
            "15minute": "15minute",
            "30minute": "30minute",
            "day": "1day",
        }
        breeze_interval = interval_map.get(interval, "5minute")
        ex = _EXCHANGE_MAP.get(exchange, "NSE")
        try:
            resp = self._breeze.get_historical_data_v2(
                interval=breeze_interval,
                from_date=from_date.strftime("%Y-%m-%dT00:00:00.000Z"),
                to_date=to_date.strftime("%Y-%m-%dT00:00:00.000Z"),
                stock_code=symbol,
                exchange_code=ex,
                product_type="cash",
            )
            records = resp.get("Success", [])
            df = pd.DataFrame(records)
            if df.empty:
                return df
            df.rename(
                columns={
                    "datetime": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
                inplace=True,
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            return df[numeric_cols].dropna()
        except Exception as exc:
            raise BrokerDataError(
                f"get_historical_data failed for {symbol}: {exc}"
            ) from exc

    # ── Order Management ──────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_order(self, order: Order) -> str:
        ex = _EXCHANGE_MAP.get(order.exchange, "NSE")
        params = {
            "stock_code": order.symbol,
            "exchange_code": ex,
            "product": _PRODUCT_MAP[order.product],
            "action": "buy" if order.side == OrderSide.BUY else "sell",
            "order_type": _ORDER_TYPE_MAP[order.order_type],
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price else "0",
            "stoploss": str(order.trigger_price) if order.trigger_price else "0",
            "right": "others",
            "strike_price": "0",
        }
        try:
            resp = self._breeze.place_order(**params)
            if resp.get("Status") == 200:
                order_id = str(resp.get("Success", {}).get("order_id", ""))
                logger.info(
                    "ICICI order placed | {} {} {} | ID: {}",
                    order.side.value,
                    order.symbol,
                    order.quantity,
                    order_id,
                )
                return order_id
            raise BrokerOrderError(
                f"ICICI order rejected: {resp.get('Error', 'Unknown')}", order
            )
        except BrokerOrderError:
            raise
        except Exception as exc:
            raise BrokerOrderError(
                f"ICICI place_order failed: {exc}", order
            ) from exc

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None,
    ) -> bool:
        params: dict = {"order_id": order_id}
        if price is not None:
            params["price"] = str(price)
        if trigger_price is not None:
            params["stoploss"] = str(trigger_price)
        if quantity is not None:
            params["quantity"] = str(quantity)
        try:
            resp = self._breeze.modify_order(**params)
            return resp.get("Status") == 200
        except Exception as exc:
            logger.error("ICICI modify_order failed | {}", exc)
            return False

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = self._breeze.cancel_order(order_id=order_id)
            return resp.get("Status") == 200
        except Exception as exc:
            logger.error("ICICI cancel_order failed | {}", exc)
            return False

    def get_order_status(self, order_id: str) -> Order:
        try:
            resp = self._breeze.get_order_detail(order_id=order_id)
            raw = resp.get("Success", [{}])[0]
            return self._parse_order(raw)
        except Exception as exc:
            raise BrokerDataError(f"get_order_status failed: {exc}") from exc

    def get_all_orders(self) -> list[Order]:
        try:
            resp = self._breeze.get_order_list(
                exchange_code="NSE",
                from_date=datetime.now().strftime("%Y-%m-%dT00:00:00.000Z"),
                to_date=datetime.now().strftime("%Y-%m-%dT23:59:59.000Z"),
            )
            return [self._parse_order(o) for o in resp.get("Success", [])]
        except Exception as exc:
            raise BrokerDataError(f"get_all_orders failed: {exc}") from exc

    # ── Positions & Funds ─────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        try:
            resp = self._breeze.get_portfolio_positions()
            positions = []
            for p in resp.get("Success", []):
                qty = int(p.get("quantity", 0))
                if qty == 0:
                    continue
                positions.append(
                    Position(
                        symbol=p.get("stock_code", ""),
                        exchange=Exchange(p.get("exchange_code", "NSE")),
                        product=ProductType.MIS,
                        quantity=qty,
                        average_price=float(p.get("average_price", 0)),
                        ltp=float(p.get("ltp", 0)),
                        pnl=float(p.get("profit_loss", 0)),
                    )
                )
            return positions
        except Exception as exc:
            raise BrokerDataError(f"get_positions failed: {exc}") from exc

    def get_margins(self) -> AccountMargins:
        try:
            resp = self._breeze.get_funds()
            data = resp.get("Success", {})
            available = float(data.get("net_available_for_trading", 0))
            used = float(data.get("utilised_amount", 0))
            return AccountMargins(
                available_cash=available,
                used_margin=used,
                total_balance=available + used,
            )
        except Exception as exc:
            raise BrokerDataError(f"get_margins failed: {exc}") from exc

    # ── WebSocket Streaming ───────────────────────────────────────────

    def subscribe_ticks(
        self, symbols: list[str], exchange: Exchange = Exchange.NSE
    ) -> None:
        ex = _EXCHANGE_MAP.get(exchange, "NSE")

        def _on_tick(tick):
            parsed = {
                "symbol": tick.get("stock_code", ""),
                "ltp": float(tick.get("last", 0)),
                "volume": int(tick.get("ttq", 0)),
                "oi": int(tick.get("OI", 0)),
                "timestamp": datetime.now(),
            }
            self._dispatch_tick(parsed)

        self._breeze.on_ticks = _on_tick
        self._breeze.ws_connect()

        for symbol in symbols:
            self._breeze.subscribe_feeds(
                exchange_code=ex,
                stock_code=symbol,
                product_type="cash",
                expiry_date="",
                strike_price="",
                right="",
                get_exchange_quotes=True,
                get_market_depth=False,
            )
            self._subscribed_symbols.append(symbol)
            logger.info("ICICI subscribed to ticks: {}", symbol)

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        for symbol in symbols:
            try:
                self._breeze.unsubscribe_feeds(stock_code=symbol)
                self._subscribed_symbols.remove(symbol)
            except Exception:  # noqa: BLE001
                pass

    # ── Instrument Lookup ─────────────────────────────────────────────

    def get_instrument_token(self, symbol: str, exchange: Exchange) -> int:
        # ICICI doesn't use numeric tokens — return 0 as a no-op
        return 0

    def search_instruments(self, query: str, exchange: Exchange) -> list[dict]:
        ex = _EXCHANGE_MAP.get(exchange, "NSE")
        try:
            resp = self._breeze.get_names(exchange_code=ex, stock_code=query)
            return resp.get("Success", [])
        except Exception as exc:
            raise BrokerDataError(f"search_instruments failed: {exc}") from exc

    # ── Private helpers ───────────────────────────────────────────────

    def _parse_order(self, raw: dict) -> Order:
        side_str = raw.get("action", "buy").lower()
        return Order(
            order_id=str(raw.get("order_id", "")),
            symbol=raw.get("stock_code", ""),
            exchange=Exchange(raw.get("exchange_code", "NSE")),
            side=OrderSide.BUY if side_str == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
            quantity=int(raw.get("quantity", 0)),
            price=float(raw.get("price", 0)),
            trigger_price=float(raw.get("stoploss", 0)),
            status=_STATUS_MAP.get(raw.get("status", ""), OrderStatus.PENDING),
            filled_price=float(raw.get("average_price", 0)),
            filled_quantity=int(raw.get("filled_quantity", 0)),
        )
