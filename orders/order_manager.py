"""
orders/order_manager.py — Order placement, paper trading simulation, and position lifecycle.

Responsibilities
----------------
  Signal processing   : convert strategy Signal → broker order
  Paper trading       : simulate fills at signal price (no broker API calls)
  Entry management    : MARKET entry + immediate SL-M bracket order (with 3-retry)
  Exit management     : cancel SL order, place market exit
  Partial booking     : book 50% at 1:1 RR, move SL to breakeven
  Trailing SL         : update SL order when RiskManager moves the SL
  F&O support         : build Zerodha NFO symbol, ATM strike selection
  Reconciliation      : sync positions and trailing SL every 5 min
  Square-off          : close all/one positions at EOD or shutdown

Paper vs Live
-------------
  Paper  → fills simulated at signal price; no broker API calls for orders
  Live   → MARKET entry (3 retries) → poll for fill → SL-M bracket order

Usage (from main.py)
--------------------
    om = OrderManager(broker, risk_manager, trade_logger, notifier, settings,
                      strategy=strategy)
    om.process_signal(signal)               # called after every candle signal
    om.sync_with_broker()                   # called every 5 minutes
    om.square_off_all()                     # called at EOD / shutdown
"""

from __future__ import annotations

import calendar
import threading
import time
from datetime import date, datetime, timedelta
from typing import Optional

from loguru import logger

from brokers.base_broker import (
    Exchange,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)
from risk.risk_manager import TradePosition
from strategies.base_strategy import ExitReason, Signal, SignalDirection


# Timeout waiting for a live order fill before cancelling
_FILL_TIMEOUT_SECS: float = 15.0
_FILL_POLL_INTERVAL: float = 0.5
_ORDER_RETRY_DELAY: float = 1.0   # seconds between retries
_MAX_ORDER_RETRIES: int = 3

# Zerodha monthly/weekly expiry single-char month codes for weekly symbols
_WEEKLY_MONTH_CODE: dict[int, str] = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
    7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D",
}


class OrderManager:
    """
    Translates strategy signals into broker orders and manages the full
    lifecycle of each trade.

    Thread-safe — process_signal() is called from the candle-close callback
    thread; sync_with_broker() runs in the main loop thread.
    """

    def __init__(
        self,
        broker,
        risk_manager,
        trade_logger,
        notifier,
        settings,
        strategy=None,
    ) -> None:
        self.broker = broker
        self.risk = risk_manager
        self.logger = trade_logger
        self.notifier = notifier
        self._s = settings
        self._strategy = strategy       # optional: for on_trade_entry/exit callbacks

        self._lock = threading.Lock()
        # symbol (underlying) → broker SL order ID  (live mode only)
        self._sl_order_ids: dict[str, str] = {}
        # symbol (underlying) → actual broker trading symbol
        # (same as underlying for equity; full option symbol for F&O)
        self._trade_symbols: dict[str, str] = {}

    def set_strategy(self, strategy) -> None:
        """Wire in the strategy after construction if not passed to __init__."""
        self._strategy = strategy

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API — signal routing + lifecycle
    # ═══════════════════════════════════════════════════════════════════════════

    def process_signal(self, signal: Signal) -> None:
        """
        Main entry point — called on every candle after generate_signal().

        LONG / SHORT  → _enter_trade()   (opens a new position)
        FLAT          → _exit_trade()    (closes the open position)
        HOLD          → no action
        """
        if not signal.is_valid():
            logger.warning("process_signal: invalid signal ignored | {}", signal)
            return

        if signal.direction in (SignalDirection.LONG, SignalDirection.SHORT):
            existing = self.risk.get_position(signal.symbol)
            if existing:
                logger.debug(
                    "process_signal: already in {} trade for {} — ignoring new entry",
                    existing.direction, signal.symbol,
                )
                return
            if not self.risk.is_trading_allowed():
                logger.info("process_signal: trading not allowed — skipping entry")
                return
            self._enter_trade(signal)

        elif signal.direction == SignalDirection.FLAT:
            self._exit_trade(
                symbol=signal.symbol,
                reason=signal.exit_reason or ExitReason.SIGNAL_REVERSAL,
                hint_price=signal.entry_price,
            )
        # HOLD → do nothing

    def sync_with_broker(self) -> None:
        """
        Called every 5 minutes from main.py.

        Paper mode : check partial booking trigger using last tick price.
        Live mode  : update trailing SL → modify SL order; partial booking;
                     sync realised P&L from broker.
        """
        if self._s.paper_trade:
            self._paper_check_partials()
            return
        try:
            self._live_reconcile()
        except Exception as exc:
            logger.error("sync_with_broker error: {}", exc)

    def sync_positions_with_broker(self) -> None:
        """Alias for sync_with_broker() — matches prompt spec naming."""
        self.sync_with_broker()

    def get_open_positions(self) -> list:
        """Return all open TradePosition objects tracked by RiskManager."""
        return self.risk.get_open_positions()

    def square_off_all(self) -> None:
        """Close every open position at market price (EOD or shutdown)."""
        positions = self.risk.get_open_positions()
        if not positions:
            return
        logger.info("Squaring off {} open position(s)...", len(positions))
        for pos in positions:
            self._exit_trade(
                symbol=pos.symbol,
                reason=ExitReason.TIME_SQUAREOFF,
                force=True,
            )

    def square_off_position(self, symbol: str) -> None:
        """Close a single position by underlying symbol."""
        self._exit_trade(
            symbol=symbol,
            reason=ExitReason.MANUAL,
            force=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Public order placement wrappers (convenience API for direct use)
    # ═══════════════════════════════════════════════════════════════════════════

    def place_buy_order(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        exchange: Exchange = Exchange.NSE,
        product: ProductType = ProductType.MIS,
    ) -> Optional[str]:
        """Place a buy order with retry logic. Returns broker order_id or None."""
        if self._s.paper_trade:
            logger.info("[PAPER] Simulated BUY {} qty={} @ ₹{}", symbol, quantity, price or "MARKET")
            return f"PAPER_BUY_{symbol}_{int(time.time())}"
        order = Order(
            symbol=symbol, exchange=exchange, side=OrderSide.BUY,
            order_type=order_type, product=product,
            quantity=quantity, price=price,
        )
        return self._place_with_retry(order)

    def place_sell_order(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        exchange: Exchange = Exchange.NSE,
        product: ProductType = ProductType.MIS,
    ) -> Optional[str]:
        """Place a sell order with retry logic. Returns broker order_id or None."""
        if self._s.paper_trade:
            logger.info("[PAPER] Simulated SELL {} qty={} @ ₹{}", symbol, quantity, price or "MARKET")
            return f"PAPER_SELL_{symbol}_{int(time.time())}"
        order = Order(
            symbol=symbol, exchange=exchange, side=OrderSide.SELL,
            order_type=order_type, product=product,
            quantity=quantity, price=price,
        )
        return self._place_with_retry(order)

    def place_sl_order(
        self,
        symbol: str,
        quantity: int,
        trigger_price: float,
        limit_price: float = 0.0,
        exchange: Exchange = Exchange.NSE,
        product: ProductType = ProductType.MIS,
    ) -> Optional[str]:
        """Place a SL / SL-M order. limit_price=0 → SL-M (market stop)."""
        if self._s.paper_trade:
            logger.info(
                "[PAPER] Simulated SL {} qty={} trigger=₹{}", symbol, quantity, trigger_price
            )
            return f"PAPER_SL_{symbol}_{int(time.time())}"
        order = Order(
            symbol=symbol, exchange=exchange, side=OrderSide.SELL,
            order_type=OrderType.SL if limit_price > 0 else OrderType.SL_M,
            product=product,
            quantity=quantity,
            trigger_price=trigger_price,
            price=limit_price,
        )
        return self._place_with_retry(order)

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> bool:
        """Modify an open/pending broker order."""
        if self._s.paper_trade:
            return True
        try:
            return self.broker.modify_order(
                order_id=order_id, price=new_price, quantity=new_quantity
            )
        except Exception as exc:
            logger.error("modify_order {} failed: {}", order_id, exc)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending/open broker order."""
        if self._s.paper_trade:
            return True
        try:
            return self.broker.cancel_order(order_id)
        except Exception as exc:
            logger.error("cancel_order {} failed: {}", order_id, exc)
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # F&O helpers
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def select_atm_strike(
        index_price: float,
        option_type: str,
        interval: int = 50,
    ) -> dict:
        """
        Return ATM strike and option type for a given index price.

        Parameters
        ----------
        index_price : current index LTP
        option_type : "CE" (call) or "PE" (put)
        interval    : strike spacing — Nifty=50, BankNifty=100, FinNifty=50

        Returns
        -------
        {"strike_price": 22000.0, "option_type": "CE"}
        """
        atm = round(index_price / interval) * interval
        return {"strike_price": float(atm), "option_type": option_type.upper()}

    @staticmethod
    def _build_option_symbol(
        underlying: str,
        strike: float,
        option_type: str,
        expiry: date,
    ) -> str:
        """
        Build the Zerodha NFO trading symbol for an options contract.

        Weekly expiry  (not last Thursday): NIFTY2451722000CE
        Monthly expiry (last Thursday)    : NIFTY24MAY22000CE
        """
        yr = str(expiry.year)[2:]           # "24"
        strike_str = str(int(strike))       # "22000"
        opt = option_type.upper()           # "CE"

        # Last Thursday of the month = monthly expiry
        last_day = calendar.monthrange(expiry.year, expiry.month)[1]
        last_thu = last_day - (date(expiry.year, expiry.month, last_day).weekday() - 3) % 7

        if expiry.day == last_thu:
            mon = expiry.strftime("%b").upper()   # "MAY"
            return f"{underlying.upper()}{yr}{mon}{strike_str}{opt}"
        else:
            mc = _WEEKLY_MONTH_CODE[expiry.month]
            return f"{underlying.upper()}{yr}{mc}{expiry.day:02d}{strike_str}{opt}"

    @staticmethod
    def _nearest_expiry() -> date:
        """Return the nearest upcoming NSE weekly expiry (Thursday)."""
        today = date.today()
        days = (3 - today.weekday()) % 7   # days until next Thursday (3 = Thursday)
        if days == 0:
            # Today is Thursday — use today unless we're past 3:30 PM
            now = datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                days = 7
        return today + timedelta(days=days)

    def get_option_chain_ltp(
        self,
        underlying: str,
        expiry: date,
        strikes: list[float],
        option_type: str,
    ) -> dict[float, float]:
        """
        Fetch LTP for a list of strikes.
        Returns {strike_price: ltp}.  Silently skips symbols that fail.
        """
        prices: dict[float, float] = {}
        for strike in strikes:
            sym = self._build_option_symbol(underlying, strike, option_type, expiry)
            try:
                if self._s.paper_trade:
                    prices[strike] = 0.0   # paper mode: no live data
                else:
                    prices[strike] = self.broker.get_ltp(sym, Exchange.NFO)
            except Exception as exc:
                logger.debug("get_option_chain_ltp: {} failed: {}", sym, exc)
        return prices

    # ═══════════════════════════════════════════════════════════════════════════
    # Entry flow (internal)
    # ═══════════════════════════════════════════════════════════════════════════

    def _enter_trade(self, signal: Signal) -> None:
        direction_str = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"

        # ── Resolve F&O trading symbol if options trade ───────────────────────
        trade_symbol = signal.symbol
        trade_exchange = signal.exchange
        if signal.option_type and signal.strike_price:
            expiry = (
                datetime.strptime(signal.expiry, "%Y-%m-%d").date()
                if signal.expiry
                else self._nearest_expiry()
            )
            trade_symbol = self._build_option_symbol(
                signal.symbol, signal.strike_price, signal.option_type, expiry
            )
            trade_exchange = "NFO"
            logger.info(
                "F&O entry: underlying={} → trading_symbol={} expiry={}",
                signal.symbol, trade_symbol, expiry,
            )

        # ── Position sizing ───────────────────────────────────────────────────
        qty = self.risk.calculate_quantity(
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss,
            symbol=signal.symbol,
        )
        if qty <= 0:
            logger.warning("_enter_trade: qty=0 for {} — skipping", signal.symbol)
            return

        # ── Place entry order ─────────────────────────────────────────────────
        filled_price = self._place_entry_order(signal, qty, trade_symbol, trade_exchange)
        if filled_price is None:
            logger.error("_enter_trade: entry order failed for {}", trade_symbol)
            return

        # Recompute SL / target from actual fill price (guards against slippage)
        sl = signal.stop_loss
        if sl <= 0:
            sl = self.risk.calculate_sl(
                entry_price=filled_price,
                direction=direction_str,
                method="percent",
            )
        if direction_str == "LONG" and sl >= filled_price:
            sl = round(filled_price * 0.99, 2)
        elif direction_str == "SHORT" and sl <= filled_price:
            sl = round(filled_price * 1.01, 2)

        target = self.risk.calculate_target(
            entry_price=filled_price,
            stop_loss_price=sl,
            direction=direction_str,
        )
        if target <= 0 and signal.target > 0:
            target = signal.target

        # ── Register position with RiskManager ───────────────────────────────
        pos = self.risk.open_position(
            symbol=signal.symbol,           # track by underlying always
            direction=direction_str,
            entry_price=filled_price,
            quantity=qty,
            stop_loss=sl,
            target_price=target,
            exchange=trade_exchange,
            product="MIS",
        )
        if pos is None:
            logger.warning(
                "_enter_trade: RiskManager rejected open_position for {}", signal.symbol
            )
            if not self._s.paper_trade:
                self._emergency_exit(trade_symbol, direction_str, qty, trade_exchange)
            return

        # Store broker trading symbol for exit (underlying → option symbol mapping)
        with self._lock:
            self._trade_symbols[signal.symbol] = trade_symbol

        # ── Notify strategy ───────────────────────────────────────────────────
        if self._strategy is not None:
            try:
                self._strategy.on_trade_entry(signal, filled_price)
            except Exception as exc:
                logger.warning("on_trade_entry callback error: {}", exc)

        # ── Place SL bracket order (live only) ───────────────────────────────
        if not self._s.paper_trade:
            sl_id = self._place_sl_bracket(
                broker_symbol=trade_symbol,
                direction=direction_str,
                quantity=qty,
                sl_price=sl,
                exchange=Exchange(trade_exchange),
            )
            if sl_id:
                with self._lock:
                    self._sl_order_ids[signal.symbol] = sl_id

        # ── Notifications and logging ─────────────────────────────────────────
        mode = "PAPER" if self._s.paper_trade else "LIVE"
        fo_tag = f" [{trade_symbol}]" if trade_symbol != signal.symbol else ""
        msg = (
            f"[{mode}] ENTRY {direction_str} {signal.symbol}{fo_tag} "
            f"@ ₹{filled_price:.2f} | SL ₹{sl:.2f} | Target ₹{target:.2f} "
            f"| Qty {qty} | Conf {signal.confidence}% | {signal.reason}"
        )
        self.notifier.send(msg)
        logger.info(msg)
        self.logger.log_trade({
            "type": "ENTRY",
            "symbol": signal.symbol,
            "broker_symbol": trade_symbol,
            "direction": direction_str,
            "price": filled_price,
            "quantity": qty,
            "sl": round(sl, 2),
            "target": round(target, 2),
            "confidence": signal.confidence,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # Exit flow (internal)
    # ═══════════════════════════════════════════════════════════════════════════

    def _exit_trade(
        self,
        symbol: str,
        reason: ExitReason,
        force: bool = False,
        hint_price: float = 0.0,
    ) -> None:
        pos = self.risk.get_position(symbol)
        if pos is None:
            return  # already closed / never opened

        exit_price = self._resolve_exit_price(pos, reason, hint_price)
        if exit_price <= 0:
            if force:
                exit_price = pos.entry_price
                logger.warning(
                    "_exit_trade: no price for {} — falling back to entry ₹{}",
                    symbol, exit_price,
                )
            else:
                logger.error("_exit_trade: no exit price for {} — skipping", symbol)
                return

        # Retrieve the actual broker trading symbol (option symbol or underlying)
        with self._lock:
            broker_symbol = self._trade_symbols.pop(symbol, symbol)

        # ── Cancel pending SL order (live only) ──────────────────────────────
        if not self._s.paper_trade:
            self._cancel_sl_order(symbol)

        # ── Place exit market order (live only) ──────────────────────────────
        if not self._s.paper_trade:
            exit_side = OrderSide.SELL if pos.direction == "LONG" else OrderSide.BUY
            exit_order = Order(
                symbol=broker_symbol,
                exchange=Exchange(pos.exchange),
                side=exit_side,
                order_type=OrderType.MARKET,
                product=ProductType(pos.product),
                quantity=pos.active_quantity,
            )
            order_id = self._place_with_retry(exit_order)
            if order_id is None and not force:
                logger.error("Exit order failed for {} — aborting exit", broker_symbol)
                return

        # ── Close position in RiskManager ────────────────────────────────────
        pnl = self.risk.close_position(symbol, exit_price, reason=reason.value)

        # ── Notify strategy ───────────────────────────────────────────────────
        if self._strategy is not None:
            try:
                self._strategy.on_trade_exit(reason, exit_price)
            except Exception as exc:
                logger.warning("on_trade_exit callback error: {}", exc)

        # ── Notifications and logging ─────────────────────────────────────────
        mode = "PAPER" if self._s.paper_trade else "LIVE"
        pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"−₹{abs(pnl):.2f}"
        msg = (
            f"[{mode}] EXIT {pos.direction} {symbol} "
            f"@ ₹{exit_price:.2f} | P&L {pnl_str} | {reason.value}"
        )
        self.notifier.send(msg)
        logger.info(msg)
        self.logger.log_trade({
            "type": "EXIT",
            "symbol": symbol,
            "broker_symbol": broker_symbol,
            "direction": pos.direction,
            "price": exit_price,
            "quantity": pos.active_quantity,
            "pnl": round(pnl, 2),
            "reason": reason.value,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })

    def _resolve_exit_price(
        self, pos: TradePosition, reason: ExitReason, hint: float
    ) -> float:
        """Pick the best available exit price for a given exit reason."""
        if reason in (ExitReason.STOP_LOSS_HIT, ExitReason.TRAILING_SL_HIT):
            return pos.current_sl
        if reason == ExitReason.TARGET_HIT:
            return pos.target_price
        if hint > 0:
            return hint
        if not self._s.paper_trade:
            with self._lock:
                broker_sym = self._trade_symbols.get(pos.symbol, pos.symbol)
            try:
                return self.broker.get_ltp(broker_sym, Exchange(pos.exchange))
            except Exception as exc:
                logger.warning("get_ltp failed for {}: {}", broker_sym, exc)
        return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # Order placement helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _place_with_retry(
        self, order: Order, max_retries: int = _MAX_ORDER_RETRIES
    ) -> Optional[str]:
        """
        Place a broker order with up to max_retries attempts.
        Waits _ORDER_RETRY_DELAY seconds between attempts.
        Returns order_id on success, None after all retries exhausted.
        On REJECTED status: logs reason, sends alert, does NOT retry.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                order_id = self.broker.place_order(order)
                if attempt > 1:
                    logger.info(
                        "Order placed on attempt {} | {} {} qty={}",
                        attempt, order.side.value, order.symbol, order.quantity,
                    )
                return order_id
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Order attempt {}/{} failed for {}: {}",
                    attempt, max_retries, order.symbol, exc,
                )
                if attempt < max_retries:
                    time.sleep(_ORDER_RETRY_DELAY)

        msg = (
            f"ORDER FAILED after {max_retries} retries | "
            f"{order.side.value} {order.symbol} qty={order.quantity} | {last_exc}"
        )
        logger.error(msg)
        self.notifier.send(f"[ALERT] {msg}")
        return None

    def _place_entry_order(
        self,
        signal: Signal,
        qty: int,
        trade_symbol: str = "",
        trade_exchange: str = "",
    ) -> Optional[float]:
        """
        Place entry order and return the actual fill price.
        Paper mode : return signal.entry_price immediately.
        Live mode  : place MARKET order (with retry), poll for fill.
        """
        if self._s.paper_trade:
            logger.info(
                "[PAPER] Simulated {} fill | {} @ ₹{:.2f} qty={}",
                signal.direction.value,
                trade_symbol or signal.symbol,
                signal.entry_price,
                qty,
            )
            return signal.entry_price

        sym = trade_symbol or signal.symbol
        exch = Exchange(trade_exchange or signal.exchange)
        side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL
        entry_order = Order(
            symbol=sym,
            exchange=exch,
            side=side,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
            quantity=qty,
        )
        order_id = self._place_with_retry(entry_order)
        if order_id is None:
            return None

        filled = self._wait_for_fill(order_id)
        if filled is None:
            logger.error(
                "Entry order {} not filled within {}s — cancelling",
                order_id, _FILL_TIMEOUT_SECS,
            )
            try:
                self.broker.cancel_order(order_id)
            except Exception:
                pass
        return filled

    def _wait_for_fill(self, order_id: str) -> Optional[float]:
        """Poll broker every 0.5 s until fill confirmed or timeout (15 s)."""
        deadline = time.monotonic() + _FILL_TIMEOUT_SECS
        while time.monotonic() < deadline:
            try:
                order = self.broker.get_order_status(order_id)
                if order.status == OrderStatus.COMPLETE:
                    return order.filled_price
                if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    msg = (
                        f"Order {order_id} {order.status.value} | "
                        f"{order.symbol} | reason: {order.rejection_reason}"
                    )
                    logger.error(msg)
                    self.notifier.send(f"[ALERT] {msg}")   # alert on rejection
                    return None
            except Exception as exc:
                logger.warning("get_order_status error: {}", exc)
            time.sleep(_FILL_POLL_INTERVAL)
        return None

    def _place_sl_bracket(
        self,
        broker_symbol: str,
        direction: str,
        quantity: int,
        sl_price: float,
        exchange: Exchange,
    ) -> Optional[str]:
        """
        Place SL-M bracket order after entry fill.
        LONG → SELL SL-M; SHORT → BUY SL-M.
        """
        side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
        sl_order = Order(
            symbol=broker_symbol,
            exchange=exchange,
            side=side,
            order_type=OrderType.SL_M,
            product=ProductType.MIS,
            quantity=quantity,
            trigger_price=sl_price,
        )
        order_id = self._place_with_retry(sl_order)
        if order_id:
            logger.info(
                "SL-M bracket placed | {} {} | trigger=₹{} | id={}",
                direction, broker_symbol, sl_price, order_id,
            )
        return order_id

    def _cancel_sl_order(self, symbol: str) -> None:
        """Cancel the active SL bracket order for a symbol, if any."""
        with self._lock:
            order_id = self._sl_order_ids.pop(symbol, None)
        if order_id is None:
            return
        try:
            self.broker.cancel_order(order_id)
            logger.info("SL order {} cancelled for {}", order_id, symbol)
        except Exception as exc:
            logger.warning("SL order cancel failed for {}: {}", symbol, exc)

    def _emergency_exit(
        self, broker_symbol: str, direction: str, qty: int, exchange: str
    ) -> None:
        """Place an emergency market exit for an orphaned broker position."""
        side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
        order = Order(
            symbol=broker_symbol,
            exchange=Exchange(exchange),
            side=side,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
            quantity=qty,
        )
        order_id = self._place_with_retry(order)
        if order_id:
            logger.warning(
                "Emergency exit placed | {} {} qty={}", direction, broker_symbol, qty
            )
        else:
            logger.error(
                "Emergency exit FAILED for {} — manual intervention required!", broker_symbol
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Reconciliation (internal)
    # ═══════════════════════════════════════════════════════════════════════════

    def _paper_check_partials(self) -> None:
        """
        Paper mode: check if any open position has hit the 1:1 RR partial-exit
        trigger using the last known tick price from the strategy.
        """
        for pos in self.risk.get_open_positions():
            ltp = 0.0
            if self._strategy is not None and hasattr(self._strategy, "_last_tick_price"):
                ltp = self._strategy._last_tick_price

            if ltp <= 0:
                logger.debug(
                    "[PAPER] Open: {} {} entry=₹{} sl=₹{} target=₹{}",
                    pos.direction, pos.symbol,
                    pos.entry_price, pos.current_sl, pos.target_price,
                )
                continue

            if pos.is_partial_trigger(ltp):
                booked_qty, partial_pnl = self.risk.apply_partial_booking(pos, ltp)
                if booked_qty > 0:
                    msg = (
                        f"[PAPER] PARTIAL EXIT {pos.direction} {pos.symbol} "
                        f"qty={booked_qty} @ ₹{ltp:.2f} | "
                        f"P&L ₹{partial_pnl:.2f} | SL → breakeven"
                    )
                    self.notifier.send(msg)
                    logger.info(msg)
                    self.logger.log_trade({
                        "type": "PARTIAL_EXIT",
                        "symbol": pos.symbol,
                        "direction": pos.direction,
                        "price": ltp,
                        "quantity": booked_qty,
                        "pnl": round(partial_pnl, 2),
                        "reason": "PARTIAL_1R",
                        "mode": "PAPER",
                        "timestamp": datetime.now().isoformat(),
                    })

    def _live_reconcile(self) -> None:
        """
        Live mode: for each open position, fetch LTP, apply partial booking
        if triggered, update trailing SL and modify broker SL order if moved,
        then sync daily P&L from broker.
        """
        for pos in self.risk.get_open_positions():
            with self._lock:
                broker_sym = self._trade_symbols.get(pos.symbol, pos.symbol)
            try:
                ltp = self.broker.get_ltp(broker_sym, Exchange(pos.exchange))
            except Exception as exc:
                logger.warning("get_ltp failed for {}: {}", broker_sym, exc)
                continue

            # ── Partial booking check ─────────────────────────────────────────
            if pos.is_partial_trigger(ltp):
                booked_qty, partial_pnl = self.risk.apply_partial_booking(pos, ltp)
                if booked_qty > 0:
                    exit_side = OrderSide.SELL if pos.direction == "LONG" else OrderSide.BUY
                    partial_order = Order(
                        symbol=broker_sym,
                        exchange=Exchange(pos.exchange),
                        side=exit_side,
                        order_type=OrderType.MARKET,
                        product=ProductType(pos.product),
                        quantity=booked_qty,
                    )
                    self._place_with_retry(partial_order)

                    msg = (
                        f"[LIVE] PARTIAL EXIT {pos.direction} {pos.symbol} "
                        f"qty={booked_qty} @ ₹{ltp:.2f} | "
                        f"P&L ₹{partial_pnl:.2f} | SL → breakeven ₹{pos.entry_price:.2f}"
                    )
                    self.notifier.send(msg)
                    logger.info(msg)
                    self.logger.log_trade({
                        "type": "PARTIAL_EXIT",
                        "symbol": pos.symbol,
                        "broker_symbol": broker_sym,
                        "direction": pos.direction,
                        "price": ltp,
                        "quantity": booked_qty,
                        "pnl": round(partial_pnl, 2),
                        "reason": "PARTIAL_1R",
                        "mode": "LIVE",
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Modify SL order to breakeven
                    with self._lock:
                        sl_id = self._sl_order_ids.get(pos.symbol)
                    if sl_id:
                        try:
                            self.broker.modify_order(
                                order_id=sl_id, trigger_price=pos.entry_price
                            )
                            logger.info(
                                "SL → breakeven ₹{} for {}", pos.entry_price, pos.symbol
                            )
                        except Exception as exc:
                            logger.warning(
                                "SL modify to breakeven failed for {}: {}", pos.symbol, exc
                            )

            # ── Trailing SL update ────────────────────────────────────────────
            old_sl = pos.current_sl
            new_sl = self.risk.update_trailing_sl(pos, ltp)
            if new_sl != old_sl:
                with self._lock:
                    sl_id = self._sl_order_ids.get(pos.symbol)
                if sl_id:
                    try:
                        self.broker.modify_order(
                            order_id=sl_id, trigger_price=new_sl
                        )
                        logger.info(
                            "Trailing SL modified | {} {} | ₹{:.2f} → ₹{:.2f}",
                            pos.direction, pos.symbol, old_sl, new_sl,
                        )
                    except Exception as exc:
                        logger.warning(
                            "SL order modify failed for {}: {}", pos.symbol, exc
                        )

        # ── Sync P&L from broker ──────────────────────────────────────────────
        try:
            broker_positions = self.broker.get_positions()
            realised_pnl = sum(p.pnl for p in broker_positions)
            self.risk.update_daily_pnl(realised_pnl)
        except Exception as exc:
            logger.warning("P&L sync from broker failed: {}", exc)
