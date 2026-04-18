"""
orders/order_manager.py — Order placement, paper trading simulation, and position lifecycle.

Responsibilities
----------------
  Signal processing  : convert strategy Signal → broker order
  Paper trading      : simulate fills at signal price (no broker API calls)
  Entry management   : market order + immediate SL-M bracket order
  Exit management    : cancel SL order, place market exit
  Partial booking    : book 50% at 1:1 RR, move SL to breakeven
  Trailing SL        : update SL order when RiskManager moves the SL
  Reconciliation     : sync positions and trailing SL every 5 min
  Square-off         : close all positions at EOD or shutdown

Paper vs Live
-------------
  Paper mode  → fills simulated at signal price; no broker API calls for orders
  Live mode   → MARKET entry order → poll for fill → SL-M bracket order

Usage (from main.py)
--------------------
    om = OrderManager(broker, risk_manager, trade_logger, notifier, settings,
                      strategy=strategy)
    om.process_signal(signal)       # called after every candle signal
    om.sync_with_broker()           # called every 5 minutes
    om.square_off_all()             # called at EOD / shutdown
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
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
        # symbol → broker SL order ID (only used in live mode)
        self._sl_order_ids: dict[str, str] = {}

    def set_strategy(self, strategy) -> None:
        """Wire in the strategy after construction if not passed to __init__."""
        self._strategy = strategy

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API — called from main.py
    # ═══════════════════════════════════════════════════════════════════════════

    def process_signal(self, signal: Signal) -> None:
        """
        Main entry point — called on every candle after generate_signal().

        Routing
        -------
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
                hint_price=signal.entry_price,   # strategy may embed close price here
            )
        # HOLD → do nothing

    def sync_with_broker(self) -> None:
        """
        Called every 5 minutes from main.py.

        Paper mode : log open positions; no broker calls needed.
        Live mode  : update trailing SL → modify SL order if SL moved;
                     check partial booking; sync realised P&L from broker.
        """
        if self._s.paper_trade:
            self._paper_check_partials()
            return

        try:
            self._live_reconcile()
        except Exception as exc:
            logger.error("sync_with_broker error: {}", exc)

    def square_off_all(self) -> None:
        """
        Close every open position at market price.
        Called at square-off time and on shutdown signal.
        """
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Entry flow
    # ═══════════════════════════════════════════════════════════════════════════

    def _enter_trade(self, signal: Signal) -> None:
        direction_str: str = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"

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
        filled_price = self._place_entry_order(signal, qty)
        if filled_price is None:
            logger.error("_enter_trade: entry order failed for {}", signal.symbol)
            return

        # Recompute SL / target from actual fill price (slippage may differ)
        sl = signal.stop_loss
        if sl <= 0:
            sl = self.risk.calculate_sl(
                entry_price=filled_price,
                direction=direction_str,
                method="atr",
                atr=self._s.__dict__.get("_atr", 0.0),  # best-effort; 0 falls back to percent
            )

        # Ensure SL is on the correct side after slippage
        if direction_str == "LONG" and sl >= filled_price:
            sl = filled_price * 0.99   # 1% fallback
        elif direction_str == "SHORT" and sl <= filled_price:
            sl = filled_price * 1.01

        target = self.risk.calculate_target(
            entry_price=filled_price,
            stop_loss_price=sl,
            direction=direction_str,
        )
        if target <= 0 and signal.target > 0:
            target = signal.target

        # ── Register position with RiskManager ───────────────────────────────
        pos = self.risk.open_position(
            symbol=signal.symbol,
            direction=direction_str,
            entry_price=filled_price,
            quantity=qty,
            stop_loss=sl,
            target_price=target,
            exchange=signal.exchange,
            product="MIS",
        )
        if pos is None:
            # RiskManager rejected (max positions hit in the instant between checks)
            logger.warning("_enter_trade: RiskManager rejected open_position for {}", signal.symbol)
            if not self._s.paper_trade:
                self._emergency_exit(signal.symbol, direction_str, qty, signal.exchange)
            return

        # ── Notify strategy ───────────────────────────────────────────────────
        if self._strategy is not None:
            try:
                self._strategy.on_trade_entry(signal, filled_price)
            except Exception as exc:
                logger.warning("on_trade_entry callback error: {}", exc)

        # ── Place SL bracket order (live only) ───────────────────────────────
        if not self._s.paper_trade:
            sl_id = self._place_sl_order(
                symbol=signal.symbol,
                direction=direction_str,
                quantity=qty,
                sl_price=sl,
                exchange=Exchange(signal.exchange),
            )
            if sl_id:
                with self._lock:
                    self._sl_order_ids[signal.symbol] = sl_id

        # ── Notifications and logging ─────────────────────────────────────────
        mode = "PAPER" if self._s.paper_trade else "LIVE"
        msg = (
            f"[{mode}] ENTRY {direction_str} {signal.symbol} "
            f"@ ₹{filled_price:.2f} | SL ₹{sl:.2f} | Target ₹{target:.2f} "
            f"| Qty {qty} | Conf {signal.confidence}% | {signal.reason}"
        )
        self.notifier.send(msg)
        logger.info(msg)

        self.logger.log_trade({
            "type": "ENTRY",
            "symbol": signal.symbol,
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
    # Exit flow
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

        # ── Determine exit price ──────────────────────────────────────────────
        exit_price = self._resolve_exit_price(pos, reason, hint_price)
        if exit_price <= 0:
            if force:
                exit_price = pos.entry_price   # worst-case fallback
                logger.warning(
                    "_exit_trade: could not determine exit price for {} — using entry {}",
                    symbol, exit_price,
                )
            else:
                logger.error("_exit_trade: no exit price for {} — skipping", symbol)
                return

        # ── Cancel pending SL order (live only) ──────────────────────────────
        if not self._s.paper_trade:
            self._cancel_sl_order(symbol)

        # ── Place exit market order (live only) ──────────────────────────────
        if not self._s.paper_trade:
            exit_side = OrderSide.SELL if pos.direction == "LONG" else OrderSide.BUY
            try:
                self.broker.place_market_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=pos.active_quantity,
                    exchange=Exchange(pos.exchange),
                    product=ProductType(pos.product),
                )
            except Exception as exc:
                logger.error("Exit order failed for {}: {}", symbol, exc)
                if not force:
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
        if reason == ExitReason.STOP_LOSS_HIT:
            return pos.current_sl
        if reason == ExitReason.TRAILING_SL_HIT:
            return pos.current_sl
        if reason == ExitReason.TARGET_HIT:
            return pos.target_price
        # SIGNAL_REVERSAL / TIME_SQUAREOFF / MANUAL
        if hint > 0:
            return hint
        if not self._s.paper_trade:
            try:
                return self.broker.get_ltp(pos.symbol, Exchange(pos.exchange))
            except Exception as exc:
                logger.warning("get_ltp failed for {}: {}", pos.symbol, exc)
        return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # Order placement helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _place_entry_order(self, signal: Signal, qty: int) -> Optional[float]:
        """
        Place entry order and return the actual fill price.
        Paper mode : return signal.entry_price immediately (no slippage modelled).
        Live mode  : place MARKET order, poll for fill, timeout after 15 s.
        """
        if self._s.paper_trade:
            logger.info(
                "[PAPER] Simulated {} fill | {} @ ₹{:.2f} qty={}",
                signal.direction.value, signal.symbol, signal.entry_price, qty,
            )
            return signal.entry_price

        side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL
        try:
            order_id = self.broker.place_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=qty,
                exchange=Exchange(signal.exchange),
                product=ProductType.MIS,
            )
        except Exception as exc:
            logger.error("Entry order placement failed: {}", exc)
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
        """Poll broker until fill confirmed or timeout. Returns filled_price or None."""
        deadline = time.monotonic() + _FILL_TIMEOUT_SECS
        while time.monotonic() < deadline:
            try:
                order = self.broker.get_order_status(order_id)
                if order.status == OrderStatus.COMPLETE:
                    return order.filled_price
                if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    logger.error(
                        "Order {} {}: {}",
                        order_id, order.status.value, order.rejection_reason,
                    )
                    return None
            except Exception as exc:
                logger.warning("get_order_status error: {}", exc)
            time.sleep(_FILL_POLL_INTERVAL)
        return None

    def _place_sl_order(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        sl_price: float,
        exchange: Exchange,
    ) -> Optional[str]:
        """
        Place a SL-M (Stop-Loss Market) order.
        LONG position → SELL SL-M at trigger=sl_price.
        SHORT position → BUY SL-M at trigger=sl_price.
        Returns broker order_id or None on failure.
        """
        side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
        sl_order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=OrderType.SL_M,
            product=ProductType.MIS,
            quantity=quantity,
            trigger_price=sl_price,
        )
        try:
            order_id = self.broker.place_order(sl_order)
            logger.info(
                "SL-M order placed | {} {} | trigger=₹{} | order_id={}",
                direction, symbol, sl_price, order_id,
            )
            return order_id
        except Exception as exc:
            logger.error("Failed to place SL order for {}: {}", symbol, exc)
            return None

    def _cancel_sl_order(self, symbol: str) -> None:
        """Cancel the active SL order for a symbol, if any."""
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
        self, symbol: str, direction: str, qty: int, exchange: str
    ) -> None:
        """Place a market exit to close an orphaned broker position."""
        side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
        try:
            self.broker.place_market_order(
                symbol=symbol, side=side, quantity=qty,
                exchange=Exchange(exchange), product=ProductType.MIS,
            )
            logger.warning(
                "Emergency exit placed | {} {} qty={}", direction, symbol, qty
            )
        except Exception as exc:
            logger.error("Emergency exit FAILED for {}: {}", symbol, exc)

    # ═══════════════════════════════════════════════════════════════════════════
    # Reconciliation helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _paper_check_partials(self) -> None:
        """
        Paper mode: check if any open position has hit the 1:1 RR partial-exit
        trigger.  We don't have live LTP here so we use the last known tick
        price from the strategy if available.
        """
        for pos in self.risk.get_open_positions():
            # Try to get last tick price from strategy
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

            # Partial booking check
            if pos.is_partial_trigger(ltp):
                booked_qty, partial_pnl = self.risk.apply_partial_booking(pos, ltp)
                if booked_qty > 0:
                    mode = "PAPER"
                    msg = (
                        f"[{mode}] PARTIAL EXIT {pos.direction} {pos.symbol} "
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
                        "mode": mode,
                        "timestamp": datetime.now().isoformat(),
                    })

    def _live_reconcile(self) -> None:
        """
        Live mode reconciliation.

        For each open position:
          1. Fetch current LTP.
          2. Check and apply partial booking (50% at 1:1 RR).
          3. Update trailing SL via RiskManager; if SL moved, modify broker SL order.
        Also syncs realised P&L from broker positions for accurate daily accounting.
        """
        for pos in self.risk.get_open_positions():
            try:
                ltp = self.broker.get_ltp(pos.symbol, Exchange(pos.exchange))
            except Exception as exc:
                logger.warning("get_ltp failed for {}: {}", pos.symbol, exc)
                continue

            # Partial booking check
            if pos.is_partial_trigger(ltp):
                booked_qty, partial_pnl = self.risk.apply_partial_booking(pos, ltp)
                if booked_qty > 0:
                    # Place partial exit market order
                    exit_side = OrderSide.SELL if pos.direction == "LONG" else OrderSide.BUY
                    try:
                        self.broker.place_market_order(
                            symbol=pos.symbol,
                            side=exit_side,
                            quantity=booked_qty,
                            exchange=Exchange(pos.exchange),
                            product=ProductType(pos.product),
                        )
                    except Exception as exc:
                        logger.error("Partial exit order failed for {}: {}", pos.symbol, exc)

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
                        "direction": pos.direction,
                        "price": ltp,
                        "quantity": booked_qty,
                        "pnl": round(partial_pnl, 2),
                        "reason": "PARTIAL_1R",
                        "mode": "LIVE",
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Modify SL order to breakeven after partial booking
                    with self._lock:
                        sl_order_id = self._sl_order_ids.get(pos.symbol)
                    if sl_order_id:
                        try:
                            self.broker.modify_order(
                                order_id=sl_order_id,
                                trigger_price=pos.entry_price,
                            )
                            logger.info(
                                "SL order updated to breakeven ₹{} for {}",
                                pos.entry_price, pos.symbol,
                            )
                        except Exception as exc:
                            logger.warning(
                                "SL modify to breakeven failed for {}: {}", pos.symbol, exc
                            )

            # Trailing SL update
            old_sl = pos.current_sl
            new_sl = self.risk.update_trailing_sl(pos, ltp)
            if new_sl != old_sl:
                with self._lock:
                    sl_order_id = self._sl_order_ids.get(pos.symbol)
                if sl_order_id:
                    try:
                        self.broker.modify_order(
                            order_id=sl_order_id,
                            trigger_price=new_sl,
                        )
                        logger.info(
                            "SL order modified | {} {} | ₹{:.2f} → ₹{:.2f}",
                            pos.direction, pos.symbol, old_sl, new_sl,
                        )
                    except Exception as exc:
                        logger.warning(
                            "SL order modify failed for {}: {}", pos.symbol, exc
                        )

        # Sync realised P&L from broker
        try:
            broker_positions = self.broker.get_positions()
            realised_pnl = sum(p.pnl for p in broker_positions)
            self.risk.update_daily_pnl(realised_pnl)
        except Exception as exc:
            logger.warning("P&L sync from broker failed: {}", exc)
