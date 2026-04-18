"""
risk/risk_manager.py — Complete risk management engine.

Responsibilities
----------------
  Position sizing   : capital-based lot calculation, hard INR caps
  Stop loss         : ATR / percent / points methods
  Trailing SL       : activates after min profit, moves only in favorable direction
  Partial booking   : 50% exit at 1:1 RR, move SL to breakeven, trail remainder
  Daily limits      : max loss, max profit, max trades — is_trading_allowed()
  Position tracking : owns the authoritative list of open positions

Usage (by OrderManager)
-----------------------
    rm = RiskManager(settings)
    qty  = rm.calculate_quantity(entry=22350, sl=22250, capital=rm.available_capital)
    pos  = rm.open_position(symbol, direction, entry, qty, sl, target)
    new_sl = rm.update_trailing_sl(pos, current_price, atr=55.0)
    rm.close_position(symbol, exit_price)
    rm.is_trading_allowed()          # False after daily loss cap hit
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

from loguru import logger


# ── Known F&O lot sizes ───────────────────────────────────────────────────────
# Keep updated with NSE circular changes.
_LOT_SIZES: dict[str, int] = {
    "NIFTY":      50,
    "BANKNIFTY":  15,
    "FINNIFTY":   40,
    "MIDCPNIFTY": 75,
    "SENSEX":     10,
    "BANKEX":     15,
}


# ═════════════════════════════════════════════════════════════════════════════
# TradePosition — richer than the broker's Position dataclass
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TradePosition:
    """
    A single open trade tracked by the RiskManager.

    Distinct from brokers.base_broker.Position (which is a thin broker view).
    This class owns the risk state: SL, trailing SL, target, partial booking.
    """

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    quantity: int
    entry_time: datetime
    initial_sl: float          # SL at entry — never changes
    current_sl: float          # Updated by trailing logic
    target_price: float        # Full-exit target (2:1 RR)

    exchange: str = "NSE"
    product: str = "MIS"

    # Trailing SL state
    trail_activated: bool = False
    peak_price: float = 0.0    # Highest seen for LONG, lowest for SHORT

    # Partial booking state
    partial_booked: bool = False    # True once 50% booked at 1:1 RR
    active_quantity: int = 0        # Remaining quantity after partial booking

    def __post_init__(self) -> None:
        if self.active_quantity == 0:
            self.active_quantity = self.quantity
        self.peak_price = self.entry_price

    # ── P&L helpers ───────────────────────────────────────────────────

    def current_pnl(self, current_price: float) -> float:
        """Unrealised P&L for the active (remaining) quantity."""
        mult = 1.0 if self.direction == "LONG" else -1.0
        return mult * (current_price - self.entry_price) * self.active_quantity

    def risk_per_share(self) -> float:
        """Initial risk per unit (entry − initial_sl for LONG)."""
        return abs(self.entry_price - self.initial_sl)

    def initial_risk_amount(self) -> float:
        """Total capital at risk when the trade was opened."""
        return self.risk_per_share() * self.quantity

    def reward_at_target(self) -> float:
        """Expected P&L if full target is hit."""
        mult = 1.0 if self.direction == "LONG" else -1.0
        return mult * (self.target_price - self.entry_price) * self.active_quantity

    def risk_reward_ratio(self) -> float:
        risk = self.risk_per_share()
        if risk <= 0:
            return 0.0
        reward = abs(self.target_price - self.entry_price)
        return round(reward / risk, 2)

    # ── Status checks ─────────────────────────────────────────────────

    def is_sl_hit(self, price: float) -> bool:
        if self.direction == "LONG":
            return price <= self.current_sl
        return price >= self.current_sl

    def is_target_hit(self, price: float) -> bool:
        if self.direction == "LONG":
            return price >= self.target_price
        return price <= self.target_price

    def is_partial_trigger(self, price: float, rr_trigger: float = 1.0) -> bool:
        """True when price has reached the 1:1 RR level and partial not yet booked."""
        if self.partial_booked:
            return False
        risk = self.risk_per_share()
        if risk <= 0:
            return False
        partial_target = (
            self.entry_price + rr_trigger * risk
            if self.direction == "LONG"
            else self.entry_price - rr_trigger * risk
        )
        return (
            (self.direction == "LONG" and price >= partial_target)
            or (self.direction == "SHORT" and price <= partial_target)
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "active_quantity": self.active_quantity,
            "entry_time": self.entry_time.isoformat(),
            "initial_sl": round(self.initial_sl, 2),
            "current_sl": round(self.current_sl, 2),
            "target_price": round(self.target_price, 2),
            "trail_activated": self.trail_activated,
            "partial_booked": self.partial_booked,
        }

    def __repr__(self) -> str:
        return (
            f"<TradePosition {self.direction} {self.symbol} "
            f"qty={self.active_quantity} entry={self.entry_price} "
            f"sl={self.current_sl:.2f} target={self.target_price:.2f}>"
        )


# ═════════════════════════════════════════════════════════════════════════════
# DailyStats — resets at market open each day
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DailyStats:
    date: date = field(default_factory=date.today)
    realised_pnl: float = 0.0    # Sum of closed-trade P&L
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    gross_loss: float = 0.0      # Sum of losing trades only (positive number)
    gross_profit: float = 0.0    # Sum of winning trades only

    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return round(self.trades_won / self.trades_taken * 100, 1)

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "realised_pnl": round(self.realised_pnl, 2),
            "trades_taken": self.trades_taken,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "win_rate_pct": self.win_rate(),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
        }


# ═════════════════════════════════════════════════════════════════════════════
# RiskManager
# ═════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Central risk management engine.

    Thread-safe — all public methods acquire a lock so OrderManager can call
    them from the tick callback thread without race conditions.
    """

    def __init__(self, settings) -> None:
        self._s = settings
        self._lock = threading.Lock()

        self._positions: dict[str, TradePosition] = {}   # symbol → position
        self._daily: DailyStats = DailyStats()
        self._trading_halted: bool = False   # True when daily limit hit mid-session

    # ── Position sizing ────────────────────────────────────────────────

    def calculate_quantity(
        self,
        entry_price: float,
        stop_loss_price: float,
        capital: Optional[float] = None,
        symbol: str = "",
        force_lots: bool = False,
    ) -> int:
        """
        Return the number of shares (or F&O units) to trade.

        Sizing rules applied in order:
        1. Risk per trade = min(capital × risk_pct, max_risk_inr)
        2. Shares = risk_budget / risk_per_share
        3. Round DOWN to the nearest lot size for F&O symbols
        4. Ensure at least 1 lot / 1 share

        Parameters
        ----------
        entry_price     : planned entry price
        stop_loss_price : initial SL price
        capital         : available capital (defaults to settings.trading_capital)
        symbol          : used to look up F&O lot size
        force_lots      : if True, always round to nearest lot even for equity
        """
        cap = capital if capital is not None else self._s.trading_capital
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share <= 0:
            logger.warning("calculate_quantity: zero risk_per_share, defaulting to 1")
            return 1

        # Risk budget: smallest of % and hard INR cap
        risk_budget = min(
            cap * self._s.risk_per_trade_pct / 100.0,
            self._s.max_risk_per_trade_inr,
        )

        raw_qty = int(risk_budget / risk_per_share)
        lot_size = _LOT_SIZES.get(symbol.upper(), 1)

        if lot_size > 1 or force_lots:
            lots = max(1, raw_qty // lot_size)
            qty = lots * lot_size
        else:
            qty = max(1, raw_qty)

        logger.debug(
            "Position size | {} entry={} sl={} risk_budget=₹{:.0f} qty={}",
            symbol or "equity", entry_price, stop_loss_price, risk_budget, qty,
        )
        return qty

    def lot_size_for(self, symbol: str) -> int:
        """Return the F&O lot size for a symbol (1 for equity)."""
        return _LOT_SIZES.get(symbol.upper(), 1)

    # ── Stop loss calculation ──────────────────────────────────────────

    def calculate_sl(
        self,
        entry_price: float,
        direction: Literal["LONG", "SHORT"],
        method: Literal["atr", "percent", "points"] = "atr",
        atr: float = 0.0,
        atr_multiplier: float = 1.5,
        percent: float = 1.0,
        points: float = 50.0,
    ) -> float:
        """
        Compute the initial stop-loss price.

        Methods
        -------
        atr     : entry ± (atr_multiplier × ATR)  [default, recommended]
        percent : entry ± (percent / 100 × entry)
        points  : entry ± points

        Returns the SL price (never on the wrong side of entry).
        """
        if method == "atr":
            if atr <= 0:
                logger.warning("ATR=0 in calculate_sl, falling back to percent method")
                method = "percent"
            else:
                offset = atr_multiplier * atr
        if method == "percent":
            offset = entry_price * percent / 100.0
        if method == "points":
            offset = points

        sl = (
            entry_price - offset
            if direction == "LONG"
            else entry_price + offset
        )
        return round(sl, 2)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: Literal["LONG", "SHORT"],
        rr_ratio: float = 2.0,
    ) -> float:
        """Return the target price for a given RR ratio."""
        risk = abs(entry_price - stop_loss_price)
        target = (
            entry_price + rr_ratio * risk
            if direction == "LONG"
            else entry_price - rr_ratio * risk
        )
        return round(target, 2)

    # ── Trailing stop loss ─────────────────────────────────────────────

    def update_trailing_sl(
        self,
        position: TradePosition,
        current_price: float,
        atr: float = 0.0,
        method: Literal["atr", "percent"] = "atr",
        trail_atr_multiplier: float = 1.5,
        trail_percent: float = 1.0,
        min_profit_atr_multiple: float = 1.0,
    ) -> float:
        """
        Update the trailing stop loss for an open position.

        Activation
        ----------
        Trailing only activates after the trade has moved
        `min_profit_atr_multiple × ATR` in the favorable direction.
        Before activation, the initial SL is unchanged.

        Movement rules
        --------------
        - SL can only move in the favorable direction (tighten, never widen).
        - For LONG: new_sl = current_high - trail_distance (rises with price).
        - For SHORT: new_sl = current_low + trail_distance (falls with price).

        Returns
        -------
        The new (or unchanged) SL price.  Always updates position.current_sl.
        """
        with self._lock:
            return self._do_trail(
                position, current_price, atr,
                method, trail_atr_multiplier, trail_percent,
                min_profit_atr_multiple,
            )

    def _do_trail(
        self,
        pos: TradePosition,
        price: float,
        atr: float,
        method: str,
        atr_mult: float,
        pct: float,
        min_profit_mult: float,
    ) -> float:
        # Trail distance
        if method == "atr" and atr > 0:
            trail_dist = atr_mult * atr
        else:
            trail_dist = price * pct / 100.0

        # Activation check
        if not pos.trail_activated:
            activation_dist = min_profit_mult * atr if atr > 0 else trail_dist
            profit = (
                price - pos.entry_price
                if pos.direction == "LONG"
                else pos.entry_price - price
            )
            if profit < activation_dist:
                return pos.current_sl   # not activated yet

            pos.trail_activated = True
            logger.info(
                "Trailing SL activated for {} {} | price={} entry={} profit={:.2f}",
                pos.direction, pos.symbol, price, pos.entry_price, profit,
            )

        # Update peak price
        if pos.direction == "LONG":
            if price > pos.peak_price:
                pos.peak_price = price
            new_sl = round(pos.peak_price - trail_dist, 2)
            # SL can only move UP (tighten)
            if new_sl > pos.current_sl:
                old_sl = pos.current_sl
                pos.current_sl = new_sl
                logger.info(
                    "Trailing SL updated | {} {} | {:.2f} → {:.2f} (price={} peak={})",
                    pos.direction, pos.symbol,
                    old_sl, new_sl, price, pos.peak_price,
                )
        else:  # SHORT
            if price < pos.peak_price:
                pos.peak_price = price
            new_sl = round(pos.peak_price + trail_dist, 2)
            # SL can only move DOWN (tighten)
            if new_sl < pos.current_sl:
                old_sl = pos.current_sl
                pos.current_sl = new_sl
                logger.info(
                    "Trailing SL updated | {} {} | {:.2f} → {:.2f} (price={} peak={})",
                    pos.direction, pos.symbol,
                    old_sl, new_sl, price, pos.peak_price,
                )

        return pos.current_sl

    # ── Partial booking ────────────────────────────────────────────────

    def apply_partial_booking(
        self,
        position: TradePosition,
        current_price: float,
        book_pct: float = 0.50,
    ) -> tuple[int, float]:
        """
        Book `book_pct` of the position at current_price (default 50%).
        Moves SL to breakeven and marks partial_booked = True.

        Returns
        -------
        (booked_quantity, realised_pnl)
        """
        with self._lock:
            if position.partial_booked:
                return 0, 0.0

            lot_size = self.lot_size_for(position.symbol)
            book_qty = max(lot_size, int(position.active_quantity * book_pct))
            # Round down to lot size
            if lot_size > 1:
                book_qty = (book_qty // lot_size) * lot_size
            book_qty = min(book_qty, position.active_quantity)

            mult = 1.0 if position.direction == "LONG" else -1.0
            pnl = mult * (current_price - position.entry_price) * book_qty

            position.active_quantity -= book_qty
            position.partial_booked = True
            position.current_sl = position.entry_price  # move to breakeven

            logger.info(
                "Partial booking | {} {} | qty={} @ {} | P&L=₹{:.2f} | SL→breakeven {}",
                position.direction, position.symbol,
                book_qty, current_price, pnl, position.entry_price,
            )
            return book_qty, pnl

    # ── Position lifecycle ─────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target_price: float,
        exchange: str = "NSE",
        product: str = "MIS",
    ) -> TradePosition | None:
        """
        Register a new open position.  Returns None if limits already hit.

        Call this after the broker confirms the entry order is filled.
        """
        with self._lock:
            if not self._is_allowed_locked():
                logger.warning("open_position blocked — trading not allowed")
                return None

            if len(self._positions) >= self._s.max_open_positions:
                logger.warning(
                    "open_position blocked — max_open_positions ({}) reached",
                    self._s.max_open_positions,
                )
                return None

            pos = TradePosition(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=datetime.now(),
                initial_sl=stop_loss,
                current_sl=stop_loss,
                target_price=target_price,
                exchange=exchange,
                product=product,
            )
            self._positions[symbol] = pos
            self._daily.trades_taken += 1

            logger.info(
                "Position opened | {} {} {} qty={} entry={} sl={} target={} | "
                "Daily trades: {}/{}",
                direction, symbol, exchange,
                quantity, entry_price, stop_loss, target_price,
                self._daily.trades_taken, self._s.max_trades_per_day,
            )
            return pos

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "",
    ) -> float:
        """
        Remove the position and record the realised P&L.

        Returns the realised P&L (positive = profit, negative = loss).
        """
        with self._lock:
            pos = self._positions.pop(symbol, None)
            if pos is None:
                logger.warning("close_position: no position found for {}", symbol)
                return 0.0

            mult = 1.0 if pos.direction == "LONG" else -1.0
            pnl = mult * (exit_price - pos.entry_price) * pos.active_quantity
            self._record_closed_trade(pnl)

            logger.info(
                "Position closed | {} {} | entry={} exit={} qty={} | "
                "P&L=₹{:.2f} | reason={} | Daily P&L=₹{:.2f}",
                pos.direction, symbol,
                pos.entry_price, exit_price, pos.active_quantity,
                pnl, reason or "—", self._daily.realised_pnl,
            )
            return pnl

    def get_position(self, symbol: str) -> TradePosition | None:
        return self._positions.get(symbol)

    def get_open_positions(self) -> list[TradePosition]:
        with self._lock:
            return list(self._positions.values())

    def open_position_count(self) -> int:
        return len(self._positions)

    # ── Daily risk controls ────────────────────────────────────────────

    def is_trading_allowed(self) -> bool:
        """
        Return True only when all daily risk limits are within bounds.

        Checks:
        - Daily loss has not exceeded max_daily_loss_inr
        - Daily profit target has not been hit (optional hard stop)
        - Trade count has not exceeded max_trades_per_day
        - No manual halt has been set
        """
        with self._lock:
            return self._is_allowed_locked()

    def _is_allowed_locked(self) -> bool:
        if self._trading_halted:
            return False
        s = self._s
        stats = self._daily

        if stats.realised_pnl <= -abs(s.max_daily_loss_inr):
            if not self._trading_halted:
                self._trading_halted = True
                logger.critical(
                    "DAILY LOSS CAP HIT ₹{:.2f} — trading halted for today",
                    stats.realised_pnl,
                )
            return False

        if stats.realised_pnl >= s.max_daily_profit_inr:
            if not self._trading_halted:
                self._trading_halted = True
                logger.info(
                    "Daily profit target ₹{:.2f} reached — trading stopped",
                    stats.realised_pnl,
                )
            return False

        if stats.trades_taken >= s.max_trades_per_day:
            logger.info(
                "Max trades/day ({}) reached — no new entries",
                s.max_trades_per_day,
            )
            return False

        return True

    def halt_trading(self, reason: str = "manual") -> None:
        """Manually halt all trading for the rest of the day."""
        with self._lock:
            self._trading_halted = True
            logger.warning("Trading halted: {}", reason)

    def update_daily_pnl(self, realised_pnl: float = 0.0) -> None:
        """
        Called from the main reconciliation loop (every 5 minutes).
        Pass the current session P&L from the broker for an accurate sync.
        Without an argument, uses internally tracked P&L (accurate enough
        since every close_position() updates the counter).
        """
        with self._lock:
            if realised_pnl != 0.0:
                self._daily.realised_pnl = realised_pnl
            self._ensure_daily_reset()
            self._is_allowed_locked()   # re-evaluates halt conditions

    def reset_daily(self) -> None:
        """Reset all daily counters — call at market open each morning."""
        with self._lock:
            self._daily = DailyStats()
            self._trading_halted = False
            logger.info("Daily risk counters reset")

    def get_daily_summary(self) -> dict:
        """Return a snapshot of today's risk stats (for dashboard)."""
        with self._lock:
            d = self._daily.to_dict()
            d["trading_allowed"] = self._is_allowed_locked()
            d["open_positions"] = len(self._positions)
            d["max_open_positions"] = self._s.max_open_positions
            d["max_trades_per_day"] = self._s.max_trades_per_day
            d["max_daily_loss_inr"] = self._s.max_daily_loss_inr
            d["max_daily_profit_inr"] = self._s.max_daily_profit_inr
            return d

    @property
    def available_capital(self) -> float:
        """Rough available capital estimate (settings value; broker syncs in Phase 7)."""
        return self._s.trading_capital

    # ── Private helpers ────────────────────────────────────────────────

    def _record_closed_trade(self, pnl: float) -> None:
        self._daily.realised_pnl += pnl
        self._daily.trades_taken = max(self._daily.trades_taken, 1)
        if pnl >= 0:
            self._daily.trades_won += 1
            self._daily.gross_profit += pnl
        else:
            self._daily.trades_lost += 1
            self._daily.gross_loss += abs(pnl)

    def _ensure_daily_reset(self) -> None:
        """Auto-reset stats if the date has rolled over (bot ran overnight)."""
        if self._daily.date != date.today():
            self._daily = DailyStats()
            self._trading_halted = False

    def __repr__(self) -> str:
        return (
            f"<RiskManager positions={len(self._positions)} "
            f"daily_pnl=₹{self._daily.realised_pnl:.2f} "
            f"allowed={self._is_allowed_locked()}>"
        )
