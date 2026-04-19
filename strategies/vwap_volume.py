"""
strategies/vwap_volume.py — VWAP + Volume crossover strategy.

Entry logic:
  LONG  → price crosses above VWAP + volume surge + RSI 45–65 + above 9 EMA
  SHORT → price crosses below VWAP + volume surge + RSI 35–55 + below 9 EMA

Exit logic:
  - Price crosses opposite side of VWAP
  - RSI extremes (overbought/oversold)
  - Stop loss / target hit (delegated to RiskManager)
  - Time-based square-off
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy,
    Candle,
    ExitReason,
    Signal,
    SignalDirection,
)
from strategies.indicators import (
    atr as _atr,
    ema as _ema,
    rsi as _rsi,
    sma as _sma,
    vwap as _vwap_calc,
)


# ── Strategy Parameters ───────────────────────────────────────────────────────

@dataclass
class VWAPVolumeConfig:
    """All tunable parameters — change here without touching logic."""

    # Volume
    volume_ma_period: int = 20          # SMA period for average volume
    volume_surge_multiplier: float = 2.0  # Volume must be N× the average

    # RSI
    rsi_period: int = 14
    rsi_long_min: int = 45              # LONG: RSI must be above this
    rsi_long_max: int = 65              # LONG: RSI must be below this
    rsi_short_min: int = 35             # SHORT: RSI range
    rsi_short_max: int = 55
    rsi_overbought: int = 75            # Exit long if RSI > this
    rsi_oversold: int = 25              # Exit short if RSI < this

    # EMA
    ema_period: int = 9

    # ATR (for stop loss)
    atr_period: int = 14
    sl_atr_multiplier: float = 1.5      # Stop loss = N × ATR
    rr_ratio: float = 2.0              # Reward:Risk ratio for target

    # Partial booking
    partial_exit_rr: float = 1.0        # Book 50% when RR hits 1:1

    # F&O strike selection
    strike_interval: int = 50           # Nifty=50, BankNifty=100, FinNifty=50

    warmup_candles: int = 50            # Candles needed before trading


# ── Strategy ──────────────────────────────────────────────────────────────────

class VWAPVolumeStrategy(BaseStrategy):
    """
    VWAP + Volume surge strategy for intraday trading.

    Signals are generated on every completed candle close.
    """

    def __init__(self, symbol: str, settings) -> None:
        super().__init__(symbol, settings)
        self.cfg = VWAPVolumeConfig()
        self._warmup_candles = self.cfg.warmup_candles

        # Indicator cache (updated per candle)
        self._vwap: float = 0.0
        self._rsi: float = 50.0
        self._ema9: float = 0.0
        self._atr: float = 0.0
        self._volume_ma: float = 0.0
        self._prev_close_vs_vwap: int = 0   # +1 above, -1 below, 0 init
        self._partial_booked: bool = False  # True once 1:1 RR partial exit fires
        self._last_tick_price: float = 0.0  # Updated on every tick
        self._tick_sl_breach: bool = False  # SL crossed intra-candle on a tick

        # Backtest fast-path: pre-computed indicators keyed by candle datetime.
        # Populated by precompute_indicators(); on_candle() uses it to skip
        # recomputing indicators on every bar (O(n²) → O(n) per backtest).
        self._precomputed: dict = {}

    # ── Candle processing ─────────────────────────────────────────────

    def add_candle(self, candle: Candle) -> None:
        """
        Override base: when pre-computed indicators are loaded (backtest),
        just append to history without rebuilding the whole DataFrame —
        the fast path in on_candle() does not read self._df at all.
        This eliminates the O(n²) rebuild cost over long backtests.
        """
        if self._precomputed:
            self._candle_history.append(candle)
        else:
            super().add_candle(candle)

    def precompute_indicators(self, full_df: pd.DataFrame) -> None:
        """
        Backtest fast-path: compute every indicator once on the full DataFrame
        and cache the per-bar results keyed by candle datetime. Called by the
        backtest engine before the candle loop — on_candle() then just reads
        the cached row instead of recomputing all five indicators each bar.
        """
        df = full_df.copy()
        df["vwap"], _, _ = _vwap_calc(df)
        df["ema9"]   = _ema(df["close"], self.cfg.ema_period)
        df["rsi"]    = _rsi(df["close"], self.cfg.rsi_period)
        df["atr"]    = _atr(df, self.cfg.atr_period)
        df["vol_ma"] = _sma(df["volume"], self.cfg.volume_ma_period)

        cols = ["vwap", "ema9", "rsi", "atr", "vol_ma"]
        self._precomputed = {
            ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts:
                tuple(row[c] for c in cols)
            for ts, row in df[cols].iterrows()
        }

    def on_candle(self, candle: Candle) -> None:
        """Update indicators on every completed candle."""
        self.add_candle(candle)
        self.state.candles_in_trade = (
            self.state.candles_in_trade + 1 if self.state.in_trade else 0
        )

        if not self.is_warmed_up():
            return

        # ── Fast path (backtest): read pre-computed values ────────────────
        cached = self._precomputed.get(candle.datetime)
        if cached is not None:
            v, e, r, a, vm = cached
            if not math.isnan(v):  self._vwap      = v
            if not math.isnan(e):  self._ema9      = e
            if not math.isnan(r):  self._rsi       = r
            if not math.isnan(a):  self._atr       = a
            if not math.isnan(vm): self._volume_ma = vm
        else:
            # ── Live path: recompute on the recent slice ──────────────────
            # Cap to last 200 candles — sufficient for all indicators (VWAP ≤75 bars/day,
            # RSI/ATR/EMA/SMA all ≤50) and avoids O(n²) growth.
            df = self._df.iloc[-200:].copy()
            df["vwap"], _, _ = _vwap_calc(df)
            df["ema9"]   = _ema(df["close"], self.cfg.ema_period)
            df["rsi"]    = _rsi(df["close"], self.cfg.rsi_period)
            df["atr"]    = _atr(df, self.cfg.atr_period)
            df["vol_ma"] = _sma(df["volume"], self.cfg.volume_ma_period)
            last = df.iloc[-1]
            if not math.isnan(last["vwap"]):   self._vwap      = last["vwap"]
            if not math.isnan(last["ema9"]):   self._ema9      = last["ema9"]
            if not math.isnan(last["rsi"]):    self._rsi       = last["rsi"]
            if not math.isnan(last["atr"]):    self._atr       = last["atr"]
            if not math.isnan(last["vol_ma"]): self._volume_ma = last["vol_ma"]

        # Store indicator values in state for dashboard display
        self.state.indicators = {
            "vwap": round(self._vwap, 2),
            "ema9": round(self._ema9, 2),
            "rsi": round(self._rsi, 1),
            "atr": round(self._atr, 2),
            "volume_ma": round(self._volume_ma, 0),
            "volume": candle.volume,
            "volume_ratio": round(candle.volume / self._volume_ma, 2) if self._volume_ma else 0,
        }

    def on_tick(self, tick: dict) -> None:
        """
        Called on every live price tick (high-frequency).
        Monitors intra-candle SL breaches so the exit fires at the next
        generate_signal() call rather than waiting for candle close.
        Full tick-based order execution is handled by OrderManager (Phase 7).
        """
        ltp = float(tick.get("ltp", 0))
        if ltp <= 0 or not self.state.in_trade:
            self._last_tick_price = ltp
            return

        self._last_tick_price = ltp
        sl = self.state.current_sl
        direction = self.state.trade_direction

        if direction == SignalDirection.LONG and ltp <= sl:
            if not self._tick_sl_breach:
                self._tick_sl_breach = True
                logger.warning("Tick SL breach detected @ {} (SL {})", ltp, sl)
        elif direction == SignalDirection.SHORT and ltp >= sl:
            if not self._tick_sl_breach:
                self._tick_sl_breach = True
                logger.warning("Tick SL breach detected @ {} (SL {})", ltp, sl)

    # ── Signal generation ─────────────────────────────────────────────

    def _indicators_ready(self) -> bool:
        """Return True only when every indicator holds a valid, non-zero value."""
        return (
            self._vwap > 0
            and self._atr > 0
            and self._volume_ma > 0
            and not math.isnan(self._rsi)
            and not math.isnan(self._ema9)
        )

    def generate_signal(self) -> Signal:
        """Evaluate conditions and return a Signal."""
        if not self.is_warmed_up():
            return self._flat_signal("Warming up")

        if not self._indicators_ready():
            return self._flat_signal("Indicators not ready")

        candle = self.last_candle()
        if candle is None:
            return self._flat_signal("No candle data")

        # Time guards
        now = candle.datetime.time()
        if not (self._settings.trade_start <= now <= self._settings.trade_end):
            if now >= self._settings.squareoff and self.state.in_trade:
                return self._exit_signal(ExitReason.TIME_SQUAREOFF, candle.close)
            return self._flat_signal("Outside trading hours")

        # If in trade, check exit conditions first
        if self.state.in_trade:
            # Honour any intra-candle SL breach flagged by on_tick()
            if self._tick_sl_breach:
                self._tick_sl_breach = False
                return self._exit_signal(ExitReason.STOP_LOSS_HIT, candle.close)
            exit_signal = self._check_exit_conditions(candle)
            if exit_signal:
                return exit_signal
            return Signal(
                direction=SignalDirection.HOLD,
                symbol=self.symbol,
                reason="Holding position",
            )

        # Check for new entry
        return self._check_entry_conditions(candle)

    def _check_entry_conditions(self, candle: Candle) -> Signal:
        price = candle.close
        vol_surge = candle.volume > self.cfg.volume_surge_multiplier * self._volume_ma
        prev_dir = self._prev_close_vs_vwap

        # Determine current price position vs VWAP
        above_vwap = price > self._vwap
        below_vwap = price < self._vwap

        # Update position tracker for next candle's crossover detection
        curr_dir = 1 if above_vwap else (-1 if below_vwap else 0)
        crossed_above = (prev_dir <= 0) and (curr_dir == 1)
        crossed_below = (prev_dir >= 0) and (curr_dir == -1)
        self._prev_close_vs_vwap = curr_dir

        # ── LONG conditions ──────────────────────────────────────────
        if (
            crossed_above
            and vol_surge
            and self.cfg.rsi_long_min <= self._rsi <= self.cfg.rsi_long_max
            and price > self._ema9
        ):
            sl = price - (self.cfg.sl_atr_multiplier * self._atr)
            risk = price - sl
            target = price + (self.cfg.rr_ratio * risk)
            confidence = self._confidence_score(
                crossed=True, vol_surge=vol_surge, rsi_in_range=True, ema_aligned=True
            )
            fo = self.which_strike(SignalDirection.LONG, price)
            return Signal(
                direction=SignalDirection.LONG,
                symbol=self.symbol,
                entry_price=price,
                stop_loss=round(sl, 2),
                target=round(target, 2),
                confidence=confidence,
                strategy_name=self.name,
                option_type=fo["option_type"],
                strike_price=fo["strike_price"],
                reason=(
                    f"VWAP crossover UP | RSI {self._rsi:.1f} | "
                    f"Vol {candle.volume:,} ({candle.volume / self._volume_ma:.1f}x avg) | "
                    f"Price {price} > EMA9 {self._ema9:.2f} | "
                    f"F&O: {fo['option_type']} @ {fo['strike_price']:.0f}"
                ),
            )

        # ── SHORT conditions ─────────────────────────────────────────
        if (
            crossed_below
            and vol_surge
            and self.cfg.rsi_short_min <= self._rsi <= self.cfg.rsi_short_max
            and price < self._ema9
        ):
            sl = price + (self.cfg.sl_atr_multiplier * self._atr)
            risk = sl - price
            target = price - (self.cfg.rr_ratio * risk)
            confidence = self._confidence_score(
                crossed=True, vol_surge=vol_surge, rsi_in_range=True, ema_aligned=True
            )
            fo = self.which_strike(SignalDirection.SHORT, price)
            return Signal(
                direction=SignalDirection.SHORT,
                symbol=self.symbol,
                entry_price=price,
                stop_loss=round(sl, 2),
                target=round(target, 2),
                confidence=confidence,
                strategy_name=self.name,
                option_type=fo["option_type"],
                strike_price=fo["strike_price"],
                reason=(
                    f"VWAP crossover DOWN | RSI {self._rsi:.1f} | "
                    f"Vol {candle.volume:,} ({candle.volume / self._volume_ma:.1f}x avg) | "
                    f"Price {price} < EMA9 {self._ema9:.2f} | "
                    f"F&O: {fo['option_type']} @ {fo['strike_price']:.0f}"
                ),
            )

        return self._flat_signal("No signal")

    def _check_exit_conditions(self, candle: Candle) -> Signal | None:
        price = candle.close
        direction = self.state.trade_direction
        entry = self.state.entry_price
        risk = abs(entry - self.state.current_sl)

        # ── Partial exit at 1:1 RR — move SL to breakeven ────────────────────
        # Actual quantity reduction (book 50%) is handled by OrderManager in
        # Phase 7. Here we track the event and update the SL.
        if not self._partial_booked and risk > 0:
            partial_target = (
                entry + self.cfg.partial_exit_rr * risk
                if direction == SignalDirection.LONG
                else entry - self.cfg.partial_exit_rr * risk
            )
            hit = (
                (direction == SignalDirection.LONG and price >= partial_target)
                or (direction == SignalDirection.SHORT and price <= partial_target)
            )
            if hit:
                self._partial_booked = True
                self.state.current_sl = entry  # move SL to breakeven
                logger.info(
                    "1:1 RR reached @ {} — SL moved to breakeven {}",
                    price, entry,
                )

        if direction == SignalDirection.LONG:
            # Target hit — full exit
            if self.state.current_target > 0 and price >= self.state.current_target:
                return self._exit_signal(ExitReason.TARGET_HIT, price)
            # SL hit
            if price <= self.state.current_sl:
                return self._exit_signal(ExitReason.STOP_LOSS_HIT, price)
            # VWAP cross back below → exit
            if price < self._vwap:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)
            # RSI overbought → exit
            if self._rsi > self.cfg.rsi_overbought:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)

        elif direction == SignalDirection.SHORT:
            # Target hit — full exit
            if self.state.current_target > 0 and price <= self.state.current_target:
                return self._exit_signal(ExitReason.TARGET_HIT, price)
            # SL hit
            if price >= self.state.current_sl:
                return self._exit_signal(ExitReason.STOP_LOSS_HIT, price)
            # VWAP cross back above → exit
            if price > self._vwap:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)
            # RSI oversold → exit
            if self._rsi < self.cfg.rsi_oversold:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)

        return None

    # ── Lifecycle callbacks ───────────────────────────────────────────

    def on_trade_entry(self, signal: Signal, filled_price: float) -> None:
        self.state.in_trade = True
        self.state.trade_direction = signal.direction
        self.state.entry_price = filled_price
        self.state.entry_time = datetime.now()
        self.state.current_sl = signal.stop_loss
        self.state.current_target = signal.target
        self.state.last_signal = signal
        self._partial_booked = False
        self._tick_sl_breach = False
        logger.info(
            "Trade entered | {} {} @ {} | SL: {} | Target: {}",
            signal.direction.value, self.symbol,
            filled_price, signal.stop_loss, signal.target,
        )

    def on_trade_exit(self, exit_reason: ExitReason, exit_price: float) -> None:
        entry = self.state.entry_price
        direction = self.state.trade_direction
        pnl = (exit_price - entry) if direction == SignalDirection.LONG else (entry - exit_price)
        logger.info(
            "Trade exited | {} {} | Entry: {} Exit: {} | P&L: {:.2f} | Reason: {}",
            direction.value if direction else "?", self.symbol,
            entry, exit_price, pnl, exit_reason.value,
        )
        self._partial_booked = False
        self._tick_sl_breach = False
        self.reset_state()

    def on_market_open(self) -> None:
        """Reset daily state at 9:15 AM."""
        self._prev_close_vs_vwap = 0
        self._tick_sl_breach = False
        logger.info("{} strategy reset for new trading day", self.name)

    def which_strike(
        self,
        direction: SignalDirection,
        current_price: float,
    ) -> dict:
        """
        Return the ATM strike details for F&O (options) trading.

        Used when the bot trades Call/Put options instead of the index directly.
        The Signal's option_type and strike_price fields are populated using
        this method before the signal is emitted.

        Returns
        -------
        dict with keys: strike_price (float), option_type ("CE" or "PE")

        Example (Nifty @ 22_345, interval=50):
            LONG  → {"strike_price": 22350.0, "option_type": "CE"}
            SHORT → {"strike_price": 22350.0, "option_type": "PE"}
        """
        interval = self.cfg.strike_interval
        atm = round(current_price / interval) * interval
        option_type = "CE" if direction == SignalDirection.LONG else "PE"
        return {"strike_price": float(atm), "option_type": option_type}

    # ── Private helpers ───────────────────────────────────────────────

    def _flat_signal(self, reason: str) -> Signal:
        return Signal(
            direction=SignalDirection.FLAT,
            symbol=self.symbol,
            reason=reason,
            strategy_name=self.name,
        )

    def _exit_signal(self, reason: ExitReason, price: float) -> Signal:
        return Signal(
            direction=SignalDirection.FLAT,
            symbol=self.symbol,
            entry_price=price,
            exit_reason=reason,
            reason=f"Exit: {reason.value}",
            strategy_name=self.name,
        )

    def _confidence_score(
        self,
        crossed: bool,
        vol_surge: bool,
        rsi_in_range: bool,
        ema_aligned: bool,
    ) -> int:
        """Score 0–100 based on how many conditions are strongly met."""
        score = 0
        if crossed:
            score += 30
        if vol_surge:
            vol_ratio = self.last_candle().volume / self._volume_ma if self._volume_ma else 1
            score += min(30, int(vol_ratio * 10))  # up to 30 points
        if rsi_in_range:
            score += 20
        if ema_aligned:
            score += 20
        return min(score, 100)
