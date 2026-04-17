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

from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy,
    Candle,
    ExitReason,
    InsufficientDataError,
    Signal,
    SignalDirection,
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

    warmup_candles: int = 50            # Candles needed before trading


# ── Indicator functions (pure numpy/pandas — no ta-lib) ───────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP that resets at market open each day.
    Requires a datetime index (or 'datetime' column).
    """
    if "datetime" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or 'datetime' column")

    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["datetime"])
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_x_vol = tp * df["volume"]

    vwap_values = []
    for date, group in tp_x_vol.groupby(idx.date):
        vol_group = df["volume"].loc[group.index]
        vwap_group = tp_x_vol.loc[group.index].cumsum() / vol_group.cumsum()
        vwap_values.append(vwap_group)

    return pd.concat(vwap_values).reindex(df.index)


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

    # ── Candle processing ─────────────────────────────────────────────

    def on_candle(self, candle: Candle) -> None:
        """Update indicators on every completed candle."""
        self.add_candle(candle)
        self.state.candles_in_trade = (
            self.state.candles_in_trade + 1 if self.state.in_trade else 0
        )

        if not self.is_warmed_up():
            return

        df = self._df.copy()

        # Compute indicators
        df["vwap"] = _vwap(df)
        df["ema9"] = _ema(df["close"], self.cfg.ema_period)
        df["rsi"] = _rsi(df["close"], self.cfg.rsi_period)
        df["atr"] = _atr(df, self.cfg.atr_period)
        df["vol_ma"] = _sma(df["volume"], self.cfg.volume_ma_period)

        last = df.iloc[-1]
        self._vwap = last["vwap"]
        self._ema9 = last["ema9"]
        self._rsi = last["rsi"]
        self._atr = last["atr"]
        self._volume_ma = last["vol_ma"]

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

    # ── Signal generation ─────────────────────────────────────────────

    def generate_signal(self) -> Signal:
        """Evaluate conditions and return a Signal."""
        if not self.is_warmed_up():
            return self._flat_signal("Warming up")

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
            return Signal(
                direction=SignalDirection.LONG,
                symbol=self.symbol,
                entry_price=price,
                stop_loss=round(sl, 2),
                target=round(target, 2),
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"VWAP crossover UP | RSI {self._rsi:.1f} | "
                    f"Vol {candle.volume:,} ({candle.volume / self._volume_ma:.1f}x avg) | "
                    f"Price {price} > EMA9 {self._ema9:.2f}"
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
            return Signal(
                direction=SignalDirection.SHORT,
                symbol=self.symbol,
                entry_price=price,
                stop_loss=round(sl, 2),
                target=round(target, 2),
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"VWAP crossover DOWN | RSI {self._rsi:.1f} | "
                    f"Vol {candle.volume:,} ({candle.volume / self._volume_ma:.1f}x avg) | "
                    f"Price {price} < EMA9 {self._ema9:.2f}"
                ),
            )

        return self._flat_signal("No signal")

    def _check_exit_conditions(self, candle: Candle) -> Signal | None:
        price = candle.close
        direction = self.state.trade_direction

        if direction == SignalDirection.LONG:
            # VWAP cross back below → exit
            if price < self._vwap:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)
            # RSI overbought → exit
            if self._rsi > self.cfg.rsi_overbought:
                return self._exit_signal(ExitReason.SIGNAL_REVERSAL, price)

        elif direction == SignalDirection.SHORT:
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
        self.reset_state()

    def on_market_open(self) -> None:
        """Reset daily state at 9:15 AM."""
        self._prev_close_vs_vwap = 0
        logger.info("{} strategy reset for new trading day", self.name)

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
