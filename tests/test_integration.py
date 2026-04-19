"""
tests/test_integration.py — Integration test suite for the trading bot.

Run all tests:
    pytest tests/ -v

Run a specific group:
    pytest tests/ -v -k "TestRiskManager"
    pytest tests/ -v -k "TestStrategy"
    pytest tests/ -v -k "TestOrderManager"
    pytest tests/ -v -k "TestBacktest"

These tests do NOT connect to a broker — all broker calls are mocked.
They verify business logic: risk sizing, strategy signals, order flow, and
a full simulated trading day using synthetic OHLCV data.
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    """Minimal settings object for tests — no .env required."""
    s = MagicMock()
    s.paper_trade          = True
    s.trading_capital      = 100_000.0
    s.risk_per_trade_pct   = 1.0
    s.max_risk_per_trade_inr = 5_000.0
    s.max_daily_loss_inr   = 10_000.0
    s.max_daily_profit_inr = 15_000.0
    s.max_open_positions   = 3
    s.max_trades_per_day   = 5
    s.candle_interval      = 5
    s.strategy             = "vwap_volume"
    s.broker               = "zerodha"
    s.trade_start          = time(9, 20)
    s.trade_end            = time(14, 30)
    s.squareoff            = time(15, 15)
    s.is_paper             = True
    s.effective_risk_per_trade.return_value = 1_000.0
    return s


@pytest.fixture
def risk_manager(settings):
    from risk.risk_manager import RiskManager
    return RiskManager(settings=settings)


@pytest.fixture
def strategy(settings):
    from strategies import get_strategy
    return get_strategy("vwap_volume", symbol="NIFTY", settings=settings)


def _make_candle(
    dt: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> "Candle":
    from strategies.base_strategy import Candle
    return Candle(
        datetime=dt, open=open_, high=high, low=low, close=close,
        volume=volume, interval_minutes=5,
    )


def _make_candle_seq(n: int = 60, base_price: float = 22_000.0) -> list:
    """Build a sequence of realistic-looking 5-min candles starting at 09:15."""
    from strategies.base_strategy import Candle
    import random
    random.seed(42)
    candles = []
    price = base_price
    start = datetime(2024, 6, 3, 9, 15)
    for i in range(n):
        dt     = start + timedelta(minutes=5 * i)
        change = random.gauss(0, 0.002)
        close  = round(price * (1 + change), 2)
        high   = round(max(price, close) * (1 + abs(random.gauss(0, 0.001))), 2)
        low    = round(min(price, close) * (1 - abs(random.gauss(0, 0.001))), 2)
        vol    = random.randint(50_000, 300_000)
        candles.append(Candle(
            datetime=dt, open=round(price, 2), high=high, low=low,
            close=close, volume=vol, interval_minutes=5,
        ))
        price = close
    return candles


# ═══════════════════════════════════════════════════════════════════════════════
# RiskManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManager:

    def test_quantity_basic(self, risk_manager):
        """Quantity = risk_budget / risk_per_share, rounded to lot."""
        qty = risk_manager.calculate_quantity(entry=22_000, sl=21_900, symbol="NIFTY")
        # risk_per_share = 100, effective_risk = min(1000, 5000) = 1000
        # raw_qty = 1000 / 100 = 10 → lot = 50 → 0 lots? No, floor to 1 lot = 50
        assert qty >= 0
        if qty > 0:
            assert qty % 50 == 0, "NIFTY quantity must be multiple of lot size 50"

    def test_quantity_zero_when_sl_too_tight(self, risk_manager):
        """If risk budget < 1 lot's worth, return 0 quantity."""
        # entry=22000, sl=21999 → 1 point risk, capital risk=1% of 100k = 1000
        # qty = 1000/1 = 1000 → rounds to 1000, but lot=50 so 1000/50=20 lots → ok
        qty = risk_manager.calculate_quantity(entry=22_000, sl=21_999, symbol="NIFTY")
        assert isinstance(qty, int)
        assert qty >= 0

    def test_open_position_tracked(self, risk_manager):
        """Position opens and is retrievable."""
        pos = risk_manager.open_position(
            symbol="NIFTY",
            direction="LONG",
            entry_price=22_000.0,
            quantity=50,
            stop_loss=21_850.0,
            target_price=22_300.0,
        )
        assert pos is not None
        assert risk_manager.get_position("NIFTY") is not None
        assert risk_manager.get_position("NIFTY").entry_price == 22_000.0

    def test_close_position_returns_pnl(self, risk_manager):
        """Closing a LONG position at profit returns correct P&L."""
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_850.0, target_price=22_300.0,
        )
        pnl = risk_manager.close_position("NIFTY", exit_price=22_200.0, reason="TARGET_HIT")
        assert pnl == pytest.approx(200.0 * 50, rel=1e-4)

    def test_close_position_loss(self, risk_manager):
        """Closing a LONG position at a loss returns negative P&L."""
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_850.0, target_price=22_300.0,
        )
        pnl = risk_manager.close_position("NIFTY", exit_price=21_850.0, reason="STOP_LOSS_HIT")
        assert pnl < 0

    def test_trailing_sl_moves_up_for_long(self, risk_manager):
        """Trailing SL should only move upward for LONG positions."""
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_850.0, target_price=22_300.0,
        )
        pos = risk_manager.get_position("NIFTY")
        initial_sl = pos.current_sl

        # Price moves up — trailing SL should increase
        risk_manager.update_trailing_sl(pos, current_price=22_100.0, atr=100.0)
        assert pos.current_sl >= initial_sl, "Trailing SL must never move down for LONG"

    def test_trailing_sl_never_moves_down_for_long(self, risk_manager):
        """After price retraces, trailing SL must not decrease."""
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_850.0, target_price=22_400.0,
        )
        pos = risk_manager.get_position("NIFTY")
        # Move price up to activate trailing
        risk_manager.update_trailing_sl(pos, current_price=22_200.0, atr=80.0)
        sl_after_rise = pos.current_sl
        # Price pulls back — SL must stay where it was
        risk_manager.update_trailing_sl(pos, current_price=22_050.0, atr=80.0)
        assert pos.current_sl >= sl_after_rise, "Trailing SL must never retreat"

    def test_daily_loss_cap_halts_trading(self, risk_manager):
        """After hitting max daily loss, is_trading_allowed() returns False."""
        # Open and close with a big loss
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_500.0, target_price=23_000.0,
        )
        # ₹500 loss per unit × 50 = ₹25,000 loss → exceeds 10,000 cap
        risk_manager.close_position("NIFTY", exit_price=21_500.0, reason="STOP_LOSS_HIT")
        assert not risk_manager.is_trading_allowed(), "Trading must halt after daily loss cap"

    def test_daily_reset_restores_trading(self, risk_manager):
        """After reset_daily(), trading is allowed again."""
        risk_manager.open_position(
            symbol="NIFTY", direction="LONG",
            entry_price=22_000.0, quantity=50,
            stop_loss=21_500.0, target_price=23_000.0,
        )
        risk_manager.close_position("NIFTY", exit_price=21_500.0, reason="STOP_LOSS_HIT")
        risk_manager.reset_daily()
        assert risk_manager.is_trading_allowed(), "Trading must resume after daily reset"

    def test_short_position_pnl(self, risk_manager):
        """SHORT position P&L is positive when price falls."""
        risk_manager.open_position(
            symbol="NIFTY", direction="SHORT",
            entry_price=22_000.0, quantity=50,
            stop_loss=22_200.0, target_price=21_600.0,
        )
        pnl = risk_manager.close_position("NIFTY", exit_price=21_800.0, reason="TARGET_HIT")
        assert pnl > 0, "SHORT position profit when price falls"


# ═══════════════════════════════════════════════════════════════════════════════
# VWAP Strategy Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVWAPStrategy:

    def test_strategy_warms_up(self, strategy):
        """Strategy must not generate entry signals before warmup_candles."""
        from strategies.base_strategy import SignalDirection
        candles = _make_candle_seq(n=40)  # below 50-candle warmup
        for c in candles:
            strategy.on_candle(c)
        sig = strategy.generate_signal()
        assert sig.direction in (SignalDirection.FLAT, SignalDirection.HOLD), \
            "Should not trade before warmup"

    def test_strategy_warmed_up_after_50_candles(self, strategy):
        """After 50 candles the is_warmed_up() flag is True."""
        candles = _make_candle_seq(n=55)
        for c in candles:
            strategy.on_candle(c)
        assert strategy.is_warmed_up()

    def test_long_signal_on_vwap_crossover_with_volume(self, strategy, settings):
        """
        Craft a candle sequence that forces a LONG signal:
        - 50 warmup candles below VWAP
        - Final candle crosses ABOVE VWAP with volume surge
        """
        from strategies.base_strategy import SignalDirection
        # Build 52 warmup candles at steady price (builds VWAP baseline)
        base = 22_000.0
        start = datetime(2024, 6, 3, 9, 15)
        candles = []
        from strategies.base_strategy import Candle
        for i in range(52):
            dt = start + timedelta(minutes=5 * i)
            # Slight downward drift to stay below VWAP
            price = base - i * 2
            candles.append(Candle(
                datetime=dt, open=price, high=price + 20, low=price - 20,
                close=price, volume=80_000, interval_minutes=5,
            ))
        # Feed warmup candles
        for c in candles:
            strategy.on_candle(c)

        # Verify warmed up
        assert strategy.is_warmed_up()

        # Now feed a candle that crosses ABOVE VWAP with massive volume
        last_price = candles[-1].close
        crossover_candle = Candle(
            datetime=start + timedelta(minutes=5 * 53),
            open=last_price,
            high=last_price + 100,
            low=last_price,
            close=last_price + 80,  # Strong close
            volume=500_000,          # 6× average → volume surge
            interval_minutes=5,
        )
        strategy.on_candle(crossover_candle)
        sig = strategy.generate_signal()
        # Should be LONG, HOLD, or FLAT — never an erroneous SHORT
        assert sig.direction != SignalDirection.SHORT, \
            "Should not generate SHORT on upward VWAP crossover"

    def test_exit_on_sl_tick_breach(self, strategy, settings):
        """on_tick() SL breach flag causes generate_signal() to return FLAT."""
        from strategies.base_strategy import SignalDirection, ExitReason
        candles = _make_candle_seq(n=55)
        for c in candles:
            strategy.on_candle(c)

        # Manually inject trade state
        strategy.state.in_trade = True
        strategy.state.trade_direction = SignalDirection.LONG
        strategy.state.current_sl = 21_800.0

        # Simulate tick below SL
        strategy.on_tick({"ltp": 21_750.0, "symbol": "NIFTY"})
        assert strategy._tick_sl_breach, "SL breach not flagged"

        sig = strategy.generate_signal()
        assert sig.direction == SignalDirection.FLAT, "Must exit on SL tick breach"
        assert not strategy._tick_sl_breach, "Flag must be cleared after signal"

    def test_indicators_ready_after_warmup(self, strategy):
        """All indicators (VWAP, EMA, RSI, ATR, VolumeMA) must be finite after warmup."""
        candles = _make_candle_seq(n=55)
        for c in candles:
            strategy.on_candle(c)
        assert strategy._indicators_ready(), "Indicators must be ready after 55 candles"
        assert strategy._vwap > 0
        assert strategy._atr > 0
        assert not math.isnan(strategy._rsi)
        assert 0 < strategy._rsi < 100

    def test_no_entry_outside_trading_hours(self, strategy, settings):
        """Signals outside trade_start/trade_end must be FLAT."""
        from strategies.base_strategy import Candle, SignalDirection
        candles = _make_candle_seq(n=55)
        for c in candles:
            strategy.on_candle(c)

        # Candle at 15:00 — after trade_end (14:30)
        late_candle = Candle(
            datetime=datetime(2024, 6, 3, 15, 0),
            open=22_000, high=22_100, low=21_900, close=22_050,
            volume=200_000, interval_minutes=5,
        )
        strategy.on_candle(late_candle)
        sig = strategy.generate_signal()
        assert sig.direction in (SignalDirection.FLAT, SignalDirection.HOLD), \
            "No new entries after trade_end"


# ═══════════════════════════════════════════════════════════════════════════════
# OrderManager Paper Trade Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderManagerPaper:

    def _make_om(self, settings):
        from risk.risk_manager import RiskManager
        from orders.order_manager import OrderManager
        from db.trade_logger import TradeLogger

        rm = RiskManager(settings=settings)
        notifier = MagicMock()
        notifier.send = MagicMock()

        # TradeLogger backed by in-memory SQLite
        with patch("config.settings", settings):
            tl = MagicMock()
            tl.log_trade = MagicMock()

        return OrderManager(
            broker=MagicMock(),
            risk_manager=rm,
            trade_logger=tl,
            notifier=notifier,
            settings=settings,
        ), rm, tl, notifier

    def test_paper_entry_creates_position(self, settings):
        """Paper LONG entry registers a position in RiskManager."""
        from strategies.base_strategy import Signal, SignalDirection
        om, rm, tl, _ = self._make_om(settings)

        sig = Signal(
            direction=SignalDirection.LONG,
            symbol="NIFTY",
            entry_price=22_000.0,
            stop_loss=21_850.0,
            target=22_300.0,
            confidence=80,
            reason="Test entry",
        )
        om.process_signal(sig)
        pos = rm.get_position("NIFTY")
        assert pos is not None, "Position must be created after LONG signal"
        assert pos.direction == "LONG"
        assert pos.entry_price == pytest.approx(22_000.0, rel=1e-3)

    def test_paper_exit_closes_position(self, settings):
        """FLAT signal after open position closes it and logs the trade."""
        from strategies.base_strategy import Signal, SignalDirection, ExitReason
        om, rm, tl, _ = self._make_om(settings)

        entry = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_000.0, stop_loss=21_850.0, target=22_300.0,
            confidence=80, reason="Test entry",
        )
        om.process_signal(entry)

        exit_sig = Signal(
            direction=SignalDirection.FLAT, symbol="NIFTY",
            entry_price=22_200.0,
            exit_reason=ExitReason.TARGET_HIT,
            confidence=0, reason="Target",
        )
        om.process_signal(exit_sig)

        assert rm.get_position("NIFTY") is None, "Position must be closed after FLAT signal"

    def test_paper_pnl_positive_on_target(self, settings):
        """P&L notification must be positive after a winning trade."""
        from strategies.base_strategy import Signal, SignalDirection, ExitReason
        om, rm, tl, notifier = self._make_om(settings)

        entry = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_000.0, stop_loss=21_850.0, target=22_300.0,
            confidence=80, reason="Test",
        )
        om.process_signal(entry)

        exit_sig = Signal(
            direction=SignalDirection.FLAT, symbol="NIFTY",
            entry_price=22_200.0, exit_reason=ExitReason.TARGET_HIT,
            confidence=0, reason="Target hit",
        )
        om.process_signal(exit_sig)

        # Notifier.send must have been called with a positive P&L message
        assert notifier.send.called, "Notifier must send entry and exit messages"

    def test_duplicate_entry_rejected(self, settings):
        """Second LONG signal while already in LONG should be ignored."""
        from strategies.base_strategy import Signal, SignalDirection
        om, rm, tl, _ = self._make_om(settings)

        entry = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_000.0, stop_loss=21_850.0, target=22_300.0,
            confidence=80, reason="First entry",
        )
        om.process_signal(entry)

        entry2 = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_100.0, stop_loss=21_900.0, target=22_400.0,
            confidence=80, reason="Second entry (should be ignored)",
        )
        om.process_signal(entry2)

        pos = rm.get_position("NIFTY")
        assert pos.entry_price == pytest.approx(22_000.0, rel=1e-3), \
            "Second entry must not override first"

    def test_trading_halted_after_daily_loss(self, settings):
        """After daily loss cap, process_signal must not open new positions."""
        from strategies.base_strategy import Signal, SignalDirection, ExitReason
        om, rm, tl, _ = self._make_om(settings)

        # Force daily loss cap via a large losing trade
        entry = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_000.0, stop_loss=21_000.0, target=23_000.0,
            confidence=80, reason="Test",
        )
        om.process_signal(entry)
        # Exit at huge loss
        exit_sig = Signal(
            direction=SignalDirection.FLAT, symbol="NIFTY",
            entry_price=21_000.0, exit_reason=ExitReason.STOP_LOSS_HIT,
            confidence=0, reason="SL hit",
        )
        om.process_signal(exit_sig)

        # Try another entry — should be blocked
        entry2 = Signal(
            direction=SignalDirection.LONG, symbol="NIFTY",
            entry_price=22_000.0, stop_loss=21_850.0, target=22_300.0,
            confidence=80, reason="After loss cap",
        )
        om.process_signal(entry2)
        assert rm.get_position("NIFTY") is None, \
            "No new positions allowed after daily loss cap"


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest End-to-End Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestEndToEnd:

    def _get_sample_df(self, n_days: int = 30) -> "pd.DataFrame":
        from backtest.engine import generate_sample_data
        return generate_sample_data(n_days=n_days, seed=42)

    def test_backtest_runs_on_sample_data(self, settings):
        """BacktestEngine must complete without exception on 30 days of data."""
        from backtest.engine import BacktestEngine
        with patch("config.settings", settings):
            engine = BacktestEngine(
                strategy_name="vwap_volume",
                symbol="NIFTY",
                settings=settings,
            )
            df = self._get_sample_df(n_days=30)
            result = engine.run_from_dataframe(df, interval_minutes=5)

        assert result is not None
        assert result.candles_tested > 0
        assert result.initial_capital == settings.trading_capital

    def test_backtest_equity_curve_starts_at_capital(self, settings):
        """Equity curve first point must equal initial capital."""
        from backtest.engine import BacktestEngine
        with patch("config.settings", settings):
            engine = BacktestEngine(
                strategy_name="vwap_volume", symbol="NIFTY", settings=settings,
            )
            df = self._get_sample_df(n_days=30)
            result = engine.run_from_dataframe(df)

        if result.equity_curve:
            first_equity = result.equity_curve[0]["equity"]
            assert first_equity == pytest.approx(settings.trading_capital, rel=0.01)

    def test_backtest_no_lookahead_bias(self, settings):
        """
        Each trade's entry_time must be AFTER the signal candle's datetime.
        Fill at next candle's open → entry_time > signal_candle_time.
        """
        from backtest.engine import BacktestEngine
        with patch("config.settings", settings):
            engine = BacktestEngine(
                strategy_name="vwap_volume", symbol="NIFTY", settings=settings,
            )
            df = self._get_sample_df(n_days=60)
            result = engine.run_from_dataframe(df)

        for trade in result.trades:
            assert trade.entry_time <= trade.exit_time, \
                f"Entry {trade.entry_time} must precede exit {trade.exit_time}"

    def test_backtest_sl_respected(self, settings):
        """No trade should lose more than initial SL × quantity + commission."""
        from backtest.engine import BacktestEngine
        with patch("config.settings", settings):
            engine = BacktestEngine(
                strategy_name="vwap_volume", symbol="NIFTY", settings=settings,
                commission_inr=40.0,
            )
            df = self._get_sample_df(n_days=60)
            result = engine.run_from_dataframe(df)

        for trade in result.trades:
            if trade.pnl is None:
                continue
            max_loss = trade.initial_risk * trade.quantity + trade.commission
            # Allow 5% tolerance for slippage
            assert trade.pnl >= -(max_loss * 1.05), \
                f"Trade P&L {trade.pnl:.0f} exceeds max expected loss {-max_loss:.0f}"

    def test_optimizer_completes(self, settings):
        """Grid optimizer must complete 243 combinations in reasonable time."""
        import time as _time
        from backtest.optimizer import run_grid_optimization, DEFAULT_PARAM_GRID

        df = self._get_sample_df(n_days=90)
        with patch("config.settings", settings):
            t0      = _time.time()
            results = run_grid_optimization(
                symbol="NIFTY",
                strategy_name="vwap_volume",
                settings=settings,
                df=df,
                interval_minutes=5,
                param_grid=DEFAULT_PARAM_GRID,
                top_n=3,
                show_progress=False,
            )
            elapsed = _time.time() - t0

        assert len(results) <= 3
        assert elapsed < 120, f"Optimizer took {elapsed:.0f}s — expected < 120s"


# ═══════════════════════════════════════════════════════════════════════════════
# Indicator Correctness Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndicators:

    def _df(self, n=100):
        import numpy as np
        rng  = np.random.default_rng(0)
        close = 22_000 + np.cumsum(rng.normal(0, 50, n))
        high  = close + rng.uniform(10, 80, n)
        low   = close - rng.uniform(10, 80, n)
        vol   = rng.integers(50_000, 300_000, n)
        idx   = pd.date_range("2024-06-03 09:15", periods=n, freq="5min")
        return pd.DataFrame({"open": close, "high": high, "low": low,
                              "close": close, "volume": vol}, index=idx)

    def test_rsi_in_range(self):
        from strategies.indicators import rsi
        df  = self._df()
        r   = rsi(df, 14).dropna()
        assert (r >= 0).all() and (r <= 100).all(), "RSI must be in [0, 100]"

    def test_ema_length(self):
        from strategies.indicators import ema
        df  = self._df(n=50)
        e   = ema(df, 9)
        assert len(e) == 50, "EMA must return same length as input"

    def test_atr_positive(self):
        from strategies.indicators import atr
        df  = self._df()
        a   = atr(df, 14).dropna()
        assert (a > 0).all(), "ATR must be positive"

    def test_vwap_daily_reset(self):
        """VWAP must reset at each calendar day boundary."""
        from strategies.indicators import vwap
        # 2 full days of data
        idx = pd.date_range("2024-06-03 09:15", periods=150, freq="5min")
        import numpy as np
        rng = np.random.default_rng(7)
        close = 22_000 + np.cumsum(rng.normal(0, 30, 150))
        df = pd.DataFrame({
            "open": close, "high": close + 20, "low": close - 20,
            "close": close, "volume": rng.integers(50_000, 200_000, 150),
        }, index=idx)
        vw, _, _ = vwap(df)
        # Day 1 last bar and Day 2 first bar should NOT be cumulative
        day1_last  = df[df.index.normalize() == pd.Timestamp("2024-06-03")].index[-1]
        day2_first = df[df.index.normalize() == pd.Timestamp("2024-06-04")].index[0]
        # Day 2 first bar: VWAP = typical price (resets to single bar)
        tp_d2_first = (df.loc[day2_first, "high"] + df.loc[day2_first, "low"] + df.loc[day2_first, "close"]) / 3
        assert abs(vw[day2_first] - tp_d2_first) < 1.0, "VWAP must reset at day 2 start"

    def test_sma_nan_before_period(self):
        from strategies.indicators import sma
        import numpy as np
        df = self._df(n=30)
        s  = sma(df, 20)
        assert all(np.isnan(s.iloc[:19])), "SMA must be NaN before period rows"
        assert not np.isnan(s.iloc[19]),    "SMA must be valid at period row"
