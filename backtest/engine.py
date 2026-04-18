"""
backtest/engine.py — Event-driven backtesting engine.

Uses the SAME Strategy + RiskManager interfaces as live trading.
Replays historical OHLCV candles, simulates fills, and produces
a BacktestResult with full trade log and performance metrics.

Fill model
----------
  Entry  : signal candle close ± slippage_pct
  SL hit : exact SL price (candle.low ≤ SL for LONG, high ≥ SL for SHORT)
  Target : exact target price (candle.high ≥ target for LONG)
  Exit   : signal candle close ± slippage_pct

Intra-candle priority
---------------------
  If SL and target both triggered in the same candle, SL wins (conservative).
  Partial booking (50% at 1:1 RR) is applied before the full-exit check.

Usage
-----
    from backtest.engine import BacktestEngine

    engine = BacktestEngine("vwap_volume", symbol="NIFTY", settings=settings)

    # From broker (fetches + caches via HistoricalData)
    from datetime import date
    result = engine.run_from_broker(broker, date(2024, 1, 1), date(2024, 3, 31))

    # From CSV file (columns: datetime, open, high, low, close, volume)
    result = engine.run_from_csv("nifty_5min.csv")

    result.print_summary()
    result.to_csv("trades.csv")
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import Candle, ExitReason, Signal, SignalDirection


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """One completed round-trip trade in the backtest."""

    symbol: str
    direction: str              # "LONG" | "SHORT"
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: int
    pnl: float                  # net P&L including any partial exit
    exit_reason: str
    initial_sl: float
    target_price: float
    initial_risk: float         # |entry - initial_sl| per share
    r_multiple: float           # pnl / (initial_risk × qty)
    duration_minutes: int
    partial_pnl: float = 0.0    # P&L realised on partial (50%) exit
    commission: float = 0.0     # estimated round-trip commission

    def is_winner(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": round(self.entry_price, 2),
            "exit_time": self.exit_time.isoformat(),
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "pnl": round(self.pnl, 2),
            "exit_reason": self.exit_reason,
            "initial_sl": round(self.initial_sl, 2),
            "target_price": round(self.target_price, 2),
            "r_multiple": round(self.r_multiple, 2),
            "duration_min": self.duration_minutes,
            "partial_pnl": round(self.partial_pnl, 2),
            "commission": round(self.commission, 2),
        }


@dataclass
class BacktestResult:
    """Complete backtest output — trade log + performance metrics."""

    symbol: str
    strategy_name: str
    from_date: datetime
    to_date: datetime
    interval_minutes: int
    candles_tested: int
    initial_capital: float

    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    # ── Computed on __post_init__ ──────────────────────────────────────
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_inr: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r_multiple: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_duration_minutes: float = 0.0
    total_commission: float = 0.0
    sharpe_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def __post_init__(self) -> None:
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        if not self.trades:
            return

        pnls = [t.pnl for t in self.trades]
        winners = [t for t in self.trades if t.is_winner()]
        losers = [t for t in self.trades if not t.is_winner()]

        self.total_trades = len(self.trades)
        self.winning_trades = len(winners)
        self.losing_trades = len(losers)
        self.win_rate_pct = round(self.winning_trades / self.total_trades * 100, 1)

        self.net_pnl = round(sum(pnls), 2)
        self.gross_profit = round(sum(t.pnl for t in winners), 2)
        self.gross_loss = round(abs(sum(t.pnl for t in losers)), 2)
        self.profit_factor = round(
            self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf"), 2
        )

        self.avg_win = round(self.gross_profit / self.winning_trades, 2) if winners else 0.0
        self.avg_loss = round(self.gross_loss / self.losing_trades, 2) if losers else 0.0

        self.best_trade_pnl = round(max(pnls), 2)
        self.worst_trade_pnl = round(min(pnls), 2)
        self.avg_r_multiple = round(
            sum(t.r_multiple for t in self.trades) / self.total_trades, 2
        )
        self.avg_duration_minutes = round(
            sum(t.duration_minutes for t in self.trades) / self.total_trades, 1
        )
        self.total_commission = round(sum(t.commission for t in self.trades), 2)

        # Max drawdown from equity curve
        if self.equity_curve:
            equities = [e["equity"] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = peak - eq
                max_dd = max(max_dd, dd)
            self.max_drawdown_inr = round(max_dd, 2)
            peak_equity = max(equities) if max(equities) > 0 else self.initial_capital
            self.max_drawdown_pct = round(max_dd / peak_equity * 100, 2)

        # Sharpe ratio (annualised from daily P&L)
        daily: dict[date, float] = {}
        for t in self.trades:
            d = t.exit_time.date()
            daily[d] = daily.get(d, 0.0) + t.pnl
        if len(daily) >= 5:
            vals = list(daily.values())
            mean_daily = sum(vals) / len(vals)
            variance = sum((v - mean_daily) ** 2 for v in vals) / len(vals)
            std_daily = math.sqrt(variance)
            if std_daily > 0:
                self.sharpe_ratio = round(mean_daily / std_daily * math.sqrt(252), 2)

        # Consecutive wins / losses
        max_cw = max_cl = cur_cw = cur_cl = 0
        for t in self.trades:
            if t.is_winner():
                cur_cw += 1
                cur_cl = 0
            else:
                cur_cl += 1
                cur_cw = 0
            max_cw = max(max_cw, cur_cw)
            max_cl = max(max_cl, cur_cl)
        self.max_consecutive_wins = max_cw
        self.max_consecutive_losses = max_cl

    # ── Reporting ──────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted performance summary to console."""
        sep = "═" * 56
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS — {self.strategy_name} | {self.symbol}")
        print(sep)
        print(f"  Period          : {self.from_date.date()} → {self.to_date.date()}")
        print(f"  Interval        : {self.interval_minutes}-min candles")
        print(f"  Candles tested  : {self.candles_tested:,}")
        print(f"  Initial capital : ₹{self.initial_capital:,.0f}")
        print(sep)
        print(f"  Total trades    : {self.total_trades}")
        print(f"  Winners / Losers: {self.winning_trades} / {self.losing_trades}")
        print(f"  Win rate        : {self.win_rate_pct}%")
        print(sep)
        pnl_sign = "+" if self.net_pnl >= 0 else ""
        print(f"  Net P&L         : ₹{pnl_sign}{self.net_pnl:,.2f}")
        pct_return = self.net_pnl / self.initial_capital * 100 if self.initial_capital else 0
        print(f"  Return          : {pct_return:+.2f}%")
        print(f"  Gross profit    : ₹{self.gross_profit:,.2f}")
        print(f"  Gross loss      : ₹{self.gross_loss:,.2f}")
        print(f"  Profit factor   : {self.profit_factor}")
        print(sep)
        print(f"  Max drawdown    : ₹{self.max_drawdown_inr:,.2f} ({self.max_drawdown_pct}%)")
        print(f"  Sharpe ratio    : {self.sharpe_ratio}")
        print(sep)
        print(f"  Avg win         : ₹{self.avg_win:,.2f}")
        print(f"  Avg loss        : ₹{self.avg_loss:,.2f}")
        print(f"  Avg R-multiple  : {self.avg_r_multiple}R")
        print(f"  Best trade      : ₹{self.best_trade_pnl:,.2f}")
        print(f"  Worst trade     : ₹{self.worst_trade_pnl:,.2f}")
        print(f"  Avg duration    : {self.avg_duration_minutes:.0f} min")
        print(sep)
        print(f"  Max consec wins : {self.max_consecutive_wins}")
        print(f"  Max consec loss : {self.max_consecutive_losses}")
        print(f"  Total commission: ₹{self.total_commission:,.2f}")
        print(sep)

    def to_csv(self, path: str | Path) -> None:
        """Export trade log to a CSV file."""
        path = Path(path)
        if not self.trades:
            logger.warning("BacktestResult.to_csv: no trades to export")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].to_dict().keys())
            writer.writeheader()
            writer.writerows(t.to_dict() for t in self.trades)
        logger.info("Trade log exported → {}", path)

    def to_equity_csv(self, path: str | Path) -> None:
        """Export equity curve to a CSV file."""
        path = Path(path)
        if not self.equity_curve:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["datetime", "equity", "drawdown"])
            writer.writeheader()
            writer.writerows(self.equity_curve)
        logger.info("Equity curve exported → {}", path)


# ═════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Creates a fresh strategy + RiskManager for each run so results
    are never contaminated by prior state.

    Parameters
    ----------
    strategy_name  : registered strategy key, e.g. "vwap_volume"
    symbol         : underlying symbol, e.g. "NIFTY"
    settings       : Settings object (from config.py)
    commission_inr : flat round-trip commission per trade in ₹ (default 40)
    slippage_pct   : one-way slippage as fraction, e.g. 0.0005 = 0.05%
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        settings,
        commission_inr: float = 40.0,
        slippage_pct: float = 0.0005,
    ) -> None:
        self._strategy_name = strategy_name
        self._symbol = symbol
        self._s = settings
        self._commission_inr = commission_inr
        self._slippage_pct = slippage_pct

    # ── Public entry points ────────────────────────────────────────────

    def run_from_broker(
        self,
        broker,
        from_date: date,
        to_date: date,
        interval_minutes: int = 5,
    ) -> BacktestResult:
        """
        Fetch historical OHLCV from broker (with SQLite cache) and run backtest.

        Parameters
        ----------
        broker         : connected broker instance (ZerodhaBroker)
        from_date      : backtest start date (inclusive)
        to_date        : backtest end date (inclusive)
        interval_minutes : candle size in minutes
        """
        from data.feed import HistoricalData
        from brokers.base_broker import Exchange

        logger.info(
            "Fetching historical data: {} {} → {} ({}-min)",
            self._symbol, from_date, to_date, interval_minutes,
        )
        hist = HistoricalData(broker)
        df = hist.fetch(
            symbol=self._symbol,
            exchange=Exchange.NSE,
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            interval_minutes=interval_minutes,
        )
        if df.empty:
            raise ValueError(
                f"No historical data returned for {self._symbol} "
                f"{from_date} → {to_date}"
            )
        logger.info("Data loaded: {} rows", len(df))
        return self.run_from_dataframe(df, interval_minutes)

    def run_from_csv(
        self,
        filepath: str | Path,
        interval_minutes: int = 5,
    ) -> BacktestResult:
        """
        Load OHLCV data from a CSV file and run backtest.

        Expected columns: datetime, open, high, low, close, volume
        The datetime column may be the index or a regular column.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        # Normalise column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        # Find and parse datetime column
        dt_col = next(
            (c for c in df.columns if "date" in c or "time" in c), None
        )
        if dt_col:
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "CSV must have a datetime column or a DatetimeIndex. "
                f"Columns found: {list(df.columns)}"
            )

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        logger.info("CSV loaded: {} rows from {}", len(df), path.name)
        return self.run_from_dataframe(df, interval_minutes)

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        interval_minutes: int = 5,
    ) -> BacktestResult:
        """
        Run backtest directly from a pandas DataFrame.

        Index must be DatetimeIndex.
        Columns: open, high, low, close, volume (case-insensitive).
        """
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()

        candles: list[Candle] = []
        for idx, row in df.iterrows():
            dt = idx if isinstance(idx, datetime) else pd.Timestamp(idx).to_pydatetime()
            candles.append(Candle(
                datetime=dt,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                interval_minutes=interval_minutes,
            ))

        if not candles:
            raise ValueError("DataFrame produced zero candles")

        return self._run(candles, interval_minutes)

    # ── Core simulation loop ───────────────────────────────────────────

    def _run(self, candles: list[Candle], interval_minutes: int) -> BacktestResult:
        """
        Simulate trading over the candle list.

        Creates a fresh Strategy and RiskManager for isolation.
        """
        from risk.risk_manager import RiskManager
        from strategies import get_strategy

        strategy = get_strategy(self._strategy_name, self._symbol, self._s)
        risk = RiskManager(self._s)

        logger.info(
            "Backtest starting | {} {} | {} candles | strategy={}",
            self._symbol, interval_minutes, len(candles), self._strategy_name,
        )

        completed_trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        cumulative_pnl: float = 0.0
        peak_equity: float = self._s.trading_capital

        # Active position state (separate from strategy.state for clarity)
        _pos: Optional[_ActivePos] = None

        for candle in candles:
            # ── 1. Feed candle to strategy (updates indicators) ───────────────
            strategy.on_candle(candle)
            atr = getattr(strategy, "_atr", 0.0)

            # ── 2. Intra-candle SL / target check for open position ───────────
            exited_this_candle = False
            if _pos is not None:
                exit_reason, exit_price = self._check_intra_candle(candle, _pos)
                if exit_reason:
                    trade = self._close_position(
                        _pos, exit_price, candle.datetime,
                        exit_reason, risk,
                    )
                    completed_trades.append(trade)
                    cumulative_pnl += trade.pnl
                    equity_curve.append({
                        "datetime": candle.datetime.isoformat(),
                        "equity": round(self._s.trading_capital + cumulative_pnl, 2),
                        "drawdown": round(
                            max(0, peak_equity - (self._s.trading_capital + cumulative_pnl)), 2
                        ),
                    })
                    peak_equity = max(peak_equity, self._s.trading_capital + cumulative_pnl)

                    strategy.on_trade_exit(exit_reason, exit_price)
                    _pos = None
                    exited_this_candle = True

                else:
                    # ── Partial booking check ─────────────────────────────────
                    if not _pos.partial_booked:
                        pb_price = self._partial_trigger_price(_pos)
                        if pb_price is not None:
                            candle_best = candle.high if _pos.direction == "LONG" else candle.low
                            if (
                                (_pos.direction == "LONG" and candle_best >= pb_price)
                                or (_pos.direction == "SHORT" and candle_best <= pb_price)
                            ):
                                rm_pos = risk.get_position(self._symbol)
                                if rm_pos:
                                    booked_qty, partial_pnl = risk.apply_partial_booking(
                                        rm_pos, pb_price
                                    )
                                    if booked_qty > 0:
                                        _pos.partial_pnl += partial_pnl
                                        _pos.partial_booked = True
                                        _pos.active_quantity = rm_pos.active_quantity
                                        logger.debug(
                                            "[BT] Partial exit {} @ ₹{} P&L=₹{:.2f}",
                                            self._symbol, pb_price, partial_pnl,
                                        )

                    # ── Trailing SL update ────────────────────────────────────
                    rm_pos = risk.get_position(self._symbol)
                    if rm_pos:
                        risk.update_trailing_sl(rm_pos, candle.close, atr=atr)
                        _pos.current_sl = rm_pos.current_sl

            # ── 3. Generate strategy signal ───────────────────────────────────
            signal = strategy.generate_signal()

            # ── 4. FLAT signal → close position (if not already exited) ───────
            if signal.direction == SignalDirection.FLAT and _pos is not None and not exited_this_candle:
                exit_price = candle.close * (
                    (1 - self._slippage_pct)
                    if _pos.direction == "LONG"
                    else (1 + self._slippage_pct)
                )
                reason = signal.exit_reason or ExitReason.SIGNAL_REVERSAL
                trade = self._close_position(
                    _pos, exit_price, candle.datetime, reason, risk
                )
                completed_trades.append(trade)
                cumulative_pnl += trade.pnl
                equity_curve.append({
                    "datetime": candle.datetime.isoformat(),
                    "equity": round(self._s.trading_capital + cumulative_pnl, 2),
                    "drawdown": round(
                        max(0, peak_equity - (self._s.trading_capital + cumulative_pnl)), 2
                    ),
                })
                peak_equity = max(peak_equity, self._s.trading_capital + cumulative_pnl)
                strategy.on_trade_exit(reason, exit_price)
                _pos = None

            # ── 5. Entry signal → open position ──────────────────────────────
            elif (
                signal.direction in (SignalDirection.LONG, SignalDirection.SHORT)
                and _pos is None
                and not exited_this_candle
                and risk.is_trading_allowed()
            ):
                direction = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"
                fill_price = candle.close * (
                    (1 + self._slippage_pct)
                    if direction == "LONG"
                    else (1 - self._slippage_pct)
                )

                sl = signal.stop_loss
                target = signal.target

                # Guard against degenerate SL / target
                if direction == "LONG" and sl >= fill_price:
                    sl = fill_price * 0.99
                elif direction == "SHORT" and sl <= fill_price:
                    sl = fill_price * 1.01
                if target <= 0:
                    risk_pts = abs(fill_price - sl)
                    target = (
                        fill_price + 2 * risk_pts
                        if direction == "LONG"
                        else fill_price - 2 * risk_pts
                    )

                qty = risk.calculate_quantity(fill_price, sl, symbol=self._symbol)
                if qty <= 0:
                    continue

                rm_pos = risk.open_position(
                    symbol=self._symbol,
                    direction=direction,
                    entry_price=fill_price,
                    quantity=qty,
                    stop_loss=sl,
                    target_price=target,
                )
                if rm_pos is None:
                    continue

                _pos = _ActivePos(
                    direction=direction,
                    entry_time=candle.datetime,
                    entry_price=fill_price,
                    quantity=qty,
                    active_quantity=qty,
                    initial_sl=sl,
                    current_sl=sl,
                    target_price=target,
                )

                strategy.on_trade_entry(signal, fill_price)
                logger.debug(
                    "[BT] ENTRY {} {} @ ₹{:.2f} | SL ₹{:.2f} | Target ₹{:.2f} | qty {}",
                    direction, self._symbol, fill_price, sl, target, qty,
                )

        # ── Force-close any still-open position at last candle ────────────────
        if _pos is not None and candles:
            last = candles[-1]
            trade = self._close_position(
                _pos, last.close, last.datetime,
                ExitReason.TIME_SQUAREOFF, risk,
            )
            completed_trades.append(trade)
            cumulative_pnl += trade.pnl

        logger.info(
            "Backtest complete | {} trades | Net P&L ₹{:.2f}",
            len(completed_trades), cumulative_pnl,
        )

        return BacktestResult(
            symbol=self._symbol,
            strategy_name=self._strategy_name,
            from_date=candles[0].datetime if candles else datetime.now(),
            to_date=candles[-1].datetime if candles else datetime.now(),
            interval_minutes=interval_minutes,
            candles_tested=len(candles),
            initial_capital=self._s.trading_capital,
            trades=completed_trades,
            equity_curve=equity_curve,
        )

    # ── Intra-candle simulation helpers ───────────────────────────────

    def _check_intra_candle(
        self, candle: Candle, pos: "_ActivePos"
    ) -> tuple[Optional[ExitReason], float]:
        """
        Check if SL or target was hit during this candle.

        Returns (exit_reason, exit_price) or (None, 0.0).
        SL takes priority over target (conservative).
        """
        sl_hit = (
            (pos.direction == "LONG" and candle.low <= pos.current_sl)
            or (pos.direction == "SHORT" and candle.high >= pos.current_sl)
        )
        target_hit = (
            (pos.direction == "LONG" and candle.high >= pos.target_price)
            or (pos.direction == "SHORT" and candle.low <= pos.target_price)
        )

        if sl_hit:
            reason = (
                ExitReason.TRAILING_SL_HIT
                if pos.trail_activated
                else ExitReason.STOP_LOSS_HIT
            )
            return reason, pos.current_sl

        if target_hit:
            return ExitReason.TARGET_HIT, pos.target_price

        return None, 0.0

    def _partial_trigger_price(self, pos: "_ActivePos") -> Optional[float]:
        """Return the 1:1 RR price at which 50% partial exit should fire."""
        risk_pts = abs(pos.entry_price - pos.initial_sl)
        if risk_pts <= 0:
            return None
        if pos.direction == "LONG":
            return pos.entry_price + risk_pts
        return pos.entry_price - risk_pts

    def _close_position(
        self,
        pos: "_ActivePos",
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason,
        risk,
    ) -> BacktestTrade:
        """Create a BacktestTrade, deduct commission, close in RiskManager."""
        mult = 1.0 if pos.direction == "LONG" else -1.0
        remaining_pnl = mult * (exit_price - pos.entry_price) * pos.active_quantity
        total_pnl = pos.partial_pnl + remaining_pnl - self._commission_inr

        risk_pts = abs(pos.entry_price - pos.initial_sl)
        r_multiple = (
            round(total_pnl / (risk_pts * pos.quantity), 2)
            if risk_pts > 0 and pos.quantity > 0
            else 0.0
        )

        duration = int(
            (exit_time - pos.entry_time).total_seconds() / 60
        )

        risk.close_position(self._symbol, exit_price, reason=reason.value)

        trade = BacktestTrade(
            symbol=self._symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=round(total_pnl, 2),
            exit_reason=reason.value,
            initial_sl=pos.initial_sl,
            target_price=pos.target_price,
            initial_risk=risk_pts,
            r_multiple=r_multiple,
            duration_minutes=duration,
            partial_pnl=round(pos.partial_pnl, 2),
            commission=self._commission_inr,
        )

        logger.debug(
            "[BT] EXIT {} {} @ ₹{:.2f} | P&L ₹{:.2f} | {}R | {}",
            pos.direction, self._symbol,
            exit_price, total_pnl, r_multiple, reason.value,
        )
        return trade


# ── Internal active-position state (used only within BacktestEngine) ──────────

@dataclass
class _ActivePos:
    """Mutable backtest position state (parallel to RiskManager.TradePosition)."""
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: int
    active_quantity: int
    initial_sl: float
    current_sl: float
    target_price: float
    partial_pnl: float = 0.0
    partial_booked: bool = False
    trail_activated: bool = False
