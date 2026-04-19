"""
backtest/engine.py — Event-driven backtesting engine.

Uses the SAME Strategy + RiskManager interfaces as live trading.
Replays historical OHLCV candles, simulates fills, and produces
a BacktestResult with full trade log and performance metrics.

Fill model
----------
  Entry  : NEXT candle's open ± slippage  (no lookahead bias)
  SL hit : exact SL price  (candle.low ≤ SL for LONG)
  Target : exact target price  (candle.high ≥ target for LONG)
  Exit   : signal candle close ± slippage

Intra-candle priority
---------------------
  Partial booking (50% at 1:1 RR) checked first.
  If SL and target both trigger in same candle, SL wins (conservative).

Usage
-----
    engine = BacktestEngine("vwap_volume", symbol="NIFTY", settings=settings)

    result = engine.run_from_broker(broker, date(2024, 1, 1), date(2024, 3, 31))
    result = engine.run_from_csv("nifty_5min.csv")

    result.print_summary()
    result.monthly_table()
    result.plot_charts("output/")
    result.to_csv("trades.csv")
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import Candle, ExitReason, Signal, SignalDirection

# India risk-free rate (RBI repo rate proxy)
_INDIA_RF_ANNUAL = 0.065
_INDIA_RF_DAILY = _INDIA_RF_ANNUAL / 252


# ═════════════════════════════════════════════════════════════════════════════
# BacktestTrade
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """One completed round-trip trade."""

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
    initial_risk: float         # |entry - initial_sl| per unit
    r_multiple: float           # pnl / (initial_risk × qty)
    duration_minutes: int
    partial_pnl: float = 0.0
    commission: float = 0.0

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


# ═════════════════════════════════════════════════════════════════════════════
# BacktestResult
# ═════════════════════════════════════════════════════════════════════════════

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
    # Per-candle equity curve: [{datetime, equity, drawdown}]
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
    cagr_pct: float = 0.0
    max_drawdown_inr: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_days: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r_multiple: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_duration_minutes: float = 0.0
    trades_per_day: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    total_commission: float = 0.0
    sharpe_ratio: float = 0.0       # without risk-free rate
    sharpe_ratio_rf: float = 0.0    # with India Rf = 6.5%
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    monthly_pnl: dict = field(default_factory=dict)   # "YYYY-MM" → float

    def __post_init__(self) -> None:
        self._compute_metrics()

    # ── Metrics ────────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        if not self.trades:
            return

        pnls = [t.pnl for t in self.trades]
        winners = [t for t in self.trades if t.is_winner()]
        losers  = [t for t in self.trades if not t.is_winner()]

        self.total_trades   = len(self.trades)
        self.winning_trades = len(winners)
        self.losing_trades  = len(losers)
        self.win_rate_pct   = round(self.winning_trades / self.total_trades * 100, 1)

        self.net_pnl       = round(sum(pnls), 2)
        self.gross_profit  = round(sum(t.pnl for t in winners), 2)
        self.gross_loss    = round(abs(sum(t.pnl for t in losers)), 2)
        self.profit_factor = round(
            self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf"), 2
        )

        self.avg_win  = round(self.gross_profit / self.winning_trades, 2) if winners else 0.0
        self.avg_loss = round(self.gross_loss  / self.losing_trades,  2) if losers  else 0.0

        self.best_trade_pnl  = round(max(pnls), 2)
        self.worst_trade_pnl = round(min(pnls), 2)
        self.avg_r_multiple  = round(
            sum(t.r_multiple for t in self.trades) / self.total_trades, 2
        )
        self.avg_duration_minutes = round(
            sum(t.duration_minutes for t in self.trades) / self.total_trades, 1
        )
        self.total_commission = round(sum(t.commission for t in self.trades), 2)

        # Daily P&L grouping
        daily: dict[date, float] = defaultdict(float)
        for t in self.trades:
            daily[t.exit_time.date()] += t.pnl

        if daily:
            trading_days = len(daily)
            self.trades_per_day = round(self.total_trades / trading_days, 1)
            self.best_day_pnl  = round(max(daily.values()), 2)
            self.worst_day_pnl = round(min(daily.values()), 2)

        # Monthly P&L
        monthly: dict[str, float] = defaultdict(float)
        for t in self.trades:
            monthly[t.exit_time.strftime("%Y-%m")] += t.pnl
        self.monthly_pnl = {k: round(v, 2) for k, v in sorted(monthly.items())}

        # CAGR
        years = max((self.to_date - self.from_date).days / 365.25, 1 / 252)
        if self.initial_capital > 0:
            final = self.initial_capital + self.net_pnl
            if final > 0:
                self.cagr_pct = round(((final / self.initial_capital) ** (1 / years) - 1) * 100, 2)

        # Sharpe (with and without Rf)
        if len(daily) >= 5:
            all_days_pnl = list(daily.values())
            mean_d = sum(all_days_pnl) / len(all_days_pnl)
            std_d  = math.sqrt(
                sum((v - mean_d) ** 2 for v in all_days_pnl) / len(all_days_pnl)
            )
            if std_d > 0:
                self.sharpe_ratio    = round(mean_d / std_d * math.sqrt(252), 2)
                # Excess return over risk-free rate
                cap = self.initial_capital or 1.0
                daily_returns = [p / cap for p in all_days_pnl]
                excess = [r - _INDIA_RF_DAILY for r in daily_returns]
                mean_ex = sum(excess) / len(excess)
                std_ex  = math.sqrt(sum((r - mean_ex) ** 2 for r in excess) / len(excess))
                if std_ex > 0:
                    self.sharpe_ratio_rf = round(mean_ex / std_ex * math.sqrt(252), 2)

        # Max drawdown from equity curve (value, %, and duration in days)
        if self.equity_curve:
            equities   = [e["equity"]   for e in self.equity_curve]
            timestamps = [e["datetime"] for e in self.equity_curve]

            peak_eq    = equities[0]
            peak_dt    = timestamps[0]
            max_dd     = 0.0
            max_dd_dur = 0

            for eq, ts in zip(equities, timestamps):
                if eq > peak_eq:
                    peak_eq = eq
                    peak_dt = ts
                dd = peak_eq - eq
                if dd > max_dd:
                    max_dd = dd
                    try:
                        dur = (
                            datetime.fromisoformat(ts) - datetime.fromisoformat(peak_dt)
                        ).days
                        max_dd_dur = max(max_dd_dur, dur)
                    except Exception:
                        pass

            self.max_drawdown_inr  = round(max_dd, 2)
            self.max_drawdown_days = max_dd_dur
            if max(equities) > 0:
                self.max_drawdown_pct = round(max_dd / max(equities) * 100, 2)

        # Consecutive wins / losses
        max_cw = max_cl = cur_cw = cur_cl = 0
        for t in self.trades:
            if t.is_winner():
                cur_cw += 1; cur_cl = 0
            else:
                cur_cl += 1; cur_cw = 0
            max_cw = max(max_cw, cur_cw)
            max_cl = max(max_cl, cur_cl)
        self.max_consecutive_wins   = max_cw
        self.max_consecutive_losses = max_cl

    # ── Console output ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print formatted performance report to console."""
        sep = "═" * 58
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS — {self.strategy_name} | {self.symbol}")
        print(sep)
        print(f"  Period           : {self.from_date.date()} → {self.to_date.date()}")
        print(f"  Interval         : {self.interval_minutes}-min candles")
        print(f"  Candles tested   : {self.candles_tested:,}")
        print(f"  Initial capital  : ₹{self.initial_capital:,.0f}")
        print(sep)
        print(f"  Total trades     : {self.total_trades}")
        print(f"  Winners / Losers : {self.winning_trades} / {self.losing_trades}")
        print(f"  Win rate         : {self.win_rate_pct}%")
        print(f"  Trades / day     : {self.trades_per_day}")
        print(sep)
        sign = "+" if self.net_pnl >= 0 else ""
        ret  = self.net_pnl / self.initial_capital * 100 if self.initial_capital else 0
        print(f"  Net P&L          : ₹{sign}{self.net_pnl:,.2f}  ({ret:+.2f}%)")
        print(f"  CAGR             : {self.cagr_pct:+.2f}%")
        print(f"  Gross profit     : ₹{self.gross_profit:,.2f}")
        print(f"  Gross loss       : ₹{self.gross_loss:,.2f}")
        print(f"  Profit factor    : {self.profit_factor}")
        print(sep)
        print(f"  Max drawdown     : ₹{self.max_drawdown_inr:,.2f}  ({self.max_drawdown_pct}%)")
        print(f"  Max DD duration  : {self.max_drawdown_days} days")
        print(f"  Sharpe (no Rf)   : {self.sharpe_ratio}")
        print(f"  Sharpe (Rf 6.5%) : {self.sharpe_ratio_rf}")
        print(sep)
        print(f"  Avg win          : ₹{self.avg_win:,.2f}")
        print(f"  Avg loss         : ₹{self.avg_loss:,.2f}")
        print(f"  Avg R-multiple   : {self.avg_r_multiple}R")
        print(f"  Best trade       : ₹{self.best_trade_pnl:,.2f}")
        print(f"  Worst trade      : ₹{self.worst_trade_pnl:,.2f}")
        print(f"  Avg duration     : {self.avg_duration_minutes:.0f} min")
        print(sep)
        print(f"  Best day         : ₹{self.best_day_pnl:,.2f}")
        print(f"  Worst day        : ₹{self.worst_day_pnl:,.2f}")
        print(f"  Max consec wins  : {self.max_consecutive_wins}")
        print(f"  Max consec loss  : {self.max_consecutive_losses}")
        print(f"  Total commission : ₹{self.total_commission:,.2f}")
        print(sep)
        if self.monthly_pnl:
            self.monthly_table()

    def monthly_table(self) -> None:
        """Print monthly P&L breakdown."""
        if not self.monthly_pnl:
            return

        # Count trades per month
        month_trades: dict[str, int] = defaultdict(int)
        for t in self.trades:
            month_trades[t.exit_time.strftime("%Y-%m")] += 1

        print(f"\n  {'Month':<10}  {'P&L':>12}  {'Trades':>7}  {'Result'}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*6}")
        for month, pnl in sorted(self.monthly_pnl.items()):
            sign   = "+" if pnl >= 0 else ""
            result = "✅" if pnl >= 0 else "❌"
            n      = month_trades.get(month, 0)
            print(f"  {month:<10}  ₹{sign}{pnl:>10,.2f}  {n:>7}  {result}")

        total = sum(self.monthly_pnl.values())
        pos   = sum(1 for v in self.monthly_pnl.values() if v >= 0)
        neg   = len(self.monthly_pnl) - pos
        print(f"  {'─'*10}  {'─'*12}  {'─'*7}")
        print(f"  {'TOTAL':<10}  ₹{total:>+10,.2f}  {pos} up / {neg} down")
        print()

    # ── Chart generation ───────────────────────────────────────────────

    def plot_charts(self, output_dir: str = ".") -> Optional[str]:
        """
        Generate equity curve and drawdown charts using matplotlib.
        Returns the output path, or None if matplotlib is not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping charts. "
                "Run: pip install matplotlib"
            )
            return None

        if not self.equity_curve:
            logger.warning("plot_charts: equity curve is empty")
            return None

        dates      = [datetime.fromisoformat(e["datetime"]) for e in self.equity_curve]
        equities   = [e["equity"]   for e in self.equity_curve]
        drawdowns  = [e["drawdown"] for e in self.equity_curve]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(
            f"Backtest — {self.strategy_name} | {self.symbol}  "
            f"({self.from_date.date()} → {self.to_date.date()})",
            fontsize=12,
        )

        # ── Equity curve ───────────────────────────────────────────────
        ax1.plot(dates, equities, color="#1976D2", linewidth=1.2, label="Equity")
        ax1.axhline(
            y=self.initial_capital, color="gray",
            linestyle="--", alpha=0.6, linewidth=0.8, label="Initial capital",
        )
        ax1.fill_between(
            dates, self.initial_capital, equities,
            where=[e >= self.initial_capital for e in equities],
            alpha=0.15, color="green",
        )
        ax1.fill_between(
            dates, self.initial_capital, equities,
            where=[e < self.initial_capital for e in equities],
            alpha=0.15, color="red",
        )
        ax1.set_ylabel("Portfolio Value (₹)")
        ax1.set_title(
            f"Net P&L ₹{self.net_pnl:+,.0f}  |  CAGR {self.cagr_pct:+.1f}%  |  "
            f"Sharpe(Rf) {self.sharpe_ratio_rf}",
            fontsize=10,
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
        )

        # ── Drawdown ───────────────────────────────────────────────────
        dd_neg = [-d for d in drawdowns]
        ax2.fill_between(dates, 0, dd_neg, color="#D32F2F", alpha=0.5, label="Drawdown")
        ax2.plot(dates, dd_neg, color="#D32F2F", linewidth=0.6)
        ax2.set_ylabel("Drawdown (₹)")
        ax2.set_title(
            f"Max Drawdown ₹{self.max_drawdown_inr:,.0f}  ({self.max_drawdown_pct}%)  "
            f"| Duration {self.max_drawdown_days} days",
            fontsize=10,
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
        )

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        out_path = Path(output_dir) / "equity_curve.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved → {}", out_path)
        return str(out_path)

    # ── Export ─────────────────────────────────────────────────────────

    def to_csv(self, path: str | Path) -> None:
        """Export trade-by-trade log to CSV."""
        path = Path(path)
        if not self.trades:
            logger.warning("BacktestResult.to_csv: no trades to export")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].to_dict().keys())
            writer.writeheader()
            writer.writerows(t.to_dict() for t in self.trades)
        logger.info("Trade log → {}", path)

    def to_equity_csv(self, path: str | Path) -> None:
        """Export per-candle equity curve to CSV."""
        path = Path(path)
        if not self.equity_curve:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["datetime", "equity", "drawdown"])
            writer.writeheader()
            writer.writerows(self.equity_curve)
        logger.info("Equity curve → {}", path)


# ═════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Creates a fresh Strategy + RiskManager for each run to prevent
    state contamination between runs.

    Parameters
    ----------
    strategy_name    : registered key, e.g. "vwap_volume"
    symbol           : underlying symbol, e.g. "NIFTY"
    settings         : Settings object from config.py
    commission_inr   : flat round-trip commission per trade (₹20 entry + ₹20 exit = ₹40)
    slippage_pct     : one-way slippage fraction (0.0005 = 0.05%)
    next_candle_fill : True → fill at next candle open (no lookahead bias) [recommended]
    strategy_config  : dict of VWAPVolumeConfig attribute overrides (used by optimizer)
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        settings,
        commission_inr: float = 40.0,
        slippage_pct: float = 0.0005,
        next_candle_fill: bool = True,
        strategy_config: Optional[dict] = None,
    ) -> None:
        self._strategy_name = strategy_name
        self._symbol        = symbol
        self._s             = settings
        self._commission    = commission_inr
        self._slippage      = slippage_pct
        self._next_fill     = next_candle_fill
        self._cfg_overrides = strategy_config or {}

    # ── Public entry points ────────────────────────────────────────────

    def run_from_broker(
        self,
        broker,
        from_date: date,
        to_date: date,
        interval_minutes: int = 5,
    ) -> BacktestResult:
        """Fetch via HistoricalData (SQLite-cached) and run backtest."""
        from data.feed import HistoricalData
        from brokers.base_broker import Exchange

        hist = HistoricalData(broker)
        df   = hist.fetch(
            symbol=self._symbol,
            exchange=Exchange.NSE,
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            interval_minutes=interval_minutes,
        )
        if df.empty:
            raise ValueError(
                f"No data for {self._symbol} {from_date} → {to_date}"
            )
        return self.run_from_dataframe(df, interval_minutes)

    def run_from_csv(
        self, filepath: str | Path, interval_minutes: int = 5
    ) -> BacktestResult:
        """Load OHLCV CSV and run backtest. Columns: datetime, open, high, low, close, volume."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        if dt_col:
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"CSV must have a datetime column. Found: {list(df.columns)}"
            )

        missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        return self.run_from_dataframe(df, interval_minutes)

    def run_from_dataframe(
        self, df: pd.DataFrame, interval_minutes: int = 5
    ) -> BacktestResult:
        """Run backtest from a pre-loaded DataFrame (DatetimeIndex required)."""
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
        return self._run(candles, interval_minutes, full_df=df)

    # ── Core simulation loop ───────────────────────────────────────────

    def _run(
        self,
        candles: list[Candle],
        interval_minutes: int,
        full_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Feed candles one-by-one through strategy, simulate fills and exits."""
        from risk.risk_manager import RiskManager
        from strategies import get_strategy

        # Fresh instances for clean run
        strategy = get_strategy(self._strategy_name, self._symbol, self._s)
        # Apply parameter overrides (e.g., from optimizer grid search)
        if self._cfg_overrides and hasattr(strategy, "cfg"):
            for key, val in self._cfg_overrides.items():
                if hasattr(strategy.cfg, key):
                    setattr(strategy.cfg, key, val)

        # Precompute indicators once over full history — massive speedup
        # for long backtests and grid optimization (O(n²) → O(n)).
        if full_df is not None and hasattr(strategy, "precompute_indicators"):
            strategy.precompute_indicators(full_df)

        risk = RiskManager(self._s)

        logger.info(
            "Backtest | {} | {}-min | {} candles | fills={}",
            self._symbol, interval_minutes, len(candles),
            "next-open" if self._next_fill else "signal-close",
        )

        completed: list[BacktestTrade] = []
        equity_curve: list[dict]       = []
        realised_pnl: float            = 0.0
        peak_equity: float             = self._s.trading_capital

        _pos: Optional[_ActivePos]     = None
        _pending: Optional[Signal]     = None   # entry signal waiting for next candle
        _last_date: Optional[date]     = None   # tracks trading-day boundary

        for candle in candles:
            exited_this = False

            # ── 0. Reset daily risk counters at start of each new trading day ─
            candle_date = candle.datetime.date()
            if candle_date != _last_date:
                risk.reset_daily()
                _last_date = candle_date

            # ── 1. Fill pending entry signal at this candle's OPEN ──────────
            if _pending is not None and _pos is None:
                direction = "LONG" if _pending.direction == SignalDirection.LONG else "SHORT"
                slip      = self._slippage if direction == "LONG" else -self._slippage
                fill      = round(candle.open * (1 + slip), 2)

                sl     = _pending.stop_loss
                target = _pending.target

                if direction == "LONG"  and sl >= fill: sl = round(fill * 0.99, 2)
                if direction == "SHORT" and sl <= fill: sl = round(fill * 1.01, 2)
                if target <= 0:
                    risk_pts = abs(fill - sl)
                    target = (fill + 2 * risk_pts) if direction == "LONG" else (fill - 2 * risk_pts)

                qty = risk.calculate_quantity(fill, sl, symbol=self._symbol)
                if qty > 0:
                    rm_pos = risk.open_position(
                        symbol=self._symbol,
                        direction=direction,
                        entry_price=fill,
                        quantity=qty,
                        stop_loss=sl,
                        target_price=target,
                    )
                    if rm_pos is not None:
                        _pos = _ActivePos(
                            direction=direction,
                            entry_time=candle.datetime,
                            entry_price=fill,
                            quantity=qty,
                            active_quantity=qty,
                            initial_sl=sl,
                            current_sl=sl,
                            target_price=target,
                        )
                        strategy.on_trade_entry(_pending, fill)
                        logger.debug(
                            "[BT] ENTRY {} @ ₹{:.2f} | SL ₹{:.2f} | T ₹{:.2f} | qty {}",
                            direction, fill, sl, target, qty,
                        )
                _pending = None

            # ── 2. Feed candle to strategy ──────────────────────────────────
            strategy.on_candle(candle)
            atr = getattr(strategy, "_atr", 0.0)

            # ── 3. Intra-candle SL / target check ──────────────────────────
            if _pos is not None:
                reason, ep = self._check_intra_candle(candle, _pos)

                if reason is None and not _pos.partial_booked:
                    # Check partial booking at 1:1 RR
                    pb = self._partial_trigger(candle, _pos)
                    if pb is not None:
                        rm_pos = risk.get_position(self._symbol)
                        if rm_pos:
                            bqty, ppnl = risk.apply_partial_booking(rm_pos, pb)
                            if bqty > 0:
                                _pos.partial_pnl    += ppnl
                                _pos.partial_booked  = True
                                _pos.active_quantity = rm_pos.active_quantity
                                logger.debug(
                                    "[BT] PARTIAL @ ₹{:.2f} P&L ₹{:.2f}", pb, ppnl
                                )

                if reason is not None:
                    trade = self._close_pos(_pos, ep, candle.datetime, reason, risk)
                    realised_pnl += trade.pnl
                    completed.append(trade)
                    strategy.on_trade_exit(reason, ep)
                    _pos = None
                    exited_this = True
                else:
                    # Update trailing SL
                    rm_pos = risk.get_position(self._symbol)
                    if rm_pos:
                        risk.update_trailing_sl(rm_pos, candle.close, atr=atr)
                        _pos.current_sl     = rm_pos.current_sl
                        _pos.trail_activated = rm_pos.trail_activated

            # ── 4. Generate strategy signal ─────────────────────────────────
            signal = strategy.generate_signal()

            # ── 5. FLAT signal → exit at close ──────────────────────────────
            if signal.direction == SignalDirection.FLAT and _pos is not None and not exited_this:
                slip   = self._slippage if _pos.direction == "LONG" else -self._slippage
                ep     = round(candle.close * (1 - slip), 2)
                reason = signal.exit_reason or ExitReason.SIGNAL_REVERSAL
                trade  = self._close_pos(_pos, ep, candle.datetime, reason, risk)
                realised_pnl += trade.pnl
                completed.append(trade)
                strategy.on_trade_exit(reason, ep)
                _pos = None

            # ── 6. Entry signal → buffer (next-open) or fill immediately ────
            elif (
                signal.direction in (SignalDirection.LONG, SignalDirection.SHORT)
                and _pos is None
                and not exited_this
                and risk.is_trading_allowed()
            ):
                if self._next_fill:
                    _pending = signal                  # fill at NEXT candle's open
                else:
                    # Legacy: fill immediately at close
                    direction = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"
                    slip      = self._slippage if direction == "LONG" else -self._slippage
                    fill      = round(candle.close * (1 + slip), 2)
                    sl, target = signal.stop_loss, signal.target
                    if direction == "LONG"  and sl >= fill: sl = round(fill * 0.99, 2)
                    if direction == "SHORT" and sl <= fill: sl = round(fill * 1.01, 2)
                    if target <= 0:
                        rp = abs(fill - sl)
                        target = (fill + 2*rp) if direction == "LONG" else (fill - 2*rp)
                    qty = risk.calculate_quantity(fill, sl, symbol=self._symbol)
                    if qty > 0:
                        rm_pos = risk.open_position(
                            symbol=self._symbol, direction=direction,
                            entry_price=fill, quantity=qty,
                            stop_loss=sl, target_price=target,
                        )
                        if rm_pos:
                            _pos = _ActivePos(
                                direction=direction,
                                entry_time=candle.datetime,
                                entry_price=fill,
                                quantity=qty,
                                active_quantity=qty,
                                initial_sl=sl,
                                current_sl=sl,
                                target_price=target,
                            )
                            strategy.on_trade_entry(signal, fill)

            # ── 7. Per-candle equity update (realized + unrealized) ──────────
            unrealized = 0.0
            if _pos is not None:
                mult       = 1.0 if _pos.direction == "LONG" else -1.0
                unrealized = mult * (candle.close - _pos.entry_price) * _pos.active_quantity

            current_eq = self._s.trading_capital + realised_pnl + unrealized
            peak_equity = max(peak_equity, current_eq)
            equity_curve.append({
                "datetime": candle.datetime.isoformat(),
                "equity":   round(current_eq, 2),
                "drawdown": round(max(0.0, peak_equity - current_eq), 2),
            })

        # ── Force-close any open position at last candle ─────────────────────
        if _pos is not None and candles:
            last  = candles[-1]
            trade = self._close_pos(
                _pos, last.close, last.datetime, ExitReason.TIME_SQUAREOFF, risk
            )
            realised_pnl += trade.pnl
            completed.append(trade)

        logger.info(
            "Backtest done | {} trades | Net P&L ₹{:.2f}",
            len(completed), realised_pnl,
        )

        return BacktestResult(
            symbol=self._symbol,
            strategy_name=self._strategy_name,
            from_date=candles[0].datetime,
            to_date=candles[-1].datetime,
            interval_minutes=interval_minutes,
            candles_tested=len(candles),
            initial_capital=self._s.trading_capital,
            trades=completed,
            equity_curve=equity_curve,
        )

    # ── Simulation helpers ─────────────────────────────────────────────

    def _check_intra_candle(
        self, candle: Candle, pos: "_ActivePos"
    ) -> tuple[Optional[ExitReason], float]:
        """Return (exit_reason, price) if SL or target was crossed this candle."""
        sl_hit = (
            (pos.direction == "LONG"  and candle.low  <= pos.current_sl)
            or (pos.direction == "SHORT" and candle.high >= pos.current_sl)
        )
        tgt_hit = (
            (pos.direction == "LONG"  and candle.high >= pos.target_price)
            or (pos.direction == "SHORT" and candle.low  <= pos.target_price)
        )

        if sl_hit:
            reason = ExitReason.TRAILING_SL_HIT if pos.trail_activated else ExitReason.STOP_LOSS_HIT
            return reason, pos.current_sl

        if tgt_hit:
            return ExitReason.TARGET_HIT, pos.target_price

        return None, 0.0

    def _partial_trigger(
        self, candle: Candle, pos: "_ActivePos"
    ) -> Optional[float]:
        """Return the partial-exit price if 1:1 RR was reached this candle, else None."""
        risk_pts = abs(pos.entry_price - pos.initial_sl)
        if risk_pts <= 0:
            return None
        if pos.direction == "LONG":
            pb_price = pos.entry_price + risk_pts
            return pb_price if candle.high >= pb_price else None
        else:
            pb_price = pos.entry_price - risk_pts
            return pb_price if candle.low <= pb_price else None

    def _close_pos(
        self,
        pos: "_ActivePos",
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason,
        risk,
    ) -> BacktestTrade:
        mult          = 1.0 if pos.direction == "LONG" else -1.0
        remaining_pnl = mult * (exit_price - pos.entry_price) * pos.active_quantity
        total_pnl     = pos.partial_pnl + remaining_pnl - self._commission

        risk_pts  = abs(pos.entry_price - pos.initial_sl)
        r_mult    = round(total_pnl / (risk_pts * pos.quantity), 2) if risk_pts > 0 and pos.quantity > 0 else 0.0
        duration  = int((exit_time - pos.entry_time).total_seconds() / 60)

        risk.close_position(self._symbol, exit_price, reason=reason.value)

        logger.debug(
            "[BT] EXIT {} @ ₹{:.2f} P&L ₹{:.2f} {}R | {}",
            pos.direction, exit_price, total_pnl, r_mult, reason.value,
        )

        return BacktestTrade(
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
            r_multiple=r_mult,
            duration_minutes=duration,
            partial_pnl=round(pos.partial_pnl, 2),
            commission=self._commission,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Internal active-position state
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _ActivePos:
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


# ═════════════════════════════════════════════════════════════════════════════
# Sample data generator (offline testing without a broker)
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample_data(
    n_days: int = 90,
    interval_minutes: int = 5,
    starting_price: float = 22_000.0,
    daily_drift: float = 0.0003,       # slight upward drift
    daily_volatility: float = 0.012,   # ~1.2% daily vol (realistic for Nifty)
    output_csv: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic NSE-like OHLCV data using geometric Brownian motion
    with intraday patterns (higher vol at open/close).

    Returns a DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    Optionally saves to CSV if output_csv is specified.

    Use this when you don't have real historical data to test the engine.
    """
    random.seed(seed)

    candles_per_day   = 375 // interval_minutes   # NSE session = 9:15–15:30 = 375 min
    total_candles     = n_days * candles_per_day
    candle_vol        = daily_volatility / math.sqrt(candles_per_day)
    candle_drift      = daily_drift / candles_per_day

    rows  = []
    price = starting_price
    start = datetime(2024, 1, 1, 9, 15)

    for day in range(n_days):
        # Skip weekends
        day_dt = start + timedelta(days=day)
        if day_dt.weekday() >= 5:
            continue

        daily_open = price
        day_volume_scale = random.gauss(1.0, 0.3)  # some days more active

        for bar in range(candles_per_day):
            t = day_dt + timedelta(minutes=bar * interval_minutes)

            # Intraday volatility scaling (higher at open/close)
            bar_pct = bar / candles_per_day
            vol_mult = 1.5 if bar_pct < 0.1 or bar_pct > 0.9 else 1.0

            ret   = candle_drift + candle_vol * vol_mult * random.gauss(0, 1)
            close = round(price * (1 + ret), 2)
            open_ = round(price, 2)
            high  = round(max(open_, close) * (1 + abs(random.gauss(0, candle_vol * 0.5))), 2)
            low   = round(min(open_, close) * (1 - abs(random.gauss(0, candle_vol * 0.5))), 2)
            vol   = max(1, int(random.gauss(50_000, 15_000) * day_volume_scale))

            rows.append({
                "datetime": t,
                "open": open_, "high": high, "low": low, "close": close,
                "volume": vol,
            })
            price = close

    df = pd.DataFrame(rows)
    df = df.set_index("datetime")

    if output_csv:
        df.to_csv(output_csv)
        logger.info("Sample data saved → {} ({} rows)", output_csv, len(df))

    return df
