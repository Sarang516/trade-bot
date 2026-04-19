"""
db/trade_logger.py — SQLite trade journal.

Tables
------
trades          Round-trip trade log (entry + exit in one row).
daily_summary   Computed EOD stats — call compute_daily_summary() at market close.
bot_logs        Structured log stream written by dashboard / main loop.
config_history  Before/after record of every parameter change.

Usage (from OrderManager / dashboard)
--------------------------------------
    tl = TradeLogger()
    tl.log_trade({"type": "ENTRY", "symbol": ..., "price": ..., ...})
    tl.log_trade({"type": "EXIT",  "symbol": ..., "price": ..., "pnl": ...})
    tl.get_today_summary()
    tl.get_daily_pnl(days=30)
    tl.get_equity_curve()
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator, Optional

from loguru import logger
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer, String, Text,
    create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ── ORM models ────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _Trade(_Base):
    __tablename__ = "trades"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    date        = Column(Date,    nullable=False, index=True)
    symbol      = Column(String(32), nullable=False)
    direction   = Column(String(8),  nullable=False)   # LONG / SHORT
    entry_price = Column(Float,  nullable=False)
    exit_price  = Column(Float)
    quantity    = Column(Integer, default=0)
    pnl         = Column(Float)
    entry_time  = Column(DateTime, nullable=False, index=True)
    exit_time   = Column(DateTime, index=True)
    exit_reason = Column(String(32))
    initial_sl  = Column(Float)
    target      = Column(Float)
    strategy_params = Column(Text)   # JSON — confidence, broker_symbol, etc.
    commission  = Column(Float, default=0.0)
    is_paper    = Column(Boolean, default=True)


class _DailySummary(_Base):
    __tablename__ = "daily_summary"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    date            = Column(Date, nullable=False, unique=True, index=True)
    total_trades    = Column(Integer, default=0)
    winning_trades  = Column(Integer, default=0)
    losing_trades   = Column(Integer, default=0)
    gross_profit    = Column(Float,   default=0.0)
    gross_loss      = Column(Float,   default=0.0)
    net_pnl         = Column(Float,   default=0.0)
    win_rate_pct    = Column(Float,   default=0.0)
    best_trade_pnl  = Column(Float,   default=0.0)
    worst_trade_pnl = Column(Float,   default=0.0)


class _BotLog(_Base):
    __tablename__ = "bot_logs"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level     = Column(String(16), nullable=False)
    message   = Column(Text, nullable=False)


class _ConfigHistory(_Base):
    __tablename__ = "config_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
    param_name = Column(String(64), nullable=False)
    old_value  = Column(Text)
    new_value  = Column(Text)
    changed_by = Column(String(32), default="dashboard")


# ── TradeLogger ───────────────────────────────────────────────────────────────

class TradeLogger:
    """Thread-safe SQLite trade journal."""

    def __init__(self) -> None:
        from config import settings
        db_path = settings.log_dir / "trades.db"
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._lock = threading.Lock()
        # symbol → open trade id — used to match EXIT to its ENTRY row
        self._open_ids: dict[str, int] = {}

    # ── Session helper ─────────────────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._lock:
            s = self._Session()
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

    # ── Trade lifecycle ────────────────────────────────────────────────────────

    def log_trade(self, data: dict) -> None:
        """
        Central entry point — called by OrderManager with type=ENTRY/EXIT/PARTIAL_EXIT.

        ENTRY        → inserts a new row; records id in _open_ids[symbol].
        EXIT         → updates the open row with exit price, P&L, reason.
        PARTIAL_EXIT → logged to bot_logs only (main row stays open).
        """
        trade_type = data.get("type", "")

        if trade_type == "ENTRY":
            entry_time = _parse_dt(data.get("timestamp")) or datetime.now()
            with self._session() as s:
                trade = _Trade(
                    date=entry_time.date(),
                    symbol=data.get("symbol", ""),
                    direction=data.get("direction", "LONG"),
                    entry_price=data.get("price", 0.0),
                    quantity=data.get("quantity", 0),
                    initial_sl=data.get("sl"),
                    target=data.get("target"),
                    entry_time=entry_time,
                    strategy_params=json.dumps({
                        "confidence": data.get("confidence"),
                        "broker_symbol": data.get("broker_symbol"),
                    }),
                    is_paper=(data.get("mode", "PAPER") == "PAPER"),
                )
                s.add(trade)
                s.flush()
                self._open_ids[data.get("symbol", "")] = trade.id

        elif trade_type == "EXIT":
            exit_time = _parse_dt(data.get("timestamp")) or datetime.now()
            symbol    = data.get("symbol", "")
            trade_id  = self._open_ids.pop(symbol, None)

            with self._session() as s:
                trade = (
                    s.get(_Trade, trade_id) if trade_id else
                    s.query(_Trade)
                     .filter(_Trade.symbol == symbol, _Trade.exit_time.is_(None))
                     .order_by(_Trade.entry_time.desc())
                     .first()
                )
                if trade:
                    trade.exit_price  = data.get("price")
                    trade.exit_time   = exit_time
                    trade.pnl         = data.get("pnl")
                    trade.exit_reason = data.get("reason", "UNKNOWN")
                else:
                    logger.warning("TradeLogger: no open trade found for {}", symbol)

        elif trade_type == "PARTIAL_EXIT":
            self.log_bot_event(
                "INFO",
                f"PARTIAL_EXIT {data.get('symbol')} qty={data.get('partial_qty')} "
                f"pnl=₹{data.get('partial_pnl')}",
            )

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_trades(
        self,
        from_date: Optional[date] = None,
        to_date:   Optional[date] = None,
        symbol:    Optional[str]  = None,
        direction: Optional[str]  = None,
        limit:     int = 200,
    ) -> list[dict]:
        with self._session() as s:
            q = s.query(_Trade).filter(_Trade.exit_time.isnot(None))
            if from_date:
                q = q.filter(_Trade.date >= from_date)
            if to_date:
                q = q.filter(_Trade.date <= to_date)
            if symbol:
                q = q.filter(_Trade.symbol == symbol)
            if direction:
                q = q.filter(_Trade.direction == direction.upper())
            rows = q.order_by(_Trade.entry_time.desc()).limit(limit).all()
        return [_trade_dict(r) for r in rows]

    def get_open_trades(self) -> list[dict]:
        with self._session() as s:
            rows = s.query(_Trade).filter(_Trade.exit_time.is_(None)).all()
        return [_trade_dict(r) for r in rows]

    def get_today_summary(self) -> dict:
        today = date.today()
        with self._session() as s:
            rows = (
                s.query(_Trade)
                 .filter(_Trade.date == today, _Trade.exit_time.isnot(None))
                 .all()
            )
        total  = len(rows)
        wins   = sum(1 for t in rows if t.pnl and t.pnl > 0)
        net    = round(sum(t.pnl for t in rows if t.pnl is not None), 2)
        return {
            "date":           str(today),
            "total_trades":   total,
            "winning_trades": wins,
            "losing_trades":  total - wins,
            "net_pnl":        net,
            "win_rate":       round(wins / total * 100, 1) if total else 0.0,
        }

    def get_daily_pnl(self, days: int = 30) -> list[dict]:
        with self._session() as s:
            rows = (
                s.query(
                    _Trade.date,
                    func.sum(_Trade.pnl).label("pnl"),
                    func.count(_Trade.id).label("trades"),
                )
                .filter(_Trade.exit_time.isnot(None))
                .group_by(_Trade.date)
                .order_by(_Trade.date.desc())
                .limit(days)
                .all()
            )
        return [
            {"date": str(r.date), "pnl": round(r.pnl or 0.0, 2), "trades": r.trades}
            for r in reversed(rows)
        ]

    def get_equity_curve(self, limit: int = 2000) -> list[dict]:
        from config import settings
        with self._session() as s:
            rows = (
                s.query(_Trade)
                 .filter(_Trade.exit_time.isnot(None), _Trade.pnl.isnot(None))
                 .order_by(_Trade.exit_time)
                 .limit(limit)
                 .all()
            )
        equity = settings.trading_capital
        pts: list[dict] = []
        for r in rows:
            equity += r.pnl
            pts.append({"time": r.exit_time.isoformat(), "equity": round(equity, 2)})
        return pts

    def get_recent_logs(self, level: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._session() as s:
            q = s.query(_BotLog)
            if level and level.upper() not in ("ALL", ""):
                q = q.filter(_BotLog.level == level.upper())
            rows = q.order_by(_BotLog.timestamp.desc()).limit(limit).all()
        return [
            {"time": r.timestamp.isoformat(), "level": r.level, "message": r.message}
            for r in reversed(rows)
        ]

    # ── Write helpers ──────────────────────────────────────────────────────────

    def log_bot_event(self, level: str, message: str) -> None:
        with self._session() as s:
            s.add(_BotLog(level=level.upper(), message=message, timestamp=datetime.utcnow()))

    def log_config_change(
        self,
        param_name: str,
        old_value:  Any,
        new_value:  Any,
        changed_by: str = "dashboard",
    ) -> None:
        with self._session() as s:
            s.add(_ConfigHistory(
                param_name=param_name,
                old_value=str(old_value),
                new_value=str(new_value),
                changed_by=changed_by,
                timestamp=datetime.utcnow(),
            ))

    def compute_daily_summary(self, target_date: Optional[date] = None) -> None:
        """Upsert the daily_summary row. Call this at market close."""
        d = target_date or date.today()
        with self._session() as s:
            rows = (
                s.query(_Trade)
                 .filter(_Trade.date == d, _Trade.exit_time.isnot(None))
                 .all()
            )
            if not rows:
                return
            wins   = [t for t in rows if t.pnl and t.pnl > 0]
            losses = [t for t in rows if not t.pnl or t.pnl <= 0]
            pnls   = [t.pnl for t in rows if t.pnl is not None]
            net    = sum(pnls)

            row = s.query(_DailySummary).filter(_DailySummary.date == d).first()
            if row is None:
                row = _DailySummary(date=d)
                s.add(row)
            row.total_trades    = len(rows)
            row.winning_trades  = len(wins)
            row.losing_trades   = len(losses)
            row.gross_profit    = round(sum(t.pnl for t in wins if t.pnl), 2)
            row.gross_loss      = round(abs(sum(t.pnl for t in losses if t.pnl)), 2)
            row.net_pnl         = round(net, 2)
            row.win_rate_pct    = round(len(wins) / len(rows) * 100, 1)
            row.best_trade_pnl  = round(max(pnls), 2)
            row.worst_trade_pnl = round(min(pnls), 2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _trade_dict(t: _Trade) -> dict:
    return {
        "id":          t.id,
        "date":        str(t.date),
        "symbol":      t.symbol,
        "direction":   t.direction,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "quantity":    t.quantity,
        "pnl":         round(t.pnl, 2) if t.pnl is not None else None,
        "entry_time":  t.entry_time.isoformat() if t.entry_time else None,
        "exit_time":   t.exit_time.isoformat()  if t.exit_time  else None,
        "exit_reason": t.exit_reason,
        "initial_sl":  t.initial_sl,
        "target":      t.target,
        "commission":  t.commission or 0.0,
        "is_paper":    t.is_paper,
    }
