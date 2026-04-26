"""
data/parameter_registry.py - Parameter Performance Registry.

Tracks every backtest and live trading session's parameters alongside
the market regime and resulting P&L. Over time this builds a database
that answers: "what parameters work best in BEAR markets for RELIANCE?"

Tables
------
  param_runs      One row per backtest / live session with full parameter snapshot
  regime_summary  Aggregated best-params per (symbol, regime) — updated on each run

Usage
-----
    from data.parameter_registry import ParameterRegistry
    reg = ParameterRegistry()

    # Log a completed backtest
    reg.log_run(
        symbol="RELIANCE",
        regime="BEAR",
        interval_minutes=15,
        params={"volume_surge_multiplier": 2.5, "rsi_long_min": 50, ...},
        result={"net_pnl": 5000, "win_rate_pct": 38.0, "total_trades": 22, ...},
        source="backtest",
    )

    # Query best params for a regime
    best = reg.best_params("RELIANCE", "BEAR")
    # Returns: {"volume_surge_multiplier": 2.5, ...} or None

    # Print full report
    reg.print_report()
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional, Iterator

from loguru import logger
from sqlalchemy import (
    Column, Date, DateTime, Float, Integer, String, Text,
    create_engine, func, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ── DB location ───────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.resolve()
_DB_PATH = _ROOT / "db" / "parameter_registry.db"


# ── ORM Models ────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _ParamRun(_Base):
    """One recorded run (backtest or live session)."""
    __tablename__ = "param_runs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_date        = Column(Date,     nullable=False, index=True)
    symbol          = Column(String,   nullable=False, index=True)
    regime          = Column(String,   nullable=False, index=True)
    interval_min    = Column(Integer,  nullable=False)
    source          = Column(String,   nullable=False)  # "backtest" | "live"

    # Period covered
    period_from     = Column(String,   nullable=True)   # YYYY-MM-DD
    period_to       = Column(String,   nullable=True)

    # Strategy parameters (stored as JSON for flexibility)
    params_json     = Column(Text,     nullable=False)

    # Results
    total_trades    = Column(Integer,  default=0)
    win_rate_pct    = Column(Float,    default=0.0)
    net_pnl         = Column(Float,    default=0.0)
    profit_factor   = Column(Float,    default=0.0)
    sharpe_ratio    = Column(Float,    default=0.0)
    max_drawdown_pct= Column(Float,    default=0.0)
    cagr_pct        = Column(Float,    default=0.0)

    notes           = Column(Text,     nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class _RegimeSummary(_Base):
    """Aggregated best-params per (symbol, regime) — denormalised for fast lookup."""
    __tablename__ = "regime_summary"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String,  nullable=False, index=True)
    regime          = Column(String,  nullable=False, index=True)
    best_run_id     = Column(Integer, nullable=True)
    best_sharpe     = Column(Float,   default=0.0)
    best_win_rate   = Column(Float,   default=0.0)
    best_net_pnl    = Column(Float,   default=0.0)
    best_params_json= Column(Text,    nullable=True)
    run_count       = Column(Integer, default=0)
    last_updated    = Column(DateTime,default=datetime.utcnow)


# ── Registry class ────────────────────────────────────────────────────────────

class ParameterRegistry:
    """
    Persistent store for parameter experiment results, keyed by regime.

    Thread-safe — uses SQLAlchemy session-per-call pattern.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        _Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_run(
        self,
        symbol: str,
        regime: str,
        interval_minutes: int,
        params: dict[str, Any],
        result: dict[str, Any],
        source: str = "backtest",
        period_from: str | None = None,
        period_to: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Record one completed backtest or live session.
        Returns the run ID.
        """
        # Only log if the run had meaningful data
        total_trades = result.get("total_trades", 0)
        if total_trades < 3 and source == "backtest":
            logger.debug("Skipping registry log - too few trades ({})", total_trades)
            return -1

        run = _ParamRun(
            run_date        = date.today(),
            symbol          = symbol.upper(),
            regime          = regime.upper(),
            interval_min    = interval_minutes,
            source          = source,
            period_from     = period_from,
            period_to       = period_to,
            params_json     = json.dumps(params, default=str),
            total_trades    = total_trades,
            win_rate_pct    = result.get("win_rate_pct", 0.0),
            net_pnl         = result.get("net_pnl", 0.0),
            profit_factor   = result.get("profit_factor", 0.0),
            sharpe_ratio    = result.get("sharpe_ratio", 0.0),
            max_drawdown_pct= result.get("max_drawdown_pct", 0.0),
            cagr_pct        = result.get("cagr_pct", 0.0),
            notes           = notes,
        )

        with self._session() as s:
            s.add(run)
            s.flush()
            run_id = run.id
            self._update_summary(s, symbol, regime, run)

        logger.info(
            "ParameterRegistry: logged run #{} | {} {} | regime={} | "
            "trades={} win={:.1f}% pnl={:.0f}",
            run_id, symbol, source, regime,
            total_trades, result.get("win_rate_pct", 0), result.get("net_pnl", 0),
        )
        return run_id

    def _update_summary(self, session: Session, symbol: str, regime: str, run: _ParamRun) -> None:
        """Update the regime_summary row if this run beats the current best Sharpe."""
        sym = symbol.upper()
        reg = regime.upper()

        existing = (
            session.query(_RegimeSummary)
            .filter_by(symbol=sym, regime=reg)
            .first()
        )

        if existing is None:
            existing = _RegimeSummary(symbol=sym, regime=reg, run_count=0)
            session.add(existing)

        existing.run_count  += 1
        existing.last_updated = datetime.utcnow()

        # Update best if this run has higher Sharpe (or first profitable run)
        current_best = existing.best_sharpe or -999.0
        if run.sharpe_ratio > current_best:
            existing.best_sharpe      = run.sharpe_ratio
            existing.best_win_rate    = run.win_rate_pct
            existing.best_net_pnl     = run.net_pnl
            existing.best_run_id      = run.id
            existing.best_params_json = run.params_json

    # ── Read ──────────────────────────────────────────────────────────────────

    def best_params(
        self,
        symbol: str,
        regime: str,
        min_trades: int = 5,
    ) -> dict[str, Any] | None:
        """
        Return the parameter dict that produced the best Sharpe for this
        (symbol, regime) combination.  Returns None if no data yet.
        """
        with self._session() as s:
            row = (
                s.query(_RegimeSummary)
                .filter_by(symbol=symbol.upper(), regime=regime.upper())
                .first()
            )
            if row and row.best_params_json:
                params = json.loads(row.best_params_json)
                logger.info(
                    "Best params for {} {}: Sharpe={:.2f} WR={:.1f}% pnl={:.0f} (from {} runs)",
                    symbol, regime,
                    row.best_sharpe, row.best_win_rate, row.best_net_pnl, row.run_count,
                )
                return params
        return None

    def get_runs(
        self,
        symbol: str | None = None,
        regime: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent runs as list of dicts, newest first."""
        with self._session() as s:
            q = s.query(_ParamRun)
            if symbol: q = q.filter(_ParamRun.symbol == symbol.upper())
            if regime: q = q.filter(_ParamRun.regime == regime.upper())
            if source: q = q.filter(_ParamRun.source == source)
            rows = q.order_by(_ParamRun.created_at.desc()).limit(limit).all()
            return [self._run_to_dict(r) for r in rows]

    def get_regime_summary(self) -> list[dict]:
        """Return summary of best params per (symbol, regime)."""
        with self._session() as s:
            rows = s.query(_RegimeSummary).order_by(
                _RegimeSummary.symbol, _RegimeSummary.regime
            ).all()
            return [
                {
                    "symbol":      r.symbol,
                    "regime":      r.regime,
                    "run_count":   r.run_count,
                    "best_sharpe": round(r.best_sharpe, 3),
                    "best_win_rate": round(r.best_win_rate, 1),
                    "best_net_pnl":  round(r.best_net_pnl, 2),
                    "best_params": json.loads(r.best_params_json) if r.best_params_json else {},
                    "last_updated": str(r.last_updated)[:16],
                }
                for r in rows
            ]

    # ── Print report ──────────────────────────────────────────────────────────

    def print_report(self, symbol: str | None = None) -> None:
        """Print a human-readable performance-by-regime table."""
        summaries = self.get_regime_summary()
        if symbol:
            summaries = [s for s in summaries if s["symbol"] == symbol.upper()]

        if not summaries:
            print("No data in parameter registry yet.")
            print("Run backtests with --log-regime to populate it.")
            return

        print()
        print("=" * 80)
        print("  Parameter Registry - Best Results by Regime")
        print("=" * 80)
        print(f"  {'Symbol':<12} {'Regime':<10} {'Runs':>5} {'Sharpe':>8} "
              f"{'WinRate':>8} {'Net P&L':>12} {'Updated'}")
        print(f"  {'-'*12} {'-'*10} {'-'*5} {'-'*8} {'-'*8} {'-'*12} {'-'*16}")
        for s in summaries:
            pnl_str = f"Rs.{s['best_net_pnl']:+,.0f}"
            print(
                f"  {s['symbol']:<12} {s['regime']:<10} {s['run_count']:>5} "
                f"{s['best_sharpe']:>8.3f} {s['best_win_rate']:>7.1f}% "
                f"{pnl_str:>12} {s['last_updated']}"
            )
        print()

        # Show best params for each row
        for s in summaries:
            if s["best_params"]:
                print(f"  Best params [{s['symbol']} / {s['regime']}]:")
                for k, v in s["best_params"].items():
                    print(f"    {k}: {v}")
                print()

    @staticmethod
    def _run_to_dict(r: _ParamRun) -> dict:
        return {
            "id":            r.id,
            "run_date":      str(r.run_date),
            "symbol":        r.symbol,
            "regime":        r.regime,
            "interval_min":  r.interval_min,
            "source":        r.source,
            "period_from":   r.period_from,
            "period_to":     r.period_to,
            "params":        json.loads(r.params_json) if r.params_json else {},
            "total_trades":  r.total_trades,
            "win_rate_pct":  r.win_rate_pct,
            "net_pnl":       r.net_pnl,
            "profit_factor": r.profit_factor,
            "sharpe_ratio":  r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown_pct,
            "cagr_pct":      r.cagr_pct,
            "notes":         r.notes,
        }
