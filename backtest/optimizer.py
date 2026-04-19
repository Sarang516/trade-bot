"""
backtest/optimizer.py — Parameter grid-search optimizer for backtesting.

Iterates over combinations of strategy parameters, runs a full backtest for
each combination, ranks by Sharpe ratio (with Rf), and flags overfitting.

Usage
-----
    from backtest.optimizer import run_grid_optimization, DEFAULT_PARAM_GRID

    results = run_grid_optimization(
        symbol="NIFTY",
        strategy_name="vwap_volume",
        settings=settings,
        df=df,                         # pre-loaded DataFrame
        interval_minutes=5,
        param_grid=DEFAULT_PARAM_GRID,
        top_n=5,
    )
    print_optimization_report(results)
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine, BacktestResult


# ── Default parameter grid ─────────────────────────────────────────────────────
# Chosen to cover realistic VWAPVolume strategy parameter ranges.
# Total combos with defaults: 3 × 3 × 3 × 3 × 3 = 243 (fast enough in pure Python)

DEFAULT_PARAM_GRID: dict[str, list] = {
    "volume_surge_multiplier": [1.5, 2.0, 2.5],
    "rsi_long_min":            [40, 45, 50],
    "rsi_long_max":            [60, 65, 70],
    "sl_atr_multiplier":       [1.0, 1.5, 2.0],
    "rr_ratio":                [1.5, 2.0, 2.5],
}

# Overfitting thresholds
_MIN_TRADES_THRESHOLD   = 30     # fewer trades = unreliable statistics
_MAX_WIN_RATE_THRESHOLD = 75.0   # win rate above this looks too good
_MAX_MONTHLY_CV         = 1.5    # coefficient of variation: high = inconsistent months


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OverfitWarning:
    """Describes a single overfitting risk factor."""
    code: str
    message: str


@dataclass
class OptimizeResult:
    """Result for one parameter combination."""
    params: dict
    result: BacktestResult
    rank: int = 0
    overfit_warnings: list[OverfitWarning] = field(default_factory=list)

    @property
    def sharpe(self) -> float:
        return self.result.sharpe_ratio_rf

    @property
    def total_trades(self) -> int:
        return self.result.total_trades

    @property
    def win_rate(self) -> float:
        return self.result.win_rate_pct

    @property
    def net_pnl(self) -> float:
        return self.result.net_pnl

    @property
    def cagr(self) -> float:
        return self.result.cagr_pct

    @property
    def is_overfit(self) -> bool:
        return len(self.overfit_warnings) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Overfitting detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_overfitting(result: BacktestResult) -> list[OverfitWarning]:
    """
    Return a list of overfitting warning flags for a backtest result.

    Criteria
    --------
    1. Too few trades  : < 30 trades → statistically unreliable
    2. Win rate too high : > 75% → strategy may be curve-fitted to data
    3. Inconsistent monthly P&L : CV > 1.5 → performance not stable
    4. Suspiciously high profit factor : > 5.0 → overfitted to favorable periods
    5. Zero losing months in multi-month test : → no stress test coverage
    """
    warnings: list[OverfitWarning] = []

    # 1. Too few trades
    if 0 < result.total_trades < _MIN_TRADES_THRESHOLD:
        warnings.append(OverfitWarning(
            code="FEW_TRADES",
            message=f"Only {result.total_trades} trades — need ≥{_MIN_TRADES_THRESHOLD} for reliable stats",
        ))

    # 2. Win rate too high
    if result.total_trades >= 10 and result.win_rate_pct > _MAX_WIN_RATE_THRESHOLD:
        warnings.append(OverfitWarning(
            code="HIGH_WIN_RATE",
            message=f"Win rate {result.win_rate_pct}% exceeds {_MAX_WIN_RATE_THRESHOLD}% — potential curve fitting",
        ))

    # 3. Inconsistent monthly P&L (high coefficient of variation)
    if len(result.monthly_pnl) >= 3:
        monthly_vals = list(result.monthly_pnl.values())
        mean_m = sum(monthly_vals) / len(monthly_vals)
        if mean_m != 0:
            std_m = math.sqrt(sum((v - mean_m) ** 2 for v in monthly_vals) / len(monthly_vals))
            cv = abs(std_m / mean_m)
            if cv > _MAX_MONTHLY_CV:
                warnings.append(OverfitWarning(
                    code="INCONSISTENT_MONTHS",
                    message=f"Monthly P&L CV={cv:.2f} > {_MAX_MONTHLY_CV} — performance not consistent across months",
                ))

    # 4. Suspiciously high profit factor
    if result.total_trades >= 10 and result.profit_factor > 5.0:
        warnings.append(OverfitWarning(
            code="HIGH_PROFIT_FACTOR",
            message=f"Profit factor {result.profit_factor} > 5.0 — may not hold on unseen data",
        ))

    # 5. No losing months in a multi-month test
    if len(result.monthly_pnl) >= 4:
        losing_months = sum(1 for v in result.monthly_pnl.values() if v < 0)
        if losing_months == 0:
            warnings.append(OverfitWarning(
                code="NO_LOSING_MONTHS",
                message="Zero losing months in test window — strategy not stress-tested",
            ))

    return warnings


# ═════════════════════════════════════════════════════════════════════════════
# Core optimizer
# ═════════════════════════════════════════════════════════════════════════════

def run_grid_optimization(
    symbol: str,
    strategy_name: str,
    settings,
    df: pd.DataFrame,
    interval_minutes: int = 5,
    param_grid: Optional[dict[str, list]] = None,
    commission_inr: float = 40.0,
    slippage_pct: float = 0.0005,
    top_n: int = 5,
    show_progress: bool = True,
) -> list[OptimizeResult]:
    """
    Run a backtest for every combination in `param_grid`, rank by Sharpe(Rf),
    and return the top `top_n` results with overfitting analysis.

    Parameters
    ----------
    symbol           : e.g. "NIFTY"
    strategy_name    : e.g. "vwap_volume"
    settings         : Settings object (config.py)
    df               : pre-loaded OHLCV DataFrame with DatetimeIndex
    interval_minutes : candle interval for backtest
    param_grid       : dict of {param_name: [values]}. Defaults to DEFAULT_PARAM_GRID.
    commission_inr   : round-trip commission per trade (₹)
    slippage_pct     : one-way slippage fraction
    top_n            : how many top results to return
    show_progress    : print progress dots to console

    Returns
    -------
    List of OptimizeResult sorted by Sharpe ratio (descending), with rank set.
    Results with zero trades are excluded.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    logger.info("Optimizer | {} param combos | symbol={} strategy={}", total, symbol, strategy_name)

    if show_progress:
        try:
            from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
            _rich = True
        except ImportError:
            _rich = False
    else:
        _rich = False

    all_results: list[OptimizeResult] = []
    t0 = time.time()

    def _run_combo(i: int, combo: tuple) -> Optional[OptimizeResult]:
        params = dict(zip(keys, combo))
        try:
            engine = BacktestEngine(
                strategy_name=strategy_name,
                symbol=symbol,
                settings=settings,
                commission_inr=commission_inr,
                slippage_pct=slippage_pct,
                next_candle_fill=True,
                strategy_config=params,
            )
            result = engine.run_from_dataframe(df.copy(), interval_minutes)
            if result.total_trades == 0:
                return None
            warnings = detect_overfitting(result)
            return OptimizeResult(params=params, result=result, overfit_warnings=warnings)
        except Exception as exc:
            logger.debug("Optimizer combo {} failed: {}", params, exc)
            return None

    if _rich:
        from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Optimizing...", total=total)
            for i, combo in enumerate(combos):
                opt = _run_combo(i, combo)
                if opt is not None:
                    all_results.append(opt)
                progress.advance(task)
    else:
        for i, combo in enumerate(combos):
            opt = _run_combo(i, combo)
            if opt is not None:
                all_results.append(opt)
            if show_progress and (i + 1) % max(1, total // 20) == 0:
                pct = (i + 1) / total * 100
                elapsed = time.time() - t0
                print(f"  {pct:.0f}% ({i+1}/{total}) — {elapsed:.1f}s elapsed", flush=True)

    elapsed_total = time.time() - t0
    logger.info(
        "Optimizer done | {} valid results / {} combos | {:.1f}s",
        len(all_results), total, elapsed_total,
    )

    # Sort by Sharpe(Rf) descending; ties broken by net P&L
    all_results.sort(key=lambda r: (r.sharpe, r.net_pnl), reverse=True)

    # Assign ranks
    for rank, res in enumerate(all_results, start=1):
        res.rank = rank

    return all_results[:top_n]


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════

def print_optimization_report(
    results: list[OptimizeResult],
    show_params: bool = True,
) -> None:
    """
    Print a ranked table of the top optimization results plus overfitting analysis.
    """
    if not results:
        print("\n[Optimizer] No valid results found.")
        return

    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print(f"  OPTIMIZATION REPORT  (top {len(results)} by Sharpe Ratio with Rf=6.5%)")
    print(sep)

    for opt in results:
        r   = opt.result
        tag = "  ⚠ OVERFIT" if opt.is_overfit else "  ✓ OK"

        print(f"\n  Rank #{opt.rank}{tag}")
        print(sep2)

        if show_params:
            param_str = "  | ".join(f"{k}={v}" for k, v in opt.params.items())
            print(f"  Params   : {param_str}")

        print(
            f"  Sharpe(Rf): {r.sharpe_ratio_rf:<7}  "
            f"CAGR: {r.cagr_pct:+.2f}%  "
            f"Net P&L: ₹{r.net_pnl:+,.0f}"
        )
        print(
            f"  Trades: {r.total_trades:<6}  "
            f"Win%: {r.win_rate_pct:<6}  "
            f"PF: {r.profit_factor:<6}  "
            f"Max DD: {r.max_drawdown_pct:.1f}%"
        )
        print(
            f"  Avg R: {r.avg_r_multiple}R  "
            f"Sharpe(raw): {r.sharpe_ratio}  "
            f"Commission: ₹{r.total_commission:,.0f}"
        )

        if opt.overfit_warnings:
            print(f"  Overfitting flags:")
            for w in opt.overfit_warnings:
                print(f"    [{w.code}] {w.message}")

    print(f"\n{sep}")

    # Overall overfitting summary
    clean   = sum(1 for o in results if not o.is_overfit)
    flagged = len(results) - clean
    print(f"  Summary: {clean} clean / {flagged} flagged out of top {len(results)} results")

    if flagged == len(results):
        print(
            "  WARNING: All top results show overfitting signs.\n"
            "  Consider: longer data range, fewer parameters, or walk-forward validation."
        )
    elif results[0].is_overfit:
        print(
            "  WARNING: #1 ranked result is flagged for overfitting.\n"
            "  Use the first clean result (lowest rank without ⚠) for live trading."
        )
    print(sep)


def best_clean_result(results: list[OptimizeResult]) -> Optional[OptimizeResult]:
    """Return the highest-ranked result with no overfitting warnings."""
    for r in results:
        if not r.is_overfit:
            return r
    return None
