"""backtest — Event-driven backtesting engine for the trading bot."""
from backtest.engine import BacktestEngine, BacktestResult, BacktestTrade, generate_sample_data
from backtest.optimizer import run_grid_optimization, print_optimization_report, OptimizeResult

__all__ = [
    "BacktestEngine", "BacktestResult", "BacktestTrade", "generate_sample_data",
    "run_grid_optimization", "print_optimization_report", "OptimizeResult",
]
