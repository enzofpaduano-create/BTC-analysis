"""Backtest layer — vectorised harness with realistic costs and walk-forward.

Public API::

    from backtest import (
        BacktestConfig, CostsConfig, WalkForwardConfig,
        Strategy, SignalFrame,
        run_backtest, BacktestResult,
        walk_forward_splits, save_html_report,
    )
"""

from backtest.config import BacktestConfig, CostsConfig, WalkForwardConfig
from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import (
    PerformanceMetrics,
    annualised_return,
    calmar_ratio,
    compute_metrics,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_metrics,
)
from backtest.optimization import grid_search, walk_forward_optimize
from backtest.plotting import plot_equity_and_drawdown, save_html_report
from backtest.strategy import SignalFrame, Strategy
from backtest.walk_forward import WalkForwardSplit, walk_forward_splits

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostsConfig",
    "PerformanceMetrics",
    "SignalFrame",
    "Strategy",
    "WalkForwardConfig",
    "WalkForwardSplit",
    "annualised_return",
    "calmar_ratio",
    "compute_metrics",
    "equity_curve",
    "grid_search",
    "max_drawdown",
    "plot_equity_and_drawdown",
    "run_backtest",
    "save_html_report",
    "sharpe_ratio",
    "sortino_ratio",
    "trade_metrics",
    "walk_forward_optimize",
    "walk_forward_splits",
]
