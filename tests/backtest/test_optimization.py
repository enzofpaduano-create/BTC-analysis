"""Tests for the grid-search + walk-forward optimisation helpers."""

from __future__ import annotations

import pandas as pd

from backtest import (
    BacktestConfig,
    WalkForwardConfig,
    grid_search,
    walk_forward_optimize,
)
from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy


class _ConfigurableStrategy(Strategy):
    """Toy strategy that opens long every ``period`` bars and exits 1 bar later."""

    name = "configurable"

    def __init__(self, *, period: int, size: float = 1.0) -> None:
        self.period = period
        self._size = size

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        n = len(features)
        df = pd.DataFrame(
            dict.fromkeys(SIGNAL_COLUMNS, 0.0),
            index=features.index,
        )
        for col in ("entry_long", "exit_long", "entry_short", "exit_short"):
            df[col] = False
        df["size"] = 0.0
        for i in range(self.period, n - 1, self.period * 2):
            df.iloc[i, df.columns.get_loc("entry_long")] = True
            df.iloc[i, df.columns.get_loc("size")] = self._size
            df.iloc[i + 1, df.columns.get_loc("exit_long")] = True
        return SignalFrame(df=df[list(SIGNAL_COLUMNS)])


def test_grid_search_runs_one_per_combination(synthetic_ohlcv: pd.DataFrame) -> None:
    cfg = BacktestConfig(initial_capital=10_000.0, bar_minutes=1)
    grid = {"period": [10, 20, 30], "size": [0.5, 1.0]}
    out = grid_search(
        ohlcv=synthetic_ohlcv,
        features=synthetic_ohlcv,
        strategy_factory=_ConfigurableStrategy,
        param_grid=grid,
        cfg=cfg,
    )
    # 3 × 2 = 6 trials.
    assert len(out) == 6
    assert {"period", "size", "sharpe", "max_drawdown", "n_trades"}.issubset(out.columns)


def test_grid_search_sorts_by_metric(synthetic_ohlcv: pd.DataFrame) -> None:
    cfg = BacktestConfig(initial_capital=10_000.0, bar_minutes=1)
    grid = {"period": [10, 50]}
    out = grid_search(
        ohlcv=synthetic_ohlcv,
        features=synthetic_ohlcv,
        strategy_factory=_ConfigurableStrategy,
        param_grid=grid,
        cfg=cfg,
        metric="sharpe",
    )
    # Sharpe should be sorted descending.
    assert out["sharpe"].is_monotonic_decreasing


def test_walk_forward_optimize_returns_one_row_per_split(synthetic_ohlcv: pd.DataFrame) -> None:
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        bar_minutes=1,
        walk_forward=WalkForwardConfig(
            train_size=200,
            test_size=100,
            step_size=100,
            purge=0,
            embargo=0,
        ),
    )
    grid = {"period": [10, 20]}
    out = walk_forward_optimize(
        ohlcv=synthetic_ohlcv,
        features=synthetic_ohlcv,
        strategy_factory=_ConfigurableStrategy,
        param_grid=grid,
        cfg=cfg,
    )
    # synthetic_ohlcv has 600 bars → 4 WF splits at this config.
    assert len(out) >= 2
    assert {"period", "oos_sharpe", "oos_max_drawdown", "is_sharpe"}.issubset(out.columns)
