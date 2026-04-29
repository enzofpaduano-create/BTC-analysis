"""Strategy parameter optimization with walk-forward validation.

Two-step API:
    1. ``grid_search`` — exhaustive sweep of a parameter grid on a fixed
       (in-sample) range. Fast, but vulnerable to overfit if the grid is
       big and the dataset is short.
    2. ``walk_forward_optimize`` — refits the optimum on each train window,
       evaluates it on the following test window, aggregates out-of-sample
       metrics. Slower but trustworthy.

Both return a ``pandas.DataFrame`` of trial results so you can sort,
filter, plot, or export.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from itertools import product
from typing import Any

import pandas as pd
from loguru import logger

from backtest.config import BacktestConfig
from backtest.engine import run_backtest
from backtest.strategy import Strategy
from backtest.walk_forward import walk_forward_splits

# A factory takes a dict of params and returns a fresh Strategy instance.
StrategyFactory = Callable[..., Strategy]


def grid_search(
    *,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Iterable[Any]],
    cfg: BacktestConfig,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Run one backtest per parameter combination on the full series.

    Args:
        ohlcv: OHLCV data.
        features: Pre-computed features (computed once outside).
        strategy_factory: Callable that returns a fresh ``Strategy`` given
            keyword arguments matching ``param_grid`` keys.
        param_grid: Dict ``{param_name: iterable_of_values}``. The cartesian
            product is searched.
        cfg: Backtest config (costs, capital, etc.).
        metric: Metric name to surface as ``best_metric`` for sorting.

    Returns:
        DataFrame with one row per param combo. Columns include every
        param plus all metrics from ``PerformanceMetrics`` plus ``trial``.
    """
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    rows: list[dict[str, Any]] = []

    for trial_idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo, strict=True))
        strat = strategy_factory(**params)
        res = run_backtest(ohlcv=ohlcv, features=features, strategy=strat, cfg=cfg)
        row: dict[str, Any] = {"trial": trial_idx, **params, **res.metrics.to_dict()}
        rows.append(row)
        logger.debug(
            "trial {}/{} params={} {}={:.3f}",
            trial_idx + 1,
            None,
            params,
            metric,
            float(row.get(metric, 0.0)),
        )

    df = pd.DataFrame(rows)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    return df


def walk_forward_optimize(
    *,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Iterable[Any]],
    cfg: BacktestConfig,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Walk-forward grid search.

    For each (train, test) split:
        1. Run ``grid_search`` on the train slice → pick the best params.
        2. Re-evaluate that strategy on the test slice.
        3. Record the test metrics.

    Aggregating the test metrics gives a realistic out-of-sample estimate.
    Returns one row per split with the chosen params and OOS metrics.
    """
    rows: list[dict[str, Any]] = []
    for split_idx, split in enumerate(walk_forward_splits(len(ohlcv), cfg.walk_forward)):
        train_ohlcv = ohlcv.iloc[split.train_start : split.train_end].reset_index(drop=True)
        train_feat = features.iloc[split.train_start : split.train_end].reset_index(drop=True)
        test_ohlcv = ohlcv.iloc[split.test_start : split.test_end].reset_index(drop=True)
        test_feat = features.iloc[split.test_start : split.test_end].reset_index(drop=True)

        # In-sample best.
        in_sample = grid_search(
            ohlcv=train_ohlcv,
            features=train_feat,
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            cfg=cfg,
            metric=metric,
        )
        if in_sample.empty:
            continue
        best = in_sample.iloc[0]
        # Restore the original Python type — pandas widens ints to float64
        # when reading back as a Series, which breaks consumers like ``range``.
        best_params: dict[str, Any] = {}
        for k, vals in param_grid.items():
            sample = next(iter(vals))
            raw = best[k]
            if isinstance(sample, bool):
                best_params[k] = bool(raw)
            elif isinstance(sample, int):
                best_params[k] = int(raw)
            elif isinstance(sample, float):
                best_params[k] = float(raw)
            else:
                best_params[k] = raw

        # Out-of-sample evaluation.
        oos_strat = strategy_factory(**best_params)
        oos_res = run_backtest(ohlcv=test_ohlcv, features=test_feat, strategy=oos_strat, cfg=cfg)
        rows.append(
            {
                "split": split_idx,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                **best_params,
                **{f"is_{k}": v for k, v in best.to_dict().items() if k in {metric, "n_trades"}},
                **{f"oos_{k}": v for k, v in oos_res.metrics.to_dict().items()},
            }
        )
        logger.info(
            "WF split {}: best params {} → IS {}={:.2f}, OOS {}={:.2f}",
            split_idx,
            best_params,
            metric,
            float(best.get(metric, 0.0)),
            metric,
            float(oos_res.metrics.to_dict().get(metric, 0.0)),
        )
    return pd.DataFrame(rows)


def _params_as_dict(obj: object) -> dict[str, Any]:
    """Helper for callers that prefer to pass a dataclass instance."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError("Expected a dataclass instance")
