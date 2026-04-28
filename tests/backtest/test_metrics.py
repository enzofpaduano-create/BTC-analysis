"""Known-answer tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import (
    annualised_return,
    calmar_ratio,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_metrics,
)


def test_sharpe_zero_for_zero_returns() -> None:
    rets = pd.Series([0.0] * 100)
    assert sharpe_ratio(rets, bars_per_year=525_600) == 0.0


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(1e-4, 5e-4, 1_000))
    assert sharpe_ratio(rets, bars_per_year=525_600) > 0


def test_sortino_only_penalises_downside() -> None:
    """Series with only positive returns → infinite or 0 Sortino (no downside)."""
    rets = pd.Series([0.001, 0.002, 0.0015])
    s = sortino_ratio(rets, bars_per_year=525_600)
    # No downside variance → return 0 by convention.
    assert s == 0.0


def test_max_drawdown_simple() -> None:
    """Equity 1.0 → 1.2 → 0.9 → 1.1 has a drawdown of (0.9 / 1.2 - 1) = -25 %."""
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    rets = eq.pct_change().fillna(0.0)
    mdd, _ = max_drawdown(rets)
    assert abs(mdd - (0.9 / 1.2 - 1.0)) < 1e-9


def test_calmar_signs() -> None:
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(2e-4, 1e-3, 2000))
    c = calmar_ratio(rets, bars_per_year=525_600)
    assert c != 0


def test_annualised_return_constant_1pct_per_year() -> None:
    """A 1 %/year compounded return should give CAGR ≈ 0.01."""
    bpy = 252.0  # daily bars
    daily = (1.01) ** (1 / bpy) - 1
    rets = pd.Series([daily] * int(bpy * 3))
    cagr = annualised_return(rets, bars_per_year=bpy)
    assert abs(cagr - 0.01) < 1e-6


def test_equity_curve_compounds() -> None:
    rets = pd.Series([0.1, -0.1, 0.1])
    eq = equity_curve(rets, initial=100.0)
    expected = 100.0 * 1.1 * 0.9 * 1.1
    assert abs(eq.iloc[-1] - expected) < 1e-9


def test_trade_metrics_basic() -> None:
    pnls = pd.Series([10.0, -5.0, 8.0, -3.0, 0.0])
    durs = pd.Series([3, 5, 4, 2, 1])
    m = trade_metrics(pnls, durs)
    assert m["n_trades"] == 5
    assert abs(m["win_rate"] - 2 / 5) < 1e-9
    assert abs(m["profit_factor"] - 18.0 / 8.0) < 1e-9
    assert abs(m["expectancy"] - 2.0) < 1e-9
