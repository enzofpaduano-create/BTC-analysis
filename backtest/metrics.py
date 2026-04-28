"""Performance metrics computed on a per-bar return series.

All inputs are pandas Series of simple per-bar returns ``(equity_t / equity_{t-1} - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary statistics of a backtest equity curve."""

    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float  # negative number, e.g. -0.12 = -12%
    max_drawdown_duration: int  # bars
    cagr: float
    total_return: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_trade_duration: float
    exposure: float  # fraction of bars with non-zero position

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "cagr": self.cagr,
            "total_return": self.total_return,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_trade_duration": self.avg_trade_duration,
            "exposure": self.exposure,
        }


# -- Risk-adjusted return ratios ------------------------------------------


def sharpe_ratio(returns: pd.Series, *, bars_per_year: float, rf_per_bar: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Returns 0 for an empty / constant series.
    """
    excess = returns.dropna() - rf_per_bar
    sigma = excess.std(ddof=1)
    if not np.isfinite(sigma) or sigma == 0.0 or len(excess) < 2:  # noqa: PLR2004 — need ≥2 obs
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series, *, bars_per_year: float, rf_per_bar: float = 0.0) -> float:
    """Annualised Sortino — penalises only downside variance."""
    excess = returns.dropna() - rf_per_bar
    downside = excess[excess < 0]
    if len(downside) < 2:  # noqa: PLR2004 — need ≥2 downside obs
        return 0.0
    sigma_d = downside.std(ddof=1)
    if not np.isfinite(sigma_d) or sigma_d == 0.0:
        return 0.0
    return float(excess.mean() / sigma_d * np.sqrt(bars_per_year))


# -- Drawdown -------------------------------------------------------------


def equity_curve(returns: pd.Series, *, initial: float = 1.0) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() * initial


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """Returns (max_drawdown_pct, duration_in_bars).

    ``max_drawdown_pct`` is negative or zero.
    """
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0, 0
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    mdd = float(dd.min())
    if mdd == 0.0:
        return 0.0, 0
    # Find duration of the deepest drawdown — from previous peak to recovery.
    end_idx = int(dd.idxmin()) if eq.index.dtype != object else dd.values.argmin()
    if isinstance(end_idx, int) and 0 <= end_idx < len(eq):
        running_arr = running_max.to_numpy()
        eq_arr = eq.to_numpy()
        peak_value = running_arr[end_idx]
        # Walk left to find peak start (where equity == peak_value).
        start = end_idx
        while start > 0 and eq_arr[start - 1] != peak_value:
            start -= 1
        # Walk right to find recovery (eq returns to peak), or end.
        recover = end_idx
        while recover < len(eq) - 1 and eq_arr[recover] < peak_value:
            recover += 1
        duration = int(recover - start)
    else:
        duration = int(len(eq))
    return mdd, duration


def calmar_ratio(returns: pd.Series, *, bars_per_year: float) -> float:
    """CAGR / |max_drawdown|."""
    cagr = annualised_return(returns, bars_per_year=bars_per_year)
    mdd, _ = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return float(cagr / abs(mdd))


def annualised_return(returns: pd.Series, *, bars_per_year: float) -> float:
    rets = returns.dropna()
    if len(rets) < 2:  # noqa: PLR2004 — need ≥2 obs to annualize
        return 0.0
    total = float((1.0 + rets).prod() - 1.0)
    years = len(rets) / bars_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total) ** (1.0 / years) - 1.0)


# -- Trade-level metrics --------------------------------------------------


def trade_metrics(trade_pnls: pd.Series, trade_durations: pd.Series) -> dict[str, float]:
    """Compute win-rate / profit-factor / expectancy from per-trade PnLs."""
    pnls = trade_pnls.dropna()
    n = int(len(pnls))
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade_duration": 0.0,
        }
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / n)
    sum_wins = float(wins.sum())
    sum_losses = float(-losses.sum())
    profit_factor = float(sum_wins / sum_losses) if sum_losses > 0 else float("inf")
    expectancy = float(pnls.mean())
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_duration": float(trade_durations.mean()) if len(trade_durations) else 0.0,
    }


def compute_metrics(
    *,
    returns: pd.Series,
    trade_pnls: pd.Series,
    trade_durations: pd.Series,
    position_open: pd.Series,
    bars_per_year: float,
) -> PerformanceMetrics:
    """Bundle every metric into a single dataclass."""
    eq = equity_curve(returns)
    total_return = float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0
    mdd, mdd_dur = max_drawdown(returns)
    trade_stats = trade_metrics(trade_pnls, trade_durations)
    exposure = float(position_open.astype(bool).mean()) if len(position_open) else 0.0
    return PerformanceMetrics(
        sharpe=sharpe_ratio(returns, bars_per_year=bars_per_year),
        sortino=sortino_ratio(returns, bars_per_year=bars_per_year),
        calmar=calmar_ratio(returns, bars_per_year=bars_per_year),
        max_drawdown=mdd,
        max_drawdown_duration=mdd_dur,
        cagr=annualised_return(returns, bars_per_year=bars_per_year),
        total_return=total_return,
        n_trades=int(trade_stats["n_trades"]),
        win_rate=float(trade_stats["win_rate"]),
        profit_factor=float(trade_stats["profit_factor"]),
        expectancy=float(trade_stats["expectancy"]),
        avg_trade_duration=float(trade_stats["avg_trade_duration"]),
        exposure=exposure,
    )
