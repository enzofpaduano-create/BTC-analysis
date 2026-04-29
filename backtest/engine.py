"""Vectorised backtest engine.

Strict semantics — preventing look-ahead bias:
    - Strategies emit signals at bar ``i`` based on features ``[:i+1]``.
    - The engine fills any entry/exit at bar ``i+1``'s close, NEVER at the
      same-bar close. This embodies the "act on next bar" rule that any
      live system has to obey: a model running at the close of bar ``i``
      can only place an order that fills at or after bar ``i+1``.

Position model: at most one open position at a time (long or short),
sized as a fraction of the latest equity. Re-entries are allowed only
after the previous position is closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from backtest.config import BacktestConfig
from backtest.costs import apply_funding, fill_price_with_costs
from backtest.metrics import PerformanceMetrics, compute_metrics, equity_curve
from backtest.strategy import Strategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of one backtest run."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series  # signed BTC held at the end of each bar
    trades: pd.DataFrame  # columns: entry_idx, exit_idx, side, size, pnl, duration
    metrics: PerformanceMetrics

    def summary(self) -> str:
        m = self.metrics
        return (
            f"trades={m.n_trades} sharpe={m.sharpe:.2f} sortino={m.sortino:.2f} "
            f"calmar={m.calmar:.2f} mdd={m.max_drawdown:.2%} "
            f"win_rate={m.win_rate:.1%} pf={m.profit_factor:.2f} "
            f"total_ret={m.total_return:.2%} exposure={m.exposure:.1%}"
        )


def run_backtest(
    *,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    strategy: Strategy,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Run a single backtest.

    Args:
        ohlcv: Raw OHLCV with at least ``close`` and ``volume``. Used for
            execution prices and slippage scaling.
        features: Pre-computed features fed to ``strategy.generate_signals``.
        strategy: Concrete ``Strategy`` instance.
        cfg: Backtest configuration.

    Returns:
        ``BacktestResult`` with the equity curve, per-bar returns, position
        history, the trade ledger, and aggregated metrics.

    Notes:
        - Entries/exits at bar ``i`` fill at ``ohlcv['close'].iloc[i + 1]``.
        - Last bar's open signals are dropped (no future bar to fill on).
    """
    if not ohlcv.index.equals(features.index):
        raise ValueError("`ohlcv` and `features` must share the same index")
    if not pd.api.types.is_datetime64_any_dtype(ohlcv["timestamp"]):
        raise TypeError("`ohlcv['timestamp']` must be datetime")

    sig = strategy.generate_signals(features)
    sig.validate()
    sigs = sig.df

    n = len(ohlcv)
    close = ohlcv["close"].to_numpy()
    volume = ohlcv["volume"].to_numpy()
    bars_per_year = (365 * 24 * 60) / cfg.bar_minutes

    # Per-bar tracking arrays.
    position_size = np.zeros(n)  # signed BTC held at END of bar i
    cash = np.full(n, cfg.initial_capital)
    fees = np.zeros(n)
    funding_cost = np.zeros(n)

    open_long = np.asarray(sigs["entry_long"].fillna(False).to_numpy(), dtype=bool)
    close_long = np.asarray(sigs["exit_long"].fillna(False).to_numpy(), dtype=bool)
    open_short = np.asarray(sigs["entry_short"].fillna(False).to_numpy(), dtype=bool)
    close_short = np.asarray(sigs["exit_short"].fillna(False).to_numpy(), dtype=bool)
    sizes = np.asarray(sigs["size"].fillna(0.0).to_numpy(), dtype=float)

    # Trade ledger.
    trades_records: list[dict[str, float | int]] = []
    open_trade: dict[str, float | int] | None = None

    funding_per_bar = (cfg.costs.funding_annual_bps * 1e-4) / bars_per_year

    for i in range(n):
        # Carry the position forward. Unrealised PnL is captured automatically
        # via `equity = cash + position_size * close`, so cash MUST NOT be
        # bumped by mark-to-market — that would double-count price moves.
        prev_pos = position_size[i - 1] if i > 0 else 0.0
        cur_close = close[i]

        # Funding on the position held over the bar (long & short symmetric).
        funding_cost[i] = abs(prev_pos) * cur_close * funding_per_bar
        position_size[i] = prev_pos
        cash[i] = (cash[i - 1] if i > 0 else cfg.initial_capital) - funding_cost[i]

        # Decide actions for bar i — fills happen at bar i+1's close.
        # We process the *signals from the previous bar* now to keep the
        # "act on next bar" semantics clean.
        if i == 0:
            continue
        sig_idx = i - 1  # signals from bar i-1 fill at close of bar i

        side: int = 0
        size_btc = 0.0
        action = ""

        # 1) close existing position if signalled
        if position_size[i] > 0 and close_long[sig_idx]:
            side = -1
            size_btc = position_size[i]
            action = "close_long"
        elif position_size[i] < 0 and close_short[sig_idx]:
            side = 1
            size_btc = -position_size[i]
            action = "close_short"

        # 2) open a new position if flat AND an entry fired
        if action == "" and position_size[i] == 0:
            if open_long[sig_idx]:
                side = 1
                action = "open_long"
            elif open_short[sig_idx]:
                side = -1
                action = "open_short"
            if side != 0:
                # Volatility / fraction-of-equity sizing: ``size`` ∈ [0, 1].
                target_frac = max(0.0, float(sizes[sig_idx]))
                size_btc = (cash[i] * target_frac) / cur_close

        if action == "" or size_btc <= 0:
            continue

        # Apply costs at the fill price.
        fill = fill_price_with_costs(
            mid_price=float(cur_close),
            size=float(size_btc),
            bar_volume=float(volume[i]),
            side=side,
            cfg=cfg.costs,
        )
        cash[i] -= fill.fee_paid
        fees[i] = fill.fee_paid

        if action.startswith("open"):
            position_size[i] = side * size_btc
            # `side` already encodes buy(+1) / sell(-1): a buy debits cash,
            # a sell credits it.
            cash[i] -= side * size_btc * fill.fill_price
            open_trade = {
                "entry_idx": int(i),
                "side": int(side),
                "entry_price": float(fill.fill_price),
                "size": float(size_btc),
            }
        else:  # close
            assert open_trade is not None, "close without open"
            position_size[i] = 0.0
            # Same formula as open — `side` is the closing direction:
            # close_long has side=-1 (we sell, cash IN); close_short has side=+1 (cash OUT).
            cash[i] -= side * size_btc * fill.fill_price
            entry_price = float(open_trade["entry_price"])
            trade_side = int(open_trade["side"])
            pnl_trade = (
                trade_side * (float(fill.fill_price) - entry_price) * float(open_trade["size"])
            )
            trades_records.append(
                {
                    "entry_idx": int(open_trade["entry_idx"]),
                    "exit_idx": int(i),
                    "side": trade_side,
                    "size": float(open_trade["size"]),
                    "entry_price": entry_price,
                    "exit_price": float(fill.fill_price),
                    "pnl": float(pnl_trade),
                    "duration": int(i - int(open_trade["entry_idx"])),
                }
            )
            open_trade = None

    # Equity at bar i = cash + position MTM at close.
    equity = cash + position_size * close
    equity_series = pd.Series(equity, index=ohlcv.index, name="equity")
    rets = equity_series.pct_change().fillna(0.0)
    rets.name = "ret"

    trades = pd.DataFrame(trades_records)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "entry_idx",
                "exit_idx",
                "side",
                "size",
                "entry_price",
                "exit_price",
                "pnl",
                "duration",
            ]
        )

    metrics = compute_metrics(
        returns=rets,
        trade_pnls=trades["pnl"] if not trades.empty else pd.Series(dtype=float),
        trade_durations=trades["duration"] if not trades.empty else pd.Series(dtype=float),
        position_open=pd.Series(position_size != 0, index=ohlcv.index),
        bars_per_year=bars_per_year,
    )

    logger.info(
        "Backtest done — equity {:.2f} → {:.2f} ({} trades)",
        cfg.initial_capital,
        float(equity_series.iloc[-1]),
        len(trades),
    )

    return BacktestResult(
        equity=equity_series,
        returns=rets,
        positions=pd.Series(position_size, index=ohlcv.index, name="position"),
        trades=trades,
        metrics=metrics,
    )


# Keep references to silence importer/linters.
_ = (apply_funding, equity_curve)
