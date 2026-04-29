"""End-to-end tests of the backtest engine — including the no-leak assertion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest
from backtest.config import CostsConfig
from backtest.strategy import SIGNAL_COLUMNS


def _empty_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        dict.fromkeys(SIGNAL_COLUMNS[:-1], False) | {"size": 0.0},
        index=index,
    )


def test_backtest_with_no_signals_keeps_capital(
    synthetic_ohlcv: pd.DataFrame,
    fixed_signals_factory: type,
) -> None:
    sig = _empty_signals(synthetic_ohlcv.index)
    strat = fixed_signals_factory(sig)
    cfg = BacktestConfig(initial_capital=10_000.0, bar_minutes=1)

    res = run_backtest(ohlcv=synthetic_ohlcv, features=synthetic_ohlcv, strategy=strat, cfg=cfg)
    assert res.metrics.n_trades == 0
    assert abs(float(res.equity.iloc[-1]) - 10_000.0) < 1e-6
    assert (res.positions == 0).all()


def test_single_long_trade_realises_market_move(
    synthetic_ohlcv: pd.DataFrame,
    fixed_signals_factory: type,
) -> None:
    """Buy at bar 50, sell at bar 200 — equity gain ≈ (exit-entry)*size - costs."""
    sig = _empty_signals(synthetic_ohlcv.index)
    sig.loc[synthetic_ohlcv.index[50], "entry_long"] = True
    sig.loc[synthetic_ohlcv.index[50], "size"] = 1.0  # use full equity
    sig.loc[synthetic_ohlcv.index[200], "exit_long"] = True
    strat = fixed_signals_factory(sig)
    # Zero costs so the equity arithmetic is exact (1 BTC × price move).
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        bar_minutes=1,
        costs=CostsConfig(
            spread_bps=0.0,
            taker_fee_bps=0.0,
            slippage_bps_fixed=0.0,
            slippage_prop_coeff_bps=0.0,
            funding_annual_bps=0.0,
        ),
    )

    res = run_backtest(ohlcv=synthetic_ohlcv, features=synthetic_ohlcv, strategy=strat, cfg=cfg)
    assert res.metrics.n_trades == 1
    trade = res.trades.iloc[0]
    # The strategy fired at bar 50 → fill at bar 51's close. Same for exit.
    assert int(trade["entry_idx"]) == 51
    assert int(trade["exit_idx"]) == 201
    assert int(trade["side"]) == 1

    # Cost-free equity must equal initial + (exit_close - entry_close) * size.
    entry_close = float(synthetic_ohlcv["close"].iloc[51])
    exit_close = float(synthetic_ohlcv["close"].iloc[201])
    size_btc = 10_000.0 / entry_close  # full-equity sizing at entry close
    expected_pnl = (exit_close - entry_close) * size_btc
    final_equity = float(res.equity.iloc[-1])
    assert abs(final_equity - (10_000.0 + expected_pnl)) < 1e-6, (
        f"equity accounting wrong: got {final_equity:.4f}, "
        f"expected {10_000.0 + expected_pnl:.4f}"
    )


def test_single_short_trade_pnl_correct(
    synthetic_ohlcv: pd.DataFrame,
    fixed_signals_factory: type,
) -> None:
    """A short profits when the price falls."""
    # Force the price to fall after bar 50 so the short is profitable.
    df = synthetic_ohlcv.copy()
    factor = np.linspace(1.0, 0.95, len(df))
    df.loc[df.index[51:], "close"] = df["close"].iloc[51:].to_numpy() * factor[51:]

    sig = _empty_signals(df.index)
    sig.loc[df.index[50], "entry_short"] = True
    sig.loc[df.index[50], "size"] = 1.0
    sig.loc[df.index[200], "exit_short"] = True
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        bar_minutes=1,
        costs=CostsConfig(
            spread_bps=0.0,
            taker_fee_bps=0.0,
            slippage_bps_fixed=0.0,
            slippage_prop_coeff_bps=0.0,
            funding_annual_bps=0.0,
        ),
    )
    res = run_backtest(ohlcv=df, features=df, strategy=fixed_signals_factory(sig), cfg=cfg)

    entry_close = float(df["close"].iloc[51])
    exit_close = float(df["close"].iloc[201])
    size_btc = 10_000.0 / entry_close
    expected_pnl = (entry_close - exit_close) * size_btc  # profit when price drops
    final_equity = float(res.equity.iloc[-1])
    assert exit_close < entry_close  # sanity
    assert expected_pnl > 0  # short was profitable
    assert abs(final_equity - (10_000.0 + expected_pnl)) < 1e-6


def test_no_lookahead_truncation_invariance(
    synthetic_ohlcv: pd.DataFrame,
    fixed_signals_factory: type,
) -> None:
    """Engine result on a truncated frame must match the full run on the
    overlapping bars.

    We pre-compute signals on the FULL frame so the strategy itself can't
    introduce a leak — what we're testing here is the engine's bookkeeping.
    """
    n_trunc = 400

    sig_full = _empty_signals(synthetic_ohlcv.index)
    sig_full.loc[synthetic_ohlcv.index[100], "entry_long"] = True
    sig_full.loc[synthetic_ohlcv.index[100], "size"] = 1.0
    sig_full.loc[synthetic_ohlcv.index[300], "exit_long"] = True

    cfg = BacktestConfig(initial_capital=10_000.0, bar_minutes=1)

    full_strat = fixed_signals_factory(sig_full)
    res_full = run_backtest(
        ohlcv=synthetic_ohlcv, features=synthetic_ohlcv, strategy=full_strat, cfg=cfg
    )

    trunc_ohlcv = synthetic_ohlcv.iloc[:n_trunc].copy().reset_index(drop=True)
    trunc_sig = sig_full.iloc[:n_trunc].copy().reset_index(drop=True)
    trunc_strat = fixed_signals_factory(trunc_sig)
    res_trunc = run_backtest(ohlcv=trunc_ohlcv, features=trunc_ohlcv, strategy=trunc_strat, cfg=cfg)

    eq_full = res_full.equity.iloc[:n_trunc].to_numpy()
    eq_trunc = res_trunc.equity.to_numpy()
    assert np.allclose(eq_full, eq_trunc, rtol=1e-12, atol=1e-9), (
        f"engine leaked future info: max |Δ| = " f"{np.max(np.abs(eq_full - eq_trunc)):.3e}"
    )
    pos_full = res_full.positions.iloc[:n_trunc].to_numpy()
    pos_trunc = res_trunc.positions.to_numpy()
    assert np.array_equal(pos_full, pos_trunc), "position history diverged"


def test_metrics_dict_round_trips(
    synthetic_ohlcv: pd.DataFrame,
    fixed_signals_factory: type,
) -> None:
    """Sanity: the metrics dataclass exports cleanly."""
    sig = _empty_signals(synthetic_ohlcv.index)
    strat = fixed_signals_factory(sig)
    cfg = BacktestConfig(initial_capital=10_000.0, bar_minutes=1)
    res = run_backtest(ohlcv=synthetic_ohlcv, features=synthetic_ohlcv, strategy=strat, cfg=cfg)
    d = res.metrics.to_dict()
    assert "sharpe" in d and "max_drawdown" in d and "exposure" in d
