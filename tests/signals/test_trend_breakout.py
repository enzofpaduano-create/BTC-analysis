"""Tests for the trend-breakout (Donchian + ADX + HMM bull) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import TrendBreakoutADXHMM


def _hand_crafted_features(n: int = 200) -> pd.DataFrame:
    """Setup: at bar 50, close breaks above donchian_high, ADX is high, regime=bull."""
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.full(n, 60_000.0)
    high = np.full(n, 60_010.0)
    low = np.full(n, 59_990.0)
    d_high = np.full(n, 60_050.0)
    d_low = np.full(n, 59_950.0)
    adx = np.full(n, 30.0)  # strong trend
    regime = np.full(n, 2, dtype=np.int64)  # bull everywhere
    atr = np.full(n, 50.0)
    vol_60m = np.full(n, 0.5)

    # Long entry trigger at bar 50
    close[50] = 60_100.0  # > donchian_high

    return pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "high": high,
            "low": low,
            "donchian_high": d_high,
            "donchian_low": d_low,
            "adx": adx,
            "regime_hmm": regime,
            "atr": atr,
            "vol_60m": vol_60m,
        }
    )


def test_long_entry_when_breakout_with_adx_and_bull() -> None:
    feat = _hand_crafted_features()
    sig = TrendBreakoutADXHMM().generate_signals(feat).df
    assert sig["entry_long"].iloc[50]


def test_no_entry_when_adx_too_low() -> None:
    feat = _hand_crafted_features()
    feat["adx"] = 15.0  # below threshold
    sig = TrendBreakoutADXHMM().generate_signals(feat).df
    assert not sig["entry_long"].any()


def test_no_long_in_non_bull_regime() -> None:
    feat = _hand_crafted_features()
    feat["regime_hmm"] = 1  # range, not bull
    sig = TrendBreakoutADXHMM().generate_signals(feat).df
    assert not sig["entry_long"].any()


def test_short_entry_in_bear_regime() -> None:
    n = 200
    feat = _hand_crafted_features(n=n)
    feat["regime_hmm"] = 0  # bear
    feat.loc[feat.index[50], "close"] = 59_900.0  # < donchian_low
    sig = TrendBreakoutADXHMM().generate_signals(feat).df
    assert sig["entry_short"].iloc[50]


def test_atr_trailing_stop_triggers_exit() -> None:
    n = 200
    feat = _hand_crafted_features(n=n)
    # After entry at 50, drive close higher, then back below trailing stop.
    feat.loc[feat.index[51:60], "close"] = 60_300.0  # ride the trend up
    feat.loc[feat.index[60], "close"] = 60_140.0  # 60_300 - 3*50 = 60_150 → 60_140 < 60_150
    sig = TrendBreakoutADXHMM(atr_trail_mult=3.0).generate_signals(feat).df
    assert sig["entry_long"].iloc[50]
    assert sig["exit_long"].iloc[60]


def test_adx_exit_when_trend_dies() -> None:
    n = 200
    feat = _hand_crafted_features(n=n)
    # After entry, ADX collapses below the exit threshold.
    feat.loc[feat.index[55], "adx"] = 10.0
    sig = TrendBreakoutADXHMM(adx_exit_threshold=18.0).generate_signals(feat).df
    assert sig["entry_long"].iloc[50]
    assert sig["exit_long"].iloc[55]


def test_no_lookahead_truncation_invariance() -> None:
    feat_full = _hand_crafted_features(n=200)
    feat_trunc = feat_full.iloc[:120].copy().reset_index(drop=True)
    feat_full_idx = feat_full.copy().reset_index(drop=True)

    strat = TrendBreakoutADXHMM()
    sig_full = strat.generate_signals(feat_full_idx).df.iloc[:120].reset_index(drop=True)
    sig_trunc = strat.generate_signals(feat_trunc).df.reset_index(drop=True)

    pd.testing.assert_frame_equal(sig_full, sig_trunc)
