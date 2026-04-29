"""Tests for the baseline Bollinger + HMM mean-reversion strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals import MeanReversionBollingerHMM
from signals.sizing import vol_target_size


def _hand_crafted_features(n: int = 100) -> pd.DataFrame:
    """Build a deterministic features frame the strategy can act on.

    Parameters chosen so that around bar 50 a long signal *should* fire:
    close drops below bb_lower, RSI < 30, regime != bear (1=range).
    """
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Keep default close strictly between bb_lower (59_900) and bb_mid (60_000)
    # so neither entry nor exit triggers on its own.
    close = np.full(n, 59_950.0)
    bb_lower = np.full(n, 59_900.0)
    bb_mid = np.full(n, 60_000.0)
    bb_upper = np.full(n, 60_100.0)
    rsi = np.full(n, 50.0)
    regime = np.full(n, 1, dtype=np.int64)  # 1 = range
    atr = np.full(n, 50.0)
    vol_60m = np.full(n, 0.5)  # 50 % annualised — typical BTC

    # Long entry trigger at bar 50
    close[50] = 59_850.0
    rsi[50] = 25.0
    # Mean revert at bar 60
    close[60] = 60_010.0  # crosses bb_mid

    return pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "rsi_14": rsi,
            "regime_hmm": regime,
            "atr": atr,
            "vol_60m": vol_60m,
        }
    )


def test_long_entry_when_conditions_met() -> None:
    feat = _hand_crafted_features()
    strat = MeanReversionBollingerHMM()
    sig = strat.generate_signals(feat).df
    assert sig["entry_long"].iloc[50]
    assert not sig["entry_long"].drop(index=feat.index[50]).any()
    assert 0 < sig["size"].iloc[50] <= 1.0


def test_exit_at_bollinger_mid() -> None:
    feat = _hand_crafted_features()
    strat = MeanReversionBollingerHMM()
    sig = strat.generate_signals(feat).df
    # Entry at 50, mid-revert at 60.
    assert sig["exit_long"].iloc[60]


def test_no_entry_in_bear_regime() -> None:
    feat = _hand_crafted_features()
    feat["regime_hmm"] = 0  # bear everywhere
    sig = MeanReversionBollingerHMM().generate_signals(feat).df
    assert not sig["entry_long"].any()


def test_no_short_in_bull_regime() -> None:
    feat = _hand_crafted_features()
    feat["regime_hmm"] = 2  # bull everywhere
    feat.loc[feat.index[50], "close"] = 60_150.0  # touch upper
    feat.loc[feat.index[50], "rsi_14"] = 75.0
    sig = MeanReversionBollingerHMM().generate_signals(feat).df
    assert not sig["entry_short"].any()


def test_atr_stop_triggers_exit() -> None:
    """A sharp move below entry - 2*ATR should force a stop exit."""
    feat = _hand_crafted_features(n=80)
    # Force a stop-out at bar 55 (entry at 50, ATR=50 → stop at 59,750)
    feat.loc[feat.index[55], "close"] = 59_500.0
    sig = MeanReversionBollingerHMM(atr_stop_mult=2.0).generate_signals(feat).df
    assert sig["entry_long"].iloc[50]
    assert sig["exit_long"].iloc[55]


def test_timeout_exit() -> None:
    """If neither target nor stop is hit, strategy exits on timeout."""
    n = 200
    feat = _hand_crafted_features(n=n)
    # After bar 50, lock close just below bb_mid so target never triggers,
    # and well above the stop so we exit purely on timeout.
    feat.loc[feat.index[51:], "close"] = 59_900.0  # never reaches mid
    feat.loc[feat.index[50], "close"] = 59_850.0  # entry trigger only at 50
    sig = MeanReversionBollingerHMM(timeout_bars=10).generate_signals(feat).df
    assert sig["entry_long"].iloc[50]
    assert sig["exit_long"].iloc[60]


def test_no_overlapping_positions() -> None:
    """Strategy must never emit a second entry while a position is open."""
    n = 200
    feat = _hand_crafted_features(n=n)
    # Multiple touches of bb_lower with low RSI between bar 50 and 60.
    for i in range(50, 60):
        feat.loc[feat.index[i], "close"] = 59_850.0
        feat.loc[feat.index[i], "rsi_14"] = 25.0
    sig = MeanReversionBollingerHMM().generate_signals(feat).df
    # Only the first one should trigger.
    assert sig["entry_long"].sum() == 1
    assert sig["entry_long"].iloc[50]


def test_strategy_respects_warmup_nans() -> None:
    """Bars where required inputs are NaN must produce no signal."""
    feat = _hand_crafted_features()
    feat.loc[feat.index[:30], ["bb_lower", "bb_upper", "bb_mid"]] = np.nan
    sig = MeanReversionBollingerHMM().generate_signals(feat).df
    assert not sig["entry_long"].iloc[:30].any()


def test_no_lookahead_truncation_invariance() -> None:
    """Strategy signals at bar i must be identical between truncated/full runs."""
    feat_full = _hand_crafted_features(n=200)
    feat_trunc = feat_full.iloc[:120].copy().reset_index(drop=True)
    feat_full_idx = feat_full.copy().reset_index(drop=True)

    strat = MeanReversionBollingerHMM()
    sig_full = strat.generate_signals(feat_full_idx).df.iloc[:120].reset_index(drop=True)
    sig_trunc = strat.generate_signals(feat_trunc).df.reset_index(drop=True)

    pd.testing.assert_frame_equal(sig_full, sig_trunc)


# -- vol_target_size --


@pytest.mark.parametrize(
    "vol,target,expected",
    [
        (0.5, 0.05, 0.10),  # 0.05 / 0.5 = 0.10
        (1.0, 0.10, 0.10),  # exact match
        (0.0, 0.10, 0.0),  # invalid vol → 0
        (np.nan, 0.10, 0.0),  # NaN vol → 0
    ],
)
def test_vol_target_size(vol: float, target: float, expected: float) -> None:
    s = vol_target_size(realized_vol_annualized=vol, target_vol_per_trade=target)
    assert abs(s - expected) < 1e-9


def test_vol_target_size_capped() -> None:
    s = vol_target_size(realized_vol_annualized=0.05, target_vol_per_trade=0.50, max_size=1.0)
    assert s == 1.0  # raw would be 10, capped at 1
