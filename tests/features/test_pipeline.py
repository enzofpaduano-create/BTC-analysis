"""End-to-end test of the features pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from features import FeaturesConfig, compute_features


def test_compute_features_runs(synthetic_btc_m1: pd.DataFrame) -> None:
    cfg = FeaturesConfig(bar_minutes=1)
    cfg.volatility.garch_min_obs = 300
    cfg.volatility.garch_refit_every = 500
    cfg.regime.hmm_min_obs = 300
    cfg.regime.hmm_refit_every = 500
    cfg.regime.cp_window = 300
    cfg.regime.cp_refit_every = 100

    out = compute_features(synthetic_btc_m1, cfg)
    assert len(out) == len(synthetic_btc_m1)
    # A handful of expected columns from each module:
    for col in (
        "rsi_14",
        "macd",
        "atr",
        "vwap",
        "ema_50",
        "vol_60m",
        "garch_vol_1step",
        "zscore_ema50",
        "kalman_trend",
        "regime_hmm",
        "cp_segment",
    ):
        assert col in out.columns, f"missing {col}"


def test_compute_features_rejects_naive_timestamp(small_ohlcv: pd.DataFrame) -> None:
    bad = small_ohlcv.copy()
    bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        compute_features(bad)


def test_compute_features_rejects_unsorted(small_ohlcv: pd.DataFrame) -> None:
    bad = small_ohlcv.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        compute_features(bad)
