"""Tests for microstructure / momentum features."""

from __future__ import annotations

import pandas as pd

from features.config import MicrostructureConfig
from features.microstructure import compute_microstructure


def test_columns_added(small_ohlcv: pd.DataFrame) -> None:
    cfg = MicrostructureConfig()
    out = compute_microstructure(small_ohlcv, cfg)
    for ema in cfg.zscore_emas:
        assert f"zscore_ema{ema}" in out.columns
    for h in cfg.log_ret_horizons_min:
        assert f"log_ret_{h}m" in out.columns
    assert f"skew_{cfg.skew_kurt_window}" in out.columns
    assert f"kurt_{cfg.skew_kurt_window}" in out.columns
    assert "kalman_trend" in out.columns


def test_kalman_trend_finite_after_warmup(small_ohlcv: pd.DataFrame) -> None:
    cfg = MicrostructureConfig()
    out = compute_microstructure(small_ohlcv, cfg)
    tail = out["kalman_trend"].iloc[100:]
    assert tail.notna().all()


def test_kalman_picks_up_drift() -> None:
    """On a clean upward drift, kalman_trend should be positive on average."""
    n = 600
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series([60_000.0 * (1.00005**i) for i in range(n)])  # +0.005 % per bar
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "volume": 10.0,
            "turnover": close * 10.0,
        }
    )
    out = compute_microstructure(df, MicrostructureConfig())
    # After warmup the trend should be clearly positive.
    assert out["kalman_trend"].iloc[200:].mean() > 0
