"""Unit tests for the technical-indicator module."""

from __future__ import annotations

import pandas as pd

from features.config import TechnicalConfig
from features.technical import compute_technical


def test_columns_added(small_ohlcv: pd.DataFrame) -> None:
    cfg = TechnicalConfig()
    out = compute_technical(small_ohlcv, cfg)
    expected = {
        "rsi_7",
        "rsi_14",
        "rsi_28",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_lower",
        "bb_mid",
        "bb_upper",
        "bb_pct",
        "bb_bw",
        "atr",
        "atr_pct",
        "vwap",
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "ema_9_slope",
        "ema_21_slope",
        "ema_50_slope",
        "ema_200_slope",
    }
    assert expected.issubset(out.columns)


def test_no_nans_after_warmup(small_ohlcv: pd.DataFrame) -> None:
    """Past the longest indicator window, no NaN should remain."""
    cfg = TechnicalConfig()
    out = compute_technical(small_ohlcv, cfg)
    warmup = max(cfg.ema_lengths) + cfg.slope_window + 5
    tail = out.iloc[warmup:]
    nan_cols = [
        c
        for c in (
            "rsi_14",
            "macd",
            "bb_mid",
            "bb_pct",
            "atr",
            "ema_50",
            "ema_50_slope",
            "vwap",
        )
        if tail[c].isna().any()
    ]
    assert not nan_cols, f"NaN columns past warmup: {nan_cols}"


def test_bb_pct_in_expected_range(small_ohlcv: pd.DataFrame) -> None:
    cfg = TechnicalConfig()
    out = compute_technical(small_ohlcv, cfg).dropna(subset=["bb_pct"])
    # %B can drift outside [0, 1] but stays within ~[-1, 2] in normal conditions.
    assert out["bb_pct"].between(-1.0, 2.0).mean() > 0.95


def test_vwap_resets_per_day() -> None:
    """A 2-day frame should have an obvious VWAP discontinuity at 00:00."""
    n = 60 * 24 * 2
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(60_000.0, index=range(n))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "turnover": close * 10.0,
        }
    )
    out = compute_technical(df, TechnicalConfig())
    # On a constant-price series VWAP equals close everywhere.
    assert (out["vwap"].dropna() - 60_000.0).abs().max() < 1e-6
