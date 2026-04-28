"""Tests for the quality-check helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.quality import (
    detect_gaps,
    detect_outliers,
    detect_zero_volume,
    run_quality_checks,
)


def test_no_outliers_on_smooth_series(synthetic_ohlcv: pd.DataFrame) -> None:
    assert detect_outliers(synthetic_ohlcv) == []


def test_outlier_is_flagged() -> None:
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 60_000.0 + np.cumsum(rng.normal(0.0, 5.0, size=n))
    close[200] *= 1.30  # +30 % spike vs prior tick — well past 8σ
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
            "turnover": close * 10.0,
        }
    )
    flagged = detect_outliers(df, window=100, n_sigma=8.0)
    assert 200 in flagged


def test_no_gaps_on_contiguous_series(synthetic_ohlcv: pd.DataFrame) -> None:
    assert detect_gaps(synthetic_ohlcv, timeframe="1") == []


def test_gap_is_detected_when_minute_missing() -> None:
    ts = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:05"],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "turnover": 1.0,
        }
    )
    gaps = detect_gaps(df, timeframe="1")
    assert len(gaps) == 1
    assert gaps[0][0] == ts[1]
    assert gaps[0][1] == ts[2]


def test_zero_volume_flag(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[5, "volume"] = 0.0
    df.loc[42, "volume"] = 0.0
    assert detect_zero_volume(df) == [5, 42]


def test_run_quality_checks_returns_clean_report(synthetic_ohlcv: pd.DataFrame) -> None:
    report = run_quality_checks(synthetic_ohlcv, timeframe="1", log_warnings=False)
    assert report.is_clean
    assert report.n_rows == len(synthetic_ohlcv)
    assert "rows=200" in report.summary()
