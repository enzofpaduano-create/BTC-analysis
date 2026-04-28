"""Fixtures for the features tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_btc_m1() -> pd.DataFrame:
    """A reproducible 2000-bar OHLCV series resembling BTC M1.

    Built from a Gaussian-random-walk on log-price with a vol regime
    switch at bar 1000 (so HMM/PELT have something to detect).
    """
    rng = np.random.default_rng(42)
    n = 2000
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    # Two-regime vol: low vol then high vol.
    sigma = np.concatenate(
        [
            np.full(1000, 0.0005),  # low vol
            np.full(1000, 0.0020),  # 4x higher
        ]
    )
    log_ret = rng.normal(0.0, sigma)
    log_price = np.cumsum(log_ret) + np.log(60_000.0)
    close = np.exp(log_price)
    high = close * (1 + rng.uniform(0, 0.0003, size=n))
    low = close * (1 - rng.uniform(0, 0.0003, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(1.0, 100.0, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover": close * volume,
        }
    )


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """A cheap 600-bar OHLCV used by fast unit tests of individual modules."""
    rng = np.random.default_rng(0)
    n = 600
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    log_ret = rng.normal(0.0, 0.001, size=n)
    close = np.exp(np.cumsum(log_ret)) * 60_000.0
    high = close * 1.0005
    low = close * 0.9995
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 10.0,
            "turnover": close * 10.0,
        }
    )
