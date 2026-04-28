"""Fixtures for backtest tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Deterministic ~600-bar OHLCV in UTC with mild trending noise."""
    rng = np.random.default_rng(7)
    n = 600
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    log_ret = rng.normal(1e-4, 0.0008, size=n)  # slight upward drift
    close = np.exp(np.cumsum(log_ret)) * 60_000.0
    high = close * (1 + rng.uniform(0, 0.0005, size=n))
    low = close * (1 - rng.uniform(0, 0.0005, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(5.0, 50.0, size=n)
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


class FixedSignalsStrategy(Strategy):
    """Test helper — emits a pre-defined signal frame."""

    name = "fixed"

    def __init__(self, signals_df: pd.DataFrame) -> None:
        self._signals = signals_df

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        df = self._signals.reindex(features.index, fill_value=False)
        if "size" in self._signals.columns:
            df["size"] = self._signals["size"].reindex(features.index, fill_value=0.0)
        else:
            df["size"] = 1.0
        return SignalFrame(df=df[list(SIGNAL_COLUMNS)])


@pytest.fixture
def fixed_signals_factory() -> type[FixedSignalsStrategy]:
    return FixedSignalsStrategy
