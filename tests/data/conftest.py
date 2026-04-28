"""Fixtures for the data-layer tests."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest


def _row(ts_ms: int, price: float, volume: float = 10.0) -> list[str]:
    """Build one Bybit-shaped kline row (all strings, as Bybit returns)."""
    return [
        str(ts_ms),
        f"{price:.2f}",
        f"{price + 5:.2f}",
        f"{price - 5:.2f}",
        f"{price + 2:.2f}",
        f"{volume:.4f}",
        f"{price * volume:.4f}",
    ]


@pytest.fixture
def make_bybit_rows() -> Callable[..., list[list[str]]]:
    """Factory returning Bybit-shaped kline rows in *descending* time order."""

    def _factory(
        n: int = 100,
        start_ts_ms: int = 1_700_000_000_000,
        interval_ms: int = 60_000,
        base_price: float = 60_000.0,
    ) -> list[list[str]]:
        rows = [_row(start_ts_ms + i * interval_ms, base_price + i) for i in range(n)]
        rows.reverse()  # Bybit returns newest first
        return rows

    return _factory


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """A clean 200-row OHLCV DataFrame on M1 starting 2024-01-01 UTC."""
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(60_000.0, index=range(n)) + pd.Series(range(n)).astype(float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 1.0,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": 10.0,
            "turnover": close * 10.0,
        }
    )
