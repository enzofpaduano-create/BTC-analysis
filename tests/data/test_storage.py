"""Tests for the Parquet partitioned store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.storage import ParquetStore


def test_write_then_read_roundtrip(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    store.write("BTCUSDT", "1", synthetic_ohlcv)

    df = store.read("BTCUSDT", "1")
    assert len(df) == len(synthetic_ohlcv)
    assert list(df.columns) == list(synthetic_ohlcv.columns)
    pd.testing.assert_series_equal(
        df["close"].reset_index(drop=True),
        synthetic_ohlcv["close"].reset_index(drop=True),
    )


def test_partition_layout_uses_year_month(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    store.write("BTCUSDT", "1", synthetic_ohlcv)

    expected = tmp_path / "symbol=BTCUSDT" / "timeframe=1" / "year_month=2024-01" / "data.parquet"
    assert expected.exists()


def test_write_upserts_on_overlap(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    """Re-writing the same timestamps with new prices replaces them."""
    store = ParquetStore(tmp_path)
    store.write("BTCUSDT", "1", synthetic_ohlcv)

    overlap = synthetic_ohlcv.iloc[:50].copy()
    overlap.loc[:, "close"] = 99_999.0
    store.write("BTCUSDT", "1", overlap)

    df = store.read("BTCUSDT", "1").sort_values("timestamp").reset_index(drop=True)
    assert len(df) == len(synthetic_ohlcv)  # no duplicates
    assert (df["close"].iloc[:50] == 99_999.0).all()


def test_read_with_time_bounds(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    store.write("BTCUSDT", "1", synthetic_ohlcv)

    start = synthetic_ohlcv["timestamp"].iloc[50]
    end = synthetic_ohlcv["timestamp"].iloc[150]
    df = store.read("BTCUSDT", "1", start=start, end=end)
    assert len(df) == 101  # inclusive
    assert df["timestamp"].min() == start
    assert df["timestamp"].max() == end


def test_read_missing_partition_returns_empty(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    df = store.read("BTCUSDT", "1")
    assert df.empty
    assert "timestamp" in df.columns


def test_latest_timestamp(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    assert store.latest_timestamp("BTCUSDT", "1") is None

    store.write("BTCUSDT", "1", synthetic_ohlcv)
    assert store.latest_timestamp("BTCUSDT", "1") == synthetic_ohlcv["timestamp"].max()


def test_write_rejects_naive_timestamp(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    bad = synthetic_ohlcv.copy()
    bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="tz-aware"):
        store.write("BTCUSDT", "1", bad)


def test_write_rejects_missing_columns(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    bad = synthetic_ohlcv.drop(columns=["turnover"])
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="missing columns"):
        store.write("BTCUSDT", "1", bad)


def test_data_spans_two_months_creates_two_partitions(tmp_path: Path) -> None:
    """A frame straddling a month boundary lands in two partition dirs."""
    n = 60 * 24 * 3  # 3 days at 1min
    ts = pd.date_range("2024-01-30 00:00", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 60_000.0,
            "high": 60_010.0,
            "low": 59_990.0,
            "close": 60_005.0,
            "volume": 10.0,
            "turnover": 600_050.0,
        }
    )
    store = ParquetStore(tmp_path)
    store.write("BTCUSDT", "1", df)

    jan = tmp_path / "symbol=BTCUSDT" / "timeframe=1" / "year_month=2024-01" / "data.parquet"
    feb = tmp_path / "symbol=BTCUSDT" / "timeframe=1" / "year_month=2024-02" / "data.parquet"
    assert jan.exists()
    assert feb.exists()
