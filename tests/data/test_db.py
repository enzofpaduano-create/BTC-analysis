"""Tests for the DuckDB wrapper over the Parquet store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.db import DuckDBStore
from data.storage import ParquetStore


def test_duckdb_reads_partitioned_parquet(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    parquet_root = tmp_path / "parquet"
    db_path = tmp_path / "db" / "test.duckdb"
    ParquetStore(parquet_root).write("BTCUSDT", "1", synthetic_ohlcv)

    with DuckDBStore(parquet_root, db_path) as db:
        df = db.query(
            "SELECT count(*) AS n FROM ohlcv WHERE symbol = 'BTCUSDT' AND timeframe = '1'"
        )
        assert int(df["n"].iloc[0]) == len(synthetic_ohlcv)


def test_duckdb_head(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> None:
    parquet_root = tmp_path / "parquet"
    ParquetStore(parquet_root).write("BTCUSDT", "1", synthetic_ohlcv)

    with DuckDBStore(parquet_root) as db:
        head = db.head("BTCUSDT", "1", n=5)
    assert len(head) == 5
    assert list(head.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]
