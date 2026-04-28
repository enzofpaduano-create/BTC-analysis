"""Data layer — Bybit ingestion, Parquet storage, DuckDB queries.

Public 3-line API::

    from data import download_history, read_history, ParquetStore

    store = ParquetStore("./data_store/parquet")
    df = download_history(symbol="BTCUSDT", timeframe="1",
                          start="2024-04-28", end="2026-04-28", store=store)
    df2 = read_history(symbol="BTCUSDT", timeframe="1",
                       start="2024-04-28", end="2026-04-28",
                       parquet_root="./data_store/parquet")
"""

from data.client import BybitClient, KlineRequest, bybit_client
from data.db import DuckDBStore
from data.download import download_history, read_history
from data.quality import QualityReport, run_quality_checks
from data.storage import ParquetStore
from data.stream import stream_klines

__all__ = [
    "BybitClient",
    "DuckDBStore",
    "KlineRequest",
    "ParquetStore",
    "QualityReport",
    "bybit_client",
    "download_history",
    "read_history",
    "run_quality_checks",
    "stream_klines",
]
