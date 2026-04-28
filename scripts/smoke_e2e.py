"""End-to-end smoke test of the data layer against the real Bybit mainnet.

Downloads ~1 day of BTCUSDT M1 from Bybit (no API key required for public
klines), saves to a temp Parquet store, reads it back, and prints a brief
summary. Intentionally fetches a small range to stay polite.

Run::

    uv run python scripts/smoke_e2e.py
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from core.logging import setup_logging
from data import BybitClient, DuckDBStore, ParquetStore, download_history, read_history


def main() -> None:
    setup_logging()

    out = Path("./data_store/smoke")
    if out.exists():
        shutil.rmtree(out)
    parquet_root = out / "parquet"
    duckdb_path = out / "db" / "btc.duckdb"
    store = ParquetStore(parquet_root)

    end = datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(hours=24)
    logger.info("Smoke: downloading BTCUSDT M1 from {} to {}", start, end)

    # Use mainnet for real public-kline data (testnet returns synthetic prices
    # with zero volume). No API key needed for public klines.
    with BybitClient(testnet=False) as client:
        df = download_history(
            symbol="BTCUSDT",
            timeframe="1",
            start=start,
            end=end,
            store=store,
            client=client,
        )
    logger.info("Smoke: downloaded {} bars; head=\n{}", len(df), df.head(3))

    df_back = read_history(
        symbol="BTCUSDT",
        timeframe="1",
        start=start,
        end=end,
        parquet_root=parquet_root,
    )
    assert len(df_back) == len(
        df
    ), f"roundtrip mismatch: downloaded={len(df)} read_back={len(df_back)}"
    logger.success("Parquet roundtrip OK ({} bars)", len(df_back))

    with DuckDBStore(parquet_root, duckdb_path) as db:
        out_df = db.query(
            "SELECT min(timestamp) AS first, max(timestamp) AS last, "
            "count(*) AS n FROM ohlcv WHERE symbol='BTCUSDT' AND timeframe='1'"
        )
    logger.success("DuckDB query OK: {}", out_df.to_dict("records")[0])


if __name__ == "__main__":
    main()
