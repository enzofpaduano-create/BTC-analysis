"""DuckDB wrapper for ad-hoc SQL queries against the Parquet store.

DuckDB reads Parquet directly via Hive-style partition discovery, so we
don't load anything into the database — we just expose a SQL view over
the partitioned files. Cheap and fast.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger


class DuckDBStore:
    """Lightweight DuckDB connection bound to the Parquet store.

    Args:
        parquet_root: Same root used by `ParquetStore`.
        db_path: Path to the DuckDB file (created on first use). Use
            `:memory:` for ephemeral connections.
    """

    def __init__(self, parquet_root: Path, db_path: Path | str = ":memory:") -> None:
        self.parquet_root = Path(parquet_root)
        self.db_path = db_path if db_path == ":memory:" else Path(db_path)
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None

    # -- Lifecycle --------------------------------------------------------

    def open(self) -> None:
        if self._conn is not None:
            return
        path = str(self.db_path)
        self._conn = duckdb.connect(path)
        logger.debug("DuckDB connection opened at {}", path)

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def __enter__(self) -> DuckDBStore:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- Queries ----------------------------------------------------------

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return the result as a pandas DataFrame.

        Inside the SQL, refer to the OHLCV store as the table function
        ``ohlcv()`` (registered automatically). Example::

            store.query("SELECT * FROM ohlcv() WHERE symbol='BTCUSDT' LIMIT 5")

        Args:
            sql: Any valid DuckDB SQL.

        Returns:
            Pandas DataFrame with the query result.
        """
        self._ensure_view()
        return self.conn.execute(sql).fetch_df()

    def _ensure_view(self) -> None:
        """Create or replace the `ohlcv` view that scans all partitions."""
        glob = str(self.parquet_root / "symbol=*" / "timeframe=*" / "year_month=*" / "data.parquet")
        # `hive_partitioning=1` makes DuckDB extract `symbol`, `timeframe`,
        # `year_month` from the directory names automatically.
        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW ohlcv AS
            SELECT *
            FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
            """
        )

    def head(self, symbol: str, timeframe: str, n: int = 10) -> pd.DataFrame:
        """Convenience: first ``n`` rows for a (symbol, timeframe)."""
        return self.query(
            f"""
            SELECT timestamp, open, high, low, close, volume, turnover
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            ORDER BY timestamp
            LIMIT {n}
            """
        )
