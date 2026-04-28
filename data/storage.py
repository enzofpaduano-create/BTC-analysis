"""Parquet-backed OHLCV storage, partitioned by symbol/timeframe/year-month.

Layout::

    {root}/
      symbol=BTCUSDT/
        timeframe=1/
          year_month=2024-03/
            data.parquet
          year_month=2024-04/
            data.parquet
        timeframe=5/
          ...

One file per (symbol, timeframe, year-month) partition. Writes use
"upsert" semantics: rows already present (same `timestamp`) are
overwritten by the incoming rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from data.schemas import OHLCV_COLUMNS


class ParquetStore:
    """Read/write OHLCV partitions on the local filesystem.

    Args:
        root: Base directory. Created on demand.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- Path helpers -----------------------------------------------------

    def partition_dir(self, symbol: str, timeframe: str, year_month: str) -> Path:
        """Resolve the directory for a given partition."""
        return (
            self.root / f"symbol={symbol}" / f"timeframe={timeframe}" / f"year_month={year_month}"
        )

    def partition_file(self, symbol: str, timeframe: str, year_month: str) -> Path:
        return self.partition_dir(symbol, timeframe, year_month) / "data.parquet"

    # -- Write ------------------------------------------------------------

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Upsert an OHLCV DataFrame into the right monthly partitions.

        Args:
            symbol: Asset symbol (e.g. "BTCUSDT").
            timeframe: Bybit timeframe code (e.g. "1", "5", "60", "D").
            df: DataFrame with columns matching `OHLCV_COLUMNS`. The
                `timestamp` column must be tz-aware UTC.
        """
        if df.empty:
            return
        _validate_ohlcv(df)

        # Group rows by year-month and write each partition.
        ym = df["timestamp"].dt.strftime("%Y-%m")
        for year_month, chunk in df.groupby(ym, sort=True):
            self._write_partition(symbol, timeframe, str(year_month), chunk)

    def _write_partition(
        self, symbol: str, timeframe: str, year_month: str, new_rows: pd.DataFrame
    ) -> None:
        path = self.partition_file(symbol, timeframe, year_month)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = (
                combined.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        else:
            combined = new_rows.sort_values("timestamp").reset_index(drop=True)

        combined.to_parquet(path, index=False)
        logger.debug(
            "Wrote {} rows to {} (partition has {} rows total)",
            len(new_rows),
            path.relative_to(self.root),
            len(combined),
        )

    # -- Read -------------------------------------------------------------

    def read(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read a contiguous slice of OHLCV from disk.

        Args:
            symbol: Asset symbol.
            timeframe: Bybit timeframe code.
            start: Inclusive lower bound (UTC). Reads from earliest if None.
            end: Inclusive upper bound (UTC). Reads to latest if None.

        Returns:
            DataFrame sorted by timestamp ascending. Empty if no data.
        """
        timeframe_dir = self.root / f"symbol={symbol}" / f"timeframe={timeframe}"
        if not timeframe_dir.exists():
            return _empty_ohlcv()

        frames: list[pd.DataFrame] = []
        for ym_dir in sorted(timeframe_dir.glob("year_month=*")):
            ym = ym_dir.name.split("=", 1)[1]
            if not _partition_intersects(ym, start, end):
                continue
            file_path = ym_dir / "data.parquet"
            if file_path.exists():
                frames.append(pd.read_parquet(file_path))

        if not frames:
            return _empty_ohlcv()

        df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        return df.reset_index(drop=True)

    def latest_timestamp(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        """Return the most recent stored timestamp, or None if empty."""
        df = self.read(symbol, timeframe)
        if df.empty:
            return None
        return pd.Timestamp(df["timestamp"].max())


# -- Internal helpers ------------------------------------------------------


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS}).astype(
        {"timestamp": "datetime64[ns, UTC]"}
    )


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {sorted(missing)}")
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise TypeError("`timestamp` must be a datetime dtype")
    if ts.dt.tz is None:
        raise ValueError("`timestamp` must be tz-aware (UTC)")


def _partition_intersects(
    year_month: str, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> bool:
    """Cheap pre-filter: does this YYYY-MM partition possibly contain rows?"""
    part_start = pd.Timestamp(f"{year_month}-01", tz="UTC")
    part_end = part_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    if start is not None and part_end < start:
        return False
    return not (end is not None and part_start > end)
