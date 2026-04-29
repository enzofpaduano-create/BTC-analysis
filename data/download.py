"""Historical OHLCV downloader with pagination, dedup, and resume."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from data.client import BybitClient, KlineRequest
from data.quality import run_quality_checks
from data.schemas import (
    OHLCV_COLUMNS,
    TIMEFRAME_MINUTES,
    Category,
    Timeframe,
)
from data.storage import ParquetStore


def _to_utc_ms(value: pd.Timestamp | datetime | str) -> int:
    """Coerce any reasonable date input to a UTC millisecond timestamp."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _bybit_rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    """Convert Bybit's raw kline rows to a canonical OHLCV DataFrame.

    Bybit returns each row as 7 strings:
    ``[start_ms, open, high, low, close, volume, turnover]``
    in *descending* time order. We reverse to ascending, parse types,
    and convert the timestamp to UTC tz-aware.
    """
    if not rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS)).astype(
            {"timestamp": "datetime64[ns, UTC]"}
        )
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df.sort_values("timestamp").reset_index(drop=True)


def download_history(
    *,
    symbol: str,
    timeframe: Timeframe,
    start: pd.Timestamp | datetime | str,
    end: pd.Timestamp | datetime | str | None = None,
    category: Category = "linear",
    store: ParquetStore | None = None,
    client: BybitClient | None = None,
    batch_limit: int = 1000,
    quality_checks: bool = True,
) -> pd.DataFrame:
    """Download OHLCV history from Bybit and persist it to Parquet.

    Resumable: if ``store`` already contains bars for ``(symbol, timeframe)``,
    the download starts from ``max(stored_timestamp + 1 interval, start)``
    so the call is idempotent and incremental.

    Args:
        symbol: Asset symbol, e.g. ``"BTCUSDT"``.
        timeframe: Bybit timeframe code (``"1"``, ``"5"``, …, ``"D"``).
        start: Lower bound (inclusive). Naive datetimes are treated as UTC.
        end: Upper bound (inclusive). Defaults to "now" UTC.
        category: ``"linear"`` perpetual, ``"spot"``, or ``"inverse"``.
        store: Where to persist. Required for resume; optional for ad-hoc.
        client: Reuse an open `BybitClient`; one is created if None.
        batch_limit: Bybit returns at most 1000 rows per call.
        quality_checks: Run outlier / gap / zero-volume checks at the end.

    Returns:
        The newly downloaded rows (ascending by timestamp). Existing rows
        already present in ``store`` are NOT re-fetched.
    """
    end_ts = pd.Timestamp(end if end is not None else datetime.now(UTC))
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    requested_start = pd.Timestamp(start)
    if requested_start.tzinfo is None:
        requested_start = requested_start.tz_localize("UTC")

    # Resume from disk if anything is already stored. NOTE: this only extends
    # the upper bound — if `requested_start` predates the earliest stored bar,
    # we DO NOT backfill the gap automatically. Delete the partition or call
    # `store.read(...)` to get whatever is on disk.
    effective_start = requested_start
    if store is not None:
        latest = store.latest_timestamp(symbol, timeframe)
        if latest is not None:
            interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
            resume_from = latest + interval
            if resume_from > effective_start:
                logger.info(
                    "Resuming {} {}: last stored={}, fetching from {}",
                    symbol,
                    timeframe,
                    latest,
                    resume_from,
                )
                # Warn if the user expected a window that starts BEFORE what
                # the store currently covers — those older bars won't be
                # fetched by this call.
                earliest_stored = store.read(symbol, timeframe).get("timestamp")
                if (
                    earliest_stored is not None
                    and not earliest_stored.empty
                    and requested_start < earliest_stored.iloc[0]
                ):
                    logger.warning(
                        "Requested start {} is BEFORE earliest stored {}. "
                        "Backward gap will NOT be filled — delete the partition "
                        "to refetch from scratch.",
                        requested_start,
                        earliest_stored.iloc[0],
                    )
                effective_start = resume_from

    if effective_start >= end_ts:
        logger.info("Nothing to download for {} {} — already up to date", symbol, timeframe)
        return pd.DataFrame(columns=list(OHLCV_COLUMNS)).astype(
            {"timestamp": "datetime64[ns, UTC]"}
        )

    owns_client = client is None
    if client is None:
        client = BybitClient()
        client.open()

    try:
        all_chunks: list[pd.DataFrame] = []
        start_ms = _to_utc_ms(effective_start)
        cursor_end_ms = _to_utc_ms(end_ts)

        # Bybit returns up to `limit` rows newest-first within [start, end].
        # To cover a long range we page BACKWARD: take the oldest bar of each
        # batch and use it as the next call's upper bound, until we reach `start`.
        while cursor_end_ms > start_ms:
            req = KlineRequest(
                category=category,
                symbol=symbol,
                interval=timeframe,
                start_ms=start_ms,
                end_ms=cursor_end_ms,
                limit=batch_limit,
            )
            rows = client.get_kline(req)
            if not rows:
                break

            chunk = _bybit_rows_to_df(rows)
            all_chunks.append(chunk)

            oldest_ms = int(chunk["timestamp"].min().timestamp() * 1000)
            new_cursor = oldest_ms - 1
            if new_cursor >= cursor_end_ms:
                # Defensive: avoid infinite loop if Bybit returns the same page.
                logger.warning("Cursor did not advance, stopping pagination")
                break
            cursor_end_ms = new_cursor

            logger.debug(
                "Fetched {} rows down to {} ({} chunks total)",
                len(chunk),
                chunk["timestamp"].min(),
                len(all_chunks),
            )
    finally:
        if owns_client:
            client.close()

    if not all_chunks:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS)).astype(
            {"timestamp": "datetime64[ns, UTC]"}
        )

    df = (
        pd.concat(all_chunks, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Trim to requested range (Bybit can return a tail outside [start, end]).
    df = df[(df["timestamp"] >= requested_start) & (df["timestamp"] <= end_ts)].reset_index(
        drop=True
    )

    if store is not None and not df.empty:
        store.write(symbol, timeframe, df)

    if quality_checks and not df.empty:
        report = run_quality_checks(df, timeframe=timeframe)
        logger.info("Quality: {}", report.summary())

    logger.info(
        "Downloaded {} bars for {} {} ({} → {})",
        len(df),
        symbol,
        timeframe,
        df["timestamp"].min() if not df.empty else "—",
        df["timestamp"].max() if not df.empty else "—",
    )
    return df


def read_history(
    *,
    symbol: str,
    timeframe: Timeframe,
    start: pd.Timestamp | datetime | str | None = None,
    end: pd.Timestamp | datetime | str | None = None,
    parquet_root: Path,
) -> pd.DataFrame:
    """Read OHLCV from the local Parquet store.

    Convenience wrapper around `ParquetStore.read` — exists so consumers
    can stay with the `from data import ...` 3-line API.
    """
    store = ParquetStore(parquet_root)
    s = pd.Timestamp(start) if start is not None else None
    e = pd.Timestamp(end) if end is not None else None
    if s is not None and s.tzinfo is None:
        s = s.tz_localize("UTC")
    if e is not None and e.tzinfo is None:
        e = e.tz_localize("UTC")
    return store.read(symbol, timeframe, start=s, end=e)
