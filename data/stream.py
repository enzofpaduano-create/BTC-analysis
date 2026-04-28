"""Polling-based "stream" of fresh OHLCV bars.

Strategy: poll Bybit for the last few klines every N seconds. Anything
newer than the last bar we've seen is forwarded to the user's callback
and persisted to the Parquet store (if provided).

This is intentionally simpler than a WebSocket subscription:
  - REST polling has no socket-lifecycle issues.
  - We only care about *closed* bars (callback fires on confirmed bars only).
  - Latency is bounded by the poll interval, which is fine for short-term
    trading on M1/M5 timeframes (alerts, not HFT).

A WS-based variant can be added later if sub-second alerts are needed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event
from typing import Any

import pandas as pd
from loguru import logger

from data.client import BybitClient, KlineRequest
from data.download import _bybit_rows_to_df
from data.schemas import TIMEFRAME_MINUTES, Category, Timeframe
from data.storage import ParquetStore

BarCallback = Callable[[pd.DataFrame], None]


def stream_klines(
    *,
    symbol: str,
    timeframe: Timeframe,
    on_bar: BarCallback,
    category: Category = "linear",
    poll_interval_s: float | None = None,
    lookback: int = 5,
    store: ParquetStore | None = None,
    stop_event: Event | None = None,
    client: BybitClient | None = None,
) -> None:
    """Block forever, calling ``on_bar(df)`` for every newly closed bar.

    Args:
        symbol: Asset symbol.
        timeframe: Bybit interval code.
        on_bar: User callback invoked with a 1-row DataFrame per new bar.
        category: ``"linear"`` (default), ``"spot"``, or ``"inverse"``.
        poll_interval_s: Seconds between polls. Defaults to 1/4 of the
            timeframe (e.g. 15 s for M1, 75 s for M5) to balance freshness
            and rate-limit headroom.
        lookback: How many recent bars to fetch each poll. ≥2 is required
            so we always overlap the previous poll's last bar.
        store: If given, new closed bars are written to disk too.
        stop_event: Optional `threading.Event` to break the loop cleanly.
        client: Reuse an open `BybitClient`; one is created if None.
    """
    if lookback < 2:  # noqa: PLR2004 — need ≥2 to overlap consecutive polls
        raise ValueError("lookback must be >= 2 to detect new bars reliably")

    interval_min = TIMEFRAME_MINUTES[timeframe]
    if poll_interval_s is None:
        poll_interval_s = max(2.0, interval_min * 60.0 / 4)

    stop_event = stop_event or Event()
    last_seen_ts: pd.Timestamp | None = None

    owns_client = client is None
    if client is None:
        client = BybitClient()
        client.open()

    logger.info(
        "Streaming {} {} (poll every {:.0f}s, lookback={})",
        symbol,
        timeframe,
        poll_interval_s,
        lookback,
    )

    try:
        while not stop_event.is_set():
            poll_start = time.monotonic()
            try:
                bars = _poll_recent_bars(
                    client=client,
                    category=category,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback=lookback,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Polling failure (will retry): {}", exc)
                _wait(stop_event, poll_interval_s)
                continue

            if bars.empty:
                _wait(stop_event, poll_interval_s)
                continue

            # The latest bar in the response may still be open (forming).
            # We only emit *closed* bars: drop the most recent one if its
            # close time is in the future.
            now = pd.Timestamp.now(tz="UTC")
            interval = pd.Timedelta(minutes=interval_min)
            closed = bars[bars["timestamp"] + interval <= now]

            new_bars = (
                closed if last_seen_ts is None else closed[closed["timestamp"] > last_seen_ts]
            )

            if not new_bars.empty:
                if store is not None:
                    store.write(symbol, timeframe, new_bars)
                for _, row in new_bars.iterrows():
                    on_bar(pd.DataFrame([row]))
                last_seen_ts = pd.Timestamp(new_bars["timestamp"].max())
                logger.debug(
                    "Emitted {} new closed bar(s); last_seen_ts={}",
                    len(new_bars),
                    last_seen_ts,
                )

            elapsed = time.monotonic() - poll_start
            _wait(stop_event, max(0.0, poll_interval_s - elapsed))
    finally:
        if owns_client:
            client.close()


def _poll_recent_bars(
    *,
    client: BybitClient,
    category: Category,
    symbol: str,
    timeframe: Timeframe,
    lookback: int,
) -> pd.DataFrame:
    """Fetch the most recent ``lookback`` bars."""
    interval_min = TIMEFRAME_MINUTES[timeframe]
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = now_ms - (interval_min * 60_000 * (lookback + 2))
    rows = client.get_kline(
        KlineRequest(
            category=category,
            symbol=symbol,
            interval=timeframe,
            start_ms=start_ms,
            end_ms=now_ms,
            limit=max(lookback + 2, 10),
        )
    )
    return _bybit_rows_to_df(rows)


def _wait(stop_event: Event, seconds: float) -> None:
    """Sleep responsively to a stop signal."""
    if seconds <= 0:
        return
    stop_event.wait(seconds)


# Re-export for convenient mocking in tests.
__all__: list[str] = ["BarCallback", "stream_klines"]


def _silence_unused(*_: Any) -> None:  # pragma: no cover
    pass
