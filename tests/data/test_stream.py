"""Test the polling-based stream emits new closed bars and stops cleanly."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pandas as pd
import pytest

from data.client import BybitClient, KlineRequest
from data.storage import ParquetStore
from data.stream import stream_klines


def _kline_row(ts_ms: int, price: float) -> list[str]:
    return [
        str(ts_ms),
        f"{price:.2f}",
        f"{price + 5:.2f}",
        f"{price - 5:.2f}",
        f"{price + 2:.2f}",
        "10",
        f"{price * 10:.2f}",
    ]


def test_stream_emits_new_closed_bars_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two polls return successively newer bars; callback fires on each new one."""
    interval_ms = 60_000
    # Anchor 10 minutes in the past so all bars are "closed" relative to wall clock.
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    base_ms = now_ms - 10 * interval_ms

    poll_responses: list[list[list[str]]] = [
        [
            _kline_row(base_ms, 60_000.0),
            _kline_row(base_ms + interval_ms, 60_001.0),
        ],
        [
            _kline_row(base_ms + interval_ms, 60_001.0),  # overlap
            _kline_row(base_ms + 2 * interval_ms, 60_002.0),
        ],
    ]
    poll_responses = [list(reversed(p)) for p in poll_responses]  # Bybit DESC order

    call_count = {"n": 0}
    stop = Event()

    def fake_open(self: BybitClient) -> None:
        return None

    def fake_close(self: BybitClient) -> None:
        return None

    def fake_get_kline(self: BybitClient, _req: KlineRequest) -> list[list[str]]:
        idx = min(call_count["n"], len(poll_responses) - 1)
        call_count["n"] += 1
        # After we've served both response sets, signal the loop to stop.
        if call_count["n"] >= len(poll_responses) + 1:
            stop.set()
        return poll_responses[idx]

    monkeypatch.setattr(BybitClient, "open", fake_open)
    monkeypatch.setattr(BybitClient, "close", fake_close)
    monkeypatch.setattr(BybitClient, "get_kline", fake_get_kline)

    received: list[pd.DataFrame] = []

    def on_bar(df: pd.DataFrame) -> None:
        received.append(df)

    store = ParquetStore(tmp_path)
    thread = Thread(
        target=stream_klines,
        kwargs={
            "symbol": "BTCUSDT",
            "timeframe": "1",
            "on_bar": on_bar,
            "poll_interval_s": 0.05,
            "lookback": 3,
            "store": store,
            "stop_event": stop,
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "stream did not stop in time"

    assert call_count["n"] >= 2
    # 3 distinct closed bars across two polls (one is overlap → deduped).
    received_concat = pd.concat(received, ignore_index=True) if received else pd.DataFrame()
    assert received_concat["timestamp"].is_unique
    assert len(received_concat) == 3
    # And the store ended up with the same 3 rows.
    df_disk = store.read("BTCUSDT", "1")
    assert len(df_disk) == 3
