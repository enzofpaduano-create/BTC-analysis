"""Tests for the historical downloader (uses a mocked BybitClient)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data.client import BybitClient, KlineRequest
from data.download import download_history
from data.storage import ParquetStore


class FakeClient:
    """Stand-in for BybitClient that pages through a fixed kline list."""

    def __init__(self, all_rows: list[list[str]], page_size: int = 50) -> None:
        # `all_rows` are in DESCENDING time order (Bybit shape).
        self._asc = sorted(all_rows, key=lambda r: int(r[0]))
        self._page_size = page_size
        self.calls: list[KlineRequest] = []

    # Lifecycle no-ops so the downloader's `owns_client` path is exercised.
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_kline(self, req: KlineRequest) -> list[list[str]]:
        self.calls.append(req)
        # Bybit-faithful: take the NEWEST `page_size` rows in [start, end],
        # returned in DESC order.
        in_range = [r for r in self._asc if req.start_ms <= int(r[0]) <= req.end_ms]
        page = in_range[-self._page_size :]
        return list(reversed(page))


@pytest.fixture
def fake_client_factory(
    monkeypatch: pytest.MonkeyPatch,
    make_bybit_rows: Callable[..., list[list[str]]],
) -> Callable[..., FakeClient]:
    """Patch `BybitClient.__init__/open/close/get_kline` to use FakeClient."""

    def _factory(n: int = 250, page_size: int = 50) -> FakeClient:
        rows = make_bybit_rows(n=n)
        fake = FakeClient(rows, page_size=page_size)

        def _init(self: BybitClient, **_kw: Any) -> None:
            self._fake = fake  # type: ignore[attr-defined]

        def _open(self: BybitClient) -> None:
            return None

        def _close(self: BybitClient) -> None:
            return None

        def _get(self: BybitClient, req: KlineRequest) -> list[list[str]]:
            return fake.get_kline(req)

        monkeypatch.setattr(BybitClient, "__init__", _init)
        monkeypatch.setattr(BybitClient, "open", _open)
        monkeypatch.setattr(BybitClient, "close", _close)
        monkeypatch.setattr(BybitClient, "get_kline", _get)
        return fake

    return _factory


def test_downloader_paginates_until_end(
    tmp_path: Path,
    fake_client_factory: Callable[..., FakeClient],
) -> None:
    fake = fake_client_factory(n=250, page_size=50)
    store = ParquetStore(tmp_path)

    df = download_history(
        symbol="BTCUSDT",
        timeframe="1",
        start="2023-11-14",  # well before fixture start
        end="2023-11-15 12:00",  # well after
        store=store,
    )
    assert len(df) == 250
    # 5 pages of 50 + maybe one empty terminator
    assert len(fake.calls) >= 5


def test_downloader_dedups_overlapping_pages(
    tmp_path: Path,
    fake_client_factory: Callable[..., FakeClient],
) -> None:
    """If the fake returns overlapping pages, the result is unique."""
    fake = fake_client_factory(n=120, page_size=50)
    store = ParquetStore(tmp_path)
    df = download_history(
        symbol="BTCUSDT",
        timeframe="1",
        start="2023-11-14",
        end="2023-11-15 12:00",
        store=store,
    )
    assert df["timestamp"].is_unique
    assert len(df) == 120
    assert len(fake.calls) >= 1


def test_downloader_resumes_from_disk(
    tmp_path: Path,
    fake_client_factory: Callable[..., FakeClient],
) -> None:
    """Second call only fetches bars after the last stored timestamp."""
    fake_client_factory(n=250, page_size=100)
    store = ParquetStore(tmp_path)

    df1 = download_history(
        symbol="BTCUSDT",
        timeframe="1",
        start="2023-11-14",
        end="2023-11-15 12:00",
        store=store,
    )
    n_first = len(df1)
    assert n_first > 0

    # Second call: nothing newer in fixture → empty result, but no crash.
    df2 = download_history(
        symbol="BTCUSDT",
        timeframe="1",
        start="2023-11-14",
        end="2023-11-15 12:00",
        store=store,
    )
    assert len(df2) == 0

    # Storage still contains the original rows.
    full = store.read("BTCUSDT", "1")
    assert len(full) == n_first


def test_downloader_writes_partitioned_parquet(
    tmp_path: Path,
    fake_client_factory: Callable[..., FakeClient],
) -> None:
    fake_client_factory(n=120, page_size=120)
    store = ParquetStore(tmp_path)
    download_history(
        symbol="BTCUSDT",
        timeframe="1",
        start="2023-11-14",
        end="2023-11-15 12:00",
        store=store,
    )
    parts = list(tmp_path.rglob("data.parquet"))
    assert parts, "no Parquet partitions written"
    # Round-trip via DataFrame.
    df = store.read("BTCUSDT", "1")
    assert df["timestamp"].dt.tz is not None
    assert df["close"].dtype == "float64"
