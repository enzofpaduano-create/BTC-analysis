"""Thin wrapper around `pybit.unified_trading.HTTP` with a context manager.

Public klines do NOT require API credentials, so the historical downloader
can run anonymously against mainnet. Credentials are still optional for
authenticated endpoints we'll need later (orders, positions, account).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pybit.unified_trading import HTTP

from core.settings import get_settings
from data.schemas import Category, Timeframe


@dataclass(frozen=True, slots=True)
class KlineRequest:
    """Parameters for a single Bybit kline batch request."""

    category: Category
    symbol: str
    interval: Timeframe
    start_ms: int
    end_ms: int
    limit: int = 1000  # Bybit max per call


class BybitClient:
    """Wraps a `pybit` HTTP session.

    Use as a context manager to ensure the underlying `requests.Session`
    is properly closed::

        with BybitClient() as client:
            batch = client.get_kline(KlineRequest(...))

    Args:
        testnet: Override `BYBIT_TESTNET` from settings. Public market data
            is identical between mainnet and testnet but mainnet history
            is much longer — use mainnet for downloads.
        api_key: Override `BYBIT_API_KEY` from settings.
        api_secret: Override `BYBIT_API_SECRET` from settings.
        max_retries: How many times to retry transient HTTP errors per call.
        backoff_base: Initial backoff delay (seconds), doubles each retry.
    """

    def __init__(
        self,
        *,
        testnet: bool | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
    ) -> None:
        settings = get_settings()
        self._testnet = settings.bybit_testnet if testnet is None else testnet
        self._api_key = settings.bybit_api_key if api_key is None else api_key
        self._api_secret = settings.bybit_api_secret if api_secret is None else api_secret
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._session: HTTP | None = None

    # -- Lifecycle --------------------------------------------------------

    def open(self) -> None:
        """Create the underlying HTTP session if not already open."""
        if self._session is not None:
            return
        kwargs: dict[str, Any] = {"testnet": self._testnet}
        if self._api_key and self._api_secret:
            kwargs["api_key"] = self._api_key
            kwargs["api_secret"] = self._api_secret
        self._session = HTTP(**kwargs)
        logger.debug("Bybit HTTP session opened (testnet={})", self._testnet)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is None:
            return
        # pybit's HTTP wraps a requests.Session in `client`; close it if exposed.
        client = getattr(self._session, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
        self._session = None
        logger.debug("Bybit HTTP session closed")

    def __enter__(self) -> BybitClient:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- Endpoints --------------------------------------------------------

    def get_kline(self, req: KlineRequest) -> list[list[str]]:
        """Fetch one page of klines.

        Returns the raw `result.list` from Bybit, in the order Bybit returns
        it (newest first, 7 string fields per row:
        `[start_ms, open, high, low, close, volume, turnover]`).

        Retries with exponential backoff on transient errors. Raises on
        permanent failures or after `max_retries`.
        """
        if self._session is None:
            raise RuntimeError("BybitClient must be used as a context manager")

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get_kline(
                    category=req.category,
                    symbol=req.symbol,
                    interval=req.interval,
                    start=req.start_ms,
                    end=req.end_ms,
                    limit=req.limit,
                )
                ret_code = resp.get("retCode", -1)
                if ret_code != 0:
                    raise RuntimeError(f"Bybit error {ret_code}: {resp.get('retMsg')}")
                rows = resp.get("result", {}).get("list", [])
                if not isinstance(rows, list):
                    raise RuntimeError(f"Unexpected Bybit response shape: {resp!r}")
                return rows
            except Exception as exc:  # noqa: BLE001 — we re-raise after retries
                last_exc = exc
                delay = self._backoff_base * (2**attempt)
                logger.warning(
                    "Bybit kline call failed (attempt {}/{}): {} — retry in {:.1f}s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        assert last_exc is not None
        raise last_exc


@contextmanager
def bybit_client(**kwargs: Any) -> Iterator[BybitClient]:
    """Sugar: ``with bybit_client() as c: ...``."""
    client = BybitClient(**kwargs)
    client.open()
    try:
        yield client
    finally:
        client.close()
