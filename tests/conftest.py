"""Pytest configuration shared by all tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> Iterator[None]:
    """Run every test with a clean Bybit/log env so .env doesn't leak in."""
    for key in (
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "BYBIT_TESTNET",
        "LOG_LEVEL",
        "DATA_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    # Force pydantic-settings to ignore the on-disk .env during tests.
    monkeypatch.chdir(os.fspath(tmp_path))
    yield
