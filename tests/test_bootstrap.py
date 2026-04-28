"""Smoke tests for Étape 1 — verify packages import and config loads cleanly."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from core.logging import setup_logging
from core.settings import Settings, get_settings


@pytest.mark.parametrize(
    "package",
    ["core", "data", "features", "signals", "backtest", "live"],
)
def test_layer_packages_importable(package: str) -> None:
    """All architectural layers import without errors."""
    module = importlib.import_module(package)
    assert module is not None


def test_settings_defaults_when_no_env() -> None:
    """Without env vars or .env, settings fall back to safe defaults."""
    get_settings.cache_clear()
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.bybit_testnet is True  # paper by default — never accidentally live
    assert s.bybit_api_key == ""
    assert s.log_level == "INFO"
    assert isinstance(s.data_dir, Path)


def test_settings_picks_up_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars override defaults."""
    monkeypatch.setenv("BYBIT_API_KEY", "test-key")
    monkeypatch.setenv("BYBIT_TESTNET", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()

    s = get_settings()
    assert s.bybit_api_key == "test-key"
    assert s.bybit_testnet is False
    assert s.log_level == "DEBUG"


def test_logging_setup_idempotent() -> None:
    """Calling setup_logging multiple times does not raise nor duplicate handlers."""
    setup_logging()
    setup_logging()
    setup_logging()
