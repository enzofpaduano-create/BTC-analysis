"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Centralised, validated application settings.

    All values are loaded from environment variables, with `.env` as fallback.
    Keys are case-insensitive. Unknown env vars are ignored to keep the
    surface stable when extra OS variables are present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Bybit -------------------------------------------------------------
    bybit_api_key: str = Field(default="", description="Bybit API key.")
    bybit_api_secret: str = Field(default="", description="Bybit API secret.")
    bybit_testnet: bool = Field(
        default=True,
        description="Use Bybit testnet (paper trading). Set to false only when going live.",
    )

    # -- Logging -----------------------------------------------------------
    log_level: LogLevel = Field(default="INFO", description="Loguru log level.")

    # -- Storage -----------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./data_store"),
        description="Root directory for Parquet partitions and DuckDB database.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings, cached for the process lifetime.

    Returns:
        The validated `Settings` instance.
    """
    return Settings()
