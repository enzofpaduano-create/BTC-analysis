"""Typed schemas for the data layer.

Conventions:
    - All timestamps are UTC, tz-aware.
    - OHLCV DataFrames have columns: timestamp, open, high, low, close, volume, turnover.
    - Timeframe codes follow Bybit's convention (kept verbatim to avoid ambiguity).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Bybit kline interval codes — keep as strings since "D"/"W"/"M" cannot
# be expressed as ints. Numeric codes ("1", "5", …) follow the same convention
# for consistency.
Timeframe = Literal["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
Category = Literal["linear", "inverse", "spot"]

# Map a Bybit timeframe to its duration in minutes. Used for gap detection.
# `M` (monthly) is approximate — we don't gap-check monthly bars.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 1440,
    "W": 10080,
    "M": 43200,
}

# Canonical OHLCV columns in their canonical order.
OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
)


class CostsConfig(BaseModel):
    """Trading-cost parameters used by the backtest harness (étape 4)."""

    model_config = ConfigDict(extra="forbid")

    taker_fee_bps: float = Field(ge=0)
    maker_fee_bps: float = Field(ge=0)
    slippage_bps_fixed: float = Field(ge=0)
    slippage_bps_proportional: float = Field(ge=0)
    funding_8h_bps_avg: float = Field(default=0.0)


class StorageConfig(BaseModel):
    """Filesystem locations for Parquet partitions and DuckDB."""

    model_config = ConfigDict(extra="forbid")

    parquet_root: Path
    duckdb_path: Path


class AssetMeta(BaseModel):
    """Static metadata for one tradable asset."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    category: Category = "linear"
    timeframes: list[Timeframe]


class AssetConfig(BaseModel):
    """Full asset YAML config (mirrors `config/btcusdt.example.yaml`)."""

    model_config = ConfigDict(extra="forbid")

    asset: AssetMeta
    costs: CostsConfig
    storage: StorageConfig
