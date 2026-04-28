"""Tests for pydantic config schemas + the example YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from data.schemas import AssetConfig


def test_example_yaml_roundtrips() -> None:
    """The shipped example config validates against AssetConfig."""
    path = Path(__file__).resolve().parents[2] / "config" / "btcusdt.example.yaml"
    raw = yaml.safe_load(path.read_text())
    cfg = AssetConfig.model_validate(raw)
    assert cfg.asset.symbol == "BTCUSDT"
    assert cfg.asset.category == "linear"
    assert "1" in cfg.asset.timeframes
    assert cfg.costs.taker_fee_bps > 0


def test_extra_fields_rejected() -> None:
    raw = {
        "asset": {"symbol": "BTCUSDT", "timeframes": ["1"]},
        "costs": {
            "taker_fee_bps": 5.5,
            "maker_fee_bps": 1.0,
            "slippage_bps_fixed": 1.0,
            "slippage_bps_proportional": 0.5,
        },
        "storage": {
            "parquet_root": "./pq",
            "duckdb_path": "./db.duckdb",
        },
        "bogus": 42,  # extra at top level
    }
    with pytest.raises(ValidationError):
        AssetConfig.model_validate(raw)


def test_negative_fees_rejected() -> None:
    raw = {
        "asset": {"symbol": "BTCUSDT", "timeframes": ["1"]},
        "costs": {
            "taker_fee_bps": -1.0,  # invalid
            "maker_fee_bps": 1.0,
            "slippage_bps_fixed": 1.0,
            "slippage_bps_proportional": 0.5,
        },
        "storage": {"parquet_root": "./pq", "duckdb_path": "./db.duckdb"},
    }
    with pytest.raises(ValidationError):
        AssetConfig.model_validate(raw)
