"""Orchestrator: ``compute_features(df, config)`` runs the full pipeline."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from data.schemas import OHLCV_COLUMNS
from features.config import FeaturesConfig
from features.microstructure import compute_microstructure
from features.regime import compute_regime
from features.technical import compute_technical
from features.volatility import compute_volatility


def compute_features(df: pd.DataFrame, cfg: FeaturesConfig | None = None) -> pd.DataFrame:
    """Build the full feature DataFrame from raw OHLCV.

    Args:
        df: OHLCV DataFrame with the canonical columns
            (``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``,
            ``turnover``). The ``timestamp`` must be UTC tz-aware and sorted.
        cfg: Feature configuration. Defaults to ``FeaturesConfig()``.

    Returns:
        New DataFrame with all original columns plus features. NaNs are
        expected during the warmup region (longest indicator window).
    """
    cfg = cfg or FeaturesConfig()
    _validate_input(df)

    logger.info(
        "compute_features: {} bars, bar_minutes={}",
        len(df),
        cfg.bar_minutes,
    )
    out = compute_technical(df, cfg.technical)
    out = compute_volatility(out, cfg.volatility, bar_minutes=cfg.bar_minutes)
    out = compute_microstructure(out, cfg.microstructure)
    return compute_regime(out, cfg.regime, bar_minutes=cfg.bar_minutes)


def _validate_input(df: pd.DataFrame) -> None:
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"compute_features: missing columns {sorted(missing)}")
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise TypeError("compute_features: `timestamp` must be a datetime column")
    if ts.dt.tz is None:
        raise ValueError("compute_features: `timestamp` must be tz-aware (UTC)")
    if not ts.is_monotonic_increasing:
        raise ValueError("compute_features: rows must be sorted by ascending timestamp")
