"""Tests for volatility features."""

from __future__ import annotations

import pandas as pd

from features.config import VolatilityConfig
from features.volatility import compute_volatility


def test_realized_vol_columns(synthetic_btc_m1: pd.DataFrame) -> None:
    cfg = VolatilityConfig()
    out = compute_volatility(synthetic_btc_m1, cfg, bar_minutes=1)
    for w in cfg.realized_windows_min:
        assert f"vol_{w}m" in out.columns
    assert "vol_ratio_short_long" in out.columns
    assert "garch_vol_1step" in out.columns


def test_realized_vol_responds_to_regime_change(synthetic_btc_m1: pd.DataFrame) -> None:
    """Vol on the high-vol half of the fixture must exceed the low-vol half."""
    cfg = VolatilityConfig()
    out = compute_volatility(synthetic_btc_m1, cfg, bar_minutes=1)
    low = out["vol_60m"].iloc[100:1000].mean()
    high = out["vol_60m"].iloc[1100:].mean()
    assert high > 2 * low


def test_garch_produces_finite_forecasts(synthetic_btc_m1: pd.DataFrame) -> None:
    """After warmup, GARCH should output a sensible positive vol."""
    cfg = VolatilityConfig(garch_min_obs=300, garch_refit_every=500)
    out = compute_volatility(synthetic_btc_m1, cfg, bar_minutes=1)
    tail = out["garch_vol_1step"].iloc[500:].dropna()
    assert len(tail) > 100
    assert (tail > 0).all()
    # Annualized BTC vol is typically in [0.1, 5.0]; loose sanity.
    assert tail.between(0.05, 10.0).mean() > 0.95
