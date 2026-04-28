"""Tests for regime / change-point features."""

from __future__ import annotations

import pandas as pd

from features.config import RegimeConfig
from features.regime import compute_regime


def test_columns_added(synthetic_btc_m1: pd.DataFrame) -> None:
    cfg = RegimeConfig(hmm_min_obs=300, hmm_refit_every=500)
    out = compute_regime(synthetic_btc_m1, cfg, bar_minutes=1)
    assert "regime_hmm" in out.columns
    assert "regime_hmm_proba" in out.columns
    assert "cp_segment" in out.columns


def test_hmm_assigns_states_after_warmup(synthetic_btc_m1: pd.DataFrame) -> None:
    cfg = RegimeConfig(hmm_min_obs=300, hmm_refit_every=500)
    out = compute_regime(synthetic_btc_m1, cfg, bar_minutes=1)
    valid = out["regime_hmm"].iloc[400:]
    assert (valid != -1).mean() > 0.8
    assert out["regime_hmm_proba"].iloc[400:].dropna().between(0.0, 1.0).all()


def test_change_point_detects_regime_switch(synthetic_btc_m1: pd.DataFrame) -> None:
    """The fixture flips vol at bar 1000 — segment id should increase after."""
    cfg = RegimeConfig(cp_window=500, cp_refit_every=100, cp_penalty=2.0, cp_min_size=20)
    out = compute_regime(synthetic_btc_m1, cfg, bar_minutes=1)
    seg_before = out["cp_segment"].iloc[800]
    seg_after = out["cp_segment"].iloc[1900]
    assert seg_after > seg_before
