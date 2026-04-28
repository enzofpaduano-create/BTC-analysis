"""Pydantic config for the backtest engine.

Costs are split into spread / slippage / funding so each can be modelled
realistically per asset and per execution style.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CostsConfig(BaseModel):
    """Per-asset trading costs.

    Fees are basis-points (bps) of notional. Slippage has a fixed bps
    component plus a "proportional" one that scales with trade size relative
    to bar volume — bigger orders eat more of the book.
    """

    model_config = ConfigDict(extra="forbid")

    # Half-spread paid on every fill (taker market order assumption).
    spread_bps: float = Field(default=2.0, ge=0)
    # Exchange fees (taker side — we model market orders).
    taker_fee_bps: float = Field(default=5.5, ge=0)
    # Slippage = fixed_bps + prop_coeff * (trade_size_btc / bar_volume_btc).
    slippage_bps_fixed: float = Field(default=1.0, ge=0)
    slippage_prop_coeff_bps: float = Field(default=5.0, ge=0)
    # Annualised funding rate for perpetuals (continuous approximation).
    funding_annual_bps: float = Field(default=10.0, ge=0)


class WalkForwardConfig(BaseModel):
    """Rolling walk-forward + purge/embargo parameters."""

    model_config = ConfigDict(extra="forbid")

    train_size: int = Field(default=10_000, ge=1)
    test_size: int = Field(default=2_000, ge=1)
    step_size: int = Field(default=2_000, ge=1)
    # Purge: drop training samples whose label horizon overlaps test window.
    purge: int = Field(default=0, ge=0)
    # Embargo: drop a buffer between train and test to break serial autocorrelation.
    embargo: int = Field(default=0, ge=0)


class BacktestConfig(BaseModel):
    """Top-level backtest config."""

    model_config = ConfigDict(extra="forbid")

    initial_capital: float = Field(default=10_000.0, gt=0)
    bar_minutes: int = Field(default=1, ge=1)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
