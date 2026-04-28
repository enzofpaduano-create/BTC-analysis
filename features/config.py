"""Pydantic config for the features pipeline.

All thresholds / windows live here so the user can tune per asset/timeframe
without touching feature code. ``compute_features`` accepts a `FeaturesConfig`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TechnicalConfig(BaseModel):
    """Indicator parameters (pandas_ta-style)."""

    model_config = ConfigDict(extra="forbid")

    rsi_lengths: list[int] = Field(default_factory=lambda: [7, 14, 28])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_length: int = 20
    bb_std: float = 2.0
    atr_length: int = 14
    ema_lengths: list[int] = Field(default_factory=lambda: [9, 21, 50, 200])
    slope_window: int = 5  # bars used to compute the EMA slope


class VolatilityConfig(BaseModel):
    """Realized vol windows + GARCH walk-forward parameters."""

    model_config = ConfigDict(extra="forbid")

    # Realized-vol rolling windows in MINUTES (mapped to bars at runtime).
    realized_windows_min: list[int] = Field(default_factory=lambda: [15, 60, 240, 1440])
    short_window_min: int = 15
    long_window_min: int = 240
    # GARCH(1,1) refits every N bars on the running prefix, then propagates.
    garch_refit_every: int = 1000
    garch_min_obs: int = 500


class RegimeConfig(BaseModel):
    """HMM regimes + change-point detection."""

    model_config = ConfigDict(extra="forbid")

    hmm_n_states: int = 3
    hmm_refit_every: int = 1000
    hmm_min_obs: int = 500
    # PELT change-point on a rolling window — full-prefix PELT is O(n²) and too slow.
    cp_window: int = 1000
    cp_refit_every: int = 200
    cp_penalty: float = 5.0
    cp_min_size: int = 20


class MicrostructureConfig(BaseModel):
    """Z-scores, Kalman trend, log-returns, rolling moments."""

    model_config = ConfigDict(extra="forbid")

    zscore_emas: list[int] = Field(default_factory=lambda: [50, 200])
    zscore_window: int = 100
    log_ret_horizons_min: list[int] = Field(default_factory=lambda: [1, 5, 15, 60])
    skew_kurt_window: int = 60
    # Local linear-trend Kalman: q_level / q_trend / r_obs noise variances.
    kalman_q_level: float = 1e-4
    kalman_q_trend: float = 1e-6
    kalman_r_obs: float = 1.0


class FeaturesConfig(BaseModel):
    """Top-level features config."""

    model_config = ConfigDict(extra="forbid")

    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    microstructure: MicrostructureConfig = Field(default_factory=MicrostructureConfig)
    # Bar duration in minutes — needed to map "X minutes" windows to bar counts.
    bar_minutes: int = 1
