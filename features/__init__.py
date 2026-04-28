"""Feature engineering layer — indicators, volatility, regimes, microstructure.

Public API::

    from features import compute_features, FeaturesConfig

    cfg = FeaturesConfig(bar_minutes=5)  # tune per timeframe
    feat = compute_features(ohlcv_df, cfg)
"""

from features.config import (
    FeaturesConfig,
    MicrostructureConfig,
    RegimeConfig,
    TechnicalConfig,
    VolatilityConfig,
)
from features.microstructure import compute_microstructure
from features.pipeline import compute_features
from features.regime import compute_regime
from features.technical import compute_technical
from features.volatility import compute_volatility

__all__ = [
    "FeaturesConfig",
    "MicrostructureConfig",
    "RegimeConfig",
    "TechnicalConfig",
    "VolatilityConfig",
    "compute_features",
    "compute_microstructure",
    "compute_regime",
    "compute_technical",
    "compute_volatility",
]
