"""Signals layer — strategies, scoring, position sizing.

Public API::

    from signals import MeanReversionBollingerHMM, vol_target_size
"""

from signals.mean_reversion import MeanReversionBollingerHMM
from signals.sizing import vol_target_size
from signals.trend_breakout import TrendBreakoutADXHMM

__all__ = ["MeanReversionBollingerHMM", "TrendBreakoutADXHMM", "vol_target_size"]
