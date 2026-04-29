"""Position sizing helpers."""

from __future__ import annotations

import numpy as np


def vol_target_size(
    *,
    realized_vol_annualized: float,
    target_vol_per_trade: float,
    max_size: float = 1.0,
) -> float:
    """Vol-targeting: size = ``target_vol / realized_vol``.

    Args:
        realized_vol_annualized: Recent realized vol of the asset (annualised
            standard deviation of log-returns, e.g. 0.6 for BTC).
        target_vol_per_trade: Desired annualised vol of the trade equity
            curve (e.g. 0.10 = 10 %/year). The user spec says
            "target 1 % vol per trade" — interpret as 1 % per-bar or 100 %
            annualised depending on context. We expose this as an explicit
            parameter; the caller picks the convention.
        max_size: Cap on the fraction of equity allocated (default 1.0 =
            full equity, i.e. no leverage).

    Returns:
        Fraction of equity to allocate (0..max_size). Returns 0 if the
        realized vol is non-finite or non-positive.
    """
    if not np.isfinite(realized_vol_annualized) or realized_vol_annualized <= 0:
        return 0.0
    raw = target_vol_per_trade / realized_vol_annualized
    return float(min(max(raw, 0.0), max_size))
