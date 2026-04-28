"""Trading-cost model: spread + slippage + funding.

Effective fill price for a market order::

    buy:  fill = mid * (1 + spread/2 + slippage)
    sell: fill = mid * (1 - spread/2 - slippage)

Slippage = fixed_bps + prop_coeff * (size_btc / volume_btc).
Funding is a continuous annualised rate, applied per bar of held position.

All inputs are bps (basis points), 1 bps = 0.01 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.config import CostsConfig

_BPS = 1e-4


@dataclass(frozen=True, slots=True)
class Fill:
    """Net result of a fill given mid-price, size, and bar volume."""

    fill_price: float
    fee_paid: float  # currency, signed positive (cost to trader)
    slippage_bps: float


def fill_price_with_costs(
    *,
    mid_price: float,
    size: float,
    bar_volume: float,
    side: int,  # +1 buy, -1 sell
    cfg: CostsConfig,
) -> Fill:
    """Resolve the effective fill price and explicit fee for a single trade.

    Args:
        mid_price: Reference (close) price.
        size: Trade size in base units (e.g. BTC). Positive.
        bar_volume: Volume traded on that bar in same units.
        side: +1 for buy, -1 for sell.
        cfg: Cost parameters.

    Returns:
        ``Fill`` with the executed price, fee paid (in quote currency),
        and the realised slippage in bps for telemetry.
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    size = abs(size)
    bar_volume = max(bar_volume, 1e-12)

    slippage_bps = cfg.slippage_bps_fixed + cfg.slippage_prop_coeff_bps * (size / bar_volume)
    half_spread_bps = cfg.spread_bps / 2.0
    price_penalty = (half_spread_bps + slippage_bps) * _BPS
    fill_price = mid_price * (1.0 + side * price_penalty)
    notional = size * fill_price
    fee_paid = notional * cfg.taker_fee_bps * _BPS
    return Fill(fill_price=fill_price, fee_paid=fee_paid, slippage_bps=slippage_bps)


def funding_per_bar_bps(cfg: CostsConfig, bar_minutes: int) -> float:
    """Per-bar funding cost in bps (long pays positive, short receives positive)."""
    bars_per_year = (365 * 24 * 60) / bar_minutes
    return cfg.funding_annual_bps / bars_per_year


def apply_funding(notional_held: pd.Series, cfg: CostsConfig, bar_minutes: int) -> pd.Series:
    """Per-bar funding cost in quote currency.

    Args:
        notional_held: Signed notional held at each bar (positive = long).
        cfg: Cost params.
        bar_minutes: Bar duration in minutes.

    Returns:
        Series of per-bar funding cost (positive = paid by trader).
        We model symmetric funding: the rate applies to ``|notional|``.
    """
    rate = funding_per_bar_bps(cfg, bar_minutes) * _BPS
    return notional_held.abs() * rate


def fill_prices_vectorized(
    *,
    mid: pd.Series,
    sizes: pd.Series,
    bar_volume: pd.Series,
    sides: pd.Series,
    cfg: CostsConfig,
) -> tuple[pd.Series, pd.Series]:
    """Vectorised version of :func:`fill_price_with_costs`.

    Returns ``(fill_prices, fees)`` aligned to the input index. Bars where
    ``sizes == 0`` get the mid price and zero fee. Cleaner for the engine
    loop than calling the scalar helper N times.
    """
    sizes_abs = sizes.abs()
    safe_vol = bar_volume.replace(0.0, 1e-12)
    slippage_bps = cfg.slippage_bps_fixed + cfg.slippage_prop_coeff_bps * (sizes_abs / safe_vol)
    half_spread_bps = cfg.spread_bps / 2.0
    penalty = (half_spread_bps + slippage_bps) * _BPS
    fill = mid * (1.0 + sides.astype(float) * penalty)
    notional = sizes_abs * fill
    fees = notional * cfg.taker_fee_bps * _BPS
    # Where size==0, no trade — no penalty, no fee.
    no_trade = sizes_abs == 0.0
    fill = fill.where(~no_trade, mid)
    fees = fees.where(~no_trade, 0.0)
    return fill.astype(float), fees.astype(float).fillna(0.0)


# Keep numpy in scope for typecheckers — re-export.
_ = np
