"""Tests for the cost model."""

from __future__ import annotations

import pandas as pd

from backtest.config import CostsConfig
from backtest.costs import apply_funding, fill_price_with_costs, funding_per_bar_bps


def test_buy_pays_half_spread_plus_slippage() -> None:
    cfg = CostsConfig(
        spread_bps=10.0,  # 5 bps half-spread
        taker_fee_bps=0.0,
        slippage_bps_fixed=2.0,
        slippage_prop_coeff_bps=0.0,
        funding_annual_bps=0.0,
    )
    fill = fill_price_with_costs(mid_price=60_000.0, size=1.0, bar_volume=100.0, side=1, cfg=cfg)
    expected_penalty_bps = 5.0 + 2.0
    expected_price = 60_000.0 * (1 + expected_penalty_bps * 1e-4)
    assert abs(fill.fill_price - expected_price) < 1e-6
    assert fill.slippage_bps == 2.0


def test_sell_receives_less_than_mid() -> None:
    cfg = CostsConfig(spread_bps=4.0, slippage_bps_fixed=0.0, slippage_prop_coeff_bps=0.0)
    fill = fill_price_with_costs(mid_price=60_000.0, size=1.0, bar_volume=100.0, side=-1, cfg=cfg)
    assert fill.fill_price < 60_000.0


def test_proportional_slippage_scales_with_size() -> None:
    cfg = CostsConfig(spread_bps=0.0, slippage_bps_fixed=0.0, slippage_prop_coeff_bps=10.0)
    small = fill_price_with_costs(mid_price=60_000.0, side=1, size=1.0, bar_volume=100.0, cfg=cfg)
    big = fill_price_with_costs(mid_price=60_000.0, side=1, size=10.0, bar_volume=100.0, cfg=cfg)
    assert big.slippage_bps == small.slippage_bps * 10


def test_taker_fee_is_correct_fraction_of_notional() -> None:
    cfg = CostsConfig(
        spread_bps=0.0,
        slippage_bps_fixed=0.0,
        slippage_prop_coeff_bps=0.0,
        taker_fee_bps=5.0,
    )
    fill = fill_price_with_costs(mid_price=60_000.0, size=2.0, bar_volume=100.0, side=1, cfg=cfg)
    # notional = 2 * 60_000 = 120_000 ; fee = 120_000 * 0.0005 = 60.0
    assert abs(fill.fee_paid - 60.0) < 1e-6


def test_funding_per_bar_bps_matches_annual() -> None:
    cfg = CostsConfig(funding_annual_bps=525.6)  # exactly 1 bps per minute
    per_bar = funding_per_bar_bps(cfg, bar_minutes=1)
    assert abs(per_bar - 0.001) < 1e-6


def test_apply_funding_only_charges_when_position_held() -> None:
    cfg = CostsConfig(funding_annual_bps=10.0)
    notional = pd.Series([0.0, 1000.0, -500.0, 0.0])
    cost = apply_funding(notional, cfg, bar_minutes=1)
    # First and last bars carry no notional → no funding.
    assert cost.iloc[0] == 0.0
    assert cost.iloc[-1] == 0.0
    # Symmetric magnitude for long vs short.
    assert cost.iloc[1] > 0
    assert cost.iloc[2] > 0
