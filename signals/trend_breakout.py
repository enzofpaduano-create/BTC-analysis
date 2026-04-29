"""Trend-following: Donchian breakout filtered by ADX and HMM regime.

Rules:
    Long entry  : close > donchian_high AND adx > adx_threshold AND regime == bull (2)
    Short entry : close < donchian_low  AND adx > adx_threshold AND regime == bear (0)
    Exit        : ATR×N trailing stop OR ADX < adx_exit_threshold OR timeout

Sizing: vol-targeting (same as MeanReversionBollingerHMM).

Causality: same approach — strategy state is built from values at bar ``i``,
exits trigger from values at bar ``i``, the engine handles the next-bar fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy
from signals.sizing import vol_target_size

_BEAR_LABEL = 0
_BULL_LABEL = 2

_REQUIRED = (
    "close",
    "high",
    "low",
    "donchian_high",
    "donchian_low",
    "adx",
    "regime_hmm",
    "atr",
    "vol_60m",
)


@dataclass
class _OpenTrade:
    side: int  # +1 long, -1 short
    entry_idx: int
    entry_close: float
    atr_at_entry: float
    extreme_close: float  # running max (long) or min (short) since entry


@dataclass
class TrendBreakoutADXHMM(Strategy):
    """Donchian breakout + ADX filter + HMM regime alignment."""

    name: str = "trend_breakout_adx_hmm"

    # Entry filters
    adx_threshold: float = 25.0
    adx_exit_threshold: float = 18.0

    # Exit
    atr_trail_mult: float = 3.0
    timeout_bars: int = 288  # 24h on M5

    # Vol-targeting
    target_vol_per_trade: float = 0.01
    max_size: float = 1.0

    _trace: list[dict[str, float]] = field(default_factory=list, repr=False)

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        missing = [c for c in _REQUIRED if c not in features.columns]
        if missing:
            raise ValueError(f"Strategy needs feature columns: {missing}")

        n = len(features)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)
        size = np.zeros(n, dtype=float)

        close = features["close"].to_numpy()
        d_high = features["donchian_high"].to_numpy()
        d_low = features["donchian_low"].to_numpy()
        adx = features["adx"].to_numpy()
        regime = features["regime_hmm"].to_numpy()
        atr = features["atr"].to_numpy()
        vol_60m = features["vol_60m"].to_numpy()

        open_trade: _OpenTrade | None = None

        for i in range(n):
            inputs = (close[i], d_high[i], d_low[i], adx[i], atr[i], vol_60m[i])
            if any(not np.isfinite(x) for x in inputs):
                continue
            if regime[i] < 0:
                continue

            # 1) Manage open position first.
            if open_trade is not None:
                # Update running extreme.
                if open_trade.side > 0:
                    open_trade.extreme_close = max(open_trade.extreme_close, float(close[i]))
                else:
                    open_trade.extreme_close = min(open_trade.extreme_close, float(close[i]))

                bars_since_entry = i - open_trade.entry_idx
                trail_dist = self.atr_trail_mult * open_trade.atr_at_entry
                hit_trail = (
                    close[i] <= open_trade.extreme_close - trail_dist
                    if open_trade.side > 0
                    else close[i] >= open_trade.extreme_close + trail_dist
                )
                hit_adx_exit = adx[i] < self.adx_exit_threshold
                hit_timeout = bars_since_entry >= self.timeout_bars

                if hit_trail or hit_adx_exit or hit_timeout:
                    if open_trade.side > 0:
                        exit_long[i] = True
                    else:
                        exit_short[i] = True
                    open_trade = None
                    continue

            # 2) Entry only if flat.
            if open_trade is not None:
                continue

            wants_long = (
                close[i] > d_high[i] and adx[i] > self.adx_threshold and regime[i] == _BULL_LABEL
            )
            wants_short = (
                close[i] < d_low[i] and adx[i] > self.adx_threshold and regime[i] == _BEAR_LABEL
            )

            if wants_long:
                entry_long[i] = True
                size[i] = vol_target_size(
                    realized_vol_annualized=float(vol_60m[i]),
                    target_vol_per_trade=self.target_vol_per_trade,
                    max_size=self.max_size,
                )
                open_trade = _OpenTrade(
                    side=1,
                    entry_idx=i,
                    entry_close=float(close[i]),
                    atr_at_entry=float(atr[i]),
                    extreme_close=float(close[i]),
                )
            elif wants_short:
                entry_short[i] = True
                size[i] = vol_target_size(
                    realized_vol_annualized=float(vol_60m[i]),
                    target_vol_per_trade=self.target_vol_per_trade,
                    max_size=self.max_size,
                )
                open_trade = _OpenTrade(
                    side=-1,
                    entry_idx=i,
                    entry_close=float(close[i]),
                    atr_at_entry=float(atr[i]),
                    extreme_close=float(close[i]),
                )

        sig_df = pd.DataFrame(
            {
                "entry_long": entry_long,
                "exit_long": exit_long,
                "entry_short": entry_short,
                "exit_short": exit_short,
                "size": size,
            },
            index=features.index,
            columns=list(SIGNAL_COLUMNS),
        )
        return SignalFrame(df=sig_df)
