"""Baseline strategy : Bollinger mean-reversion filtered by HMM regime.

Rules (from the user spec):
    Long entry  : close ≤ BB_lower AND regime != bear AND RSI_14 < 30
    Short entry : close ≥ BB_upper AND regime != bull AND RSI_14 > 70
    Exit        : revert to BB_mid OR ATR×N stop OR timeout

Sizing:
    Vol-targeting : ``size = target_vol / realized_vol_60m``, capped at 1.

Causality:
    Each signal at bar ``i`` uses only ``features.iloc[: i + 1]``. The
    strategy holds purely-internal state to track the *signalled* trade
    (entry index, entry close, ATR-at-entry) — these are decided at
    bar ``i`` from inputs at bar ``i``. The engine handles the actual
    next-bar fill independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy
from signals.sizing import vol_target_size

# Bull/range/bear convention from features.regime: state 0 = bear, n-1 = bull.
_BEAR_LABEL = 0
_BULL_LABEL = 2

# Required input columns
_REQUIRED = (
    "close",
    "bb_lower",
    "bb_mid",
    "bb_upper",
    "rsi_14",
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


@dataclass
class MeanReversionBollingerHMM(Strategy):
    """Mean-reversion on Bollinger Bands, filtered by HMM regime."""

    name: str = "mean_reversion_bb_hmm"

    # Entry thresholds
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0

    # Exit
    atr_stop_mult: float = 2.0
    timeout_bars: int = 48  # 4 hours on M5

    # Vol-targeting
    target_vol_per_trade: float = 0.01  # 1 % "per trade" target
    max_size: float = 1.0

    # Internal — populated each call
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
        bb_lower = features["bb_lower"].to_numpy()
        bb_mid = features["bb_mid"].to_numpy()
        bb_upper = features["bb_upper"].to_numpy()
        rsi = features["rsi_14"].to_numpy()
        regime = features["regime_hmm"].to_numpy()
        atr = features["atr"].to_numpy()
        vol_60m = features["vol_60m"].to_numpy()

        open_trade: _OpenTrade | None = None

        for i in range(n):
            # Skip warmup bars where any required feature is NaN.
            inputs = (close[i], bb_lower[i], bb_mid[i], bb_upper[i], rsi[i], atr[i], vol_60m[i])
            if any(not np.isfinite(x) for x in inputs):
                continue
            if regime[i] < 0:  # HMM not yet fit
                continue

            # 1) Exit checks (priority over entry — never flip on the same bar).
            if open_trade is not None:
                bars_since_entry = i - open_trade.entry_idx
                stop_dist = self.atr_stop_mult * open_trade.atr_at_entry
                hit_target = close[i] >= bb_mid[i] if open_trade.side > 0 else close[i] <= bb_mid[i]
                hit_stop = (
                    close[i] <= open_trade.entry_close - stop_dist
                    if open_trade.side > 0
                    else close[i] >= open_trade.entry_close + stop_dist
                )
                hit_timeout = bars_since_entry >= self.timeout_bars

                if hit_target or hit_stop or hit_timeout:
                    if open_trade.side > 0:
                        exit_long[i] = True
                    else:
                        exit_short[i] = True
                    open_trade = None
                    continue  # no new entry on the same bar as the exit signal

            # 2) Entry checks (only if flat).
            if open_trade is not None:
                continue

            wants_long = (
                close[i] <= bb_lower[i]
                and regime[i] != _BEAR_LABEL
                and rsi[i] < self.rsi_long_threshold
            )
            wants_short = (
                close[i] >= bb_upper[i]
                and regime[i] != _BULL_LABEL
                and rsi[i] > self.rsi_short_threshold
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
