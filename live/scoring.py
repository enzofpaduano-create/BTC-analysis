"""Composite scoring across multiple strategies.

A "score" is a number in [-1, 1] representing the consensus signal:
    +1.0 = all strategies agree on a long
    -1.0 = all strategies agree on a short
     0.0 = no signal / disagreement

Each strategy contributes its latest entry signal weighted by:
    - the strategy's static weight (set per strategy)
    - the signal's ``size`` (vol-targeted intensity)
    - a regime-confidence multiplier from ``regime_hmm_proba``

We do NOT trade on this score — it is purely an alerting heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.strategy import Strategy

# Below this absolute score the consensus is too weak to call a direction.
DIRECTION_THRESHOLD = 0.2

# Default trade-plan multiples (× ATR), applied when the score is actionable.
# Tuned wider after audit on 54 live alerts (M5, 11 days) — narrower stops
# were getting hit by wicks, narrower TPs left money on the table.
SL_ATR_MULT = 3.0
TP_ATR_MULTS: tuple[float, float, float] = (2.0, 4.0, 6.0)


@dataclass(frozen=True, slots=True)
class StrategyScore:
    """Score breakdown for one strategy at one bar."""

    strategy_name: str
    direction: int  # +1 long, -1 short, 0 flat
    size: float  # vol-targeted intensity, [0, 1]
    raw_score: float  # direction * size (before weighting)


@dataclass(frozen=True, slots=True)
class CompositeScore:
    """Aggregate score across all active strategies at the latest bar."""

    timestamp: pd.Timestamp  # bar OPEN time (start of the candle)
    score: float  # [-1, 1]
    components: list[StrategyScore]
    regime_label: int
    regime_proba: float
    symbol: str = ""
    bar_minutes: int = 5  # candle duration — used to compute close time for display
    # Trade plan — populated only when the score crosses the alert threshold.
    # Values are absolute prices in the same unit as ``close``.
    entry: float | None = None
    sl: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    atr_at_entry: float | None = None

    def close_time(self) -> pd.Timestamp:
        """Return the time at which this bar closed (= when the alert fires)."""
        return self.timestamp + pd.Timedelta(minutes=self.bar_minutes)

    def direction(self) -> int:
        if self.score > DIRECTION_THRESHOLD:
            return 1
        if self.score < -DIRECTION_THRESHOLD:
            return -1
        return 0

    def label(self) -> str:
        return {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.direction()]

    def action(self) -> str:
        """Imperative action word — what the human should consider doing."""
        return {1: "BUY", -1: "SELL", 0: "WAIT"}[self.direction()]

    def rating(self) -> int:
        """Conviction rating in [1, 10] — magnitude of the composite score.

        ``|score| 0.3`` → 3/10 (just past the alert threshold).
        ``|score| 1.0`` → 10/10 (every strategy at full size).

        Always at least 1 so the rating is meaningful in messages even at
        the threshold edge.
        """
        return max(1, min(10, round(abs(self.score) * 10)))


def score_latest_bar(
    *,
    features: pd.DataFrame,
    strategies: list[tuple[Strategy, float]],
    symbol: str = "",
    bar_minutes: int = 5,
) -> CompositeScore:
    """Compute the composite score for the last bar of ``features``.

    Args:
        features: Pre-computed features. Must contain at least
            ``regime_hmm`` and ``regime_hmm_proba``.
        strategies: List of ``(strategy, weight)`` tuples. Weights are
            normalised so they sum to 1.
        symbol: Asset symbol to embed in the resulting score (purely
            informational — used by sinks for their messages).

    Returns:
        ``CompositeScore`` summarising direction + intensity.
    """
    if not strategies:
        raise ValueError("at least one strategy required")
    if features.empty:
        raise ValueError("features is empty")

    weight_sum = sum(w for _, w in strategies) or 1.0
    components: list[StrategyScore] = []
    composite = 0.0

    for strat, weight in strategies:
        sigs = strat.generate_signals(features).df.iloc[-1]
        direction = 0
        if bool(sigs["entry_long"]):
            direction = 1
        elif bool(sigs["entry_short"]):
            direction = -1
        size = float(sigs["size"])
        raw = direction * size
        components.append(
            StrategyScore(
                strategy_name=strat.name,
                direction=direction,
                size=size,
                raw_score=raw,
            )
        )
        composite += raw * (weight / weight_sum)

    last = features.iloc[-1]
    regime_label = int(last["regime_hmm"]) if "regime_hmm" in features.columns else -1
    regime_proba = (
        float(last["regime_hmm_proba"])
        if "regime_hmm_proba" in features.columns and pd.notna(last["regime_hmm_proba"])
        else 0.0
    )

    # Pre-compute trade plan if the score is actionable. Use ATR — direction
    # of stop is opposite to the trade direction, TPs are with the trade.
    score_value = float(composite)
    entry = sl = tp1 = tp2 = tp3 = atr_val = None
    if score_value > DIRECTION_THRESHOLD:
        direction = 1
    elif score_value < -DIRECTION_THRESHOLD:
        direction = -1
    else:
        direction = 0
    if (
        direction != 0
        and "atr" in features.columns
        and pd.notna(last["atr"])
        and pd.notna(last["close"])
    ):
        atr_val = float(last["atr"])
        entry = float(last["close"])
        sl = entry - direction * SL_ATR_MULT * atr_val
        tp1 = entry + direction * TP_ATR_MULTS[0] * atr_val
        tp2 = entry + direction * TP_ATR_MULTS[1] * atr_val
        tp3 = entry + direction * TP_ATR_MULTS[2] * atr_val

    return CompositeScore(
        timestamp=pd.Timestamp(last["timestamp"]),
        score=score_value,
        components=components,
        regime_label=regime_label,
        regime_proba=regime_proba,
        symbol=symbol,
        bar_minutes=bar_minutes,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        atr_at_entry=atr_val,
    )
