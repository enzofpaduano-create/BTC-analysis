"""Tests for composite scoring across strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy
from live.scoring import score_latest_bar


class _AlwaysLong(Strategy):
    name = "always_long"

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        n = len(features)
        df = pd.DataFrame(
            {
                "entry_long": np.ones(n, dtype=bool),
                "exit_long": np.zeros(n, dtype=bool),
                "entry_short": np.zeros(n, dtype=bool),
                "exit_short": np.zeros(n, dtype=bool),
                "size": np.full(n, 0.5),
            },
            index=features.index,
            columns=list(SIGNAL_COLUMNS),
        )
        return SignalFrame(df=df)


class _AlwaysShort(Strategy):
    name = "always_short"

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        n = len(features)
        df = pd.DataFrame(
            {
                "entry_long": np.zeros(n, dtype=bool),
                "exit_long": np.zeros(n, dtype=bool),
                "entry_short": np.ones(n, dtype=bool),
                "exit_short": np.zeros(n, dtype=bool),
                "size": np.full(n, 0.5),
            },
            index=features.index,
            columns=list(SIGNAL_COLUMNS),
        )
        return SignalFrame(df=df)


def _features(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "regime_hmm": [2] * n,
            "regime_hmm_proba": [0.8] * n,
        }
    )


def test_two_long_strategies_score_positive() -> None:
    feat = _features()
    score = score_latest_bar(
        features=feat,
        strategies=[(_AlwaysLong(), 1.0), (_AlwaysLong(), 1.0)],
    )
    # Both fire +1 with size 0.5 → composite = 0.5 (weighted average).
    assert abs(score.score - 0.5) < 1e-9
    assert score.label() == "LONG"


def test_long_and_short_cancel_out() -> None:
    feat = _features()
    score = score_latest_bar(
        features=feat,
        strategies=[(_AlwaysLong(), 1.0), (_AlwaysShort(), 1.0)],
    )
    assert abs(score.score) < 1e-9
    assert score.label() == "FLAT"


def test_weights_are_normalised() -> None:
    """Heavier weight on the long strategy → score leans long."""
    feat = _features()
    score = score_latest_bar(
        features=feat,
        strategies=[(_AlwaysLong(), 3.0), (_AlwaysShort(), 1.0)],
    )
    # raw = 0.5 long, -0.5 short. Normalised weights: 0.75 and 0.25.
    # composite = 0.75 * 0.5 + 0.25 * (-0.5) = 0.375 - 0.125 = 0.25
    assert abs(score.score - 0.25) < 1e-9
    assert score.direction() == 1


def test_score_includes_regime_metadata() -> None:
    feat = _features()
    score = score_latest_bar(features=feat, strategies=[(_AlwaysLong(), 1.0)])
    assert score.regime_label == 2
    assert score.regime_proba == 0.8


def test_score_carries_symbol() -> None:
    feat = _features()
    score = score_latest_bar(features=feat, strategies=[(_AlwaysLong(), 1.0)], symbol="BTCUSDT")
    assert score.symbol == "BTCUSDT"


def test_action_word_long_short_wait() -> None:
    feat = _features()
    long_score = score_latest_bar(features=feat, strategies=[(_AlwaysLong(), 1.0)])
    short_score = score_latest_bar(features=feat, strategies=[(_AlwaysShort(), 1.0)])
    assert long_score.action() == "BUY"
    assert short_score.action() == "SELL"
    # Cancelling each other → flat → WAIT.
    flat = score_latest_bar(features=feat, strategies=[(_AlwaysLong(), 1.0), (_AlwaysShort(), 1.0)])
    assert flat.action() == "WAIT"


def test_rating_scales_with_magnitude() -> None:
    feat = _features()
    score = score_latest_bar(features=feat, strategies=[(_AlwaysLong(), 1.0)])
    # Always-long fires +1 with size 0.5 → score 0.5 → rating 5/10.
    assert score.rating() == 5


def test_threshold_for_flat_label() -> None:
    """Tiny scores stay FLAT (avoid false alerts)."""
    feat = _features()

    class _TinyLong(Strategy):
        name = "tiny_long"

        def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
            n = len(features)
            df = pd.DataFrame(
                {
                    "entry_long": np.ones(n, dtype=bool),
                    "exit_long": np.zeros(n, dtype=bool),
                    "entry_short": np.zeros(n, dtype=bool),
                    "exit_short": np.zeros(n, dtype=bool),
                    "size": np.full(n, 0.1),  # very small intensity
                },
                index=features.index,
                columns=list(SIGNAL_COLUMNS),
            )
            return SignalFrame(df=df)

    score = score_latest_bar(features=feat, strategies=[(_TinyLong(), 1.0)])
    assert abs(score.score - 0.1) < 1e-9
    # 0.1 < 0.2 threshold → label flat.
    assert score.label() == "FLAT"
