"""Tests for the outcome tracker."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from live.scoring import CompositeScore, StrategyScore
from live.tracker import AlertTracker, TrackerOutcome


def _score(
    *,
    side: int,
    entry: float = 100.0,
    sl_pct: float = 0.02,
    rating: int = 5,
    score: float = 0.5,
    ts: str = "2024-01-01 00:00",
) -> CompositeScore:
    sign = side
    return CompositeScore(
        timestamp=pd.Timestamp(ts, tz="UTC"),
        score=sign * score,
        components=[StrategyScore("toy", side, 1.0, sign * 1.0)],
        regime_label=2 if side > 0 else 0,
        regime_proba=0.9,
        symbol="BTCUSDT",
        entry=entry,
        sl=entry - sign * sl_pct * entry,  # 2% against the trade
        tp1=entry + sign * 0.01 * entry,
        tp2=entry + sign * 0.02 * entry,
        tp3=entry + sign * 0.03 * entry,
        atr_at_entry=1.0,
    )


def _bar(*, ts: str, high: float, low: float, close: float) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp(ts, tz="UTC"),
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_register_and_persist(tmp_path: Path) -> None:
    state = tmp_path / "pending.json"
    t = AlertTracker(state_path=state, horizon_bars=12, bar_minutes=5)
    t.register(_score(side=1))
    assert state.exists()
    # Reload — pending should survive a process restart.
    t2 = AlertTracker(state_path=state, horizon_bars=12, bar_minutes=5)
    assert len(t2._pending) == 1
    assert t2._pending[0].entry == 100.0


def test_register_skips_when_no_trade_plan(tmp_path: Path) -> None:
    t = AlertTracker(state_path=tmp_path / "p.json", horizon_bars=12, bar_minutes=5)
    bare = CompositeScore(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        score=0.5,
        components=[],
        regime_label=2,
        regime_proba=0.9,
        symbol="BTCUSDT",
    )
    t.register(bare)
    assert len(t._pending) == 0


def test_tp1_hit_long(tmp_path: Path) -> None:
    received: list[TrackerOutcome] = []
    t = AlertTracker(
        state_path=tmp_path / "p.json",
        horizon_bars=12,
        bar_minutes=5,
        outcome_sinks=[received.append],
    )
    t.register(_score(side=1, entry=100.0))  # TP1=101, TP2=102, TP3=103, SL=98
    # Bar straddles TP1 but no further.
    out = t.update_with_bar(_bar(ts="2024-01-01 00:05", high=101.5, low=99.5, close=101.2))
    # TP1 hit but not TP3 → not closed yet (we wait for TP3 or horizon).
    assert len(out) == 0
    assert 1 in t._pending[0].tps_hit


def test_sl_hit_long_closes_immediately(tmp_path: Path) -> None:
    received: list[TrackerOutcome] = []
    t = AlertTracker(
        state_path=tmp_path / "p.json",
        horizon_bars=12,
        bar_minutes=5,
        outcome_sinks=[received.append],
    )
    t.register(_score(side=1, entry=100.0, sl_pct=0.02))  # SL = 98
    out = t.update_with_bar(_bar(ts="2024-01-01 00:05", high=99.0, low=97.5, close=97.8))
    assert len(out) == 1
    assert out[0].sl_hit is True
    assert out[0].reason == "sl_hit"
    assert len(t._pending) == 0


def test_horizon_elapsed_closes_at_close(tmp_path: Path) -> None:
    received: list[TrackerOutcome] = []
    t = AlertTracker(
        state_path=tmp_path / "p.json",
        horizon_bars=2,  # very short for the test
        bar_minutes=5,
        outcome_sinks=[received.append],
    )
    t.register(_score(side=1, entry=100.0, ts="2024-01-01 00:00"))
    # Bar 1 — no TP/SL hit
    t.update_with_bar(_bar(ts="2024-01-01 00:05", high=100.5, low=99.8, close=100.2))
    # Bar 2 — horizon (10 min) reached.
    out = t.update_with_bar(_bar(ts="2024-01-01 00:10", high=100.4, low=100.0, close=100.3))
    assert len(out) == 1
    assert out[0].reason == "horizon_elapsed"
    assert out[0].final_price == 100.3
    assert abs(out[0].final_pct - 0.30) < 1e-6  # +0.30 % from 100 to 100.3


def test_tp3_hit_long_closes(tmp_path: Path) -> None:
    received: list[TrackerOutcome] = []
    t = AlertTracker(
        state_path=tmp_path / "p.json",
        horizon_bars=12,
        bar_minutes=5,
        outcome_sinks=[received.append],
    )
    t.register(_score(side=1, entry=100.0))  # TP3 = 103
    out = t.update_with_bar(_bar(ts="2024-01-01 00:05", high=103.5, low=99.5, close=103.2))
    assert len(out) == 1
    assert out[0].reason == "tp3_hit"
    assert out[0].tps_hit == [1, 2, 3]


def test_short_sl_hit(tmp_path: Path) -> None:
    """For a SHORT, SL is ABOVE entry; TPs are below."""
    t = AlertTracker(state_path=tmp_path / "p.json", horizon_bars=12, bar_minutes=5)
    t.register(_score(side=-1, entry=100.0))  # SL = 102
    # High of bar exceeds SL.
    out = t.update_with_bar(_bar(ts="2024-01-01 00:05", high=102.5, low=99.5, close=101.0))
    assert len(out) == 1
    assert out[0].sl_hit is True
    assert out[0].direction == -1


def test_mfe_mae_tracking(tmp_path: Path) -> None:
    t = AlertTracker(state_path=tmp_path / "p.json", horizon_bars=12, bar_minutes=5)
    t.register(_score(side=1, entry=100.0))
    # First bar — favorable +1 %, adverse -0.5 %
    t.update_with_bar(_bar(ts="2024-01-01 00:05", high=101.0, low=99.5, close=100.5))
    # Second bar — favorable still capped at 101, but adverse -1.5 %
    t.update_with_bar(_bar(ts="2024-01-01 00:10", high=100.8, low=98.5, close=99.0))
    p = t._pending[0]
    assert abs(p.mfe - 1.0) < 1e-6  # max +1 %
    assert abs(p.mae - (-1.5)) < 1e-6  # max adverse -1.5 %


@pytest.mark.parametrize("side", [1, -1])
def test_outcome_pct_signed_by_direction(tmp_path: Path, side: int) -> None:
    """For a SHORT, the price going DOWN should give a POSITIVE final_pct."""
    received: list[TrackerOutcome] = []
    t = AlertTracker(
        state_path=tmp_path / "p.json",
        horizon_bars=2,
        bar_minutes=5,
        outcome_sinks=[received.append],
    )
    t.register(_score(side=side, entry=100.0, ts="2024-01-01 00:00"))
    # Price moves with the trade: +0.5 % for long, -0.5 % for short.
    final_close = 100.0 + side * 0.5
    t.update_with_bar(
        _bar(
            ts="2024-01-01 00:05",
            high=final_close + 0.1,
            low=final_close - 0.1,
            close=final_close,
        )
    )
    out = t.update_with_bar(
        _bar(ts="2024-01-01 00:10", high=final_close, low=final_close, close=final_close)
    )
    assert len(out) == 1
    assert out[0].final_pct > 0  # both directions: with-trade move is positive
