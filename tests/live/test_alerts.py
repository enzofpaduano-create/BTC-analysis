"""Tests for the AlertsRunner — uses a mocked stream to avoid network."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest.strategy import SIGNAL_COLUMNS, SignalFrame, Strategy
from features import FeaturesConfig
from live.alerts import AlertConfig, AlertsRunner, jsonl_sink
from live.scoring import CompositeScore


class _FlatStrategy(Strategy):
    """Always emits no signal — no risk of an alert firing during the test."""

    name = "flat"

    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        n = len(features)
        df = pd.DataFrame(
            {
                "entry_long": np.zeros(n, dtype=bool),
                "exit_long": np.zeros(n, dtype=bool),
                "entry_short": np.zeros(n, dtype=bool),
                "exit_short": np.zeros(n, dtype=bool),
                "size": np.zeros(n),
            },
            index=features.index,
            columns=list(SIGNAL_COLUMNS),
        )
        return SignalFrame(df=df)


def _make_features(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.exp(np.cumsum(rng.normal(0.0, 0.001, n))) * 60_000.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": rng.uniform(5.0, 50.0, n),
            "turnover": close * 10.0,
        }
    )


def test_jsonl_sink_writes_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "alerts.jsonl"
    sink = jsonl_sink(path)
    score = CompositeScore(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        score=0.42,
        components=[],
        regime_label=2,
        regime_proba=0.85,
    )
    sink(score)
    sink(score)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["score"] == 0.42
    assert record["regime_label"] == 2


def test_alerts_runner_calls_sink_on_each_bar(tmp_path: Path) -> None:
    """Runner consumes the mocked stream, builds features, calls sinks."""
    received_scores: list[CompositeScore] = []

    def _capture_sink(score: CompositeScore) -> None:
        received_scores.append(score)

    pre_filled = _make_features(n=600)

    cfg = AlertConfig(
        symbol="BTCUSDT",
        timeframe="5",
        bar_minutes=5,
        warmup_bars=600,
        parquet_root=None,
    )
    feat_cfg = FeaturesConfig(bar_minutes=5)
    feat_cfg.regime.hmm_min_obs = 200
    feat_cfg.regime.hmm_refit_every = 200
    feat_cfg.regime.cp_window = 200
    feat_cfg.regime.cp_refit_every = 100
    feat_cfg.volatility.garch_min_obs = 200
    feat_cfg.volatility.garch_refit_every = 500

    runner = AlertsRunner(
        strategies=[(_FlatStrategy(), 1.0)],
        cfg=cfg,
        features_cfg=feat_cfg,
        sinks=[_capture_sink],
    )
    runner._buffer = pre_filled  # pre-fill so we don't hit the warmup guard

    # Mock stream_klines to feed two new bars then signal stop.
    stop = Event()
    last_ts = pre_filled["timestamp"].iloc[-1]
    new_bars = [
        pd.DataFrame(
            {
                "timestamp": [last_ts + pd.Timedelta(minutes=5 * (i + 1))],
                "open": [60_000.0],
                "high": [60_005.0],
                "low": [59_995.0],
                "close": [60_001.0 + i],
                "volume": [10.0],
                "turnover": [600_010.0],
            }
        )
        for i in range(2)
    ]

    def fake_stream_klines(*, on_bar, stop_event=None, **_kwargs):  # noqa: ANN001
        for b in new_bars:
            on_bar(b)
        if stop_event is not None:
            stop_event.set()

    with patch("live.alerts.stream_klines", side_effect=fake_stream_klines):
        thread = Thread(target=runner.run, kwargs={"stop_event": stop}, daemon=True)
        thread.start()
        thread.join(timeout=120.0)

    assert not thread.is_alive(), "alerts runner did not stop"
    assert len(received_scores) == 2
    for s in received_scores:
        # Flat strategy → score == 0.
        assert s.score == 0.0
        assert s.label() == "FLAT"


def test_alerts_runner_warmup_guard_skips_when_buffer_too_short() -> None:
    """If the buffer has less than min_obs, the runner does NOT call sinks."""
    received: list[CompositeScore] = []

    def sink(s: CompositeScore) -> None:
        received.append(s)

    cfg = AlertConfig(symbol="BTCUSDT", timeframe="5", bar_minutes=5, warmup_bars=100)
    feat_cfg = FeaturesConfig(bar_minutes=5)
    runner = AlertsRunner(
        strategies=[(_FlatStrategy(), 1.0)],
        cfg=cfg,
        features_cfg=feat_cfg,
        sinks=[sink],
    )
    # Buffer only has 10 bars.
    runner._buffer = _make_features(n=10)

    bar = _make_features(n=1)
    bar.loc[0, "timestamp"] = pd.Timestamp("2024-02-01", tz="UTC")
    runner._on_bar(bar)
    assert received == []
