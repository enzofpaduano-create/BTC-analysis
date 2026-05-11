"""Live alerts loop — poll Bybit, compute features, emit composite-score alerts.

Crucial: this module does NOT execute orders. It writes to:
    - the loguru logger (with WARNING level on actionable scores)
    - an optional structured JSON-lines file for downstream tooling (Telegram,
      Slack, dashboards…) — left as an integration point.

The user's brief says "alertes seulement au début" — keep it that way.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pandas as pd
from loguru import logger

from backtest.strategy import Strategy
from data.client import BybitClient
from data.schemas import Category, Timeframe
from data.storage import ParquetStore
from data.stream import stream_klines
from features import FeaturesConfig, compute_features
from live.scoring import CompositeScore, score_latest_bar
from live.tracker import AlertTracker

# Score above this magnitude triggers an actionable alert (logged at WARNING).
DEFAULT_ALERT_THRESHOLD = 0.3

AlertSink = Callable[[CompositeScore], None]


@dataclass
class AlertConfig:
    """Knobs for the alerting loop."""

    symbol: str = "BTCUSDT"
    timeframe: Timeframe = "5"
    category: Category = "linear"
    bar_minutes: int = 5
    # Number of bars kept in memory for feature warmup. Must be larger than
    # the longest indicator/regime window to avoid degenerate features.
    warmup_bars: int = 5_000
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    alert_log_path: Path | None = None  # JSON-lines file
    poll_interval_s: float | None = None
    parquet_root: Path | None = None  # if provided, fresh bars are persisted
    # Quality filters — applied AFTER scoring, BEFORE dispatching to sinks
    # and the tracker. Console keeps logging filtered alerts at DEBUG level
    # so the runner stays observable.
    skip_short: bool = False  # drop SELL signals entirely
    min_atr_pct: float = 0.0  # drop alerts where ATR/close < this fraction

    def is_actionable(self, score: CompositeScore) -> bool:
        """Whether ``score`` should fire Telegram + tracker.

        Console always sees the score for telemetry; this method controls
        the *user-facing* dispatching only.
        """
        if abs(score.score) < self.alert_threshold:
            return False
        if self.skip_short and score.direction() < 0:
            return False
        return not (
            self.min_atr_pct > 0
            and score.entry is not None
            and score.atr_at_entry is not None
            and score.entry > 0
            and (score.atr_at_entry / score.entry) < self.min_atr_pct
        )


def default_console_sink(score: CompositeScore) -> None:
    """Print a compact one-liner to the console."""
    parts = " ".join(f"{c.strategy_name}:{c.direction:+d}@{c.size:.2f}" for c in score.components)
    symbol_str = f"{score.symbol} " if score.symbol else ""
    # Use bar CLOSE time so log timestamps match when each alert is emitted.
    close_ts = score.close_time()
    msg = (
        f"[{close_ts:%Y-%m-%d %H:%M}] {symbol_str}{score.action()} "
        f"{score.rating()}/10 score={score.score:+.2f} "
        f"regime={score.regime_label}({score.regime_proba:.2f}) | {parts}"
    )
    if abs(score.score) >= DEFAULT_ALERT_THRESHOLD:
        logger.warning(msg)
    else:
        logger.info(msg)


def jsonl_sink(path: Path) -> AlertSink:
    """Return an AlertSink that appends one JSON line per call."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sink(score: CompositeScore) -> None:
        record = {
            "ts": score.timestamp.isoformat(),
            "score": score.score,
            "direction": score.direction(),
            "regime_label": score.regime_label,
            "regime_proba": score.regime_proba,
            "components": [
                {
                    "strategy": c.strategy_name,
                    "direction": c.direction,
                    "size": c.size,
                }
                for c in score.components
            ],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return _sink


class AlertsRunner:
    """Glues the data stream + features + composite scoring + sinks together.

    Usage::

        runner = AlertsRunner(strategies=[(strat1, 1.0), (strat2, 0.5)],
                              cfg=AlertConfig(symbol="BTCUSDT", timeframe="5"),
                              features_cfg=FeaturesConfig(bar_minutes=5),
                              sinks=[default_console_sink])
        runner.run()  # blocks; pass stop_event to break cleanly
    """

    def __init__(
        self,
        *,
        strategies: list[tuple[Strategy, float]],
        cfg: AlertConfig,
        features_cfg: FeaturesConfig,
        sinks: list[AlertSink] | None = None,
        tracker: AlertTracker | None = None,
        client: BybitClient | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy required")
        self.strategies = strategies
        self.cfg = cfg
        self.features_cfg = features_cfg
        self.sinks: list[AlertSink] = sinks or [default_console_sink]
        if cfg.alert_log_path is not None:
            self.sinks.append(jsonl_sink(cfg.alert_log_path))
        self.tracker = tracker
        self.client = client
        self._buffer: pd.DataFrame | None = None  # rolling OHLCV window

    def _on_bar(self, bar: pd.DataFrame) -> None:
        """Append the new closed bar, rebuild features, score, dispatch."""
        if self._buffer is None or self._buffer.empty:
            self._buffer = bar.copy()
        else:
            # Dedup / monotonic ordering guard — polling can occasionally
            # surface a bar we already have, or one slightly older than the
            # latest. We append only strictly-newer bars.
            last_ts = self._buffer["timestamp"].iloc[-1]
            new_bar = bar[bar["timestamp"] > last_ts]
            if new_bar.empty:
                logger.debug(
                    "Dropping non-monotonic bar (last={}, got={})",
                    last_ts,
                    bar["timestamp"].iloc[-1],
                )
                return
            self._buffer = pd.concat([self._buffer, new_bar], ignore_index=True)
            # Keep only the last `warmup_bars` rows.
            if len(self._buffer) > self.cfg.warmup_bars:
                self._buffer = self._buffer.iloc[-self.cfg.warmup_bars :].reset_index(drop=True)

        # Need enough history for features to be meaningful.
        min_needed = max(
            self.features_cfg.regime.hmm_min_obs,
            self.features_cfg.volatility.garch_min_obs,
            300,
        )
        if len(self._buffer) < min_needed:
            logger.debug("Buffer warming up: {}/{}", len(self._buffer), min_needed)
            return

        feat = compute_features(self._buffer, self.features_cfg)
        score = score_latest_bar(
            features=feat,
            strategies=self.strategies,
            symbol=self.cfg.symbol,
            bar_minutes=self.cfg.bar_minutes,
        )

        # Quality filters: when an alert is generated but filtered, the
        # console sink (first in the list) still logs it for observability,
        # while sinks further down (telegram, JSONL) only see the actionable
        # ones. The tracker also only registers actionable alerts.
        actionable = self.cfg.is_actionable(score)
        if not actionable and abs(score.score) >= self.cfg.alert_threshold:
            # Score crossed the alert threshold but got filtered — log why.
            reason = (
                "skip_short" if (self.cfg.skip_short and score.direction() < 0) else "atr_too_small"
            )
            logger.info(
                "Filtered alert: {} {} score={:+.2f} ATR%={:.3f}% reason={}",
                score.symbol,
                score.action(),
                score.score,
                (score.atr_at_entry / score.entry * 100)
                if (score.entry and score.atr_at_entry)
                else float("nan"),
                reason,
            )

        sinks_to_call = self.sinks if actionable else self.sinks[:1]  # console only
        for sink in sinks_to_call:
            try:
                sink(score)
            except Exception as exc:  # noqa: BLE001
                logger.error("Alert sink {} failed: {}", sink, exc)

        # Tracker: feed the new bar to close any pending outcomes, then
        # register this alert if it passed the filters.
        if self.tracker is not None:
            try:
                self.tracker.update_with_bar(self._buffer.iloc[-1])
                if actionable:
                    self.tracker.register(score)
            except Exception as exc:  # noqa: BLE001
                logger.error("Tracker error: {}", exc)

    def run(self, *, stop_event: Event | None = None) -> None:
        """Block on the stream loop until ``stop_event`` is set."""
        store = ParquetStore(self.cfg.parquet_root) if self.cfg.parquet_root else None
        self._prefill_buffer(store)
        stream_klines(
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            on_bar=self._on_bar,
            category=self.cfg.category,
            poll_interval_s=self.cfg.poll_interval_s,
            store=store,
            stop_event=stop_event,
            client=self.client,
        )

    def _prefill_buffer(self, store: ParquetStore | None) -> None:
        """If the Parquet store has history, seed the buffer to skip warmup."""
        if store is None:
            return
        df = store.read(self.cfg.symbol, self.cfg.timeframe)
        if df.empty:
            return
        self._buffer = df.iloc[-self.cfg.warmup_bars :].reset_index(drop=True)
        logger.info(
            "Pre-filled live buffer with {} bars from disk (last={})",
            len(self._buffer),
            self._buffer["timestamp"].iloc[-1],
        )
