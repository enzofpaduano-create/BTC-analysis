"""Outcome tracker — follow up each alert for ≤ 1 hour and report the result.

For every alert that crosses the score threshold, we open a "pending" record
with its entry / SL / TP1-3. On every new closed bar, we check whether any
of the levels was hit using the bar's high & low (TP can be hit intra-bar).

A pending record is closed (and a follow-up sink event fired) on the first
of these events:

    * stop loss hit
    * any TP hit (we report up to TP3 and close)
    * the configured horizon elapses (default 1h on M5 = 12 bars)

State is persisted to a JSON file so a restart of the runner doesn't lose
in-flight alerts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from live.scoring import CompositeScore

DEFAULT_HORIZON_BARS = 12  # 1 h on M5
N_TPS = 3  # number of take-profit levels


@dataclass
class PendingAlert:
    """In-flight alert being monitored."""

    alert_id: str  # ISO timestamp of the alert (= score.timestamp)
    symbol: str
    direction: int  # +1 long, -1 short
    rating: int
    score: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    expires_at: str  # ISO timestamp
    # Tracking state
    mfe: float = 0.0  # max favorable excursion (signed return %)
    mae: float = 0.0  # max adverse excursion (signed return %)
    tps_hit: list[int] = field(default_factory=list)  # e.g. [1, 2]


@dataclass(frozen=True, slots=True)
class TrackerOutcome:
    """Result fired to the user when a pending alert is closed."""

    alert_id: str
    symbol: str
    direction: int
    entry: float
    final_price: float
    final_pct: float  # % move from entry to final_price (signed by direction)
    mfe_pct: float  # max favorable excursion (positive = good for the trade)
    mae_pct: float  # max adverse excursion (negative = bad for the trade)
    tps_hit: list[int]
    sl_hit: bool
    reason: str  # "sl_hit" | "tp3_hit" | "horizon_elapsed"
    duration_minutes: int
    bar_minutes: int = 5  # bar duration — used to compute close-time displays


OutcomeSink = Callable[[TrackerOutcome], None]


class AlertTracker:
    """Stateful tracker — feed it bars + new alerts, get outcomes back."""

    def __init__(
        self,
        *,
        state_path: Path,
        horizon_bars: int = DEFAULT_HORIZON_BARS,
        bar_minutes: int = 5,
        outcome_sinks: list[OutcomeSink] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.horizon_bars = horizon_bars
        self.bar_minutes = bar_minutes
        self.outcome_sinks: list[OutcomeSink] = outcome_sinks or []
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[PendingAlert] = self._load()

    # -- Persistence ------------------------------------------------------

    def _load(self) -> list[PendingAlert]:
        if not self.state_path.exists():
            return []
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return [PendingAlert(**rec) for rec in raw]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracker state unreadable, starting fresh: {}", exc)
            return []

    def _save(self) -> None:
        data = [asdict(p) for p in self._pending]
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- Public API -------------------------------------------------------

    def register(self, score: CompositeScore) -> None:
        """Register a fresh actionable alert as pending.

        No-op if the score is not actionable or the trade plan is missing.
        """
        if score.direction() == 0:
            return
        if (
            score.entry is None
            or score.sl is None
            or score.tp1 is None
            or score.tp2 is None
            or score.tp3 is None
        ):
            logger.warning("Alert at {} has no trade plan, skipping tracking", score.timestamp)
            return

        expires = score.timestamp + pd.Timedelta(minutes=self.horizon_bars * self.bar_minutes)
        pending = PendingAlert(
            alert_id=score.timestamp.isoformat(),
            symbol=score.symbol,
            direction=score.direction(),
            rating=score.rating(),
            score=score.score,
            entry=float(score.entry),
            sl=float(score.sl),
            tp1=float(score.tp1),
            tp2=float(score.tp2),
            tp3=float(score.tp3),
            expires_at=expires.isoformat(),
        )
        self._pending.append(pending)
        self._save()
        logger.info(
            "Tracker: registered {} alert at {}, watching for {} bars",
            "BUY" if pending.direction > 0 else "SELL",
            pending.alert_id,
            self.horizon_bars,
        )

    def update_with_bar(self, bar: pd.Series) -> list[TrackerOutcome]:
        """Process one closed bar; close any pending whose criteria are met.

        Args:
            bar: A row from the OHLCV DataFrame with at least ``timestamp``,
                ``high``, ``low``, ``close``.

        Returns:
            The outcomes fired during this update (also dispatched to sinks).
        """
        outcomes: list[TrackerOutcome] = []
        bar_ts = pd.Timestamp(bar["timestamp"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")

        kept: list[PendingAlert] = []
        for p in self._pending:
            outcome = self._check_one(p, bar_ts, bar)
            if outcome is None:
                kept.append(p)
                continue
            outcomes.append(outcome)
            for sink in self.outcome_sinks:
                try:
                    sink(outcome)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Outcome sink {} failed: {}", sink, exc)

        self._pending = kept
        self._save()
        return outcomes

    # -- Internals --------------------------------------------------------

    def _check_one(
        self, p: PendingAlert, bar_ts: pd.Timestamp, bar: pd.Series
    ) -> TrackerOutcome | None:
        """Return an outcome if this alert should be closed on this bar."""
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # Update MFE / MAE every bar for telemetry.
        favorable = (high if p.direction > 0 else low) - p.entry
        adverse = (low if p.direction > 0 else high) - p.entry
        favorable_pct = (p.direction * favorable / p.entry) * 100.0
        adverse_pct = (p.direction * adverse / p.entry) * 100.0
        p.mfe = max(p.mfe, favorable_pct)
        p.mae = min(p.mae, adverse_pct)

        # Track which TPs have been touched (for telemetry / message detail).
        tp_levels = [p.tp1, p.tp2, p.tp3]
        for n, tp in enumerate(tp_levels, start=1):
            if n in p.tps_hit:
                continue
            hit = (high >= tp) if p.direction > 0 else (low <= tp)
            if hit:
                p.tps_hit.append(n)

        # Stop hit?  (Conservative: if both SL and any TP could fire on the
        # same bar, we assume SL fills first — pessimistic for the trader.)
        sl_hit = (low <= p.sl) if p.direction > 0 else (high >= p.sl)
        if sl_hit:
            return self._make_outcome(p, close, reason="sl_hit", sl_hit=True, bar_ts=bar_ts)

        # All TPs hit? Close.
        if N_TPS in p.tps_hit:
            return self._make_outcome(p, close, reason="tp3_hit", sl_hit=False, bar_ts=bar_ts)

        # Horizon elapsed? Close at the latest close.
        if bar_ts >= pd.Timestamp(p.expires_at):
            return self._make_outcome(
                p, close, reason="horizon_elapsed", sl_hit=False, bar_ts=bar_ts
            )

        return None

    def _make_outcome(
        self,
        p: PendingAlert,
        final_price: float,
        *,
        reason: str,
        sl_hit: bool,
        bar_ts: pd.Timestamp,
    ) -> TrackerOutcome:
        final_pct = (p.direction * (final_price - p.entry) / p.entry) * 100.0
        # Duration in minutes from alert ts to current bar ts.
        alert_ts = pd.Timestamp(p.alert_id)
        if alert_ts.tzinfo is None:
            alert_ts = alert_ts.tz_localize("UTC")
        duration = int((bar_ts - alert_ts).total_seconds() // 60)
        return TrackerOutcome(
            alert_id=p.alert_id,
            symbol=p.symbol,
            direction=p.direction,
            entry=p.entry,
            final_price=final_price,
            final_pct=final_pct,
            mfe_pct=p.mfe,
            mae_pct=p.mae,
            tps_hit=sorted(p.tps_hit),
            sl_hit=sl_hit,
            reason=reason,
            duration_minutes=duration,
            bar_minutes=self.bar_minutes,
        )
