"""Live layer — real-time loop, alerts only (no auto-execution).

Public API::

    from live import AlertsRunner, AlertConfig, score_latest_bar, CompositeScore
"""

from live.alerts import (
    AlertConfig,
    AlertsRunner,
    default_console_sink,
    jsonl_sink,
)
from live.scoring import CompositeScore, StrategyScore, score_latest_bar
from live.telegram import (
    TelegramConfig,
    TelegramConfigError,
    format_alert_html,
    format_outcome_html,
    telegram_outcome_sink,
    telegram_sink,
)
from live.tracker import AlertTracker, PendingAlert, TrackerOutcome

__all__ = [
    "AlertConfig",
    "AlertTracker",
    "AlertsRunner",
    "CompositeScore",
    "PendingAlert",
    "StrategyScore",
    "TelegramConfig",
    "TelegramConfigError",
    "TrackerOutcome",
    "default_console_sink",
    "format_alert_html",
    "format_outcome_html",
    "jsonl_sink",
    "score_latest_bar",
    "telegram_outcome_sink",
    "telegram_sink",
]
