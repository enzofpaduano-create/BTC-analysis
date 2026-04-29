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

__all__ = [
    "AlertConfig",
    "AlertsRunner",
    "CompositeScore",
    "StrategyScore",
    "default_console_sink",
    "jsonl_sink",
    "score_latest_bar",
]
