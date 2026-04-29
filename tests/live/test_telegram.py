"""Tests for the Telegram sink (HTTP fully mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from live.scoring import CompositeScore, StrategyScore
from live.telegram import (
    TelegramConfig,
    TelegramConfigError,
    format_alert_html,
    telegram_sink,
)


def _make_score(score: float = 0.5, direction: int = 1) -> CompositeScore:
    return CompositeScore(
        timestamp=pd.Timestamp("2026-04-29 14:30", tz="UTC"),
        score=score,
        components=[
            StrategyScore("strat_a", direction=direction, size=0.3, raw_score=direction * 0.3),
            StrategyScore("strat_b", direction=direction, size=0.2, raw_score=direction * 0.2),
        ],
        regime_label=2,
        regime_proba=0.78,
    )


# -- formatting -----------------------------------------------------------


def test_format_alert_html_long() -> None:
    msg = format_alert_html(_make_score(score=0.7, direction=1))
    assert "🟢" in msg
    assert "LONG" in msg
    assert "+0.70" in msg
    assert "strat_a" in msg
    assert "bull" in msg


def test_format_alert_html_short() -> None:
    msg = format_alert_html(_make_score(score=-0.7, direction=-1))
    assert "🔴" in msg
    assert "SHORT" in msg
    assert "-0.70" in msg


def test_format_alert_html_flat() -> None:
    msg = format_alert_html(_make_score(score=0.05, direction=0))
    assert "⚪" in msg
    assert "FLAT" in msg


# -- sink behaviour -------------------------------------------------------


def test_sink_requires_token_and_chat_id() -> None:
    with pytest.raises(TelegramConfigError):
        telegram_sink(TelegramConfig(bot_token="", chat_id="123"))
    with pytest.raises(TelegramConfigError):
        telegram_sink(TelegramConfig(bot_token="abc", chat_id=""))


def test_sink_posts_when_score_above_threshold() -> None:
    cfg = TelegramConfig(bot_token="TOKEN", chat_id="CHAT", min_score_abs=0.3)
    with patch("live.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        sink = telegram_sink(cfg)
        sink(_make_score(score=0.5))
        assert mock_post.call_count == 1
        kwargs = mock_post.call_args.kwargs
        # URL embeds the bot token.
        assert "TOKEN" in mock_post.call_args.args[0]
        # Payload carries chat_id, formatted text, HTML mode.
        assert kwargs["json"]["chat_id"] == "CHAT"
        assert "LONG" in kwargs["json"]["text"]
        assert kwargs["json"]["parse_mode"] == "HTML"


def test_sink_skips_below_threshold() -> None:
    cfg = TelegramConfig(bot_token="TOKEN", chat_id="CHAT", min_score_abs=0.5)
    with patch("live.telegram.requests.post") as mock_post:
        sink = telegram_sink(cfg)
        sink(_make_score(score=0.2))
        assert not mock_post.called


def test_sink_logs_warning_on_http_error() -> None:
    cfg = TelegramConfig(bot_token="T", chat_id="C", min_score_abs=0.0)
    with patch("live.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=429, text="Too many requests")
        sink = telegram_sink(cfg)
        # Should NOT raise — Telegram outage must not crash the runner.
        sink(_make_score(score=0.5))


def test_sink_swallows_request_exceptions() -> None:
    """Network failures are logged but never propagated."""
    cfg = TelegramConfig(bot_token="T", chat_id="C", min_score_abs=0.0)
    with patch("live.telegram.requests.post", side_effect=requests.ConnectionError("boom")):
        sink = telegram_sink(cfg)
        sink(_make_score(score=0.5))  # must not raise
