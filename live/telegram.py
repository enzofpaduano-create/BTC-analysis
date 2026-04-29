"""Telegram bot sink for the alerts runner.

Setup:
    1. Open Telegram and chat with @BotFather → ``/newbot`` → follow prompts.
       Save the bot token (looks like ``123456:ABC-DEF...``).
    2. Send any message to your new bot from your Telegram account.
    3. Find your chat id — easiest way::

           curl https://api.telegram.org/bot<TOKEN>/getUpdates

       Look for ``"chat":{"id":...}`` in the response.
    4. Put both into ``.env``::

           TELEGRAM_BOT_TOKEN=123456:ABC-...
           TELEGRAM_CHAT_ID=987654321

The sink only POSTs when ``|composite_score| ≥ min_score_abs`` so you
don't get spammed by every closed bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from loguru import logger

from live.alerts import AlertSink
from live.scoring import CompositeScore

_TELEGRAM_API_BASE = "https://api.telegram.org"
_HTTP_ERROR_FLOOR = 400


class TelegramConfigError(ValueError):
    """Raised when the bot_token or chat_id is missing/blank."""


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Connection + filtering parameters for the Telegram sink."""

    bot_token: str
    chat_id: str
    min_score_abs: float = 0.3
    timeout_s: float = 10.0
    parse_mode: str = "HTML"


def format_alert_html(score: CompositeScore) -> str:
    """Render a CompositeScore as a Telegram-friendly HTML message.

    Layout (mobile-friendly)::

        🟢 BUY BTCUSDT — 7/10
        score +0.72 • regime bull (0.82)
        2026-04-29 14:30 UTC
        ━━━━━━━━━━━━━━━━━━━━
        • mean_reversion_bb_hmm: +1 @ 0.30
        • trend_breakout_adx_hmm: +1 @ 0.35
    """
    direction = score.direction()
    emoji = {1: "🟢", -1: "🔴", 0: "⚪"}[direction]
    action = score.action()  # BUY / SELL / WAIT
    rating = score.rating()  # 1..10
    symbol = score.symbol or "—"
    regime_name = {0: "bear", 1: "range", 2: "bull"}.get(score.regime_label, "?")

    components = "\n".join(
        f"• <code>{c.strategy_name}</code>: {c.direction:+d} @ {c.size:.2f}"
        for c in score.components
    )
    return (
        f"{emoji} <b>{action} {symbol}</b> — <b>{rating}/10</b>\n"
        f"score <b>{score.score:+.2f}</b> • regime <code>{regime_name}</code> "
        f"(p={score.regime_proba:.2f})\n"
        f"<i>{score.timestamp:%Y-%m-%d %H:%M}</i> UTC\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{components}"
    )


def telegram_sink(cfg: TelegramConfig) -> AlertSink:
    """Build a sink that POSTs filtered alerts to Telegram.

    Raises:
        TelegramConfigError: If ``bot_token`` or ``chat_id`` is empty.
    """
    if not cfg.bot_token or not cfg.chat_id:
        raise TelegramConfigError("Telegram bot_token and chat_id are required")

    url = f"{_TELEGRAM_API_BASE}/bot{cfg.bot_token}/sendMessage"

    def _sink(score: CompositeScore) -> None:
        if abs(score.score) < cfg.min_score_abs:
            return
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": cfg.chat_id,
                    "text": format_alert_html(score),
                    "parse_mode": cfg.parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=cfg.timeout_s,
            )
            if response.status_code >= _HTTP_ERROR_FLOOR:
                logger.warning(
                    "Telegram API returned {}: {}",
                    response.status_code,
                    response.text[:200],
                )
        except requests.RequestException as exc:
            logger.error("Telegram POST failed: {}", exc)

    return _sink
