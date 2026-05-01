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

import pandas as pd
import requests
from loguru import logger

from live.alerts import AlertSink
from live.scoring import CompositeScore
from live.tracker import OutcomeSink, TrackerOutcome

_TELEGRAM_API_BASE = "https://api.telegram.org"
_HTTP_ERROR_FLOOR = 400
DEFAULT_DISPLAY_TZ = "Europe/Madrid"


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
    display_tz: str = DEFAULT_DISPLAY_TZ


def _localize(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Convert a UTC timestamp to the configured display timezone."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)


def format_alert_html(score: CompositeScore, *, display_tz: str = DEFAULT_DISPLAY_TZ) -> str:
    """Render a CompositeScore as a Telegram-friendly HTML message.

    Includes a trade plan (entry / SL / TP1-TP3) when ``score`` carries one.
    """
    direction = score.direction()
    emoji = {1: "🟢", -1: "🔴", 0: "⚪"}[direction]
    action = score.action()  # BUY / SELL / WAIT
    rating = score.rating()  # 1..10
    symbol = score.symbol or "—"
    regime_name = {0: "bear", 1: "range", 2: "bull"}.get(score.regime_label, "?")
    # Display the bar CLOSE time (= when the alert actually fires) rather
    # than the bar OPEN time, so the user's perceived delay is minimal.
    local_ts = _localize(score.close_time(), display_tz)
    tz_short = local_ts.strftime("%Z") or display_tz.rsplit("/", maxsplit=1)[-1]

    components = "\n".join(
        f"• <code>{c.strategy_name}</code>: {c.direction:+d}" for c in score.components
    )

    # Trade plan block (only when actionable AND ATR-derived levels available).
    plan = ""
    if (
        score.entry is not None
        and score.sl is not None
        and score.tp1 is not None
        and score.tp2 is not None
        and score.tp3 is not None
    ):
        entry = score.entry
        sl = score.sl
        tp1 = score.tp1
        tp2 = score.tp2
        tp3 = score.tp3

        def _pct(level: float) -> str:
            return f"{(level - entry) / entry * 100:+.2f} %"

        plan = (
            "\n"
            f"📍 <b>Entry</b>  <code>{entry:,.2f}</code>\n"
            f"🛡️ <b>Stop</b>   <code>{sl:,.2f}</code>  ({_pct(sl)})\n"
            f"🎯 <b>TP1</b>    <code>{tp1:,.2f}</code>  ({_pct(tp1)})  — 1R\n"
            f"🎯 <b>TP2</b>    <code>{tp2:,.2f}</code>  ({_pct(tp2)})  — 2R\n"
            f"🎯 <b>TP3</b>    <code>{tp3:,.2f}</code>  ({_pct(tp3)})  — 3R"
        )

    return (
        f"{emoji} <b>{action} {symbol}</b> — <b>{rating}/10</b>\n"
        f"score <b>{score.score:+.2f}</b> • regime <code>{regime_name}</code> "
        f"(p={score.regime_proba:.2f})\n"
        f"<i>{local_ts:%Y-%m-%d %H:%M}</i> {tz_short}"
        f"{plan}\n"
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
                    "text": format_alert_html(score, display_tz=cfg.display_tz),
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


_TP3 = 3
_TP2 = 2
_TP1 = 1


def format_outcome_html(outcome: TrackerOutcome) -> str:
    """Format a tracker outcome as a follow-up Telegram message."""
    side = "BUY" if outcome.direction > 0 else "SELL"
    if outcome.sl_hit:
        emoji = "🛑"
        verdict = "SL hit"
    elif _TP3 in outcome.tps_hit:
        emoji = "🎯🎯🎯"
        verdict = "TP3 hit (full target)"
    elif _TP2 in outcome.tps_hit:
        emoji = "🎯🎯"
        verdict = "TP2 hit (1h elapsed)"
    elif _TP1 in outcome.tps_hit:
        emoji = "🎯"
        verdict = "TP1 hit (1h elapsed)"
    else:
        emoji = "⏰"
        verdict = "1h elapsed, no TP/SL"

    sign_final = "+" if outcome.final_pct >= 0 else ""
    # alert_id is the bar OPEN ISO timestamp; advance to the close (= when
    # the alert was actually emitted) for display, in the configured TZ.
    alert_open = pd.Timestamp(outcome.alert_id)
    if alert_open.tzinfo is None:
        alert_open = alert_open.tz_localize("UTC")
    alert_close_local = _localize(
        alert_open + pd.Timedelta(minutes=outcome.bar_minutes), DEFAULT_DISPLAY_TZ
    )
    return (
        f"{emoji} <b>{side} {outcome.symbol}</b> result\n"
        f"alert at <code>{alert_close_local:%H:%M}</code> "
        f"({outcome.duration_minutes} min ago)\n"
        f"verdict: <b>{verdict}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"entry  <code>{outcome.entry:,.2f}</code>\n"
        f"final  <code>{outcome.final_price:,.2f}</code>  "
        f"<b>({sign_final}{outcome.final_pct:.2f} %)</b>\n"
        f"best   <b>{outcome.mfe_pct:+.2f} %</b>  •  "
        f"worst  <b>{outcome.mae_pct:+.2f} %</b>"
    )


def telegram_outcome_sink(cfg: TelegramConfig) -> OutcomeSink:
    """Build a sink that POSTs follow-up messages to Telegram."""
    if not cfg.bot_token or not cfg.chat_id:
        raise TelegramConfigError("Telegram bot_token and chat_id are required")

    url = f"{_TELEGRAM_API_BASE}/bot{cfg.bot_token}/sendMessage"

    def _sink(outcome: TrackerOutcome) -> None:
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": cfg.chat_id,
                    "text": format_outcome_html(outcome),
                    "parse_mode": cfg.parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=cfg.timeout_s,
            )
            if response.status_code >= _HTTP_ERROR_FLOOR:
                logger.warning(
                    "Telegram outcome POST returned {}: {}",
                    response.status_code,
                    response.text[:200],
                )
        except requests.RequestException as exc:
            logger.error("Telegram outcome POST failed: {}", exc)

    return _sink
