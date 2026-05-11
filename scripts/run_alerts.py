"""Live alerts runner — connects Bybit polling → features → composite scoring → alerts.

This script runs forever. Stop with Ctrl+C. It does NOT execute orders.

Usage::

    uv run python -m scripts.run_alerts

Outputs:
    - INFO logs on every closed bar with the composite score
    - WARNING logs whenever |score| ≥ AlertConfig.alert_threshold
    - JSONL file at reports/alerts.jsonl (one record per closed bar)

Customisation: edit the strategies + weights below.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from loguru import logger

from core.logging import setup_logging
from core.settings import get_settings
from features import FeaturesConfig
from live.alerts import AlertConfig, AlertsRunner, default_console_sink
from live.telegram import TelegramConfig, telegram_outcome_sink, telegram_sink
from live.tracker import AlertTracker
from signals import MeanReversionBollingerHMM, TrendBreakoutADXHMM

PARQUET_ROOT = Path("./data_store/parquet")
ALERTS_LOG = Path("./reports/alerts.jsonl")


def main() -> None:
    setup_logging()
    settings = get_settings()

    cfg = AlertConfig(
        symbol="BTCUSDT",
        timeframe="5",
        bar_minutes=5,
        warmup_bars=5_000,
        alert_threshold=0.3,
        alert_log_path=ALERTS_LOG,
        parquet_root=PARQUET_ROOT,
        # "Scenario J" — derived from a 54-alert live audit over 11 days.
        # On M5 the ATR is often very small (median 0.13 %) so the original
        # SL/TP geometry was getting wicked. Combined with skipping SELL
        # (which loses badly in a slow uptrend) this flips the system net
        # PnL from -2.85 % to +0.88 % over the same period.
        skip_short=True,
        min_atr_pct=0.0015,  # require ATR ≥ 0.15 % of price
    )

    feat_cfg = FeaturesConfig(bar_minutes=5)
    feat_cfg.regime.hmm_min_obs = 500
    feat_cfg.regime.hmm_refit_every = 1000
    feat_cfg.regime.cp_window = 1000
    feat_cfg.regime.cp_refit_every = 200
    feat_cfg.volatility.garch_min_obs = 500
    feat_cfg.volatility.garch_refit_every = 1000

    # Two complementary strategies: mean-reversion and trend-following.
    strategies = [
        (MeanReversionBollingerHMM(target_vol_per_trade=1.0), 1.0),
        (TrendBreakoutADXHMM(target_vol_per_trade=1.0), 1.0),
    ]

    sinks = [default_console_sink]
    outcome_sinks = []
    if settings.telegram_bot_token and settings.telegram_chat_id:
        tg_cfg = TelegramConfig(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            min_score_abs=settings.telegram_min_score_abs,
        )
        sinks.append(telegram_sink(tg_cfg))
        outcome_sinks.append(telegram_outcome_sink(tg_cfg))
        logger.info(
            "Telegram sink active (chat_id={}, min_score_abs={}, tz={})",
            settings.telegram_chat_id,
            settings.telegram_min_score_abs,
            tg_cfg.display_tz,
        )
    else:
        logger.info("Telegram sink disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")

    # Outcome tracker — reports the result of each alert after ≤ 2h on M5
    # (wider after the audit: 1h truncated many TP3 outcomes that completed
    # in the 60-120 min window).
    tracker = AlertTracker(
        state_path=Path("./data_store/pending_alerts.json"),
        horizon_bars=24,  # 2 hours on M5
        bar_minutes=cfg.bar_minutes,
        outcome_sinks=outcome_sinks,
    )

    runner = AlertsRunner(
        strategies=strategies,
        cfg=cfg,
        features_cfg=feat_cfg,
        sinks=sinks,
        tracker=tracker,
    )
    with contextlib.suppress(KeyboardInterrupt):
        runner.run()


if __name__ == "__main__":
    main()
