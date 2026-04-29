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
from live.telegram import TelegramConfig, telegram_sink
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
    if settings.telegram_bot_token and settings.telegram_chat_id:
        sinks.append(
            telegram_sink(
                TelegramConfig(
                    bot_token=settings.telegram_bot_token,
                    chat_id=settings.telegram_chat_id,
                    min_score_abs=settings.telegram_min_score_abs,
                )
            )
        )
        logger.info(
            "Telegram sink active (chat_id={}, min_score_abs={})",
            settings.telegram_chat_id,
            settings.telegram_min_score_abs,
        )
    else:
        logger.info("Telegram sink disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")

    runner = AlertsRunner(strategies=strategies, cfg=cfg, features_cfg=feat_cfg, sinks=sinks)
    with contextlib.suppress(KeyboardInterrupt):
        runner.run()


if __name__ == "__main__":
    main()
