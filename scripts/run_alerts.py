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

from core.logging import setup_logging
from features import FeaturesConfig
from live.alerts import AlertConfig, AlertsRunner
from signals import MeanReversionBollingerHMM, TrendBreakoutADXHMM

PARQUET_ROOT = Path("./data_store/parquet")
ALERTS_LOG = Path("./reports/alerts.jsonl")


def main() -> None:
    setup_logging()

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
        (MeanReversionBollingerHMM(target_vol_per_trade=0.01), 1.0),
        (TrendBreakoutADXHMM(target_vol_per_trade=0.01), 1.0),
    ]

    runner = AlertsRunner(strategies=strategies, cfg=cfg, features_cfg=feat_cfg)
    with contextlib.suppress(KeyboardInterrupt):
        runner.run()


if __name__ == "__main__":
    main()
