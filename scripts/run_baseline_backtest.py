"""End-to-end run of the baseline mean-reversion strategy.

Pipeline:
    1. Download / load 2 years of BTCUSDT M5 from Bybit mainnet.
    2. Compute the full features pipeline (technical + vol + regime + micro).
    3. Run the backtest with realistic costs (Bybit-perp typical).
    4. Save an HTML report (equity + drawdown + metrics).

Run::

    uv run python -m scripts.run_baseline_backtest

You can shorten the test by setting ``DAYS_BACK`` lower than 730 — useful
for first runs while validating the pipeline.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from backtest import BacktestConfig, run_backtest, save_html_report
from backtest.config import CostsConfig
from core.logging import setup_logging
from data import BybitClient, ParquetStore, download_history
from features import FeaturesConfig, compute_features
from signals import MeanReversionBollingerHMM

DAYS_BACK = int(os.environ.get("DAYS_BACK", "730"))  # 2 years by default
MIN_BARS_FOR_BACKTEST = 2000
SYMBOL = "BTCUSDT"
TIMEFRAME = "5"
BAR_MINUTES = 5

OUT_DIR = Path("./reports")
PARQUET_ROOT = Path("./data_store/parquet")


def main() -> None:
    setup_logging()

    end = datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS_BACK)
    logger.info("Baseline run — {} {} from {} to {}", SYMBOL, TIMEFRAME, start, end)

    # 1. Download (resumable: subsequent runs only fetch new bars).
    store = ParquetStore(PARQUET_ROOT)
    with BybitClient(testnet=False) as client:
        download_history(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            store=store,
            client=client,
            quality_checks=False,  # we'll run them indirectly via the features stage
        )
    df = store.read(SYMBOL, TIMEFRAME, start=start, end=end)
    logger.info("Loaded {} bars from disk", len(df))
    if len(df) < MIN_BARS_FOR_BACKTEST:
        logger.error("Not enough bars to run a meaningful backtest")
        return

    # 2. Compute features.
    feat_cfg = FeaturesConfig(bar_minutes=BAR_MINUTES)
    # On M5, "1 bar" = 5 minutes, so widen warmups proportionally to keep the
    # same calendar windows.
    feat_cfg.volatility.garch_min_obs = 500
    feat_cfg.volatility.garch_refit_every = 1000
    feat_cfg.regime.hmm_min_obs = 500
    feat_cfg.regime.hmm_refit_every = 1000
    feat_cfg.regime.cp_window = 1000
    feat_cfg.regime.cp_refit_every = 200

    logger.info("Computing features…")
    feat = compute_features(df, feat_cfg)

    # 3. Run the backtest.
    bt_cfg = BacktestConfig(
        initial_capital=10_000.0,
        bar_minutes=BAR_MINUTES,
        costs=CostsConfig(
            spread_bps=2.0,
            taker_fee_bps=5.5,
            slippage_bps_fixed=1.0,
            slippage_prop_coeff_bps=5.0,
            funding_annual_bps=10.0,
        ),
    )
    strat = MeanReversionBollingerHMM(
        rsi_long_threshold=30.0,
        rsi_short_threshold=70.0,
        atr_stop_mult=2.0,
        timeout_bars=48,  # 4 hours on M5
        target_vol_per_trade=0.01,
        max_size=1.0,
    )

    logger.info("Running backtest…")
    res = run_backtest(ohlcv=df, features=feat, strategy=strat, cfg=bt_cfg)
    logger.success(res.summary())

    # 4. Report.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"baseline_{SYMBOL}_M{TIMEFRAME}.html"
    save_html_report(
        res, path=out_path, title=f"{strat.name} — {SYMBOL} M{TIMEFRAME} ({DAYS_BACK} days)"
    )
    logger.success("Report written to {}", out_path)


if __name__ == "__main__":
    main()
