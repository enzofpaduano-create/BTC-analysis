"""Data quality checks for OHLCV bars.

All checks are *non-destructive*: they return a `QualityReport` and emit
warnings via loguru. The caller decides whether to drop / fill / abort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from data.schemas import TIMEFRAME_MINUTES


@dataclass
class QualityReport:
    """Summary of issues found in a batch of bars."""

    n_rows: int
    outliers_idx: list[int] = field(default_factory=list)
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    zero_volume_idx: list[int] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.outliers_idx or self.gaps or self.zero_volume_idx)

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} outliers={len(self.outliers_idx)} "
            f"gaps={len(self.gaps)} zero_vol={len(self.zero_volume_idx)}"
        )


def detect_outliers(df: pd.DataFrame, *, window: int = 100, n_sigma: float = 8.0) -> list[int]:
    """Flag bars whose log-return is beyond ``n_sigma`` of the rolling std.

    Uses log-returns so the threshold scales naturally with price level.
    A high default (8 sigma) avoids flagging routine BTC volatility — bumps
    that triggered before were typically real moves, not bad ticks.
    """
    if len(df) < window + 1:
        return []
    log_ret = np.log(df["close"]).diff()
    mu = log_ret.rolling(window).mean()
    sigma = log_ret.rolling(window).std(ddof=0)
    z = (log_ret - mu) / sigma
    flagged = z.abs() > n_sigma
    return [int(i) for i in df.index[flagged.fillna(False)].tolist()]


def detect_gaps(df: pd.DataFrame, *, timeframe: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find timestamp gaps larger than the expected interval.

    Returns a list of ``(prev_ts, next_ts)`` pairs where consecutive bars
    are more than 1.5× the expected interval apart.
    """
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None or len(df) < 2:  # noqa: PLR2004 — need ≥2 rows for diff
        return []
    expected = pd.Timedelta(minutes=minutes)
    deltas = df["timestamp"].diff()
    gaps_mask = deltas > 1.5 * expected
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in np.where(gaps_mask.fillna(False))[0]:
        pairs.append(
            (
                pd.Timestamp(df["timestamp"].iloc[i - 1]),
                pd.Timestamp(df["timestamp"].iloc[i]),
            )
        )
    return pairs


def detect_zero_volume(df: pd.DataFrame) -> list[int]:
    """Flag bars with exactly zero volume (suspect for 24/7 BTC markets)."""
    flagged = df["volume"] == 0
    return [int(i) for i in df.index[flagged].tolist()]


def run_quality_checks(
    df: pd.DataFrame,
    *,
    timeframe: str,
    log_warnings: bool = True,
) -> QualityReport:
    """Run all checks and return a report. Logs warnings by default."""
    report = QualityReport(
        n_rows=len(df),
        outliers_idx=detect_outliers(df),
        gaps=detect_gaps(df, timeframe=timeframe),
        zero_volume_idx=detect_zero_volume(df),
    )
    if log_warnings:
        if report.outliers_idx:
            logger.warning(
                "{} return outliers detected (>8σ on log-returns) — first idx: {}",
                len(report.outliers_idx),
                report.outliers_idx[:5],
            )
        if report.gaps:
            logger.warning(
                "{} candle gaps detected on {} timeframe — first: {} → {}",
                len(report.gaps),
                timeframe,
                report.gaps[0][0],
                report.gaps[0][1],
            )
        if report.zero_volume_idx:
            logger.warning(
                "{} bars with zero volume — first idx: {}",
                len(report.zero_volume_idx),
                report.zero_volume_idx[:5],
            )
    return report
