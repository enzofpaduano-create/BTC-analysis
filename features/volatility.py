"""Volatility features: realized vol on multiple windows + GARCH(1,1) walk-forward.

Causality strategy for GARCH:
    - Refit ``arch_model`` every ``refit_every`` bars on the running prefix.
    - Between refits, propagate the GARCH(1,1) recursion analytically using
      the latest fitted (ω, α, β):

          σ²[i+1] = ω + α · r[i]² + β · σ²[i]

      so we never need future data to update σ². The 1-step-ahead vol at
      bar ``i`` is the prediction for bar ``i+1`` based only on returns ≤ i.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger

from features.config import VolatilityConfig


def compute_volatility(
    df: pd.DataFrame, cfg: VolatilityConfig, *, bar_minutes: int
) -> pd.DataFrame:
    """Append vol columns to ``df``.

    Args:
        df: OHLCV with a ``close`` column.
        cfg: Volatility config.
        bar_minutes: Duration of one bar — used to map "X minutes" windows
            to a bar count (e.g. window=60 min on a 5-min bar → 12 bars).
    """
    out = df.copy()
    log_ret = np.log(out["close"]).diff()
    bars_per_year = (365 * 24 * 60) / bar_minutes

    # -- Realized vol on multiple windows ---------------------------------
    for window_min in cfg.realized_windows_min:
        bars = max(2, window_min // bar_minutes)
        col = f"vol_{window_min}m"
        out[col] = log_ret.rolling(bars).std(ddof=0) * np.sqrt(bars_per_year)

    # -- Vol ratio short / long -------------------------------------------
    short_bars = max(2, cfg.short_window_min // bar_minutes)
    long_bars = max(2, cfg.long_window_min // bar_minutes)
    short_vol = log_ret.rolling(short_bars).std(ddof=0)
    long_vol = log_ret.rolling(long_bars).std(ddof=0)
    out["vol_ratio_short_long"] = short_vol / long_vol.replace(0, np.nan)

    # -- GARCH(1,1) walk-forward forecast --------------------------------
    out["garch_vol_1step"] = _garch_walk_forward(
        log_ret,
        refit_every=cfg.garch_refit_every,
        min_obs=cfg.garch_min_obs,
        bars_per_year=bars_per_year,
    )
    return out


def _garch_walk_forward(
    log_ret: pd.Series,
    *,
    refit_every: int,
    min_obs: int,
    bars_per_year: float,
) -> pd.Series:
    """Causal 1-step-ahead GARCH(1,1) vol forecast (annualized).

    Returns a Series aligned with ``log_ret``. NaN before the first fit.
    Each value at bar ``i`` is σ̂[i+1] computed from data ≤ i.
    """
    n = len(log_ret)
    out = np.full(n, np.nan)
    if n < min_obs + 1:
        return pd.Series(out, index=log_ret.index)

    # Drop the leading NaN return (from the diff).
    r = log_ret.to_numpy()
    finite = np.isfinite(r)

    omega = alpha = beta = sigma2 = None  # current GARCH params + state
    last_fit_at = -1

    # Scale returns by 100 to keep arch's optimizer well-conditioned.
    for i in range(n):
        if not finite[i]:
            continue
        # Refit when due — uses returns up to AND INCLUDING bar i.
        if i >= min_obs and (last_fit_at < 0 or (i - last_fit_at) >= refit_every):
            sub = r[: i + 1]
            sub = sub[np.isfinite(sub)] * 100.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = arch_model(sub, vol="GARCH", p=1, q=1, mean="Zero", dist="normal").fit(
                        disp="off", show_warning=False
                    )
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                # Initialise sigma² at the latest in-sample conditional vol.
                sigma2 = float(res.conditional_volatility[-1] ** 2)
                last_fit_at = i
            except Exception as exc:  # noqa: BLE001
                logger.warning("GARCH fit failed at bar {}: {}", i, exc)
                continue

        if omega is None or alpha is None or beta is None or sigma2 is None:
            continue

        # Apply GARCH(1,1) recursion: σ²[i+1] = ω + α r[i]² + β σ²[i]
        r_i = r[i] * 100.0
        next_sigma2 = omega + alpha * r_i * r_i + beta * sigma2
        # Forecast vol per-bar, annualize.
        out[i] = (np.sqrt(next_sigma2) / 100.0) * np.sqrt(bars_per_year)
        sigma2 = next_sigma2

    return pd.Series(out, index=log_ret.index)
