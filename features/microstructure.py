"""Microstructure / momentum features.

- Z-scores of close vs EMAs
- Local-linear-trend Kalman filter on log(close)
- Multi-horizon log returns
- Rolling skewness and kurtosis

All causal: every value at bar i uses only bars[:i+1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.config import MicrostructureConfig

NDArrayF = np.ndarray[Any, np.dtype[np.float64]]


def compute_microstructure(df: pd.DataFrame, cfg: MicrostructureConfig) -> pd.DataFrame:
    """Append microstructure columns to ``df``."""
    out = df.copy()
    close = out["close"]
    log_close = np.log(close.replace(0, np.nan))

    # -- Z-scores vs EMAs --------------------------------------------------
    for ema_len in cfg.zscore_emas:
        ema = close.ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
        diff = close - ema
        sigma = diff.rolling(cfg.zscore_window).std(ddof=0)
        out[f"zscore_ema{ema_len}"] = diff / sigma.replace(0, np.nan)

    # -- Multi-horizon log returns ----------------------------------------
    for horizon_min in cfg.log_ret_horizons_min:
        horizon_bars = max(1, horizon_min)  # caller is expected to scale by bar_minutes
        out[f"log_ret_{horizon_min}m"] = log_close.diff(horizon_bars)

    # -- Rolling skew / kurt ----------------------------------------------
    log_ret_1 = log_close.diff()
    out[f"skew_{cfg.skew_kurt_window}"] = log_ret_1.rolling(cfg.skew_kurt_window).skew()
    out[f"kurt_{cfg.skew_kurt_window}"] = log_ret_1.rolling(cfg.skew_kurt_window).kurt()

    # -- Kalman local-linear-trend ---------------------------------------
    out["kalman_trend"] = _kalman_local_trend(
        log_close.to_numpy(),
        q_level=cfg.kalman_q_level,
        q_trend=cfg.kalman_q_trend,
        r_obs=cfg.kalman_r_obs,
    )
    return out


def _kalman_local_trend(obs: NDArrayF, *, q_level: float, q_trend: float, r_obs: float) -> NDArrayF:
    """Causal local-linear-trend filter without learning parameters online.

    State = [level, trend], transition::

        F = [[1, 1],
             [0, 1]]

    Observation::

        H = [1, 0]

    Returns the filtered ``trend`` component (slope per bar). Values before
    the first non-NaN observation are NaN.

    We implement the Kalman recursion by hand to guarantee strict causality
    (no smoothing pass — only the forward filter).
    """
    n = len(obs)
    out: NDArrayF = np.full(n, np.nan)
    if n == 0:
        return out

    # Process noise (Q) and observation noise (R) — fixed, no online learning.
    Q = np.array([[q_level, 0.0], [0.0, q_trend]])
    R = r_obs

    # State + covariance. Initialise as soon as we hit the first finite obs.
    x: NDArrayF = np.zeros(2)
    P = np.eye(2) * 1e3  # high uncertainty
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])

    started = False
    for i in range(n):
        z = obs[i]
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update if observation finite
        if np.isfinite(z):
            if not started:
                x = np.array([z, 0.0])
                P = np.eye(2) * 1.0
                started = True
            else:
                y = z - H @ x
                S = H @ P @ H.T + R
                K = (P @ H) / S
                x = x + K * y
                P = (np.eye(2) - np.outer(K, H)) @ P
        if started:
            out[i] = x[1]  # trend component
    return out
