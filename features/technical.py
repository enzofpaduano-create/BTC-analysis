"""Technical indicators (RSI / MACD / BB / ATR / VWAP / EMAs).

All implemented with pandas rolling/EWM ops or pandas_ta where it adds value.
Every output is a strictly causal function of past OHLCV — no `shift(-k)`
or smoothing that touches future bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from features.config import TechnicalConfig


def compute_technical(df: pd.DataFrame, cfg: TechnicalConfig) -> pd.DataFrame:
    """Append technical-indicator columns to ``df`` (returns a new DataFrame).

    Args:
        df: OHLCV with at least ``high``, ``low``, ``close``, ``volume``,
            ``timestamp``. The frame is left untouched.
        cfg: Indicator parameters.

    Returns:
        New DataFrame with the same rows plus indicator columns. NaNs in
        the warmup region are expected and *not* dropped.
    """
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    # -- RSI ---------------------------------------------------------------
    for length in cfg.rsi_lengths:
        out[f"rsi_{length}"] = ta.rsi(close, length=length)

    # -- MACD --------------------------------------------------------------
    macd = ta.macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    if macd is not None and not macd.empty:
        macd_col = f"MACD_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
        sig_col = f"MACDs_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
        hist_col = f"MACDh_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
        out["macd"] = macd[macd_col]
        out["macd_signal"] = macd[sig_col]
        out["macd_hist"] = macd[hist_col]

    # -- Bollinger Bands ---------------------------------------------------
    mid = close.rolling(cfg.bb_length).mean()
    std = close.rolling(cfg.bb_length).std(ddof=0)
    upper = mid + cfg.bb_std * std
    lower = mid - cfg.bb_std * std
    out["bb_lower"] = lower
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    # %B = (close - lower) / (upper - lower)
    band_width = upper - lower
    out["bb_pct"] = (close - lower) / band_width.replace(0, np.nan)
    # Bandwidth = (upper - lower) / mid
    out["bb_bw"] = band_width / mid.replace(0, np.nan)

    # -- ATR ---------------------------------------------------------------
    atr = ta.atr(high, low, close, length=cfg.atr_length)
    out["atr"] = atr
    out["atr_pct"] = atr / close.replace(0, np.nan)

    # -- VWAP intraday (resets each UTC day) ------------------------------
    out["vwap"] = _intraday_vwap(out)

    # -- EMAs + slopes -----------------------------------------------------
    for length in cfg.ema_lengths:
        col = f"ema_{length}"
        out[col] = close.ewm(span=length, adjust=False, min_periods=length).mean()
        # Causal slope: pct change over the last `slope_window` bars.
        out[f"{col}_slope"] = out[col].pct_change(periods=cfg.slope_window)

    return out


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative typical-price-volume / cumulative volume, resetting each UTC day.

    Causal: at bar i it only uses [day_open, i].
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day_key = df["timestamp"].dt.tz_convert("UTC").dt.date
    cum_pv = pv.groupby(day_key).cumsum()
    cum_v = df["volume"].groupby(day_key).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)
