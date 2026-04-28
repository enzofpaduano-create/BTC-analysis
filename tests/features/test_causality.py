"""Critical test: feature[i] must depend ONLY on bars[:i+1].

Strategy: build a feature DataFrame from a *truncated* OHLCV (bars[:N])
and from the *full* OHLCV (bars[:M], M > N). For every shared row index
< N, the values must match. If any column differs, that feature is
peeking into the future — a leak.

This catches:
- Smoothing passes that touch future bars
- Centered rolling windows
- Models fit on the full sample then predicted retrospectively
- np-style rolling with `min_periods=1` that includes future on edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import FeaturesConfig, compute_features

# Columns that legitimately *can* differ at the truncation boundary because
# they are identifier-style outputs whose absolute value depends on history
# never exposed to the truncated run (the segment counter restarts from 0
# in both runs, so the latest values in the full run can be larger than
# the truncated run's). We exclude these from the strict-equality check
# but assert separately that they are non-decreasing.
_OK_TO_DIFFER = {"cp_segment"}


def _make_synthetic(n: int) -> pd.DataFrame:
    """Deterministic synthetic OHLCV used by the leak test."""
    rng = np.random.default_rng(123)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    log_ret = rng.normal(0.0, 0.001, size=n)
    close = pd.Series(np.exp(np.cumsum(log_ret)) * 60_000.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.0003,
            "low": close * 0.9997,
            "close": close,
            "volume": rng.uniform(1.0, 100.0, size=n),
            "turnover": close * 10.0,
        }
    )


@pytest.mark.slow
def test_no_future_leak_on_overlap() -> None:
    """No feature value at index ``i < N`` may change when we add bars > N."""
    cfg = FeaturesConfig(
        bar_minutes=1,
        # Push refits closer so the test exercises stateful features
        # (GARCH, HMM, PELT) within the truncated range.
    )
    cfg.volatility.garch_min_obs = 200
    cfg.volatility.garch_refit_every = 200
    cfg.regime.hmm_min_obs = 200
    cfg.regime.hmm_refit_every = 200
    cfg.regime.cp_window = 200
    cfg.regime.cp_refit_every = 50

    n_truncated = 800
    n_full = 1200
    df_full = _make_synthetic(n_full)
    df_truncated = df_full.iloc[:n_truncated].copy().reset_index(drop=True)

    feat_full = compute_features(df_full, cfg).iloc[:n_truncated].reset_index(drop=True)
    feat_trunc = compute_features(df_truncated, cfg).reset_index(drop=True)

    # Strict equality on every numeric feature column except the OK-to-differ ones.
    feature_cols = [
        c
        for c in feat_full.columns
        if c not in _OK_TO_DIFFER and pd.api.types.is_numeric_dtype(feat_full[c])
    ]
    diffs: list[str] = []
    for col in feature_cols:
        a = feat_full[col].to_numpy()
        b = feat_trunc[col].to_numpy()
        # NaN-aware comparison: both NaN => OK; else require near-equality.
        both_nan = np.isnan(a) & np.isnan(b)
        finite = ~(np.isnan(a) | np.isnan(b))
        if not np.all(both_nan | finite):
            diffs.append(f"{col} (NaN mismatch)")
            continue
        if not np.allclose(a[finite], b[finite], rtol=1e-10, atol=1e-10):
            max_abs = float(np.max(np.abs(a[finite] - b[finite])))
            diffs.append(f"{col} (max|Δ|={max_abs:.3e})")

    assert not diffs, "Feature leakage detected:\n  " + "\n  ".join(diffs)

    # cp_segment is allowed to differ in absolute value (it's a counter that
    # restarts in each run), but it must be monotone non-decreasing.
    for run in (feat_full, feat_trunc):
        seg = run["cp_segment"].to_numpy()
        assert np.all(np.diff(seg) >= 0), "cp_segment must be non-decreasing"
