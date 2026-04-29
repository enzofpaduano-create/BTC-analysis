"""Market-regime features: HMM (3 states) + change-point detection (PELT).

Both are stateful and refit periodically. Causality is preserved by:
    - Refitting on the running prefix (data ≤ current bar);
    - Predicting the latest state from observations ≤ current bar;
    - Never using a smoothing pass that touches future bars.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from features.config import RegimeConfig

NDArrayF = np.ndarray[Any, np.dtype[np.float64]]
NDArrayI = np.ndarray[Any, np.dtype[np.int64]]
NDArrayB = np.ndarray[Any, np.dtype[np.bool_]]


def compute_regime(df: pd.DataFrame, cfg: RegimeConfig, *, bar_minutes: int) -> pd.DataFrame:
    """Append regime columns to ``df``.

    Output columns:
        regime_hmm        — int label in [0, n_states-1] (or -1 if not fit)
        regime_hmm_proba  — posterior probability of the chosen state
        cp_segment        — int segment id from rolling-window PELT
    """
    out = df.copy()
    log_ret = np.log(out["close"]).diff()

    # Build the HMM observation matrix [returns, log(rolling vol)].
    short_bars = max(2, 15 // bar_minutes)
    rolling_vol = log_ret.rolling(short_bars).std(ddof=0)
    log_vol = np.log(rolling_vol.replace(0, np.nan))

    obs = np.column_stack([log_ret.to_numpy(), log_vol.to_numpy()])
    labels, probas = _hmm_walk_forward(
        obs,
        n_states=cfg.hmm_n_states,
        refit_every=cfg.hmm_refit_every,
        min_obs=cfg.hmm_min_obs,
    )
    out["regime_hmm"] = labels
    out["regime_hmm_proba"] = probas

    # PELT change-point on a rolling window over log-returns.
    out["cp_segment"] = _pelt_walk_forward(
        log_ret.to_numpy(),
        window=cfg.cp_window,
        refit_every=cfg.cp_refit_every,
        penalty=cfg.cp_penalty,
        min_size=cfg.cp_min_size,
    )
    return out


# -- HMM ------------------------------------------------------------------


def _hmm_walk_forward(
    obs: NDArrayF,
    *,
    n_states: int,
    refit_every: int,
    min_obs: int,
) -> tuple[NDArrayI, NDArrayF]:
    """Refit a Gaussian HMM periodically; predict the latest state at each bar.

    States are remapped after each fit so that label ``0`` is the lowest-mean-
    return regime (bear-like) and label ``n_states - 1`` is the highest
    (bull-like). Without this, the HMM's internal ordering is arbitrary and
    can flip across refits — strategies that read the labels would behave
    inconsistently. The mapping is recomputed at each fit.

    Returns (labels, probas), both length-N arrays.

        labels[i] = canonical label (0=bear, 1=range, 2=bull) at bar i
                    using model fit at last refit ≤ i
        probas[i] = posterior probability of the chosen label

    Causal: scoring uses only ``obs[:i+1]`` and the latest fit.
    """
    n = len(obs)
    labels = np.full(n, -1, dtype=np.int64)
    probas = np.full(n, np.nan)

    finite_mask = cast(NDArrayB, np.isfinite(obs).all(axis=1))
    first_finite = int(np.argmax(finite_mask)) if finite_mask.any() else n

    model: GaussianHMM | None = None
    state_order: NDArrayI | None = None  # order[k] = model state for canonical label k
    last_fit_at = -1

    for i in range(n):
        if i < first_finite:
            continue
        # Refit when due.
        if i >= min_obs and (last_fit_at < 0 or (i - last_fit_at) >= refit_every):
            sub = obs[first_finite : i + 1]
            sub = sub[np.isfinite(sub).all(axis=1)]
            if len(sub) < min_obs:
                continue
            try:
                # hmmlearn's ConvergenceMonitor prints to stdout regardless
                # of `verbose=False`; muffle it to keep our logs clean.
                with (
                    warnings.catch_warnings(),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    warnings.simplefilter("ignore")
                    new_model = GaussianHMM(
                        n_components=n_states,
                        covariance_type="diag",
                        n_iter=50,
                        random_state=0,
                    )
                    new_model.fit(sub)
                model = new_model
                # Sort states by mean log-return (column 0 of the obs matrix).
                state_order = np.argsort(model.means_[:, 0]).astype(np.int64)
                last_fit_at = i
            except Exception as exc:  # noqa: BLE001
                logger.warning("HMM fit failed at bar {}: {}", i, exc)
                continue

        if model is None or state_order is None or not finite_mask[i]:
            continue

        # Score on the prefix and read the LAST posterior — strictly causal.
        sub = obs[first_finite : i + 1]
        sub = sub[np.isfinite(sub).all(axis=1)]
        if len(sub) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                post = model.predict_proba(sub)
            # Reorder columns to canonical [P(bear), P(range), P(bull), …].
            canonical = post[-1, state_order]
            labels[i] = int(np.argmax(canonical))
            probas[i] = float(np.max(canonical))
        except Exception:  # noqa: BLE001
            continue
    return labels, probas


# -- PELT change-point ----------------------------------------------------


def _pelt_walk_forward(
    series: NDArrayF,
    *,
    window: int,
    refit_every: int,
    penalty: float,
    min_size: int,
) -> NDArrayI:
    """Run PELT on a rolling window; output the segment id at each bar.

    "Segment id" increments each time PELT detects a new break inside
    the window. Two consecutive bars without a fit share the latest id.
    Causal: only past values feed the algorithm.
    """
    n = len(series)
    seg = np.zeros(n, dtype=np.int64)
    seg_id = 0
    last_break_in_window: int | None = None
    last_fit_at = -1

    for i in range(n):
        seg[i] = seg_id
        if i < window:
            continue
        if (i - last_fit_at) < refit_every:
            continue
        sub = series[i - window + 1 : i + 1]
        if not np.isfinite(sub).all():
            continue
        try:
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(sub)
            breaks = algo.predict(pen=penalty)  # 1-indexed, last = len(sub)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PELT failed at bar {}: {}", i, exc)
            last_fit_at = i
            continue
        last_fit_at = i

        # Map the latest break (excluding the trailing endpoint) back to
        # an absolute index. If it differs from the previous one, increment.
        interior = [b for b in breaks if b < len(sub)]
        if not interior:
            continue
        latest_in_window = interior[-1]
        latest_abs = (i - window + 1) + latest_in_window
        if last_break_in_window is None or latest_abs != last_break_in_window:
            last_break_in_window = latest_abs
            seg_id += 1
            # Only the current bar onwards gets the new id — past values stay.
            seg[i] = seg_id
    return seg
