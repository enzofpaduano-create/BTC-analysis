"""Tests for walk-forward splits."""

from __future__ import annotations

from backtest.config import WalkForwardConfig
from backtest.walk_forward import walk_forward_splits


def test_no_purge_no_embargo_basic_layout() -> None:
    cfg = WalkForwardConfig(train_size=100, test_size=20, step_size=20, purge=0, embargo=0)
    splits = list(walk_forward_splits(n=200, cfg=cfg))
    # First test starts at index 100, last must end ≤ 200.
    assert splits[0].train_start == 0
    assert splits[0].train_end == 100
    assert splits[0].test_start == 100
    assert splits[0].test_end == 120
    # Step 2:
    assert splits[1].train_start == 20
    assert splits[1].train_end == 120
    assert splits[1].test_start == 120


def test_train_test_no_overlap_with_embargo() -> None:
    cfg = WalkForwardConfig(train_size=50, test_size=20, step_size=10, purge=5, embargo=5)
    splits = list(walk_forward_splits(n=200, cfg=cfg))
    for s in splits:
        # train_end + purge + embargo ≤ test_start  (no leak, with buffer)
        assert s.train_end + cfg.purge + cfg.embargo <= s.test_start
        # No overlap
        assert s.train_end <= s.test_start
        # Sizes consistent
        assert s.train_end - s.train_start == cfg.train_size
        assert s.test_end - s.test_start == cfg.test_size


def test_no_splits_when_n_too_small() -> None:
    cfg = WalkForwardConfig(train_size=100, test_size=50, step_size=50)
    assert list(walk_forward_splits(n=120, cfg=cfg)) == []
