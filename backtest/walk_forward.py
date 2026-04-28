"""Walk-forward splits with purge + embargo.

Each split is a ``(train_idx, test_idx)`` pair of integer indices into the
input frame. The arithmetic guarantees:

    train_end + purge + embargo  <=  test_start

so any label whose horizon would peek into the test window is excluded
(the *purge*), and a buffer of ``embargo`` bars sits between train and
test to break short-range autocorrelation leaks.

For pure backtest (no ML), purge/embargo can stay at 0; they exist now
so that the same harness works once we plug in ML strategies.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from backtest.config import WalkForwardConfig


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One train/test split."""

    train_start: int  # inclusive
    train_end: int  # exclusive
    test_start: int  # inclusive
    test_end: int  # exclusive

    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)


def walk_forward_splits(n: int, cfg: WalkForwardConfig) -> Iterator[WalkForwardSplit]:
    """Yield train/test splits over a series of length ``n``.

    Layout (no purge/embargo)::

        [---- train ----][---- test ----]                  step 1
                          |---- step ---|
                         [---- train ----][---- test ----] step 2

    With purge ``p`` and embargo ``e``::

        [---- train (last p purged) ----][embargo e][---- test ----]
    """
    train = cfg.train_size
    test = cfg.test_size
    step = cfg.step_size
    purge = cfg.purge
    embargo = cfg.embargo

    test_start = train + purge + embargo
    while test_start + test <= n:
        yield WalkForwardSplit(
            train_start=test_start - train - purge - embargo,
            train_end=test_start - purge - embargo,
            test_start=test_start,
            test_end=test_start + test,
        )
        test_start += step
