"""Strategy interface + canonical signal frame.

A strategy is a pure function of features → signals. It must be strictly
causal: signal at bar ``i`` may use only ``features.iloc[: i + 1]``.

The harness asserts this by running the strategy on a truncated frame
and checking that the past signals match.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

SIGNAL_COLUMNS: tuple[str, ...] = (
    "entry_long",
    "exit_long",
    "entry_short",
    "exit_short",
    "size",
)


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """Output of a Strategy.

    Attributes:
        df: DataFrame indexed like the input features with columns
            ``entry_long`` / ``exit_long`` / ``entry_short`` / ``exit_short``
            (booleans) and ``size`` (float, fraction of equity to allocate
            on entry; e.g. 1.0 = use all equity). When no entry triggers,
            ``size`` is ignored.
    """

    df: pd.DataFrame

    def validate(self) -> None:
        missing = set(SIGNAL_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"SignalFrame missing columns: {sorted(missing)}")
        for col in ("entry_long", "exit_long", "entry_short", "exit_short"):
            if self.df[col].dtype != bool:
                raise TypeError(f"`{col}` must be bool, got {self.df[col].dtype}")
        if not pd.api.types.is_numeric_dtype(self.df["size"]):
            raise TypeError("`size` must be numeric")


class Strategy(ABC):
    """Subclass and implement ``generate_signals``."""

    name: str = "anonymous"

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> SignalFrame:
        """Compute the signal frame for the given features.

        Must be strictly causal: the signal at row ``i`` may depend only
        on ``features.iloc[: i + 1]``. The harness verifies this.
        """
