"""Weighted moving average filter for IK output.

Ported from unitreerobotics/xr_teleoperate ``utils/weighted_moving_filter.py``.
Default weights ``[0.4, 0.3, 0.2, 0.1]`` (newest first) are the upstream defaults.

Differences from upstream:
  * no interactive ``matplotlib`` plot helper (a debug aid, not part of the filter);
  * ``add_data`` accepts any ``data_size`` instead of hard-coding 14;
  * a repeated sample is still pushed into the history, so the filter cannot latch
    onto a stale pose when the solver returns the same value twice.
"""

from typing import Optional, Sequence

import numpy as np

DEFAULT_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


class WeightedMovingFilter:
    """Trailing weighted average over the last ``len(weights)`` samples of a length-``data_size`` vector."""

    def __init__(self, weights: Sequence[float] = DEFAULT_WEIGHTS, data_size: int = 14):
        if data_size < 1:
            raise ValueError("data_size must be >= 1")
        self.data_size = data_size
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()
        self._data: list[np.ndarray] = []
        self._filtered_data: Optional[np.ndarray] = None

    def add_data(self, new_data: np.ndarray) -> None:
        data = np.asarray(new_data, dtype=float).ravel()
        if data.shape[0] != self.data_size:
            raise ValueError(f"expected {self.data_size} values, got {data.shape[0]}")
        self._data.insert(0, data)
        if len(self._data) > len(self.weights):
            self._data = self._data[: len(self.weights)]
        self._filtered_data = np.average(np.array(self._data), axis=0, weights=self.weights[: len(self._data)])

    def get_filtered_data(self) -> Optional[np.ndarray]:
        return self._filtered_data
