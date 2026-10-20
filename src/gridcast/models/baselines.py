"""Baselines (§8.1). They define the bar every model must clear.

Forecasts are produced for a block of ``horizon`` steps after an origin, given the history up to and
including the origin interval. With a 12:00 origin and 72 steps:

* ``naive``          — the last observed value, flat.
* ``seasonal_naive`` — same half-hour on the most recent day that is fully observed for that slot
                       (``y[t - 48k]`` with the smallest k that reaches into the history).
* ``seasonal_naive_week`` — same half-hour one week earlier (``y[t - 336]``), always observed.
"""

from __future__ import annotations

import numpy as np


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, history[-1], dtype="float64")


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 48) -> np.ndarray:
    n = len(history)
    steps = np.arange(1, horizon + 1)
    k = np.ceil(steps / season).astype(int)  # seasons to step back so the index is <= n-1
    idx = n - 1 + steps - k * season
    return history[idx].astype("float64")


def seasonal_naive_week(history: np.ndarray, horizon: int, season: int = 336) -> np.ndarray:
    return seasonal_naive(history, horizon, season)


BASELINES = {
    "Naive": naive,
    "Seasonal-naive (day)": seasonal_naive,
    "Seasonal-naive (week)": seasonal_naive_week,
}
