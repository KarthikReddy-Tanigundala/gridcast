"""Forecast metrics (§9.2). Pure numpy; inputs are array-likes of equal length.

* **MASE** (headline): MAE scaled by the in-sample MAE of a seasonal-naive forecast (season 48 =
  same half-hour yesterday) over a trailing training window. Scale-free; < 1 means the model beats
  what a naive "same as yesterday" rule achieves in-sample.
* MAE, RMSE in target units (MW or $/MWh).
* Pinball (quantile) loss and empirical interval **coverage** for probabilistic forecasts.
* sMAPE only for demand. **MAPE is deliberately absent**: prices cross zero and go negative
  (~23% of VIC1 intervals since 2023), which makes percentage error undefined or explosive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list


def _a(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    return float(np.mean(np.abs(_a(y) - _a(yhat))))


def rmse(y: ArrayLike, yhat: ArrayLike) -> float:
    return float(np.sqrt(np.mean((_a(y) - _a(yhat)) ** 2)))


def smape(y: ArrayLike, yhat: ArrayLike) -> float:
    """Symmetric MAPE in % — only meaningful for strictly positive series (demand)."""
    y, yhat = _a(y), _a(yhat)
    return float(100 * np.mean(2 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat))))


def seasonal_naive_scale(history: ArrayLike, season: int = 48) -> float:
    """In-sample MAE of the seasonal-naive forecast: mean |y_t - y_{t-season}|."""
    h = _a(history)
    if len(h) <= season:
        raise ValueError("history shorter than the season")
    return float(np.mean(np.abs(h[season:] - h[:-season])))


def mase(y: ArrayLike, yhat: ArrayLike, scale: float) -> float:
    if not scale > 0:
        raise ValueError("MASE scale must be positive")
    return mae(y, yhat) / scale


def pinball(y: ArrayLike, q: ArrayLike, tau: float) -> float:
    """Quantile loss for quantile level ``tau`` (0.5 => half the MAE)."""
    diff = _a(y) - _a(q)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    y = _a(y)
    return float(np.mean((y >= _a(lo)) & (y <= _a(hi))))


def interval_width(lo: ArrayLike, hi: ArrayLike) -> float:
    return float(np.mean(_a(hi) - _a(lo)))


def interval_pinball(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, level: int) -> float:
    """Mean pinball loss of the two quantiles that bound a central ``level``% interval."""
    alpha = (100 - level) / 200
    return 0.5 * (pinball(y, lo, alpha) + pinball(y, hi, 1 - alpha))
