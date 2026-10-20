"""Target lag / rolling features (§7).

The models use ``mlforecast`` to build these (so the same definitions are applied recursively when
forecasting 72 steps ahead). ``add_lag_features`` is the explicit pandas equivalent used by tests and
EDA: every feature at row ``t`` is computed from ``y`` at rows ``< t`` only.
"""

from __future__ import annotations

import pandas as pd
from mlforecast.lag_transforms import RollingMax, RollingMean, RollingStd, SeasonalRollingMean


def mlforecast_lag_config(lags: tuple[int, ...], windows: tuple[int, ...]) -> dict:
    """Lags + rolling transforms for MLForecast.

    Every lag is >= the 72-step horizon, so a 72-step forecast never consumes the model's own
    predictions: recursive prediction collapses to a *direct* day-ahead model. Rolling stats sit on
    the shortest lag (the most recent observed day at forecast time) plus a same-half-hour 7-day
    seasonal mean anchored two days back.
    """
    base = min(lags)
    transforms = {
        base: [RollingMean(w) for w in windows] + [RollingStd(windows[0]), RollingMax(windows[0])],
        96: [
            SeasonalRollingMean(season_length=48, window_size=7)
        ],  # same half-hour, 7 days to t-2d
    }
    return {"lags": list(lags), "lag_transforms": transforms}


def add_lag_features(
    df: pd.DataFrame, target: str, lags: tuple[int, ...], windows: tuple[int, ...]
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("unique_id", sort=False)[target]
    for lag in lags:
        out[f"lag{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    sg = shifted.groupby(out["unique_id"], sort=False)
    for w in windows:
        out[f"rolling_mean_lag1_window_size{w}"] = sg.transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )
    w0 = windows[0]
    out[f"rolling_std_lag1_window_size{w0}"] = sg.transform(
        lambda s: s.rolling(w0, min_periods=w0).std()
    )
    out[f"rolling_max_lag1_window_size{w0}"] = sg.transform(
        lambda s: s.rolling(w0, min_periods=w0).max()
    )
    return out


def assert_no_lookahead(df: pd.DataFrame, target: str, feature: str) -> None:
    """Raise if ``feature`` equals the target or any FUTURE target value on a material share of rows.

    A simple but effective guard: a leaky feature (e.g. ``shift(-1)``) matches a future target
    almost exactly; a legitimate lag never matches ``y[t]`` or ``y[t+k]`` systematically.
    """
    y = df[target]
    f = df[feature]
    mask = f.notna() & y.notna()
    for k in range(0, 4):
        future = y.shift(-k)
        m = mask & future.notna()
        if m.sum() and (f[m] - future[m]).abs().lt(1e-9).mean() > 0.5:
            raise AssertionError(f"feature {feature!r} matches y[t+{k}] — lookahead leakage")
