"""LightGBM via ``mlforecast`` — the workhorse (§8.1 step 3).

Recursive multi-step: one LightGBM model predicts the next interval from target lags/rolling stats
(built by mlforecast) plus calendar and forecast-as-known weather; for 72 steps it feeds its own
predictions back as lags. Fitting uses only rows up to the refit origin (expanding window).

Prediction for many origins at once: each origin's trailing history becomes its own series in
``new_df`` (``unique_id = origin``) and its exogenous future in ``X_df`` — one call per refit block.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast

from gridcast import config
from gridcast.features.build_features import exog_columns
from gridcast.features.lag_features import mlforecast_lag_config

logger = logging.getLogger(__name__)

#: History rows each origin needs for its lag / rolling features.
HISTORY_ROWS = 48 * 16  # covers lag 672 (2 weeks) + rolling windows


def make_forecaster(
    objective: str = "regression", params: dict[str, Any] | None = None, alias: str = "LightGBM"
) -> MLForecast:
    s = config.get_settings()
    p = dict(s.lightgbm) | (params or {}) | {"objective": objective}
    model = lgb.LGBMRegressor(**p)
    return MLForecast(
        models={alias: model},
        freq=f"{s.data.freq_minutes}min",
        **mlforecast_lag_config(s.features.lags, s.features.rolling_windows),
    )


def fit(
    frame: pd.DataFrame,
    objective: str = "regression",
    extra_exog: list[str] | None = None,
    alias: str = "LightGBM",
) -> MLForecast:
    """``frame``: ``unique_id, ds, y`` + exog columns (rows with NaN exog are dropped)."""
    cols = ["unique_id", "ds", "y", *exog_columns(), *(extra_exog or [])]
    years = config.get_settings().lightgbm_train_years
    if years:  # recency cap: behaviour (rooftop PV, prices) drifts, so recent years matter most
        frame = frame[frame["ds"] > frame["ds"].max() - pd.Timedelta(days=round(365.25 * years))]
    train = frame[cols].dropna()
    fc = make_forecaster(objective, alias=alias)
    fc.fit(train, static_features=[])
    return fc


def predict_origins(
    fc: MLForecast, histories: pd.DataFrame, future_x: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """``histories``: ``unique_id, ds, y`` + exog (trailing rows per origin; mlforecast reads the
    exogenous column list from ``new_df``); ``future_x``: the same exog for the forecast steps."""
    cols = ["unique_id", "ds", "y", *[c for c in future_x.columns if c not in ("unique_id", "ds")]]
    return fc.predict(h=horizon, new_df=histories[cols], X_df=future_x)


def feature_importance(fc: MLForecast, alias: str = "LightGBM") -> pd.Series:
    model = fc.models_[alias]
    return pd.Series(
        model.booster_.feature_importance("gain"), index=model.booster_.feature_name()
    ).sort_values(ascending=False)
