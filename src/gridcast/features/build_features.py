"""Assemble the exogenous feature matrix (calendar + weather) — leakage-safe by construction.

Target-derived features (lags, rolling stats) are NOT built here: ``mlforecast`` builds them inside
the model so they are recomputed recursively from forecasts when predicting 72 steps ahead.
Exogenous features only use the calendar and forecast-as-known weather, which are genuinely
available for tomorrow at forecast time. No scaler is fitted anywhere (tree models need none);
the classical models fit on each training window only.
"""

from __future__ import annotations

import pandas as pd

from gridcast import config
from gridcast.features.calendar_features import CALENDAR_FEATURES, add_calendar_features
from gridcast.features.weather_features import add_weather_features, weather_feature_names


def exog_columns() -> list[str]:
    s = config.get_settings()
    trend = ["trend_years"] if s.features.trend else []
    return CALENDAR_FEATURES + trend + weather_feature_names(s.features.temp_lags_hours)


def add_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` needs ``unique_id, ds`` and the forecast-as-known weather columns."""
    s = config.get_settings()
    out = add_calendar_features(df, s.data.freq_minutes)
    out = add_weather_features(
        out, s.features.hdd_cdd_base_c, s.features.temp_lags_hours, s.data.freq_minutes
    )
    return out


def model_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Nixtla frame ``unique_id, ds, y, <exog...>`` for one target column."""
    feats = add_exog_features(df)
    cols = ["unique_id", "ds", target, *exog_columns()]
    return feats[cols].rename(columns={target: "y"})
