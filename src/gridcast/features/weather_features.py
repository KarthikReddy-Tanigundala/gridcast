"""Weather features (§7): HDD/CDD (base 18 °C), lagged temperature (thermal inertia), solar, temp×hour.

All inputs are the forecast-as-known columns built in ``build_dataset`` — for a target interval
tomorrow these are the day-before forecasts, so nothing here peeks at observed future weather.
Lags only look BACKWARDS (positive shifts) along the regular 30-minute grid.
"""

from __future__ import annotations

import pandas as pd


def add_weather_features(
    df: pd.DataFrame,
    base_c: float = 18.0,
    temp_lags_hours: tuple[int, ...] = (3, 24),
    minutes: int = 30,
) -> pd.DataFrame:
    out = df.copy()
    t = out["temperature"]
    out["hdd"] = (base_c - t).clip(lower=0).astype("float32")
    out["cdd"] = (t - base_c).clip(lower=0).astype("float32")
    steps_per_hour = 60 // minutes
    g = (
        out.groupby("unique_id", sort=False)["temperature"]
        if "unique_id" in out
        else out["temperature"]
    )
    for h in temp_lags_hours:
        out[f"temp_lag_{h}h"] = g.shift(h * steps_per_hour).astype("float32")
    # Daily max temperature so far (trailing 24h) — heat build-up drives evening cooling load.
    roll = (
        g.rolling(24 * steps_per_hour, min_periods=1).max().reset_index(level=0, drop=True)
        if "unique_id" in out
        else t.rolling(24 * steps_per_hour, min_periods=1).max()
    )
    out["temp_max_24h"] = roll.astype("float32")
    hour = out["hour"] if "hour" in out else (out["ds"] - pd.Timedelta(minutes=minutes)).dt.hour
    out["temp_x_hour"] = (t * hour).astype("float32")
    out["cdd_daytime"] = (out["cdd"] * hour.between(10, 20)).astype("float32")
    return out


def weather_feature_names(temp_lags_hours: tuple[int, ...]) -> list[str]:
    return [
        "temperature",
        "humidity",
        "cloud_cover",
        "solar_radiation",
        "wind_speed",
        "hdd",
        "cdd",
        *[f"temp_lag_{h}h" for h in temp_lags_hours],
        "temp_max_24h",
        "temp_x_hour",
        "cdd_daytime",
    ]
