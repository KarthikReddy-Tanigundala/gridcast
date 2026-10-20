"""Join actuals + weather into the 30-minute VIC1 modelling table (Nixtla long format, §6.5).

Weather columns are **forecast-as-known**: for each variable we use the day-ahead forecast
(``*_da``, issued the day before) when it exists and fall back to the observed series
(``*_obs``: historical forecast / ERA5) otherwise. ``weather_regime`` records which one applied,
so the backtest can prove the headline window uses day-ahead weather only.

Output (committed): ``data/processed/vic1_30min.parquet``
``unique_id, ds, demand_mw, rrp, settlement, temperature, humidity, cloud_cover, solar_radiation,
wind_speed, temperature_obs, weather_regime[, rooftop_pv_mw]``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.data import fetch_actuals, fetch_weather
from gridcast.utils.aest import hourly_start_to_interval_end
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

OUT = config.DATA_PROCESSED / "vic1_30min.parquet"

# Open-Meteo variable -> modelling column name
WEATHER_COLS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "solar_radiation",
    "wind_speed_10m": "wind_speed",
}


def blend_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly obs/day-ahead weather -> one forecast-as-known column per variable + regime flag."""
    out = pd.DataFrame({"ds": hourly["ds"]})
    da_flags = []
    for var, col in WEATHER_COLS.items():
        da, obs = hourly.get(f"{var}_da"), hourly[f"{var}_obs"]
        if da is None:
            da = pd.Series(np.nan, index=hourly.index)
        out[col] = da.where(da.notna(), obs)
        da_flags.append(da.notna())
    out["temperature_obs"] = hourly["temperature_2m_obs"]
    all_da = np.logical_and.reduce(da_flags)
    temp_da = da_flags[0]
    out["weather_regime"] = np.select(
        [all_da, temp_da],
        ["dayahead", "dayahead_temp_only"],
        default=hourly["obs_source"].to_numpy(),
    )
    return out


def weather_to_30min(hourly: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    blended = blend_weather(hourly)
    numeric = hourly_start_to_interval_end(blended.drop(columns="weather_regime"), minutes=minutes)
    # The regime of an interval is that of the hour it ends in (or the next hour for :30 stamps).
    regime = blended[["ds", "weather_regime"]].set_index("ds")["weather_regime"]
    numeric["weather_regime"] = regime.reindex(numeric["ds"].dt.ceil("h")).to_numpy()
    return numeric


def build(
    actuals: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    region: str,
    rooftop: pd.DataFrame | None = None,
    minutes: int = 30,
) -> pd.DataFrame:
    weather = weather_to_30min(weather_hourly, minutes)
    df = actuals.drop(columns=["n_sub_intervals"]).merge(weather, on="ds", how="left")
    if rooftop is not None:
        df = df.merge(rooftop, on="ds", how="left")
    df.insert(0, "unique_id", region)
    return df.sort_values("ds").reset_index(drop=True)


def validate(df: pd.DataFrame, minutes: int = 30) -> dict:
    expected = pd.date_range(df["ds"].min(), df["ds"].max(), freq=f"{minutes}min")
    gaps = len(expected.difference(df["ds"]))
    if gaps:
        raise ValueError(f"{gaps} missing intervals in the modelling table")
    if df["ds"].duplicated().any():
        raise ValueError("duplicate intervals")
    wx_missing = df["temperature"].isna()
    # Weather archives lag by a day; only the most recent intervals may be missing.
    if wx_missing.any() and df.loc[wx_missing, "ds"].min() < df["ds"].max() - pd.Timedelta(days=3):
        raise ValueError(
            f"weather missing inside the history: first gap {df.loc[wx_missing, 'ds'].min()}"
        )
    return {
        "rows": len(df),
        "start": str(df["ds"].min()),
        "end": str(df["ds"].max()),
        "regimes": df["weather_regime"].value_counts().to_dict(),
    }


def run() -> pd.DataFrame:
    s = config.get_settings()
    actuals = fetch_actuals.load()
    weather = fetch_weather.load()
    rooftop = fetch_actuals.load_rooftop()
    df = build(actuals, weather, s.region, rooftop, s.data.freq_minutes)
    summary = validate(df, s.data.freq_minutes)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    logger.info("Wrote %s: %s", OUT.name, summary)
    return df


if __name__ == "__main__":
    setup_logging()
    run()
