"""Fetch Melbourne weather from Open-Meteo (free, no key, CC-BY-4.0) — CLAUDE.md §3.5.

Three layers, all requested in fixed ``GMT+10`` (never a DST zone) and stored hourly:

``*_obs``  "what happened": ERA5 reanalysis before ``historical_forecast_start`` and the Historical
           Forecast API (stitched short-lead forecasts, ~hindcast quality) after it.
``*_da``   "what was forecast the day before": the Previous Runs API ``<var>_previous_day1`` series.
           This is the leakage-safe input for a day-ahead model. Measured availability for Melbourne
           (errata §16): temperature from Jan 2022, every other variable from Feb 2024.
live       the current 16-day forecast, used by ``predict.py`` for tomorrow's exogenous inputs.

Outputs (committed): ``data/raw/weather_hourly.parquet`` and ``data/raw/weather_forecast_latest.parquet``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from gridcast import config
from gridcast.data.http import get
from gridcast.utils.aest import OPEN_METEO_TZ, now_aest
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FCST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OUT = config.DATA_RAW / "weather_hourly.parquet"
LIVE_OUT = config.DATA_RAW / "weather_forecast_latest.parquet"


def _request(url: str, variables: list[str], start: date | None = None, end: date | None = None,
             extra: dict | None = None) -> pd.DataFrame:
    s = config.get_settings()
    params = {
        "latitude": s.location.latitude,
        "longitude": s.location.longitude,
        "hourly": ",".join(variables),
        "timezone": OPEN_METEO_TZ,
    }
    if start is not None:
        params |= {"start_date": start.isoformat(), "end_date": end.isoformat()}
    params |= extra or {}
    r = get(url, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    if payload.get("utc_offset_seconds") != 36000:
        raise ValueError(f"Open-Meteo returned offset {payload.get('utc_offset_seconds')}, expected UTC+10")
    df = pd.DataFrame(payload["hourly"])
    df["ds"] = pd.to_datetime(df.pop("time"))
    return df


def _yearly(start: date, end: date) -> list[tuple[date, date]]:
    spans, cur = [], start
    while cur <= end:
        stop = min(date(cur.year, 12, 31), end)
        spans.append((cur, stop))
        cur = date(cur.year + 1, 1, 1)
    return spans


def fetch_observed(start: date, end: date) -> pd.DataFrame:
    s = config.get_settings()
    variables = list(s.weather.variables)
    split = s.weather.historical_forecast_start
    frames = []
    for a, b in _yearly(start, end):
        if b < split:
            url, src = ARCHIVE_URL, "era5"
        elif a >= split:
            url, src = HIST_FCST_URL, "historical_forecast"
        else:  # spans the split: request each side separately
            frames.append(fetch_observed(a, split - pd.Timedelta(days=1)))
            a, url, src = split, HIST_FCST_URL, "historical_forecast"
        df = _request(url, variables, a, b)
        df = df.rename(columns={v: f"{v}_obs" for v in variables}).assign(obs_source=src)
        frames.append(df)
        logger.info("observed %s→%s (%s): %d hours", a, b, src, len(df))
    return pd.concat(frames, ignore_index=True)


def fetch_dayahead(start: date, end: date) -> pd.DataFrame:
    variables = list(config.get_settings().weather.variables)
    frames = []
    for a, b in _yearly(start, end):
        df = _request(PREV_RUNS_URL, [f"{v}_previous_day1" for v in variables], a, b)
        frames.append(df.rename(columns={f"{v}_previous_day1": f"{v}_da" for v in variables}))
        logger.info("day-ahead %s→%s: %d hours", a, b, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def fetch_live_forecast() -> pd.DataFrame:
    s = config.get_settings()
    df = _request(FORECAST_URL, list(s.weather.variables), extra={"forecast_days": s.weather.forecast_days, "past_days": 2})
    df["issued_at"] = now_aest().floor("h")
    return df


def run(*, full_refresh: bool = False) -> pd.DataFrame:
    s = config.get_settings()
    today = now_aest().date()
    end = today - pd.Timedelta(days=1)  # archives lag; yesterday is the last complete day
    start = s.data.history_start - pd.Timedelta(days=14)  # warm-up for lagged temperature features
    existing = None
    if OUT.exists() and not full_refresh:
        existing = pd.read_parquet(OUT)
        start = (existing["ds"].max() - pd.Timedelta(days=10)).date()  # re-pull recent (revisions)

    obs = fetch_observed(start, end)
    da = fetch_dayahead(max(start, s.weather.historical_forecast_start), end)
    df = obs.merge(da, on="ds", how="left")
    if existing is not None:
        df = pd.concat([existing[existing["ds"] < df["ds"].min()], df], ignore_index=True)
    df = df.drop_duplicates("ds", keep="last").sort_values("ds").reset_index(drop=True)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    logger.info("Wrote %s: %d hours, %s → %s", OUT.name, len(df), df["ds"].min(), df["ds"].max())

    live = fetch_live_forecast()
    live.to_parquet(LIVE_OUT, index=False)
    logger.info("Wrote %s: %d hours (issued %s)", LIVE_OUT.name, len(live), live["issued_at"].iloc[0])
    return df


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full-refresh", action="store_true")
    run(full_refresh=p.parse_args().full_refresh)


if __name__ == "__main__":
    main()
