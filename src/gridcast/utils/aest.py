"""AEMO time handling — the #1 silent-error source in NEM data (CLAUDE.md §1.2, §6.1).

Rules encoded here:

* AEMO market time is **AEST = UTC+10, fixed all year**. The NEM does not observe daylight saving.
  ``Australia/Melbourne`` / ``Australia/Sydney`` are therefore WRONG: from October to April they
  are UTC+11 and every summer interval would silently shift by an hour against weather data.
* ``SETTLEMENTDATE`` (and every AEMO interval timestamp) labels the **end** of the interval.
  The 30-minute interval labelled ``2024-01-15 00:30`` covers 00:00–00:30.
* Inside GridCast every timestamp column is **tz-naive and means AEST, interval-ending**. Naive is
  deliberate: Nixtla libraries and parquet round-trips handle naive datetimes most predictably, and
  a single convention avoids mixed-tz joins. Conversion happens only at the edges (weather APIs,
  ``now()``), through the helpers below.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

#: Fixed UTC+10 — never a DST-aware zone.
AEST = timezone(timedelta(hours=10), name="AEST")
#: Open-Meteo accepts a fixed-offset timezone string; this one never applies DST.
OPEN_METEO_TZ = "GMT+10"


def parse_aemo_timestamp(values: pd.Series | str) -> pd.Series | pd.Timestamp:
    """Parse AEMO ``SETTLEMENTDATE``-style strings (``2024/01/15 00:30:00``) as naive AEST.

    AEMO files carry no offset; the value already *is* market time, so no conversion is applied.
    """
    parsed = pd.to_datetime(values, format="mixed", dayfirst=False)
    if isinstance(parsed, pd.Series):
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(AEST).dt.tz_localize(None)
        return parsed.astype("datetime64[ns]")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(AEST).tz_localize(None)
    return parsed


def utc_to_aest(values: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """Convert UTC timestamps (naive-as-UTC or tz-aware) to naive AEST."""
    if isinstance(values, pd.Series):
        s = values if values.dt.tz is not None else values.dt.tz_localize("UTC")
        return s.dt.tz_convert(AEST).dt.tz_localize(None)
    idx = values if values.tz is not None else values.tz_localize("UTC")
    return idx.tz_convert(AEST).tz_localize(None)


def aest_to_utc(values: pd.Series) -> pd.Series:
    """Convert naive AEST timestamps to tz-aware UTC."""
    return values.dt.tz_localize(AEST).dt.tz_convert("UTC")


def now_aest() -> pd.Timestamp:
    """Current market time as a naive AEST timestamp."""
    return pd.Timestamp(datetime.now(tz=AEST)).tz_localize(None)


def interval_start(interval_end: pd.Series, minutes: int) -> pd.Series:
    """Interval-ending label → start of that interval."""
    return interval_end - pd.Timedelta(minutes=minutes)


def hourly_start_to_interval_end(
    hourly: pd.DataFrame, ts_col: str = "ds", minutes: int = 30
) -> pd.DataFrame:
    """Align hourly *instantaneous / hour-starting* weather onto AEMO interval-ending timestamps.

    Open-Meteo returns ``time`` as the start of the hour (instantaneous values for temperature,
    preceding-hour means for radiation). We upsample to ``minutes`` by time interpolation and label
    each value at the END of its interval, so a temperature measured at 13:00 lands on the
    13:00-ending interval (12:30–13:00) — i.e. it is never attached to an earlier interval.
    """
    frame = hourly.set_index(ts_col).sort_index()
    numeric = frame.select_dtypes("number")
    grid = pd.date_range(numeric.index.min(), numeric.index.max(), freq=f"{minutes}min")
    out = numeric.reindex(numeric.index.union(grid)).interpolate(method="time").reindex(grid)
    return out.rename_axis(ts_col).reset_index()


def five_min_to_interval(df: pd.DataFrame, value_cols: list[str], minutes: int = 30,
                         ts_col: str = "ds", how: str = "mean") -> pd.DataFrame:
    """Aggregate 5-minute interval-ending data to ``minutes``-interval-ending data.

    The 30-minute interval ending 00:30 contains the six 5-minute intervals ending 00:05…00:30, so
    the resample is ``closed='right', label='right'``. Under 5-minute settlement (from 1 Oct 2021)
    the 30-minute *trading* price is exactly the mean of those six dispatch prices.
    """
    rule = f"{minutes}min"
    agg = (
        df.set_index(ts_col)[value_cols]
        .resample(rule, closed="right", label="right")
        .agg(how)
    )
    counts = df.set_index(ts_col)[value_cols[0]].resample(rule, closed="right", label="right").count()
    agg["n_sub_intervals"] = counts
    return agg.reset_index()


def trading_day(ts: pd.Series) -> pd.Series:
    """NEM trading day (04:00–04:00 AEST, labelled by the start date) for interval-ending stamps."""
    return (ts - pd.Timedelta(minutes=1) - pd.Timedelta(hours=4)).dt.normalize()
