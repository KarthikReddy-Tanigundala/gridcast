"""Calendar features. Derived from the interval START (``ds - 30min``) so the interval ending
00:00 belongs to the day it actually covers (23:30–24:00), not to the next day."""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from gridcast import config

SCHOOL_TERMS = config.DATA_REFERENCE / "vic_school_terms.csv"
TREND_ORIGIN = pd.Timestamp("2019-01-01")


@lru_cache(maxsize=1)
def vic_holidays(start_year: int = 2018, end_year: int = 2028) -> frozenset:
    return frozenset(holidays.Australia(subdiv="VIC", years=range(start_year, end_year + 1)).keys())


@lru_cache(maxsize=1)
def _terms() -> pd.DataFrame:
    return pd.read_csv(SCHOOL_TERMS, parse_dates=["term_start", "term_end"])


def is_school_term(days: pd.Series) -> pd.Series:
    """True on Victorian government school term days (weekends inside a term included)."""
    terms = _terms()
    out = pd.Series(False, index=days.index)
    for start, end in zip(terms["term_start"], terms["term_end"], strict=True):
        out |= days.between(start, end)
    return out


def add_calendar_features(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    start = df["ds"] - pd.Timedelta(minutes=minutes)
    day = start.dt.normalize()
    out = df.copy()
    out["half_hour"] = (start.dt.hour * 2 + start.dt.minute // 30).astype("int8")
    out["hour"] = start.dt.hour.astype("int8")
    out["day_of_week"] = start.dt.dayofweek.astype("int8")
    out["month"] = start.dt.month.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    hol = vic_holidays()
    out["is_public_holiday"] = day.dt.date.isin(hol).astype("int8")
    out["is_non_working_day"] = ((out["is_weekend"] == 1) | (out["is_public_holiday"] == 1)).astype(
        "int8"
    )
    out["is_school_term"] = is_school_term(day).astype("int8")
    doy = start.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    out["trend_years"] = ((start - TREND_ORIGIN).dt.days / 365.25).astype("float32")
    return out


CALENDAR_FEATURES = [
    "half_hour",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_non_working_day",
    "is_school_term",
    "doy_sin",
    "doy_cos",
]
