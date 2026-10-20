"""Timezone correctness — the most important tests in the repo (CLAUDE.md §6.1)."""

import pandas as pd
import pytest

from gridcast.utils import aest


def test_aest_is_fixed_utc_plus_10_all_year():
    for month in (1, 4, 7, 10, 12):
        ts = pd.Timestamp(2024, month, 15, 12).to_pydatetime().replace(tzinfo=aest.AEST)
        assert ts.utcoffset() == pd.Timedelta(hours=10)


def test_summer_timestamp_does_not_shift():
    # 15 Jan is daylight-saving time in Melbourne (UTC+11). AEMO time is still UTC+10, so
    # 02:00 UTC must be 12:00 market time — a DST-aware zone would give 13:00.
    utc = pd.Series(pd.to_datetime(["2024-01-15 02:00"]))
    assert aest.utc_to_aest(utc).iloc[0] == pd.Timestamp("2024-01-15 12:00")
    wrong = utc.dt.tz_localize("UTC").dt.tz_convert("Australia/Melbourne").dt.tz_localize(None)
    assert wrong.iloc[0] == pd.Timestamp("2024-01-15 13:00"), "sanity: the trap is real"


def test_winter_timestamp_matches_local_time():
    utc = pd.Series(pd.to_datetime(["2024-07-15 02:00"]))
    assert aest.utc_to_aest(utc).iloc[0] == pd.Timestamp("2024-07-15 12:00")


def test_dst_transition_day_has_no_gap_or_duplicate():
    # Melbourne clocks jump 02:00 -> 03:00 on 6 Oct 2024. Market time must stay a clean 30-min grid.
    utc = pd.Series(pd.date_range("2024-10-05 12:00", "2024-10-06 12:00", freq="30min"))
    out = aest.utc_to_aest(utc)
    assert out.diff().dropna().eq(pd.Timedelta(minutes=30)).all()
    assert out.is_unique


def test_round_trip():
    s = pd.Series(pd.date_range("2024-01-01", periods=96, freq="30min"))
    assert aest.utc_to_aest(aest.aest_to_utc(s)).equals(s)


def test_parse_aemo_timestamp_keeps_market_time():
    s = aest.parse_aemo_timestamp(pd.Series(["2024/01/15 00:30:00", "2024/01/15 01:00:00"]))
    assert s.tolist() == [pd.Timestamp("2024-01-15 00:30"), pd.Timestamp("2024-01-15 01:00")]
    assert s.dt.tz is None


def test_interval_start():
    end = pd.Series([pd.Timestamp("2024-01-15 00:30")])
    assert aest.interval_start(end, 30).iloc[0] == pd.Timestamp("2024-01-15 00:00")


def test_five_min_to_30min_is_interval_ending():
    # Six 5-min intervals ending 00:05..00:30 belong to the 30-min interval ENDING 00:30.
    ts = pd.date_range("2024-01-15 00:05", periods=12, freq="5min")
    df = pd.DataFrame({"ds": ts, "rrp": [10.0] * 6 + [40.0] * 6})
    out = aest.five_min_to_interval(df, ["rrp"])
    assert out["ds"].tolist() == [
        pd.Timestamp("2024-01-15 00:30"),
        pd.Timestamp("2024-01-15 01:00"),
    ]
    assert out["rrp"].tolist() == [10.0, 40.0]
    assert out["n_sub_intervals"].tolist() == [6, 6]


def test_hourly_weather_aligned_to_interval_end():
    hourly = pd.DataFrame(
        {"ds": pd.to_datetime(["2024-01-15 12:00", "2024-01-15 13:00"]), "temp": [20.0, 22.0]}
    )
    out = aest.hourly_start_to_interval_end(hourly)
    assert out["ds"].tolist() == pd.date_range("2024-01-15 12:00", periods=3, freq="30min").tolist()
    assert out["temp"].tolist() == pytest.approx([20.0, 21.0, 22.0])


def test_trading_day_boundary_is_0400():
    ts = pd.Series(pd.to_datetime(["2024-01-15 04:00", "2024-01-15 04:30"]))
    days = aest.trading_day(ts)
    assert days.tolist() == [pd.Timestamp("2024-01-14"), pd.Timestamp("2024-01-15")]
