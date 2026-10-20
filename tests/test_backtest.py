"""Walk-forward integrity, baselines, conformal intervals and the AEMO file parser."""

import numpy as np
import pandas as pd
import pytest

from gridcast.data import fetch_aemo_forecasts as fa
from gridcast.data import storage
from gridcast.evaluation import backtest as B
from gridcast.models import baselines, conformal


def test_walk_forward_guard_rejects_lookahead():
    origins = [pd.Timestamp("2024-03-01 12:00")]
    B.assert_walk_forward(pd.Timestamp("2024-03-01 12:00"), origins)  # data to the origin: fine
    with pytest.raises(AssertionError, match="lookahead"):
        B.assert_walk_forward(pd.Timestamp("2024-03-01 12:30"), origins)


def test_origins_are_daily_at_noon():
    o = B.make_origins(pd.Timestamp("2024-02-01").date(), pd.Timestamp("2024-02-03").date(), 12)
    assert o == [pd.Timestamp(f"2024-02-0{d} 12:00") for d in (1, 2, 3)]


def test_windows_never_include_future_rows():
    ds = pd.date_range("2024-01-01 00:30", periods=48 * 20, freq="30min")
    frame = pd.DataFrame({"ds": ds, "y": np.arange(len(ds), dtype=float), "x": 1.0})
    origin = pd.Timestamp("2024-01-15 12:00")
    hist, fut = B.windows(frame, [origin], 96, 72, ["x"])
    assert hist["ds"].max() == origin
    assert fut["ds"].min() == origin + pd.Timedelta(minutes=30)
    assert fut["step"].tolist() == list(range(1, 73))


def test_seasonal_naive_uses_latest_observed_day():
    hist = np.arange(96, dtype=float)  # two days; last index 95
    f = baselines.seasonal_naive(hist, 72, 48)
    assert f[0] == hist[95 + 1 - 48]  # step 1 -> same slot yesterday
    assert f[47] == hist[95 + 48 - 48]  # step 48 -> the origin interval itself
    assert f[48] == hist[95 + 49 - 96]  # step 49 -> two days back (yesterday not observed yet)
    w = baselines.seasonal_naive_week(np.arange(400, dtype=float), 72)
    assert w[0] == 399 + 1 - 336


def test_rolling_conformal_uses_only_matured_origins():
    rows = []
    for d in range(30):
        o = pd.Timestamp("2024-01-01 12:00") + pd.Timedelta(days=d)
        for step in (25, 26):
            rows.append({"origin": o, "step": step, "y": 100.0 + d, "yhat": 100.0})  # residual = d
    out = conformal.rolling_conformal(pd.DataFrame(rows), (80,), calibration_days=56, min_days=5)
    o = pd.Timestamp(
        "2024-01-21 12:00"
    )  # day 20: calibration = days 0..18 only (>= 2 days earlier)
    r = out[(out["origin"] == o) & (out["step"] == 25)].iloc[0]
    assert r["hi80"] - r["yhat"] <= 18 + 1e-9
    assert (
        out[out["origin"] < pd.Timestamp("2024-01-07")]["lo80"].isna().all()
    )  # too little history


def test_pinball_and_coverage_on_symmetric_errors():
    rng = np.random.default_rng(1)
    rows = [
        {
            "origin": pd.Timestamp("2024-01-01 12:00") + pd.Timedelta(days=d),
            "step": 25,
            "y": rng.normal(),
            "yhat": 0.0,
        }
        for d in range(400)
    ]
    out = conformal.rolling_conformal(pd.DataFrame(rows), (80,), calibration_days=56).dropna()
    cov = ((out["y"] >= out["lo80"]) & (out["y"] <= out["hi80"])).mean()
    assert 0.72 < cov < 0.88  # calibrated: ~80% (sampling noise allowed)


MMS = """C,SETP.WORLD,DVD_PREDISPATCHREGIONSUM,AEMO,PUBLIC,2025/10/08,09:37:00,0,MONTHLY_ARCHIVE,0
I,PREDISPATCH,REGION_SOLUTION,9,PREDISPATCHSEQNO,RUNNO,REGIONID,PERIODID,INTERVENTION,TOTALDEMAND,LASTCHANGED,DATETIME
D,PREDISPATCH,REGION_SOLUTION,9,2025090125,1,VIC1,01,0,4500.5,"2025/09/01 12:02:01","2025/09/01 12:30:00"
D,PREDISPATCH,REGION_SOLUTION,9,2025090125,1,NSW1,01,0,7000,"2025/09/01 12:02:01","2025/09/01 12:30:00"
D,PREDISPATCH,REGION_SOLUTION,9,2025090125,1,VIC1,02,1,9999,"2025/09/01 12:02:01","2025/09/01 13:00:00"
D,PREDISPATCH,REGION_SOLUTION,9,2025090125,1,VIC1,02,0,4600,"2025/09/01 12:02:01","2025/09/01 13:00:00"
"""


def test_mms_parser_filters_region_and_intervention():
    rows = fa.iter_mms_records(MMS.splitlines(), "REGION_SOLUTION", "VIC1")
    df = fa.records_to_frame(rows, "TOTALDEMAND", "aemo_demand_mw")
    assert df["aemo_demand_mw"].tolist() == [4500.5, 4600.0]
    assert df["run_time"].iloc[0] == pd.Timestamp("2025-09-01 12:02:01")
    assert df["ds"].tolist() == [pd.Timestamp("2025-09-01 12:30"), pd.Timestamp("2025-09-01 13:00")]


def _pd_run(run_time: str, start: str, n: int, seq: int) -> pd.DataFrame:
    ds = pd.date_range(start, periods=n, freq="30min")
    return pd.DataFrame(
        {
            "seqno": seq,
            "run_time": pd.Timestamp(run_time),
            "ds": ds,
            "aemo_demand_mw": 5000.0,
            "aemo_rrp": 50.0,
        }
    )


def test_comparator_is_first_run_after_noon_covering_tomorrow():
    early = _pd_run("2025-09-01 12:02", "2025-09-01 12:30", 32, 1)  # stops 04:00 tomorrow
    first = _pd_run("2025-09-01 12:33", "2025-09-01 13:00", 79, 2)  # covers all of tomorrow
    later = _pd_run("2025-09-01 13:03", "2025-09-01 13:30", 78, 3)
    comp = fa.comparator_runs(pd.concat([early, first, later]))
    assert comp["run_time"].unique().tolist() == [pd.Timestamp("2025-09-01 12:33")]
    assert len(comp) == 48
    assert comp["ds"].min() == pd.Timestamp("2025-09-02 00:30")
    assert comp["ds"].max() == pd.Timestamp("2025-09-03 00:00")


def test_partitions_round_trip_and_skip_unchanged(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-12-31 23:30", "2025-01-01 00:00", "2025-01-01 00:30"]),
            "v": [1.0, 2.0, 3.0],
        }
    )
    written = storage.write_partitions(df, tmp_path, "x")
    assert sorted(p.name for p in written) == ["x_2024.parquet", "x_2025.parquet"]
    assert storage.write_partitions(df, tmp_path, "x") == []  # nothing changed -> nothing rewritten
    pd.testing.assert_frame_equal(storage.read_partitions(tmp_path, "x"), df)
