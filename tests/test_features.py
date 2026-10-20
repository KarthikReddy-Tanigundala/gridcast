"""Feature correctness and leakage tests (§12 Phase 3 gate)."""

import numpy as np
import pandas as pd
import pytest

from gridcast.features import build_features, calendar_features, lag_features, weather_features


def frame(n=48 * 10, start="2024-01-01 00:30"):
    ds = pd.date_range(start, periods=n, freq="30min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "unique_id": "VIC1",
            "ds": ds,
            "y": 5000 + 500 * np.sin(np.arange(n) * 2 * np.pi / 48) + rng.normal(0, 50, n),
            "temperature": 20 + 5 * np.sin(np.arange(n) * 2 * np.pi / 48),
            "humidity": 60.0,
            "cloud_cover": 30.0,
            "solar_radiation": 200.0,
            "wind_speed": 10.0,
        }
    )


def test_calendar_uses_interval_start():
    df = pd.DataFrame({"ds": pd.to_datetime(["2024-01-07 00:00", "2024-01-07 00:30"])})
    out = calendar_features.add_calendar_features(df)
    # 00:00-ending interval is Saturday 23:30-24:00; 00:30-ending is Sunday 00:00-00:30.
    assert out["half_hour"].tolist() == [47, 0]
    assert out["day_of_week"].tolist() == [5, 6]


def test_vic_public_holidays():
    df = pd.DataFrame(
        {"ds": pd.to_datetime(["2024-11-05 12:00", "2024-11-06 12:00"])}
    )  # Melbourne Cup
    assert calendar_features.add_calendar_features(df)["is_public_holiday"].tolist() == [1, 0]


def test_school_term_flag():
    df = pd.DataFrame(
        {"ds": pd.to_datetime(["2024-07-01 12:00", "2024-07-16 12:00"])}
    )  # holidays / term 3
    assert calendar_features.add_calendar_features(df)["is_school_term"].tolist() == [0, 1]


def test_hdd_cdd_base_18():
    df = frame(4)
    df["temperature"] = [10.0, 18.0, 25.0, 30.0]
    df = calendar_features.add_calendar_features(df)
    out = weather_features.add_weather_features(df)
    assert out["hdd"].tolist() == [8.0, 0.0, 0.0, 0.0]
    assert out["cdd"].tolist() == [0.0, 0.0, 7.0, 12.0]


def test_temperature_lags_look_backwards_only():
    df = calendar_features.add_calendar_features(frame())
    out = weather_features.add_weather_features(df, temp_lags_hours=(3,))
    assert out["temp_lag_3h"].iloc[6] == df["temperature"].iloc[0]
    assert out["temp_lag_3h"].iloc[:6].isna().all()


def test_lag_features_are_past_only():
    df = frame()
    out = lag_features.add_lag_features(df, "y", (1, 48), (48,))
    assert out["lag1"].iloc[5] == df["y"].iloc[4]
    assert out["lag48"].iloc[100] == df["y"].iloc[52]
    # Rolling mean at t covers t-48..t-1, never t itself.
    assert out["rolling_mean_lag1_window_size48"].iloc[60] == pytest.approx(
        df["y"].iloc[12:60].mean()
    )
    for col in ["lag1", "lag48", "rolling_mean_lag1_window_size48"]:
        lag_features.assert_no_lookahead(out, "y", col)


def test_leakage_guard_catches_future_feature():
    df = frame()
    df["leaky"] = df["y"].shift(-1)
    with pytest.raises(AssertionError, match="lookahead"):
        lag_features.assert_no_lookahead(df, "y", "leaky")


def test_pandas_lags_match_mlforecast():
    from mlforecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = frame()[["unique_id", "ds", "y"]]
    cfg = lag_features.mlforecast_lag_config((1, 48), (48,))
    fc = MLForecast(models=[LinearRegression()], freq="30min", **cfg)
    prep = fc.preprocess(df, dropna=False)
    ours = lag_features.add_lag_features(df, "y", (1, 48), (48,))
    for col in ["lag1", "lag48", "rolling_mean_lag1_window_size48"]:
        pd.testing.assert_series_equal(
            prep[col].reset_index(drop=True),
            ours[col].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )


def test_model_frame_has_no_target_derived_exog():
    out = build_features.model_frame(frame().rename(columns={"y": "demand_mw"}), "demand_mw")
    assert {"unique_id", "ds", "y"} <= set(out.columns)
    assert not any(c.startswith("lag") or "rolling" in c for c in out.columns)
