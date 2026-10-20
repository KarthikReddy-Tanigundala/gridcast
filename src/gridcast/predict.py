"""Live day-ahead forecast: tomorrow's 48 half-hours of VIC1 demand and price (§10.3–10.4).

Run by the daily GitHub Action at ~12:35 AEST, after AEMO's dashboard feed has today's actuals to
12:00 and AEMO's 12:30 PREDISPATCH run is published — the same 12:00-origin protocol as the backtest.
If run earlier, the origin is the latest actual interval and the forecast still targets tomorrow
(``direct`` is then False for the furthest steps, which use one level of recursion).

Inputs: committed models, ``data/processed/vic1_30min.parquet`` (history), the live Open-Meteo
forecast (tomorrow's weather), the latest PD run (AEMO's forecast for comparison).
Outputs (committed by the daily job, all small):
``data/forecasts/latest.parquet`` + ``latest.json`` (the forecast and its metadata),
``data/forecasts/latest_inputs.parquet`` (model inputs for the dashboard's what-if tab),
``data/forecasts/history.parquet`` (every issued forecast — the live track record),
``data/forecasts/recent_actuals.parquet`` (last 21 days, for the dashboard).
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from gridcast import config
from gridcast.data import fetch_aemo_forecasts, fetch_weather
from gridcast.data.build_dataset import OUT as DATASET
from gridcast.data.build_dataset import WEATHER_COLS
from gridcast.evaluation import spike_classification
from gridcast.evaluation.backtest import TARGETS
from gridcast.features.build_features import exog_columns, model_frame
from gridcast.models import conformal, ml_lightgbm
from gridcast.train import model_path
from gridcast.utils.aest import hourly_start_to_interval_end, now_aest
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

OUT = config.DATA_FORECASTS


def choose_origin(last_actual: pd.Timestamp, now: pd.Timestamp, hour: int) -> pd.Timestamp:
    """12:00 today if today's noon is observed; otherwise the latest observed interval."""
    noon = now.normalize() + pd.Timedelta(hours=hour)
    return noon if last_actual >= noon else last_actual


def live_weather() -> pd.DataFrame:
    """Live Open-Meteo forecast (incl. the past 2 days) on the 30-minute interval-ending grid."""
    live = pd.read_parquet(fetch_weather.LIVE_OUT)
    hourly = live.rename(columns={k: v for k, v in WEATHER_COLS.items()})[
        ["ds", *WEATHER_COLS.values()]
    ]
    return hourly_start_to_interval_end(hourly)


def assemble(history: pd.DataFrame, origin: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """History up to the origin + future rows (weather from the live forecast, targets NaN).

    The weather archives lag by about a day, so the last hours of history can lack weather; those
    gaps are filled from the live forecast's ``past_days`` so lagged-temperature features stay defined.
    """
    wx = live_weather()
    hist = history[history["ds"] <= origin].copy()
    gap = hist[list(WEATHER_COLS.values())].isna().any(axis=1)
    if gap.any():
        fill = hist.loc[gap, ["ds"]].merge(wx, on="ds", how="left")
        for col in WEATHER_COLS.values():
            hist.loc[gap, col] = fill[col].to_numpy()
    fut = wx[(wx["ds"] > origin) & (wx["ds"] <= end)]
    expected = pd.date_range(origin + pd.Timedelta(minutes=30), end, freq="30min")
    if len(fut) != len(expected):
        raise ValueError(f"live weather covers {len(fut)}/{len(expected)} future intervals")
    fut = fut.assign(
        unique_id=history["unique_id"].iloc[0],
        demand_mw=np.nan,
        rrp=np.nan,
        weather_regime="live_forecast",
    )
    return pd.concat([hist, fut], ignore_index=True)


def forecast_target(
    name: str, frame_all: pd.DataFrame, origin: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    target = TARGETS[name]
    fc = joblib.load(model_path(name))
    frame = model_frame(frame_all, target["col"])
    exog = exog_columns()
    hist = frame[frame["ds"] <= origin].tail(ml_lightgbm.HISTORY_ROWS)
    fut = frame[frame["ds"] > origin].head(horizon)
    pred = ml_lightgbm.predict_origins(fc, hist, fut[["unique_id", "ds", *exog]], horizon)
    return pred.rename(columns={"LightGBM": f"{name}_yhat"})[["ds", f"{name}_yhat"]]


def add_intervals(fcst: pd.DataFrame, name: str, origin_hour: int) -> pd.DataFrame:
    """Conformal bounds from the most recent backtest residuals, matched by slot in tomorrow."""
    s = config.get_settings()
    path = config.RESULTS / f"backtest_predictions_{name}.parquet"
    if not path.exists():
        return fcst
    bt = pd.read_parquet(path)
    bt = bt[bt["model"] == "LightGBM"][["origin", "step", "y", "yhat"]]
    steps_today = (24 - origin_hour) * 2
    slot = (
        (fcst["ds"] - pd.Timedelta(minutes=30)).dt.hour * 2
        + (fcst["ds"] - pd.Timedelta(minutes=30)).dt.minute // 30
        + 1
    )
    tmp = pd.DataFrame(
        {"step": steps_today + slot.to_numpy(), "yhat": fcst[f"{name}_yhat"].to_numpy()}
    )
    res = json.loads((config.RESULTS / "backtest_results.json").read_text(encoding="utf-8"))
    alphas = {int(k): v for k, v in res["targets"][name].get("aci_alphas", {}).items()} or None
    tmp = conformal.live_intervals(
        tmp, bt, s.intervals.levels, s.intervals.calibration_days, alphas
    )
    for lv in s.intervals.levels:
        fcst[f"{name}_lo{lv}"] = tmp[f"lo{lv}"].to_numpy()
        fcst[f"{name}_hi{lv}"] = tmp[f"hi{lv}"].to_numpy()
    return fcst


def spike_probability(frame_all: pd.DataFrame, targets: pd.Series) -> np.ndarray:
    art = joblib.load(model_path("spike"))
    table, feats = spike_classification.feature_table(frame_all)
    rows = table[table["ds"].isin(targets)]
    return art["model"].predict_proba(rows[art["features"]])[:, 1]


def aemo_for_tomorrow(origin: pd.Timestamp) -> pd.DataFrame | None:
    runs = fetch_aemo_forecasts.load()
    if runs is None:
        return None
    s = config.get_settings()
    comp = fetch_aemo_forecasts.comparator_runs(runs, s.backtest.origin_hour)
    comp = comp[comp["origin"] == origin.normalize() + pd.Timedelta(hours=s.backtest.origin_hour)]
    if comp.empty:  # 12:30 run not published yet: fall back to the latest run covering tomorrow
        latest = runs[runs["run_time"] == runs["run_time"].max()]
        return latest[["ds", "aemo_demand_mw", "aemo_rrp", "run_time"]]
    return comp[["ds", "aemo_demand_mw", "aemo_rrp", "run_time"]]


def run() -> dict:
    s = config.get_settings()
    history = pd.read_parquet(DATASET)
    history = history[history["demand_mw"].notna()]
    now = now_aest()
    origin = choose_origin(history["ds"].max(), now, s.backtest.origin_hour)
    tomorrow_end = origin.normalize() + pd.Timedelta(days=2)
    horizon = int((tomorrow_end - origin) / pd.Timedelta(minutes=30))
    frame_all = assemble(history, origin, tomorrow_end)

    fc = forecast_target("demand", frame_all, origin, horizon)
    fc = fc.merge(forecast_target("price", frame_all, origin, horizon), on="ds")
    fc = fc[fc["ds"] > tomorrow_end - pd.Timedelta(days=1)].reset_index(drop=True)
    for name in TARGETS:
        fc = add_intervals(fc, name, s.backtest.origin_hour)
    fc["p_spike"] = spike_probability(frame_all, fc["ds"])
    aemo = aemo_for_tomorrow(origin)
    if aemo is not None:
        fc = fc.merge(aemo.drop(columns="run_time").drop_duplicates("ds"), on="ds", how="left")
    fc.insert(0, "origin", origin)

    live = pd.read_parquet(fetch_weather.LIVE_OUT)
    meta = {
        "issued_at": str(now.floor("min")),
        "origin": str(origin),
        "target_day": str((tomorrow_end - pd.Timedelta(days=1)).date()),
        "horizon_steps": horizon,
        "direct": horizon <= s.backtest.horizon_steps,
        "weather_issued_at": str(live["issued_at"].iloc[0]),
        "aemo_run_time": None if aemo is None or aemo.empty else str(aemo["run_time"].iloc[0]),
        "models": json.loads((config.MODELS / "model_card.json").read_text())["trained_at"],
        "timezone": "AEST (UTC+10, no DST), interval-ending",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(OUT / "latest.parquet", index=False)
    (OUT / "latest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    keep = ["unique_id", "ds", "demand_mw", "rrp", *WEATHER_COLS.values(), "weather_regime"]
    frame_all[frame_all["ds"] > origin - pd.Timedelta(days=16)][keep].to_parquet(
        OUT / "latest_inputs.parquet", index=False
    )

    hist_path = OUT / "history.parquet"
    track = fc.drop(columns=[c for c in fc if c.startswith(("aemo_",))], errors="ignore")
    if hist_path.exists():
        track = pd.concat([pd.read_parquet(hist_path), track]).drop_duplicates(
            ["origin", "ds"], keep="last"
        )
    track.to_parquet(hist_path, index=False)
    recent = history[history["ds"] > history["ds"].max() - pd.Timedelta(days=21)]
    recent[["ds", "demand_mw", "rrp", "temperature_obs"]].to_parquet(
        OUT / "recent_actuals.parquet", index=False
    )
    logger.info(
        "Forecast for %s issued from origin %s: demand %.0f–%.0f MW, price %.0f–%.0f $/MWh, max P(spike) %.2f",
        meta["target_day"],
        origin,
        fc["demand_yhat"].min(),
        fc["demand_yhat"].max(),
        fc["price_yhat"].min(),
        fc["price_yhat"].max(),
        fc["p_spike"].max(),
    )
    return meta


if __name__ == "__main__":
    setup_logging()
    run()
