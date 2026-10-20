"""Cached loaders. The app reads ONLY committed files (results/, data/forecasts/, data/raw/, models/);
it never calls AEMO or Open-Meteo and never retrains."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # Streamlit Cloud does no editable install

RESULTS = ROOT / "results"
FORECASTS = ROOT / "data" / "forecasts"
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(ttl=3600)
def backtest_results() -> dict | None:
    return _json(RESULTS / "backtest_results.json")


@st.cache_data(ttl=3600)
def aemo_results() -> dict | None:
    return _json(RESULTS / "aemo_benchmark_results.json")


@st.cache_data(ttl=3600)
def spike_results() -> dict | None:
    return _json(RESULTS / "spike_results.json")


@st.cache_data(ttl=3600)
def backtest_predictions(target: str) -> pd.DataFrame | None:
    path = RESULTS / f"backtest_predictions_{target}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=600)
def latest_forecast() -> tuple[pd.DataFrame | None, dict | None]:
    path = FORECASTS / "latest.parquet"
    if not path.exists():
        return None, None
    return pd.read_parquet(path), _json(FORECASTS / "latest.json")


@st.cache_data(ttl=600)
def forecast_history() -> pd.DataFrame | None:
    path = FORECASTS / "history.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=600)
def recent_actuals() -> pd.DataFrame | None:
    path = FORECASTS / "recent_actuals.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def actuals() -> pd.DataFrame:
    files = sorted((RAW / "actuals").glob("vic1_actuals_*.parquet"))
    df = pd.concat(
        [pd.read_parquet(f, columns=["ds", "demand_mw", "rrp"]) for f in files], ignore_index=True
    )
    recent = recent_actuals()
    if recent is not None:
        df = pd.concat([df, recent[["ds", "demand_mw", "rrp"]]]).drop_duplicates("ds", keep="first")
    return df.sort_values("ds").reset_index(drop=True)


@st.cache_data(ttl=600)
def whatif_inputs() -> pd.DataFrame | None:
    path = FORECASTS / "latest_inputs.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource
def demand_model():
    import joblib

    path = MODELS / "demand_lightgbm.joblib"
    return joblib.load(path) if path.exists() else None
