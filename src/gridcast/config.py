"""Typed project settings.

Every tunable lives in ``config.yaml`` at the repo root (CLAUDE.md §13). This module loads it once
into frozen dataclasses and exposes the repo paths, so no other module hardcodes a date range, a
horizon or a model parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"

DATA = ROOT / "data"
DATA_CACHE = DATA / "cache"  # gitignored downloads
DATA_RAW = DATA / "raw"  # committed, typed VIC1 extracts
DATA_PROCESSED = DATA / "processed"
DATA_REFERENCE = DATA / "reference"
DATA_FORECASTS = DATA / "forecasts"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
FIXTURES = ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class DataSettings:
    history_start: date
    history_end: date
    freq_minutes: int
    five_min_settlement_start: date
    source: str


@dataclass(frozen=True)
class WeatherSettings:
    variables: tuple[str, ...]
    historical_forecast_start: date
    forecast_days: int


@dataclass(frozen=True)
class FeatureSettings:
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    hdd_cdd_base_c: float
    temp_lags_hours: tuple[int, ...]


@dataclass(frozen=True)
class BacktestSettings:
    eval_start: date
    eval_end: date
    origin_hour: int
    horizon_steps: int
    scored_from_step: int
    refit_every_days: int
    ml_origin_stride_days: int
    classical_origin_stride_days: int
    classical_train_days: int
    mase_season: int
    mase_scale_days: int


@dataclass(frozen=True)
class IntervalSettings:
    levels: tuple[int, ...]
    conformal_windows: int
    calibration_days: int


@dataclass(frozen=True)
class PriceSettings:
    spike_threshold: float
    market_floor: float
    market_price_cap: float
    administered_price_cap: float


@dataclass(frozen=True)
class AemoBenchmarkSettings:
    months: int
    run_hour: int
    run_minute: int


@dataclass(frozen=True)
class DeepLearningSettings:
    enabled: bool
    train_days: int
    max_steps: int


@dataclass(frozen=True)
class Settings:
    region: str
    location: Location
    data: DataSettings
    weather: WeatherSettings
    features: FeatureSettings
    backtest: BacktestSettings
    intervals: IntervalSettings
    lightgbm: dict[str, Any] = field(hash=False)
    price: PriceSettings = field(hash=False)
    aemo_benchmark: AemoBenchmarkSettings = field(hash=False)
    gates: dict[str, float] = field(hash=False)
    deep_learning: DeepLearningSettings = field(hash=False)

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.data.freq_minutes


def _d(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _tuple(values: Any) -> tuple:
    return tuple(values)


def parse_settings(raw: dict[str, Any]) -> Settings:
    """Build typed settings from a parsed config mapping (separated out so tests can override)."""
    d, w, f, b, i = raw["data"], raw["weather"], raw["features"], raw["backtest"], raw["intervals"]
    p, a, dl = raw["price"], raw["aemo_benchmark"], raw["deep_learning"]
    return Settings(
        region=raw["region"],
        location=Location(**raw["location"]),
        data=DataSettings(
            history_start=_d(d["history_start"]),
            history_end=_d(d["history_end"]),
            freq_minutes=int(d["freq_minutes"]),
            five_min_settlement_start=_d(d["five_min_settlement_start"]),
            source=d["source"],
        ),
        weather=WeatherSettings(
            variables=_tuple(w["variables"]),
            historical_forecast_start=_d(w["historical_forecast_start"]),
            forecast_days=int(w["forecast_days"]),
        ),
        features=FeatureSettings(
            lags=_tuple(f["lags"]),
            rolling_windows=_tuple(f["rolling_windows"]),
            hdd_cdd_base_c=float(f["hdd_cdd_base_c"]),
            temp_lags_hours=_tuple(f["temp_lags_hours"]),
        ),
        backtest=BacktestSettings(
            eval_start=_d(b["eval_start"]),
            eval_end=_d(b["eval_end"]),
            **{k: int(v) for k, v in b.items() if k not in ("eval_start", "eval_end")},
        ),
        intervals=IntervalSettings(
            levels=_tuple(i["levels"]),
            conformal_windows=int(i["conformal_windows"]),
            calibration_days=int(i["calibration_days"]),
        ),
        lightgbm=dict(raw["lightgbm"]),
        price=PriceSettings(**{k: float(v) for k, v in p.items()}),
        aemo_benchmark=AemoBenchmarkSettings(**{k: int(v) for k, v in a.items()}),
        gates={k: float(v) for k, v in raw["gates"].items()},
        deep_learning=DeepLearningSettings(
            enabled=bool(dl["enabled"]), train_days=int(dl["train_days"]), max_steps=int(dl["max_steps"])
        ),
    )


@lru_cache(maxsize=1)
def get_settings(path: Path = CONFIG_FILE) -> Settings:
    """Load and cache ``config.yaml``."""
    with open(path, encoding="utf-8") as fh:
        return parse_settings(yaml.safe_load(fh))
