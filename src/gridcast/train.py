"""Fit the deployed models on all available history and serialise them to ``models/`` (§10.2).

Training happens offline (here or in the weekly GitHub Action); the dashboard and the daily job
only load these artifacts — nothing is retrained in the cloud app.

Artifacts: ``demand_lightgbm.joblib``, ``price_lightgbm.joblib``, ``spike_lightgbm.joblib`` and
``model_card.json`` (training window, features, validation/backtest scores at train time).
"""

from __future__ import annotations

import json
import logging

import joblib
import lightgbm as lgb
import pandas as pd

from gridcast import config
from gridcast.data.build_dataset import OUT as DATASET
from gridcast.evaluation import spike_classification
from gridcast.evaluation.backtest import TARGETS
from gridcast.features.build_features import exog_columns, model_frame
from gridcast.models import ml_lightgbm
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def model_path(name: str):
    return config.MODELS / f"{name}_lightgbm.joblib"


def run() -> dict:
    s = config.get_settings()
    df = pd.read_parquet(DATASET).dropna(subset=["demand_mw", "rrp"])
    config.MODELS.mkdir(parents=True, exist_ok=True)
    card = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "train_start": str(df["ds"].min()),
        "train_end": str(df["ds"].max()),
        "exog_features": exog_columns(),
        "lags": list(s.features.lags),
        "models": {},
    }

    for name, target in TARGETS.items():
        frame = model_frame(df, target["col"])
        fc = ml_lightgbm.fit(frame, target["objective"], target["extra_exog"])
        joblib.dump(fc, model_path(name), compress=3)
        imp = ml_lightgbm.feature_importance(fc)
        card["models"][name] = {
            "objective": target["objective"],
            "unit": target["unit"],
            "rows": int(frame.dropna().shape[0]),
            "top_features": (imp / imp.sum()).head(10).round(4).to_dict(),
        }
        logger.info(
            "%s model -> %s (%.1f MB)",
            name,
            model_path(name).name,
            model_path(name).stat().st_size / 1e6,
        )

    table, feats = spike_classification.feature_table(df)
    train = table.dropna(subset=feats)
    params = dict(s.lightgbm) | {"objective": "binary", "n_estimators": 400}
    clf = lgb.LGBMClassifier(**params).fit(train[feats], train["rrp"] > s.price.spike_threshold)
    spike_res = config.RESULTS / "spike_results.json"
    cut = (
        json.loads(spike_res.read_text())["decision_threshold"]["model"]
        if spike_res.exists()
        else 0.5
    )
    joblib.dump(
        {
            "model": clf,
            "features": feats,
            "decision_threshold": cut,
            "spike_threshold_usd": s.price.spike_threshold,
        },
        model_path("spike"),
        compress=3,
    )
    card["models"]["spike"] = {"decision_threshold": cut, "rows": int(len(train))}

    bt = config.RESULTS / "backtest_results.json"
    if bt.exists():
        card["backtest_headline"] = {
            k: v["headline"] for k, v in json.loads(bt.read_text())["targets"].items()
        }
    (config.MODELS / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8", newline="\n"
    )
    return card


if __name__ == "__main__":
    setup_logging()
    run()
