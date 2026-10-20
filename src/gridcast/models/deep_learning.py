"""Deep-learning comparison (§8.1 step 4): did NHITS beat a well-featured LightGBM on day-ahead demand?

Optional — needs ``pip install -r requirements-dl.txt`` (neuralforecast + CPU torch) and is never
part of the deployed app. Kept deliberately cheap so it runs on a laptop CPU:

* **NHITS** with the same forecast-as-known exogenous inputs (calendar + day-before weather),
  ``input_size`` = one week, ``h`` = 72 (12:30 today → 24:00 tomorrow), MAE loss.
* Fitted on the ``deep_learning.train_days`` before each refit point, refitted every 13 weeks, and
  used to forecast the **weekly** backtest origins (the classical-model subset). Scored on tomorrow's
  48 half-hours against LightGBM on exactly the same origins.

Writes ``results/deep_learning_results.json``; the leaderboard picks it up.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from gridcast import config
from gridcast.data.build_dataset import OUT as DATASET
from gridcast.evaluation import backtest as B
from gridcast.evaluation import metrics as M
from gridcast.features.build_features import model_frame
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

FUTR = [
    "temperature",
    "hdd",
    "cdd",
    "solar_radiation",
    "cloud_cover",
    "half_hour",
    "day_of_week",
    "is_non_working_day",
    "doy_sin",
    "doy_cos",
]
REFIT_WEEKS = 13


def run() -> dict:
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError as exc:  # optional dependency
        raise SystemExit(
            "neuralforecast not installed — `pip install -r requirements-dl.txt`"
        ) from exc

    s = config.get_settings()
    bt = s.backtest
    df = pd.read_parquet(DATASET)
    frame = model_frame(df, "demand_mw")
    frame = frame[frame["ds"] <= pd.Timestamp(s.data.history_end)].reset_index(drop=True)
    frame[FUTR] = frame[FUTR].astype("float32")
    origins = B.make_origins(
        bt.eval_start, bt.eval_end, bt.origin_hour, bt.classical_origin_stride_days
    )

    preds = []
    for i in range(0, len(origins), REFIT_WEEKS):
        block = origins[i : i + REFIT_WEEKS]
        t0 = time.time()
        train = frame[
            (frame["ds"] <= block[0])
            & (frame["ds"] > block[0] - pd.Timedelta(days=s.deep_learning.train_days))
        ]
        B.assert_walk_forward(train["ds"].max(), block)
        model = NHITS(
            h=bt.horizon_steps,
            input_size=7 * 48,
            futr_exog_list=FUTR,
            max_steps=s.deep_learning.max_steps,
            batch_size=32,
            windows_batch_size=256,
            scaler_type="robust",
            random_seed=42,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        nf = NeuralForecast(models=[model], freq="30min")
        nf.fit(df=train[["unique_id", "ds", "y", *FUTR]])
        hist, fut = B.windows(frame, block, 7 * 48, bt.horizon_steps, FUTR)
        p = nf.predict(df=hist, futr_df=fut[["unique_id", "ds", *FUTR]])
        p = p.reset_index() if "unique_id" not in p.columns else p
        preds.append(
            fut[["unique_id", "ds", "step", "y"]].merge(
                p[["unique_id", "ds", "NHITS"]], on=["unique_id", "ds"]
            )
        )
        logger.info(
            "NHITS block %d (%s): %d origins, %.0fs",
            i // REFIT_WEEKS + 1,
            block[0].date(),
            len(block),
            time.time() - t0,
        )

    nh = pd.concat(preds, ignore_index=True)
    nh = nh[nh["step"] >= bt.scored_from_step]
    nh["origin"] = pd.to_datetime(nh["unique_id"])
    lgbm = pd.read_parquet(config.RESULTS / "backtest_predictions_demand.parquet")
    lgbm = lgbm[(lgbm["model"] == "LightGBM") & lgbm["origin"].isin(nh["origin"].unique())]
    both = nh.merge(lgbm[["origin", "ds", "yhat"]], on=["origin", "ds"])
    scales = B.mase_scales(
        frame, sorted(both["origin"].unique()), bt.mase_season, bt.mase_scale_days, s.steps_per_day
    )
    rows = []
    for label, col in [("NHITS", "NHITS"), ("LightGBM", "yhat")]:
        per = both.groupby("origin").apply(
            lambda g, c=col: M.mae(g["y"], g[c]), include_groups=False
        )
        rows.append(
            {
                "model": label,
                "MASE": float((per / scales.reindex(per.index)).mean()),
                "MAE": M.mae(both["y"], both[col]),
                "RMSE": M.rmse(both["y"], both[col]),
            }
        )
    res = {
        "target": "demand",
        "n_origins": int(both["origin"].nunique()),
        "refit_weeks": REFIT_WEEKS,
        "train_days": s.deep_learning.train_days,
        "max_steps": s.deep_learning.max_steps,
        "table": rows,
    }
    (config.RESULTS / "deep_learning_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8", newline="\n"
    )
    logger.info("NHITS vs LightGBM on %d weekly origins: %s", res["n_origins"], rows)
    summary = json.loads((config.RESULTS / "backtest_results.json").read_text(encoding="utf-8"))
    B.write_leaderboard(summary["targets"])
    return res


if __name__ == "__main__":
    setup_logging()
    run()
