"""Day-ahead price-spike classification: P(RRP > $300/MWh) for each half-hour tomorrow (§9.3).

Spikes are rare (a few % of VIC1 intervals), so accuracy is meaningless — a model that always says
"no spike" is ~97% accurate. We report **PR-AUC** (average precision), precision / recall / F1 at a
threshold chosen on the 2023 validation year (max F1), and the base rate.

Comparators, on the same intervals:
* **Climatology** — the training-period spike rate for that half-hour and month.
* **AEMO PREDISPATCH** — AEMO's ≈12:33 run predicting RRP > $300 (binary; one precision/recall point).

Model: LightGBM classifier on the same direct day-ahead design as the regressors (every target lag
≥ 72 steps, calendar + day-before weather), refitted monthly on an expanding window.
Output: ``results/spike_results.json`` (+ a section in ``results/leaderboard.md``).
"""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from gridcast import config
from gridcast.data import fetch_aemo_forecasts
from gridcast.data.build_dataset import OUT as DATASET
from gridcast.features.build_features import add_exog_features, exog_columns
from gridcast.features.lag_features import add_lag_features
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    s = config.get_settings()
    out = add_exog_features(df)
    lags, windows = s.features.lags, s.features.rolling_windows
    out = add_lag_features(out.rename(columns={"rrp": "y"}), "y", lags, windows)
    out = out.rename(columns={"y": "rrp"})
    # Rolling stats in add_lag_features sit on lag 1; for a direct day-ahead model they must start at
    # the shortest allowed lag. Recompute them from the base lag so nothing newer than t-72 is used.
    base = min(lags)
    shifted = out["rrp"].shift(base)
    for w in windows:
        out[f"rolling_mean_lag1_window_size{w}"] = shifted.rolling(w, min_periods=w).mean()
    out[f"rolling_std_lag1_window_size{windows[0]}"] = shifted.rolling(
        windows[0], min_periods=windows[0]
    ).std()
    out[f"rolling_max_lag1_window_size{windows[0]}"] = shifted.rolling(
        windows[0], min_periods=windows[0]
    ).max()
    out["spikes_lag_week"] = (shifted > s.price.spike_threshold).rolling(336, min_periods=48).sum()
    lag_cols = (
        [f"lag{lg}" for lg in lags]
        + [c for c in out if c.startswith("rolling_")]
        + ["spikes_lag_week"]
    )
    return out, exog_columns() + lag_cols


def _target_rows(df: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.Series:
    """Mask of rows that are 'tomorrow' for one of the origins."""
    days = pd.DatetimeIndex([o.normalize() + pd.Timedelta(days=1) for o in origins])
    start = df["ds"] - pd.Timedelta(minutes=30)
    return start.dt.normalize().isin(days)


def walk_forward(df: pd.DataFrame, feats: list[str], start, end, threshold: float) -> pd.DataFrame:
    s = config.get_settings()
    params = dict(s.lightgbm) | {"objective": "binary", "n_estimators": 400, "is_unbalance": False}
    origins = [d + pd.Timedelta(hours=s.backtest.origin_hour) for d in pd.date_range(start, end)]
    out = []
    for i in range(0, len(origins), s.backtest.refit_every_days):
        block = origins[i : i + s.backtest.refit_every_days]
        train = df[(df["ds"] <= block[0])].dropna(subset=feats)
        clf = lgb.LGBMClassifier(**params).fit(train[feats], train["rrp"] > threshold)
        test = df[_target_rows(df, block)].copy()
        test["p_model"] = clf.predict_proba(test[feats])[:, 1]
        clim = (
            train.assign(hh=train["half_hour"], m=train["month"], spike=train["rrp"] > threshold)
            .groupby(["m", "hh"])["spike"]
            .mean()
            .rename("p_clim")
        )
        test = test.merge(clim, left_on=["month", "half_hour"], right_index=True, how="left")
        test["origin"] = (
            (test["ds"] - pd.Timedelta(minutes=30)).dt.normalize()
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=s.backtest.origin_hour)
        )
        out.append(test[["origin", "ds", "rrp", "p_model", "p_clim"]])
    res = pd.concat(out, ignore_index=True)
    res["spike"] = (res["rrp"] > threshold).astype(int)
    return res


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.02, 0.9, 89)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _point(y, yhat) -> dict:
    return {
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }


def run() -> dict:
    s = config.get_settings()
    thr = s.price.spike_threshold
    df = pd.read_parquet(DATASET)
    df = df[df["ds"] <= pd.Timestamp(s.data.history_end)].reset_index(drop=True)
    table, feats = feature_table(df)

    val = walk_forward(table, feats, s.backtest.validation_start, s.backtest.validation_end, thr)
    cut = best_f1_threshold(val["spike"].to_numpy(), val["p_model"].to_numpy())
    cut_clim = best_f1_threshold(val["spike"].to_numpy(), val["p_clim"].fillna(0).to_numpy())
    logger.info("validation 2023: threshold %.2f (model), %.2f (climatology)", cut, cut_clim)

    ev = walk_forward(table, feats, s.backtest.eval_start, s.backtest.eval_end, thr)
    y = ev["spike"].to_numpy()
    res = {
        "threshold_usd": thr,
        "eval_window": [str(ev["origin"].min()), str(ev["origin"].max())],
        "n_intervals": int(len(ev)),
        "n_spikes": int(y.sum()),
        "base_rate": float(y.mean()),
        "decision_threshold": {
            "model": cut,
            "climatology": cut_clim,
            "chosen_on": "2023 validation year (max F1)",
        },
        "model": {
            "pr_auc": float(average_precision_score(y, ev["p_model"])),
            **_point(y, ev["p_model"] >= cut),
        },
        "climatology": {
            "pr_auc": float(average_precision_score(y, ev["p_clim"].fillna(0))),
            **_point(y, ev["p_clim"].fillna(0) >= cut_clim),
        },
    }
    pd_runs = fetch_aemo_forecasts.load()
    if pd_runs is not None:
        comp = fetch_aemo_forecasts.comparator_runs(pd_runs, s.backtest.origin_hour)
        both = ev.merge(comp[["origin", "ds", "aemo_rrp"]], on=["origin", "ds"])
        if len(both):
            yb = both["spike"].to_numpy()
            res["aemo_window"] = {
                "n_intervals": int(len(both)),
                "n_spikes": int(yb.sum()),
                "model": {
                    "pr_auc": float(average_precision_score(yb, both["p_model"])),
                    **_point(yb, both["p_model"] >= cut),
                },
                "aemo_predispatch": _point(yb, both["aemo_rrp"] > thr),
            }
    ev.assign(
        p_model=ev["p_model"].astype("float32"), p_clim=ev["p_clim"].astype("float32")
    ).to_parquet(config.RESULTS / "spike_predictions.parquet", index=False)
    (config.RESULTS / "spike_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8", newline="\n"
    )
    logger.info(
        "spikes: base rate %.2f%%, PR-AUC model %.3f vs climatology %.3f",
        100 * res["base_rate"],
        res["model"]["pr_auc"],
        res["climatology"]["pr_auc"],
    )
    refresh_leaderboard()
    return res


def refresh_leaderboard() -> None:
    """Re-render results/leaderboard.md so it includes this module's section."""
    from gridcast.evaluation.backtest import write_leaderboard

    path = config.RESULTS / "backtest_results.json"
    if path.exists():
        write_leaderboard(json.loads(path.read_text(encoding="utf-8"))["targets"])


if __name__ == "__main__":
    setup_logging()
    run()
