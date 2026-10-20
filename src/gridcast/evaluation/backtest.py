"""Walk-forward backtest (§9.1) → ``results/leaderboard.md`` + ``backtest_results.json``.

Protocol (fixed in ``config.yaml`` before any result was seen):

* **Origins**: every day in the evaluation window at 12:00 AEST — the realistic day-ahead decision
  point. The model sees actuals up to the interval ending 12:00 and forecasts 72 steps; only steps
  25–72 (tomorrow 00:30–24:00) are scored.
* **Expanding window**: LightGBM is refitted every ``refit_every_days`` on all data up to the first
  origin of each block, then predicts every origin in the block. ``assert_walk_forward`` fails the
  run if any training row is later than the origin it is used for.
* **Weather**: forecast-as-known only (day-before forecasts) — every eval interval is regime
  ``dayahead``, asserted below.
* **Classical models** are expensive, so they run on weekly origins with a trailing window; a
  second table compares *all* models on exactly those origins.
* **MASE** per origin = MAE / in-sample seasonal-naive (48) MAE over the trailing 28 days.
* **Intervals**: rolling split-conformal (``models/conformal.py``); coverage measured out of sample.
* Headline model rule (fixed in advance): LightGBM. Gate: demand MASE < ``gates.demand_lightgbm_mase_max``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.data.build_dataset import OUT as DATASET
from gridcast.evaluation import metrics as M
from gridcast.features.build_features import exog_columns, model_frame
from gridcast.models import baselines, classical, conformal, ml_lightgbm
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

TARGETS = {
    "demand": {"col": "demand_mw", "objective": "regression", "unit": "MW", "extra_exog": []},
    # Price: L1 objective (robust to spikes). Coupling our own demand forecast in as a feature was
    # tested on the 2023 validation year and gave no gain (MASE 0.705 both ways), so it is off;
    # set extra_exog=["demand_forecast"] to re-enable the two-stage model.
    "price": {"col": "rrp", "objective": "l1", "unit": "$/MWh", "extra_exog": []},
}
HEADLINE = "LightGBM"
CLASSICAL = ["AutoETS", "MSTL-ETS", "MSTL-ARIMA", "MSTL-ARIMAX"]


# ----------------------------------------------------------------------------------- utilities
def make_origins(start: date, end: date, hour: int, stride_days: int = 1) -> list[pd.Timestamp]:
    days = pd.date_range(start, end, freq=f"{stride_days}D")
    return [d + pd.Timedelta(hours=hour) for d in days]


def assert_walk_forward(train_max_ds: pd.Timestamp, origins: list[pd.Timestamp]) -> None:
    """Leakage guard: a model may only be used for origins at or after its last training row."""
    first = min(origins)
    if train_max_ds > first:
        raise AssertionError(
            f"lookahead: trained on data to {train_max_ds} but used for origin {first}"
        )


def windows(
    frame: pd.DataFrame,
    origins: list[pd.Timestamp],
    n_hist: int,
    horizon: int,
    future_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin trailing history (``unique_id = origin``) and future exog/truth blocks."""
    pos = pd.Series(np.arange(len(frame)), index=frame["ds"])
    hist, fut = [], []
    for o in origins:
        i = int(pos[o])
        uid = o.strftime("%Y-%m-%d %H:%M")
        h = frame.iloc[i - n_hist + 1 : i + 1][["ds", "y", *future_cols]].assign(unique_id=uid)
        f = frame.iloc[i + 1 : i + 1 + horizon][["ds", "y", *future_cols]].assign(unique_id=uid)
        f["step"] = np.arange(1, len(f) + 1)
        hist.append(h)
        fut.append(f)
    return pd.concat(hist, ignore_index=True), pd.concat(fut, ignore_index=True)


def mase_scales(
    frame: pd.DataFrame, origins: list[pd.Timestamp], season: int, days: int, steps_per_day: int
) -> pd.Series:
    pos = pd.Series(np.arange(len(frame)), index=frame["ds"])
    y = frame["y"].to_numpy()
    n = days * steps_per_day
    return pd.Series(
        {
            o: M.seasonal_naive_scale(y[int(pos[o]) - n + 1 : int(pos[o]) + 1], season)
            for o in origins
        }
    )


# ------------------------------------------------------------------------------------ models
def run_baselines(hist: pd.DataFrame, fut: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = []
    for uid, h in hist.groupby("unique_id", sort=False):
        y = h["y"].to_numpy()
        f = fut[fut["unique_id"] == uid][["unique_id", "ds", "step", "y"]].copy()
        for name, fn in baselines.BASELINES.items():
            out.append(f.assign(model=name, yhat=fn(y, horizon)[: len(f)]))
    return pd.concat(out, ignore_index=True)


def run_lightgbm(
    frame: pd.DataFrame,
    origins: list[pd.Timestamp],
    target: dict,
    bt,
    horizon: int,
    demand_fcst: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    exog = exog_columns() + target["extra_exog"]
    out, importance = [], None
    blocks, cur = [], []  # refit cadence in calendar days, independent of the origin stride
    for o in origins:
        if cur and (o - cur[0]).days >= bt.refit_every_days:
            blocks.append(cur)
            cur = []
        cur.append(o)
    blocks.append(cur)
    for b, block in enumerate(blocks, 1):
        t0 = time.time()
        train = frame[frame["ds"] <= block[0]]
        assert_walk_forward(train["ds"].max(), block)
        fc = ml_lightgbm.fit(train, target["objective"], target["extra_exog"])
        hist, fut = windows(frame, block, ml_lightgbm.HISTORY_ROWS, horizon, exog)
        if demand_fcst is not None:  # price: replace actual demand with OUR forecast of it
            fut = fut.drop(columns="demand_forecast").merge(
                demand_fcst, on=["unique_id", "ds"], how="left", validate="one_to_one"
            )
        x = fut[["unique_id", "ds", *exog]]
        pred = ml_lightgbm.predict_origins(fc, hist, x, horizon)
        out.append(
            fut[["unique_id", "ds", "step", "y"]]
            .merge(pred, on=["unique_id", "ds"])
            .rename(columns={HEADLINE: "yhat"})
            .assign(model=HEADLINE)
        )
        imp = ml_lightgbm.feature_importance(fc)
        importance = imp if importance is None else importance.add(imp, fill_value=0)
        logger.info(
            "LightGBM block %d/%d (%s, train %d rows) %.0fs",
            b,
            len(blocks),
            block[0].date(),
            len(train),
            time.time() - t0,
        )
    return pd.concat(out, ignore_index=True), (importance / importance.sum()).sort_values(
        ascending=False
    )


def run_classical(
    frame: pd.DataFrame, origins: list[pd.Timestamp], bt, horizon: int, steps_per_day: int
) -> pd.DataFrame:
    t0 = time.time()
    hist, fut = windows(
        frame, origins, bt.classical_train_days * steps_per_day, horizon, classical.EXOG
    )
    pred = classical.forecast_windows(hist, fut, horizon)
    long = pred.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="yhat")
    logger.info("Classical models on %d origins: %.0fs", len(origins), time.time() - t0)
    return fut[["unique_id", "ds", "step", "y"]].merge(long, on=["unique_id", "ds"])


# --------------------------------------------------------------------------------- scoring
def score(
    preds: pd.DataFrame, scales: pd.Series, levels: tuple[int, ...], positive: bool
) -> pd.DataFrame:
    rows = []
    for model, g in preds.groupby("model", sort=False):
        per_origin = g.groupby("origin").apply(
            lambda x: M.mae(x["y"], x["yhat"]), include_groups=False
        )
        row = {
            "model": model,
            "origins": int(g["origin"].nunique()),
            "MASE": float((per_origin / scales.reindex(per_origin.index)).mean()),
            "MAE": M.mae(g["y"], g["yhat"]),
            "RMSE": M.rmse(g["y"], g["yhat"]),
        }
        if positive:
            row["sMAPE"] = M.smape(g["y"], g["yhat"])
        for lv in levels:
            lo, hi = f"lo{lv}", f"hi{lv}"
            if lo in g and g[lo].notna().any():
                ok = g[lo].notna()
                row[f"pinball{lv}"] = M.interval_pinball(
                    g.loc[ok, "y"], g.loc[ok, lo], g.loc[ok, hi], lv
                )
                row[f"coverage{lv}"] = M.coverage(g.loc[ok, "y"], g.loc[ok, lo], g.loc[ok, hi])
                row[f"width{lv}"] = M.interval_width(g.loc[ok, lo], g.loc[ok, hi])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)


def add_intervals(
    preds: pd.DataFrame,
    models: list[str],
    levels: tuple[int, ...],
    cal_days: int,
    gamma: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Rolling (adaptive) conformal bounds per model; returns the final working alphas per model."""
    parts, state = [], {}
    for model, g in preds.groupby("model", sort=False):
        if model in models:
            g, state[model] = conformal.rolling_conformal(
                g, levels, cal_days, gamma=gamma, return_state=True
            )
        parts.append(g)
    return pd.concat(parts, ignore_index=True), state


# ---------------------------------------------------------------------------------- driver
def backtest_target(
    name: str,
    df: pd.DataFrame,
    *,
    smoke: bool = False,
    classical_on: bool = True,
    demand_preds: pd.DataFrame | None = None,
    validation: bool = False,
) -> dict:
    s = config.get_settings()
    bt, spd = s.backtest, s.steps_per_day
    target = TARGETS[name]
    frame_src = df.assign(demand_forecast=df["demand_mw"]) if target["extra_exog"] else df
    frame = model_frame(frame_src, target["col"])
    for col in target["extra_exog"]:
        frame[col] = frame_src[col].to_numpy()
    frame = frame[frame["ds"] <= pd.Timestamp(s.data.history_end)].reset_index(drop=True)

    start, end, stride = bt.eval_start, bt.eval_end, bt.ml_origin_stride_days
    if validation:  # design selection happens here, never on the eval window
        start, end, stride = bt.validation_start, bt.validation_end, bt.validation_stride_days
    if smoke:
        start = (pd.Timestamp(end) - pd.Timedelta(days=41)).date()
    origins = make_origins(start, end, bt.origin_hour, stride)
    horizon = bt.horizon_steps

    eval_rows = df[(df["ds"] > origins[0]) & (df["ds"] <= origins[-1] + pd.Timedelta(days=2))]
    regimes = eval_rows["weather_regime"].value_counts().to_dict()
    if set(regimes) - {"dayahead"} and not (smoke or validation):
        raise AssertionError(f"eval window uses non-day-ahead weather: {regimes}")

    hist, fut = windows(frame, origins, ml_lightgbm.HISTORY_ROWS, horizon, [])
    parts = [run_baselines(hist, fut, horizon)]
    demand_fcst = None
    if demand_preds is not None and "demand_forecast" in target["extra_exog"]:
        demand_fcst = demand_preds[demand_preds["model"] == HEADLINE][
            ["unique_id", "ds", "yhat"]
        ].rename(columns={"yhat": "demand_forecast"})
    lgb_preds, importance = run_lightgbm(frame, origins, target, bt, horizon, demand_fcst)
    parts.append(lgb_preds)
    daily = pd.concat(parts, ignore_index=True)
    daily["origin"] = pd.to_datetime(daily["unique_id"])

    weekly_origins = origins[:: max(1, bt.classical_origin_stride_days // stride)]
    if smoke:
        weekly_origins = weekly_origins[-3:]
    weekly = None
    if classical_on:
        weekly = run_classical(frame, weekly_origins, bt, horizon, spd)
        weekly["origin"] = pd.to_datetime(weekly["unique_id"])

    scored_daily = daily[daily["step"] >= bt.scored_from_step]
    interval_models = [HEADLINE, "Seasonal-naive (week)"]
    scored_daily, aci_state = add_intervals(
        scored_daily,
        interval_models,
        s.intervals.levels,
        s.intervals.calibration_days,
        s.intervals.aci_gamma,
    )
    scales = mase_scales(frame, origins, bt.mase_season, bt.mase_scale_days, spd)
    positive = name == "demand"
    table_daily = score(scored_daily, scales, s.intervals.levels, positive)

    table_weekly = None
    if weekly is not None:
        w_origins = set(weekly["origin"].unique())
        common = pd.concat(
            [
                scored_daily[scored_daily["origin"].isin(w_origins)],
                weekly[weekly["step"] >= bt.scored_from_step],
            ],
            ignore_index=True,
        )
        table_weekly = score(common, scales, (), positive)

    by_hour = (
        scored_daily.assign(hour=(scored_daily["ds"] - pd.Timedelta(minutes=30)).dt.hour)
        .groupby(["model", "hour"])
        .apply(lambda g: M.mae(g["y"], g["yhat"]), include_groups=False)
        .unstack("model")
    )
    headline = table_daily.set_index("model").loc[HEADLINE]
    result = {
        "target": name,
        "unit": target["unit"],
        "eval_start": str(origins[0]),
        "eval_end": str(origins[-1]),
        "n_origins": len(origins),
        "n_classical_origins": len(weekly_origins) if classical_on else 0,
        "weather_regimes_in_eval": regimes,
        "leaderboard_daily": table_daily.to_dict("records"),
        "leaderboard_weekly": None if table_weekly is None else table_weekly.to_dict("records"),
        "mae_by_hour": by_hour.round(2).reset_index().to_dict("records"),
        "feature_importance": importance.head(20).round(4).to_dict(),
        "headline": {
            "model": HEADLINE,
            "MASE": float(headline["MASE"]),
            "MAE": float(headline["MAE"]),
        },
        # Final adaptive-conformal miscoverage levels — the live forecast continues from these.
        "aci_alphas": {str(lv): float(a) for lv, a in aci_state.get(HEADLINE, {}).items()},
    }
    if name == "demand":
        gate = s.gates["demand_lightgbm_mase_max"]
        result["gate"] = {"threshold": gate, "passed": bool(headline["MASE"] < gate)}
    keep = ["origin", "ds", "step", "model", "y", "yhat"] + [
        c for c in scored_daily if c.startswith(("lo", "hi"))
    ]
    preds_out = scored_daily[keep]
    if weekly is not None:
        preds_out = pd.concat(
            [preds_out, weekly[weekly["step"] >= bt.scored_from_step][keep[:6]]], ignore_index=True
        )
    return {"result": result, "preds": preds_out, "all_daily": daily}


# ------------------------------------------------------------------------------- reporting
def _fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}"


def _extra_sections() -> list[str]:
    """Deep-learning, spike and AEMO summaries, when their result files exist."""
    lines: list[str] = []
    dl = config.RESULTS / "deep_learning_results.json"
    if dl.exists():
        r = json.loads(dl.read_text(encoding="utf-8"))
        lines += [
            f"## Deep learning vs LightGBM (demand, {r['n_origins']} weekly origins)",
            "",
            f"NHITS (neuralforecast, CPU) trained on the trailing {r['train_days']} days, refitted every "
            f"{r['refit_weeks']} weeks, same day-before weather inputs.",
            "",
            "| Model | MASE | MAE (MW) | RMSE (MW) |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {x['model']} | {x['MASE']:.3f} | {x['MAE']:,.0f} | {x['RMSE']:,.0f} |"
            for x in r["table"]
        ]
        lines.append("")
    sp = config.RESULTS / "spike_results.json"
    if sp.exists():
        r = json.loads(sp.read_text(encoding="utf-8"))
        lines += [
            f"## Price spikes: P(RRP > ${r['threshold_usd']:.0f}/MWh) tomorrow",
            "",
            f"{r['n_intervals']:,} half-hours, {r['n_spikes']} spikes (base rate {r['base_rate']:.2%}). "
            "Decision thresholds chosen on the 2023 validation year (max F1). Accuracy is not reported: "
            "always predicting 'no spike' would score ~99%.",
            "",
            "| Classifier | PR-AUC | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for label, key in [
            ("GridCast LightGBM classifier", "model"),
            ("Climatology (month × half-hour rate)", "climatology"),
        ]:
            x = r[key]
            lines.append(
                f"| {label} | {x['pr_auc']:.3f} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} |"
            )
        if "aemo_window" in r:
            w = r["aemo_window"]
            lines += [
                "",
                f"On the {w['n_intervals']:,} half-hours where AEMO's 12:33 PD run is available ({w['n_spikes']} spikes):",
                "",
                "| Classifier | PR-AUC | Precision | Recall | F1 |",
                "|---|---|---|---|---|",
                f"| GridCast | {w['model']['pr_auc']:.3f} | {w['model']['precision']:.3f} | {w['model']['recall']:.3f} | {w['model']['f1']:.3f} |",
                f"| AEMO PREDISPATCH (RRP > $300) | — | {w['aemo_predispatch']['precision']:.3f} | "
                f"{w['aemo_predispatch']['recall']:.3f} | {w['aemo_predispatch']['f1']:.3f} |",
            ]
        lines.append("")
    ab = config.RESULTS / "aemo_benchmark_results.json"
    if ab.exists():
        lines += [
            "## GridCast vs AEMO PREDISPATCH",
            "",
            "See [aemo_benchmark.md](aemo_benchmark.md) for the full comparison.",
            "",
        ]
    return lines


def write_leaderboard(results: dict) -> str:
    s = config.get_settings()
    lines = [
        "# GridCast leaderboard",
        "",
        f"_Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m gridcast.evaluation.backtest`. "
        "All numbers are measured out of sample by walk-forward backtesting; nothing is hand-entered._",
        "",
        "**Protocol.** One forecast per day issued at **12:00 AEST** for the whole of the **next day** "
        "(48 half-hours, 12.5–36 h ahead) — the day-ahead setting. Expanding-window refits every "
        f"{s.backtest.refit_every_days} days; weather inputs are the forecasts issued the day before "
        "(no hindsight weather). **MASE** = MAE ÷ in-sample MAE of a same-half-hour-yesterday forecast over "
        "the trailing 28 days (< 1 beats that naive rule). Intervals: rolling split-conformal with adaptive (ACI) correction; coverage measured, not assumed.",
        "",
    ]
    for name, r in results.items():
        unit = r["unit"]
        nd = 0 if unit == "MW" else 1
        lines += [
            f"## {name.capitalize()} ({unit}) — {r['n_origins']} daily origins, "
            f"{r['eval_start'][:10]} → {r['eval_end'][:10]}",
            "",
        ]
        if name == "price":
            lines += [
                "> MAPE is not reported for price: VIC1 prices are ≤ 0 in ~23% of intervals, so "
                "percentage error is undefined. MASE and MAE are used instead.",
                "",
            ]
        hdr = "| Model | MASE | MAE | RMSE |" + (" sMAPE % |" if name == "demand" else "")
        hdr += " Pinball (80%) | 80% coverage | 95% coverage | 80% width |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for row in r["leaderboard_daily"]:
            cells = [
                f"**{row['model']}**" if row["model"] == HEADLINE else row["model"],
                _fmt(row["MASE"]),
                _fmt(row["MAE"], nd),
                _fmt(row["RMSE"], nd),
            ]
            if name == "demand":
                cells.append(_fmt(row.get("sMAPE"), 2))
            cells += [
                _fmt(row.get("pinball80"), nd),
                _fmt(row.get("coverage80"), 3),
                _fmt(row.get("coverage95"), 3),
                _fmt(row.get("width80"), nd),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if r.get("gate"):
            g = r["gate"]
            lines += [
                "",
                f"Gate: LightGBM demand MASE < {g['threshold']} → "
                f"**{'PASSED' if g['passed'] else 'NOT MET'}** ({r['headline']['MASE']:.3f}).",
            ]
        if r.get("leaderboard_weekly"):
            lines += [
                "",
                f"**All models incl. classical, on {r['n_classical_origins']} weekly origins** "
                "(classical models are refitted per origin on a trailing "
                f"{s.backtest.classical_train_days}-day window):",
                "",
                "| Model | MASE | MAE | RMSE |",
                "|---|---|---|---|",
            ]
            for row in r["leaderboard_weekly"]:
                lines.append(
                    f"| {row['model']} | {_fmt(row['MASE'])} | {_fmt(row['MAE'], nd)} | {_fmt(row['RMSE'], nd)} |"
                )
        lines.append("")
    lines += _extra_sections()
    out = "\n".join(lines) + "\n"
    (config.RESULTS / "leaderboard.md").write_text(out, encoding="utf-8", newline="\n")
    return out


def save(name: str, out: dict) -> None:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS / "backtest_results.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    summary["config"] = {
        "backtest": asdict(config.get_settings().backtest),
        "intervals": asdict(config.get_settings().intervals),
        "lightgbm": config.get_settings().lightgbm,
    }
    summary.setdefault("targets", {})[name] = out["result"]
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n")
    preds = out["preds"].copy()
    for c in preds.select_dtypes("float64"):
        preds[c] = preds[c].astype("float32")
    preds.to_parquet(config.RESULTS / f"backtest_predictions_{name}.parquet", index=False)
    write_leaderboard(summary["targets"])


def run(targets: list[str], *, smoke: bool = False, classical_on: bool = True) -> dict:
    df = pd.read_parquet(DATASET)
    outs = {}
    demand_all = None
    for name in targets:
        t0 = time.time()
        if name == "price" and TARGETS["price"]["extra_exog"]:
            if demand_all is None:
                path = config.RESULTS / "backtest_demand_alldaily.parquet"
                demand_all = pd.read_parquet(path) if path.exists() else None
            if demand_all is None:
                raise RuntimeError(
                    "price backtest needs the demand backtest first (run --target demand)"
                )
        out = backtest_target(
            name,
            df,
            smoke=smoke,
            classical_on=classical_on,
            demand_preds=demand_all if name == "price" else None,
        )
        if name == "demand":
            demand_all = out["all_daily"][["unique_id", "ds", "step", "model", "yhat"]]
            if not smoke:
                demand_all[demand_all["model"] == HEADLINE].to_parquet(
                    config.RESULTS / "backtest_demand_alldaily.parquet", index=False
                )
        if not smoke:
            save(name, out)
        r = out["result"]
        logger.info(
            "%s done in %.1f min — LightGBM MASE %.3f, MAE %.1f",
            name,
            (time.time() - t0) / 60,
            r["headline"]["MASE"],
            r["headline"]["MAE"],
        )
        outs[name] = out
    return outs


def validate(targets: list[str], classical_on: bool = False, label: str = "current config") -> dict:
    """Score the validation year (weekly origins). Lags, training window and features are chosen
    here BEFORE the evaluation window is scored; every run is appended to results/validation.json."""
    df = pd.read_parquet(DATASET)
    path = config.RESULTS / "validation.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    demand = None
    for name in targets:
        out = backtest_target(
            name, df, classical_on=classical_on, validation=True, demand_preds=demand
        )
        if name == "demand":
            demand = out["all_daily"][["unique_id", "ds", "step", "model", "yhat"]]
        r = out["result"]
        record["runs"].append(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "label": label,
                "target": name,
                "window": [r["eval_start"], r["eval_end"]],
                "n_origins": r["n_origins"],
                "headline": r["headline"],
                "leaderboard": r["leaderboard_daily"],
            }
        )
        logger.info(
            "VALIDATION %s [%s]: LightGBM MASE %.3f MAE %.1f",
            name,
            label,
            r["headline"]["MASE"],
            r["headline"]["MAE"],
        )
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8", newline="\n")
    return record


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--target", choices=["demand", "price", "all"], default="all")
    p.add_argument("--smoke", action="store_true", help="last ~6 weeks only; writes nothing (CI)")
    p.add_argument("--no-classical", action="store_true")
    p.add_argument(
        "--validation",
        action="store_true",
        help="score the 2023 validation year (design selection) -> results/validation.json",
    )
    a = p.parse_args()
    targets = ["demand", "price"] if a.target == "all" else [a.target]
    if a.validation:
        validate(targets, classical_on=not a.no_classical)
        return
    run(targets, smoke=a.smoke, classical_on=not a.no_classical)


if __name__ == "__main__":
    main()
