"""GridCast vs AEMO's own official forecast (PREDISPATCH) — the differentiator (§9.4).

For every day in the benchmark window we compare, on the SAME 48 target intervals (tomorrow
00:30–24:00) and with the SAME metrics:

* **GridCast** — the walk-forward backtest forecast issued at 12:00 AEST (``backtest_predictions_*``).
* **AEMO PD**  — the first PREDISPATCH run that covers all of tomorrow, published ≈12:33 AEST.
  AEMO therefore has ~30 minutes more information, plus dispatch-level inputs (unit availability,
  bids, interconnector limits) that a public model cannot see.
* **Blend**    — the simple average of the two. Forecast combination is cheap and often beats both;
  reported to show whether GridCast carries information AEMO's forecast does not.

Also reported: AEMO's error as a function of lead time across *all* PD runs (forecast convergence),
with GridCast's day-ahead lead-time band marked for context.

Outputs: ``results/aemo_benchmark_results.json``, ``results/aemo_benchmark.md``.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from gridcast import config
from gridcast.data import fetch_actuals, fetch_aemo_forecasts
from gridcast.evaluation import metrics as M
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

AEMO_COL = {"demand": "aemo_demand_mw", "price": "aemo_rrp"}
UNIT = {"demand": "MW", "price": "$/MWh"}
OURS = "LightGBM"


def _scales(actuals: pd.DataFrame, origins: list[pd.Timestamp], col: str) -> pd.Series:
    s = config.get_settings().backtest
    y = actuals.set_index("ds")[col]
    n = s.mase_scale_days * 48
    out = {}
    for o in origins:
        hist = y.loc[:o].to_numpy()[-n:]
        out[o] = M.seasonal_naive_scale(hist, s.mase_season)
    return pd.Series(out)


def compare(target: str, preds: pd.DataFrame, comp: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    col = {"demand": "demand_mw", "price": "rrp"}[target]
    ours = preds[preds["model"] == OURS][["origin", "ds", "y", "yhat"]]
    snw = preds[preds["model"] == "Seasonal-naive (week)"][["origin", "ds", "yhat"]].rename(
        columns={"yhat": "snaive"}
    )
    aemo = comp[["origin", "ds", AEMO_COL[target]]].rename(columns={AEMO_COL[target]: "aemo"})
    df = ours.merge(aemo, on=["origin", "ds"]).merge(snw, on=["origin", "ds"], how="left")
    df = df.dropna(subset=["y", "yhat", "aemo"])
    df["blend"] = (df["yhat"] + df["aemo"]) / 2
    origins = sorted(df["origin"].unique())
    scales = _scales(actuals, [pd.Timestamp(o) for o in origins], col)
    models = {
        "GridCast (LightGBM)": "yhat",
        "AEMO PREDISPATCH": "aemo",
        "Blend (mean of both)": "blend",
        "Seasonal-naive (week)": "snaive",
    }
    rows = []
    for label, c in models.items():
        per_origin = df.groupby("origin").apply(
            lambda g, c=c: M.mae(g["y"], g[c]), include_groups=False
        )
        rows.append(
            {
                "model": label,
                "MASE": float((per_origin / scales.reindex(per_origin.index)).mean()),
                "MAE": M.mae(df["y"], df[c]),
                "MedAE": float((df["y"] - df[c]).abs().median()),
                "RMSE": M.rmse(df["y"], df[c]),
                "bias": float((df[c] - df["y"]).mean()),
            }
        )
    daily = df.groupby("origin").apply(
        lambda g: pd.Series({"ours": M.mae(g["y"], g["yhat"]), "aemo": M.mae(g["y"], g["aemo"])}),
        include_groups=False,
    )
    hour = (df["ds"] - pd.Timedelta(minutes=30)).dt.hour
    by_hour = df.groupby(hour).apply(
        lambda g: pd.Series(
            {
                "gridcast": M.mae(g["y"], g["yhat"]),
                "aemo": M.mae(g["y"], g["aemo"]),
                "blend": M.mae(g["y"], g["blend"]),
            }
        ),
        include_groups=False,
    )
    month = df["origin"].dt.to_period("M").astype(str)
    by_month = df.groupby(month).apply(
        lambda g: pd.Series(
            {"gridcast": M.mae(g["y"], g["yhat"]), "aemo": M.mae(g["y"], g["aemo"])}
        ),
        include_groups=False,
    )
    extra = {}
    if target == "price":
        err = (df["aemo"] - df["y"]).abs().sort_values(ascending=False)
        extra = {
            "pd_spike_share": float((df["aemo"] > 300).mean()),
            "actual_spike_share": float((df["y"] > 300).mean()),
            "aemo_top1pct_error_share": float(err.head(max(1, len(err) // 100)).sum() / err.sum()),
        }
    return {
        **extra,
        "target": target,
        "unit": UNIT[target],
        "n_days": len(origins),
        "window": [str(min(origins))[:10], str(max(origins))[:10]],
        "table": rows,
        "gridcast_win_rate": float((daily["ours"] < daily["aemo"]).mean()),
        "mae_by_hour": by_hour.round(2).reset_index(names="hour").to_dict("records"),
        "mae_by_month": by_month.round(2).reset_index(names="month").to_dict("records"),
    }


def convergence(pd_runs: pd.DataFrame, actuals: pd.DataFrame) -> list[dict]:
    """AEMO MAE by lead time (hours from publication to interval end), all runs."""
    df = pd_runs.merge(actuals[["ds", "demand_mw", "rrp"]], on="ds")
    df["lead_h"] = ((df["ds"] - df["run_time"]).dt.total_seconds() / 3600).round().astype(int)
    df = df[(df["lead_h"] >= 1) & (df["lead_h"] <= 40)]
    out = df.groupby("lead_h").apply(
        lambda g: pd.Series(
            {
                "demand_mae": M.mae(g["demand_mw"], g["aemo_demand_mw"]),
                "price_mae": M.mae(g["rrp"], g["aemo_rrp"]),
                "n": len(g),
            }
        ),
        include_groups=False,
    )
    return out.round(2).reset_index().to_dict("records")


def write_markdown(res: dict) -> str:
    lines = [
        "# GridCast vs AEMO PREDISPATCH",
        "",
        f"_Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m gridcast.evaluation.aemo_benchmark`._",
        "",
        "Same target intervals (tomorrow 00:30–24:00), same metrics. GridCast is issued at 12:00 AEST; "
        "the comparator is AEMO's first PREDISPATCH run covering all of tomorrow (published ≈12:33 AEST), "
        "so AEMO has ~30 min more data plus dispatch-level inputs (bids, outages, interconnector limits) "
        "that no public model sees. Days where that run was late (>1 h) are excluded.",
        "",
    ]
    for t in ("demand", "price"):
        if t not in res:
            continue
        r = res[t]
        nd = 0 if t == "demand" else 1
        lines += [
            f"## {t.capitalize()} ({r['unit']}) — {r['n_days']} days, {r['window'][0]} → {r['window'][1]}",
            "",
            "| Forecast | MASE | MAE | Median AE | RMSE | Bias |",
            "|---|---|---|---|---|---|",
        ]
        for row in r["table"]:
            lines.append(
                f"| {row['model']} | {row['MASE']:.3f} | {row['MAE']:,.{nd}f} | {row['MedAE']:,.{nd}f} | {row['RMSE']:,.{nd}f} | {row['bias']:+,.{nd}f} |"
            )
        lines += [
            "",
            f"GridCast beats AEMO on **{r['gridcast_win_rate']:.0%}** of days (daily MAE).",
            "",
        ]
        if t == "price" and "pd_spike_share" in r:
            lines += [
                f"> Reading the price result: AEMO's day-ahead PD shows RRP > $300 in {r['pd_spike_share']:.1%} of "
                f"half-hours vs {r['actual_spike_share']:.2%} actually; {r['aemo_top1pct_error_share']:.0%} of its absolute "
                "error comes from the top 1% of intervals. PREDISPATCH prices tomorrow on *today's* bids and reserve "
                "outlook — participants rebid before dispatch — so it is a dispatch scenario (high spike recall, low "
                "precision), not a statistical expectation. GridCast's L1 model targets the median, so it wins on MAE; "
                "compare the median-AE column and the spike table in the leaderboard for the fuller picture.",
                "",
            ]
    lines += [
        "## AEMO forecast convergence",
        "",
        "AEMO's error shrinks as the interval approaches — the market operator's edge is largest at "
        "short lead times. GridCast's day-ahead forecasts sit at 12.5–36 h lead.",
        "",
        "| Lead (h) | AEMO demand MAE (MW) | AEMO price MAE ($/MWh) |",
        "|---|---|---|",
    ]
    for row in res["convergence"]:
        if row["lead_h"] in (1, 2, 4, 6, 12, 18, 24, 30, 36):
            lines.append(
                f"| {row['lead_h']} | {row['demand_mae']:,.0f} | {row['price_mae']:,.1f} |"
            )
    out = "\n".join(lines) + "\n"
    (config.RESULTS / "aemo_benchmark.md").write_text(out, encoding="utf-8", newline="\n")
    return out


def run() -> dict:
    pd_runs = fetch_aemo_forecasts.load()
    if pd_runs is None:
        raise FileNotFoundError(
            "no PREDISPATCH data — run `python -m gridcast.data.fetch_aemo_forecasts`"
        )
    s = config.get_settings()
    actuals = fetch_actuals.load()
    actuals = actuals[actuals["ds"] <= pd.Timestamp(s.data.history_end)]
    comp = fetch_aemo_forecasts.comparator_runs(pd_runs, s.backtest.origin_hour)
    res: dict = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    for target in ("demand", "price"):
        path = config.RESULTS / f"backtest_predictions_{target}.parquet"
        if not path.exists():
            logger.warning("skip %s: run the backtest first", target)
            continue
        preds = pd.read_parquet(path)
        res[target] = compare(target, preds, comp, actuals)
        t = {r["model"]: r for r in res[target]["table"]}
        logger.info(
            "%s: GridCast MAE %.1f vs AEMO %.1f (blend %.1f), win rate %.0f%%",
            target,
            t["GridCast (LightGBM)"]["MAE"],
            t["AEMO PREDISPATCH"]["MAE"],
            t["Blend (mean of both)"]["MAE"],
            100 * res[target]["gridcast_win_rate"],
        )
    res["convergence"] = convergence(
        pd_runs[pd_runs["ds"] <= pd.Timestamp(s.data.history_end)], actuals
    )
    (config.RESULTS / "aemo_benchmark_results.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    write_markdown(res)
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
