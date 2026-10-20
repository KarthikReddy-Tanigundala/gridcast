"""Calibrated prediction intervals by split-conformal prediction (§8.2).

Method — horizon-stratified, rolling split conformal:

1. For every past forecast we know its signed residual ``e = y - yhat`` at each step ``h``.
2. For a new origin, take the residuals of the previous ``calibration_days`` origins whose targets
   were **fully observed before this origin** (origins at least 2 days earlier — a forecast issued
   yesterday still has unobserved targets today), at the same step ``h``.
3. The interval is ``yhat + [q_{alpha/2}(e), q_{1-alpha/2}(e)]`` using empirical residual quantiles
   (asymmetric, so right-skewed price errors get a longer upper tail).

Distribution-free and adaptive to drift; coverage is then *measured* out of sample, never assumed.
The same function calibrates the live forecast from the most recent backtest residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_quantiles(
    resid: pd.DataFrame, levels: tuple[int, ...], alphas: dict[int, float] | None = None
) -> pd.DataFrame:
    """``resid``: ``step, e`` -> one row per step with ``lo{L}`` / ``hi{L}`` offsets.

    ``alphas`` overrides the miscoverage used for level ``L`` (adaptive conformal); default 1 - L/100.
    """
    out = {}
    g = resid.groupby("step")["e"]
    for lv in levels:
        a = (alphas or {}).get(lv, 1 - lv / 100)
        a = float(np.clip(a, 0.002, 0.998))
        out[f"lo{lv}"] = g.quantile(a / 2)
        out[f"hi{lv}"] = g.quantile(1 - a / 2)
    return pd.DataFrame(out)


def rolling_conformal(
    preds: pd.DataFrame,
    levels: tuple[int, ...],
    calibration_days: int,
    min_days: int = 14,
    gamma: float = 0.0,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Add ``lo{L}``/``hi{L}`` columns to backtest predictions of ONE model.

    ``preds``: ``origin, step, y, yhat`` with daily origins. Origins with fewer than ``min_days`` of
    usable calibration history get NaN bounds (excluded from coverage, reported as such).

    ``gamma > 0`` enables **adaptive conformal inference** (Gibbs & Candes, 2021): after each origin's
    targets are observed (2 days later), the working miscoverage moves by
    ``gamma * (target_alpha - observed_miss_rate)``, so persistent under-coverage widens later
    intervals. ``return_state`` also returns the final working alphas (used for live intervals).
    """
    preds = preds.copy()
    preds["e"] = preds["y"] - preds["yhat"]
    origins = [pd.Timestamp(o) for o in np.sort(preds["origin"].unique())]
    alphas = {lv: 1 - lv / 100 for lv in levels}
    pending: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    pieces = []
    for o in origins:
        # Apply ACI updates for every earlier origin whose targets are now fully observed.
        while pending and pending[0][0] + pd.Timedelta(days=2) <= o:
            _, done = pending.pop(0)
            for lv in levels:
                miss = 1 - ((done["y"] >= done[f"lo{lv}"]) & (done["y"] <= done[f"hi{lv}"])).mean()
                alphas[lv] += gamma * ((1 - lv / 100) - miss)
        lo_bound = o - pd.Timedelta(days=calibration_days + 1)
        cal = preds[(preds["origin"] <= o - pd.Timedelta(days=2)) & (preds["origin"] > lo_bound)]
        cur = preds[preds["origin"] == o]
        if cal["origin"].nunique() < min_days:
            q = pd.DataFrame(
                index=cur["step"].unique(),
                columns=[f"{p}{lv}" for lv in levels for p in ("lo", "hi")],
                dtype=float,
            )
        else:
            q = residual_quantiles(cal[["step", "e"]], levels, alphas if gamma else None)
        cur = cur.merge(q, left_on="step", right_index=True, how="left")
        for lv in levels:
            cur[f"lo{lv}"] = cur["yhat"] + cur[f"lo{lv}"]
            cur[f"hi{lv}"] = cur["yhat"] + cur[f"hi{lv}"]
        if gamma and cur[f"lo{levels[0]}"].notna().all():
            pending.append((o, cur))
        pieces.append(cur)
    out = pd.concat(pieces, ignore_index=True).drop(columns="e")
    return (out, alphas) if return_state else out


def live_intervals(
    forecast: pd.DataFrame,
    recent_preds: pd.DataFrame,
    levels: tuple[int, ...],
    calibration_days: int,
    alphas: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Intervals for a live forecast (``step, yhat``) from the latest backtest residuals, using the
    backtest's final adaptive alphas when given."""
    last = recent_preds["origin"].max()
    cal = recent_preds[recent_preds["origin"] > last - pd.Timedelta(days=calibration_days)]
    q = residual_quantiles(cal.assign(e=cal["y"] - cal["yhat"])[["step", "e"]], levels, alphas)
    out = forecast.merge(q, left_on="step", right_index=True, how="left")
    for lv in levels:
        out[f"lo{lv}"] = out["yhat"] + out[f"lo{lv}"]
        out[f"hi{lv}"] = out["yhat"] + out[f"hi{lv}"]
    return out
