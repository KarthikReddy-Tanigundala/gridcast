# ADR-001 — Direct day-ahead LightGBM (all target lags ≥ the horizon)

**Status:** accepted · 2026-10-20

## Context
The forecast is issued at 12:00 AEST for tomorrow (steps 25–72 from the origin). The spec's lags
(1, 48, 336) make an mlforecast LightGBM **recursive**: to predict step 72 it feeds its own predictions
back as lag 1. On the first trial the model put 97% of its gain on lag 1/lag 2, and the error grew
from ~230 MW at the first steps to ~560 MW by tomorrow afternoon.

## Decision
Use only lags ≥ 72 steps (72, 96, 144, 336, 672) with rolling statistics anchored on the shortest
lag, plus a same-half-hour 7-day seasonal mean anchored two days back. With every lag at least as
long as the horizon, the 72-step forecast never consumes a prediction: it is a **direct** model,
trained once per refit and valid for every step. Add a `trend_years` feature so trees can separate
the rooftop-PV era from earlier years.

Chosen on the **2023 validation year** (weekly origins):

| Variant | MASE | MAE (MW) |
|---|---|---|
| Recursive lags (1, 2, 48, 96, 336), all history | 0.564 | 242 |
| Direct lags, all history | 0.517 | 222 |
| Direct lags, last 2 years only | 0.508 | 219 |
| **Direct lags + trend, all history** | **0.472** | **204** |
| Direct lags + trend, last 2 years | 0.504 | 219 |

## Consequences
- One model serves the whole horizon; the backtest and live forecast are cheap.
- The model cannot use this morning's demand directly (the freshest lag is 36 h old). The weather
  forecast carries most of that information; a per-horizon direct ensemble (`max_horizon`) is the
  natural next step if intraday forecasts are ever needed.
