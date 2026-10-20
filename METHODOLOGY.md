# GridCast — Methodology

## 1. The forecasting problem

**Day-ahead, VIC1, half-hourly.** At **12:00 AEST** each day, forecast every half-hour of **tomorrow**
(00:30–24:00, interval-ending): 48 values, 12.5–36 hours ahead, for

- **operational demand** (AEMO `TOTALDEMAND`, MW). This excludes rooftop PV, so midday demand is
  suppressed on sunny days. The all-time low in our data is **1,244 MW on 27 Dec 2025**.
- **spot price** (`RRP`, $/MWh). VIC1 prices are **≤ 0 in roughly 23% of half-hours** since 2023, so
  no log transforms and no MAPE.
- **price spikes**: P(RRP > $300/MWh) per half-hour.

The 12:00 cut-off matches the real day-ahead decision point. It is also when AEMO publishes the first
PREDISPATCH run that covers all of tomorrow (≈12:33), which makes the comparison with AEMO
like-for-like.

## 2. Data & time

- **Actuals.** AEMO Aggregated Price & Demand, VIC1, from 2019. From 1 Oct 2021, 5-minute values
  are averaged into 30-minute intervals (`closed='right', label='right'`).
- **Time zone.** All times are **AEST = UTC+10 fixed**. The NEM does not observe daylight saving;
  `Australia/Melbourne` would shift every summer interval by an hour. Tests pin this down.
- **Weather.** Open-Meteo for Melbourne (−37.81, 144.96). Each interval uses the forecast issued the
  day before (Previous Runs API) when available, otherwise near-analysis history. The evaluation
  window (from 2024-02-01) is **100% day-before forecasts**, and the backtest asserts it.

## 3. Models

| Family | Models | Notes |
|---|---|---|
| Baselines | Naive; Seasonal-naive (day); Seasonal-naive (week) | Define the bar. "Day" uses the most recent *observed* same half-hour. |
| Classical | AutoETS; MSTL-ETS; MSTL-ARIMA; **MSTL-ARIMAX** (temperature, HDD, CDD) | Refitted per origin on a trailing 56-day window (statsforecast). |
| ML (headline) | **LightGBM** via mlforecast | Direct day-ahead: all target lags ≥ 72 steps; calendar, VIC holidays, school terms, trend, day-before weather (temperature, HDD/CDD base 18 °C, lagged and max temperature, solar, cloud, humidity, wind, temp×hour). Demand: L2 objective. Price: L1 objective. |
| Deep learning (optional) | NHITS (neuralforecast) | Same exogenous inputs; weekly origins; CPU budget. |
| Classification | LightGBM classifier | P(RRP > $300), same direct design, monthly refits. |

## 4. Backtesting

- **Walk-forward.** There is one origin per day at 12:00 from 2024-02-01 to 2026-09-29 (972 days).
  LightGBM is refitted every 28 days on an expanding window. A guard (`assert_walk_forward`) fails
  the run if a model has seen data after an origin it is used for.
- **Validation before evaluation.** Every design choice was made on the **2023 validation year**:
  lags, trend feature, training window, price objective, demand-coupling feature, conformal window
  and ACI step. The full log is in `results/validation.json`, and the evaluation window was scored
  afterwards.
  - Honesty note: a first smoke run on the last six weeks of the evaluation window was looked at
    before this protocol was in place. Its preferred variant (a 2-year training window) was
    *rejected* by the validation year, and the validation choice was kept.
  - The 80% under-coverage seen on the first full evaluation run prompted the interval
    investigation, but the fix (ACI γ) was chosen on validation data only.
- **Classical models** cost too much to run daily, so they run on every 7th origin. A second table
  compares *all* models on exactly those origins.

## 5. Metrics

| Metric | Definition | Why |
|---|---|---|
| **MASE** (headline) | MAE ÷ in-sample MAE of the same-half-hour-yesterday forecast over the 28 days before the origin, averaged over origins | Scale-free; < 1 means better than the naive "same as yesterday" rule |
| MAE, RMSE | in MW or $/MWh | Interpretable; RMSE weights spikes |
| sMAPE | demand only | Familiar to stakeholders. **MAPE is not used for price**: it is undefined at zero and explodes near it |
| Pinball loss | averaged over the two quantiles bounding the 80% interval | Proper scoring rule for quantiles |
| Coverage, width | share of actuals inside the 80% / 95% intervals; mean width | Calibration *and* sharpness |
| PR-AUC, precision / recall / F1 | spike classifier | Class imbalance: accuracy would be ~99% for "never spike" |

## 6. Intervals

Rolling, horizon-stratified **split-conformal** works as follows. For origin *o* and step *h*, the
bounds are the forecast plus the empirical residual quantiles at step *h* from origins in
(*o*−57 d, *o*−2 d]. Those are the origins whose targets are already fully observed.

**Adaptive conformal inference** is applied on top: after each origin matures, the working
miscoverage moves by γ (α − miss rate), with γ = 0.02. Coverage is reported as measured. See
[ADR-003](docs/adr/ADR-003-adaptive-conformal.md).

## 7. AEMO benchmark

GridCast at 12:00 is compared with AEMO's ≈12:33 PREDISPATCH run on the same 48 target intervals
over the last 12 months, using the same metrics. A 50/50 blend tests whether GridCast carries
information that AEMO's forecast does not. AEMO's error by lead time across all runs ("convergence")
shows where the operator's information edge is largest.

AEMO is expected to win on price: it sees bids, outages and network constraints. Any result is
reported as measured.

## 8. Reproducing

```bash
python -m gridcast.data.fetch_actuals && python -m gridcast.data.fetch_weather
python -m gridcast.data.fetch_aemo_forecasts
python -m gridcast.data.build_dataset
python -m gridcast.evaluation.backtest --validation   # design selection (2023)
python -m gridcast.evaluation.backtest                # evaluation window -> results/leaderboard.md
python -m gridcast.evaluation.aemo_benchmark
python -m gridcast.evaluation.spike_classification
```

The committed raw extracts make `build_dataset` and everything after it reproducible offline.
LightGBM runs with `deterministic=true, force_row_wise=true`, so reruns on the same platform give
identical numbers.
