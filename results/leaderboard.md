# GridCast leaderboard

_Generated 2026-10-20 10:48 UTC by `python -m gridcast.evaluation.backtest`. All numbers are measured out of sample by walk-forward backtesting; nothing is hand-entered._

**Protocol.** One forecast per day issued at **12:00 AEST** for the whole of the **next day** (48 half-hours, 12.5–36 h ahead) — the day-ahead setting. Expanding-window refits every 28 days; weather inputs are the forecasts issued the day before (no hindsight weather). **MASE** = MAE ÷ in-sample MAE of a same-half-hour-yesterday forecast over the trailing 28 days (< 1 beats that naive rule). Intervals: rolling split-conformal with adaptive (ACI) correction; coverage measured, not assumed.

## Demand (MW) — 972 daily origins, 2024-02-01 → 2026-09-29

| Model | MASE | MAE | RMSE | sMAPE % | Pinball (80%) | 80% coverage | 95% coverage | 80% width |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.482 | 234 | 329 | 5.06 | 55 | 0.798 | 0.947 | 790 |
| Seasonal-naive (week) | 1.159 | 583 | 872 | 12.40 | 146 | 0.796 | 0.947 | 2,030 |
| Seasonal-naive (day) | 1.188 | 590 | 878 | 12.51 | — | — | — | — |
| Naive | 2.524 | 1,233 | 1,532 | 28.51 | — | — | — | — |

Gate: LightGBM demand MASE < 0.6 → **PASSED** (0.482).

**All models incl. classical, on 139 weekly origins** (classical models are refitted per origin on a trailing 56-day window):

| Model | MASE | MAE | RMSE |
|---|---|---|---|
| LightGBM | 0.479 | 232 | 321 |
| Seasonal-naive (day) | 0.926 | 464 | 715 |
| Seasonal-naive (week) | 1.150 | 578 | 862 |
| MSTL-ARIMAX | 1.285 | 645 | 862 |
| MSTL-ARIMA | 1.364 | 684 | 909 |
| MSTL-ETS | 1.469 | 735 | 952 |
| Naive | 2.310 | 1,143 | 1,404 |
| AutoETS | 2.542 | 1,283 | 1,575 |

## Price ($/MWh) — 972 daily origins, 2024-02-01 → 2026-09-29

> MAPE is not reported for price: VIC1 prices are ≤ 0 in ~23% of intervals, so percentage error is undefined. MASE and MAE are used instead.

| Model | MASE | MAE | RMSE | Pinball (80%) | 80% coverage | 95% coverage | 80% width |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.816 | 43.6 | 288.4 | 12.4 | 0.798 | 0.948 | 125.7 |
| Seasonal-naive (day) | 1.231 | 69.1 | 409.7 | — | — | — | — |
| Seasonal-naive (week) | 1.384 | 79.1 | 409.4 | 24.2 | 0.801 | 0.948 | 256.8 |
| Naive | 1.553 | 80.7 | 303.7 | — | — | — | — |

**All models incl. classical, on 139 weekly origins** (classical models are refitted per origin on a trailing 56-day window):

| Model | MASE | MAE | RMSE |
|---|---|---|---|
| LightGBM | 0.677 | 35.8 | 55.2 |
| MSTL-ARIMA | 1.008 | 57.0 | 114.6 |
| MSTL-ETS | 1.017 | 57.2 | 114.8 |
| MSTL-ARIMAX | 1.037 | 59.8 | 115.9 |
| Seasonal-naive (day) | 1.088 | 57.5 | 166.8 |
| Seasonal-naive (week) | 1.198 | 64.1 | 94.6 |
| Naive | 1.364 | 70.8 | 95.9 |
| AutoETS | 1.366 | 70.8 | 95.8 |

## Deep learning vs LightGBM (demand, 139 weekly origins)

NHITS (neuralforecast, CPU) trained on the trailing 365 days, refitted every 13 weeks, same day-before weather inputs.

| Model | MASE | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| NHITS | 1.081 | 530 | 686 |
| LightGBM | 0.479 | 232 | 321 |

## Price spikes: P(RRP > $300/MWh) tomorrow

46,656 half-hours, 426 spikes (base rate 0.91%). Decision thresholds chosen on the 2023 validation year (max F1). Accuracy is not reported: always predicting 'no spike' would score ~99%.

| Classifier | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| GridCast LightGBM classifier | 0.183 | 0.241 | 0.340 | 0.282 |
| Climatology (month × half-hour rate) | 0.058 | 0.222 | 0.009 | 0.018 |

On the 17,472 half-hours where AEMO's 12:33 PD run is available (34 spikes):

| Classifier | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| GridCast | 0.227 | 0.333 | 0.118 | 0.174 |
| AEMO PREDISPATCH (RRP > $300) | — | 0.032 | 0.735 | 0.061 |

## GridCast vs AEMO PREDISPATCH

See [aemo_benchmark.md](aemo_benchmark.md) for the full comparison.

