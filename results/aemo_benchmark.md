# GridCast vs AEMO PREDISPATCH

_Generated 2026-10-20 10:42 UTC by `python -m gridcast.evaluation.aemo_benchmark`._

Same target intervals (tomorrow 00:30–24:00), same metrics. GridCast is issued at 12:00 AEST; the comparator is AEMO's first PREDISPATCH run covering all of tomorrow (published ≈12:33 AEST), so AEMO has ~30 min more data plus dispatch-level inputs (bids, outages, interconnector limits) that no public model sees. Days where that run was late (>1 h) are excluded.

## Demand (MW) — 364 days, 2025-10-01 → 2026-09-29

| Forecast | MASE | MAE | Median AE | RMSE | Bias |
|---|---|---|---|---|---|
| GridCast (LightGBM) | 0.496 | 245 | 170 | 351 | -29 |
| AEMO PREDISPATCH | 0.415 | 203 | 139 | 289 | -66 |
| Blend (mean of both) | 0.391 | 192 | 131 | 275 | -47 |
| Seasonal-naive (week) | 1.159 | 585 | 378 | 877 | -5 |

GridCast beats AEMO on **37%** of days (daily MAE).

## Price ($/MWh) — 364 days, 2025-10-01 → 2026-09-29

| Forecast | MASE | MAE | Median AE | RMSE | Bias |
|---|---|---|---|---|---|
| GridCast (LightGBM) | 0.732 | 29.6 | 22.0 | 61.5 | -1.0 |
| AEMO PREDISPATCH | 2.003 | 85.7 | 16.9 | 870.2 | +53.8 |
| Blend (mean of both) | 1.195 | 50.6 | 17.1 | 434.9 | +26.4 |
| Seasonal-naive (week) | 1.315 | 54.0 | 41.0 | 97.6 | -0.1 |

GridCast beats AEMO on **56%** of days (daily MAE).

> Reading the price result: AEMO's day-ahead PD shows RRP > $300 in 4.5% of half-hours vs 0.19% actually; 56% of its absolute error comes from the top 1% of intervals. PREDISPATCH prices tomorrow on *today's* bids and reserve outlook — participants rebid before dispatch — so it is a dispatch scenario (high spike recall, low precision), not a statistical expectation. GridCast's L1 model targets the median, so it wins on MAE; compare the median-AE column and the spike table in the leaderboard for the fuller picture.

## AEMO forecast convergence

AEMO's error shrinks as the interval approaches — the market operator's edge is largest at short lead times. GridCast's day-ahead forecasts sit at 12.5–36 h lead.

| Lead (h) | AEMO demand MAE (MW) | AEMO price MAE ($/MWh) |
|---|---|---|
| 1 | 103 | 14.0 |
| 2 | 145 | 16.6 |
| 4 | 175 | 20.1 |
| 6 | 186 | 21.4 |
| 12 | 194 | 23.4 |
| 18 | 209 | 33.3 |
| 24 | 193 | 50.2 |
| 30 | 126 | 96.5 |
| 36 | 103 | 18.2 |
