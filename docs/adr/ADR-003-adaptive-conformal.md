# ADR-003 — Prediction intervals: rolling split-conformal with adaptive correction

**Status:** accepted · 2026-10-20

## Context
Point forecasts are not enough for electricity, least of all for price. Interval methods considered:
- quantile regression (separate LightGBM per quantile)
- mlforecast's built-in conformal
- MAPIE
- a hand-rolled rolling conformal

The backtest also needs intervals at every origin, built only from residuals that had been
**observed** by that origin.

## Decision
Use **horizon-stratified rolling split-conformal**:
- Each origin takes the empirical quantiles of the signed residuals at the same step.
- The residuals come from the previous 56 days of origins, counting only those at least two days
  old, so their targets are fully observed.
- Bounds are asymmetric, so the right-skewed price errors get a longer upper tail.

On top of that, apply **adaptive conformal inference** (Gibbs & Candès, 2021). After each origin
matures, the working miscoverage moves by γ·(α − observed miss rate).

Chosen on the 2023 validation year:

| Setting | Demand 80% / 95% coverage | Price 80% / 95% coverage |
|---|---|---|
| Rolling conformal, 56 d | 0.738 / 0.892 | 0.782 / 0.926 |
| + ACI γ = 0.02 | **0.783 / 0.942** | **0.803 / 0.952** |

## Consequences
- Distribution-free, with no per-quantile models. Coverage is **measured** out of sample and
  reported next to interval width and pinball loss.
- The live forecast continues from the backtest's final adaptive α.
- ACI trades a little width for coverage, and its guarantee is long-run average, not conditional.
