# ADR-004 — 30-minute grain and day-before weather as the only evaluation weather

**Status:** accepted · 2026-10-20

## Context
- AEMO settles 5-minutely since 1 Oct 2021, and AEMO's day-ahead PREDISPATCH is half-hourly.
- Weather is the main demand driver. If evaluation weather is better than what was available at
  forecast time, the backtest looks better than the live model will be.
- The spec suggested Open-Meteo's Historical Forecast API. That API stitches the first hours of
  successive model runs, so it is closer to an analysis than to a day-ahead forecast.

## Decision
- **Grain:** model at 30 minutes, interval-ending AEST. From Oct 2021 the 30-minute price and demand
  are the mean of six 5-minute values (unit-tested).
- **Weather:** use Open-Meteo **Previous Runs** `*_previous_day1` (the forecast issued the day before)
  wherever it exists. Fall back to Historical Forecast / ERA5 only for older training history.
- **Regime tracking:** `weather_regime` records the source for every interval.
- **Evaluation window:** availability was measured for Melbourne (temperature from Jan 2022, every
  other variable from Feb 2024), so the evaluation window starts **2024-02-01**. The backtest asserts
  that every evaluation interval is `dayahead`.

## Consequences
- The headline numbers are what a forecaster could actually have achieved.
- Training rows before Feb 2024 carry slightly better-than-forecast weather. That can only make the
  model trust weather a little too much, and the out-of-sample evaluation captures the effect.
