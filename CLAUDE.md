# CLAUDE.md — Project: GridCast
## Australian NEM Electricity Demand & Price Forecasting System (VIC1)
### Master Build Instructions for Claude Code (Opus 4.8)

---

## 0. PURPOSE & MISSION

You are building **GridCast** — a production-grade, fully-deployed, end-to-end time-series
forecasting system for the Australian National Electricity Market (NEM), focused on **Victorian
(VIC1) operational demand and spot price**, with rigorous backtesting, probabilistic forecasts,
scenario analysis, and — the standout feature — **benchmarking against AEMO's own official
forecasts**.

This is **supporting project 3 of 5** in Karthik Reddy Tanigundala's flagship data portfolio.
Karthik has **prior production experience in Australian/NZ retail energy analytics** (ReAmped
Energy, Electric Kiwi via EnergyTech Global), so the energy domain is a genuine personal fit —
this project deepens an existing strength rather than inventing a new one.

It targets Data Scientist / Data Analyst roles at Melbourne employers including AGL, AusNet,
Origin Energy, EnergyAustralia, the broader AEMO ecosystem, and renewable-asset / energy-trading
firms, plus general data-science roles that ask for time-series forecasting.

The project must read as the work of a **senior data scientist with 5–10 years of industry plus
insider energy-domain experience**. Every modelling choice, every domain fact encoded, and every
evaluation decision must reflect that standard.

**What separates this from the thousands of forecasting tutorial clones (build ALL of these):**
1. **MASE < 1** — beats a strong seasonal-naive baseline (the only metric that proves real skill)
2. **Benchmarking against AEMO's own official forecasts** via `nemseer` — almost no portfolio
   does this; it reads as genuine insider work
3. **Calibrated probabilistic forecasts** (prediction intervals with measured coverage), not just
   point forecasts — essential because electricity prices are spiky and volatile
4. **Rigorous walk-forward backtesting** with explicit leakage prevention
5. A **clean, config-driven, tested, reproducible pipeline** — not a single notebook
6. A **model leaderboard** comparing baseline → classical → ML → deep learning
7. A live, public, daily-refreshed dashboard

**The output of this project is:**
1. A fully deployed, publicly accessible forecasting dashboard (Streamlit Community Cloud)
2. A polished GitHub repository with a standout README (architecture diagram + forecast GIF +
   leaderboard + AEMO-benchmark chart)
3. A reproducible forecasting pipeline with committed model artifacts
4. A GitHub Actions workflow that refreshes data and forecasts daily
5. All assets demonstrating senior-level time-series engineering and energy-domain knowledge

---

## 1. CRITICAL CONSTRAINTS & DOMAIN FACTS (READ FIRST)

### 1.1 Free-tier only
The entire project runs free. Data (AEMO, Open-Meteo) is free and public. Deployment (Streamlit
Community Cloud) is free. No API keys are required for any data source.

### 1.2 Domain facts that MUST be encoded (insider credibility — getting these wrong is fatal)

| Fact | Implication for the build |
|---|---|
| **AEMO timestamps are AEST (UTC+10, fixed, NO daylight saving), interval-ENDING** | NEVER use "Australia/Sydney" — it silently shifts every summer interval by one hour. Use fixed UTC+10. SETTLEMENTDATE = interval end. This is the #1 silent-error source. |
| **5-minute settlement since 1 October 2021** | Dispatch price (`DISPATCHPRICE.RRP`) is 5-minutely. Pre-Oct-2021 trading price = average of six 5-min dispatch prices. Handle both regimes explicitly. |
| **Negative prices are common** | In 2024, negative prices were ~15% of all NEM prices (up from 3.5% in 2020), driven by excess solar. **NEVER apply a log transform without an offset.** Models must tolerate negatives. |
| **Market Price Cap (MPC) is very high and rising** | MPC rose to $20,300/MWh on 1 July 2025 (was $17,500 in Q2 2025), trending toward ~$22,800/MWh by 2027. Market floor = −$1,000/MWh. Price spikes reach the cap. |
| **Administered Price Cap (APC) = $600/MWh** | Set by AEMC rule (7 Dec 2023), triggered when the rolling 7-day Cumulative Price Threshold (CPT) is breached. |
| **Operational vs underlying demand** | AEMO operational demand EXCLUDES rooftop PV; underlying demand includes it. Rooftop PV pushes down midday operational demand. VIC hit an all-time-low minimum operational demand of 1,594 MW on 31 Dec 2023. Forecast OPERATIONAL demand (that's what `TOTALDEMAND` is). |
| **MAPE is undefined/unstable for price** | Price crosses zero and goes negative → MAPE breaks. Use MASE and MAE for price; flag this explicitly in the README. |

### 1.3 The differentiator, restated
`nemseer` exposes AEMO's *own* official forecasts (P5MIN, PREDISPATCH, PDPASA, STPASA, MTPASA).
GridCast must benchmark its forecasts against these and answer: *"Can a graduate's model match
the market operator's own short-horizon forecast?"* Report honestly where AEMO wins — an honest
negative result is itself a senior signal.

---

## 2. HIGH-LEVEL ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE A: DATA PIPELINE (offline, cached, re-runnable daily)          │
│                                                                        │
│  AEMO actuals          AEMO official forecasts      Weather           │
│  (nemosis)             (nemseer)                    (Open-Meteo)       │
│  DISPATCHPRICE         P5MIN, PREDISPATCH           ERA5 historical    │
│  DISPATCHREGIONSUM     (RRP + demand forecasts)     + Historical       │
│  ROOFTOP_PV_ACTUAL                                  Forecast API       │
│         │                     │                          │            │
│         └─────────────────────┼──────────────────────────┘            │
│                               ▼                                        │
│              [data pipeline] → typed parquet cache                    │
│              VIC1 filter · INTERVENTION=0 · AEST handling              │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE B: FEATURE ENGINEERING                                         │
│  calendar (hour, half-hour, DoW, month, VIC holidays, school terms)   │
│  lags (1, 48, 336) · rolling mean/std/max                            │
│  weather (temp, HDD/CDD base 18°C, lagged temp, solar, temp×hour)    │
│  demand→price coupling features                                       │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE C: MODELLING PROGRESSION (each must beat the previous)         │
│                                                                        │
│  1. Baselines: naive, seasonal-naive (→ defines MASE denominator)     │
│  2. Classical: AutoARIMA/SARIMAX, AutoETS (statsforecast)            │
│  3. ML workhorse: LightGBM (mlforecast) + lags + weather + calendar   │
│  4. Deep learning: NHITS, TFT, TiDE (neuralforecast)                 │
│                                                                        │
│  + Conformal / quantile prediction intervals (calibrated)            │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE D: BACKTESTING & EVALUATION                                   │
│  walk-forward (expanding/sliding), no leakage                        │
│  MASE (primary), MAE, RMSE, pinball loss + interval coverage          │
│  spike classification: precision/recall/F1, PR-AUC                    │
│  → model leaderboard + GridCast-vs-AEMO benchmark                     │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE E: DEPLOYMENT (Streamlit Community Cloud, free)               │
│  Tabs: actuals · forecast+intervals · backtest/leaderboard ·          │
│        GridCast-vs-AEMO · scenario what-if                            │
│  GitHub Actions cron: daily data refresh → regenerate forecasts       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. DATA SOURCES — EXACT SPECIFICATIONS

### 3.1 PRIMARY: NEMOSIS (historical actuals)

```bash
pip install nemosis
```

```python
from nemosis import dynamic_data_compiler

# 5-minute dispatch price (RRP) — the core price target
price = dynamic_data_compiler(
    "2017/01/01 00:00:00", "2025/01/01 00:00:00", "DISPATCHPRICE", "/path/to/cache"
)
# Filter: REGIONID == 'VIC1' AND INTERVENTION == 0

# Regional demand
demand = dynamic_data_compiler(
    "2017/01/01 00:00:00", "2025/01/01 00:00:00", "DISPATCHREGIONSUM", "/path/to/cache"
)
# TOTALDEMAND field, REGIONID == 'VIC1', INTERVENTION == 0

# Rooftop PV actuals (for underlying vs operational analysis)
rooftop = dynamic_data_compiler(
    "2017/01/01 00:00:00", "2025/01/01 00:00:00", "ROOFTOP_PV_ACTUAL", "/path/to/cache"
)
```

**Key tables:** `DISPATCHPRICE` (5-min price, field `RRP`), `TRADINGPRICE` (settlement price),
`DISPATCHREGIONSUM` (field `TOTALDEMAND`), `ROOFTOP_PV_ACTUAL`.
**Always filter:** `REGIONID == 'VIC1'` and `INTERVENTION == 0`.
**Caching:** nemosis downloads zips from NEMWeb, extracts CSV, caches as parquet. Use
`cache_compiler` for typed parquet caches. Subsequent queries read from cache (fast).
**Known gap:** nemosis bidding data has a gap Mar 2021–Jul 2024 — not needed for demand/price
core, but flag if ever used.

### 3.2 DIFFERENTIATOR: NEMSEER (AEMO's official forecasts)

```bash
pip install nemseer  # 1.0.7+
```

```python
from nemseer import compile_data, download_raw_data, generate_runtimes

# AEMO ahead-process forecasts: P5MIN (~1hr ahead), PREDISPATCH (to end of next trading day),
# PDPASA, STPASA, MTPASA
# Price RRP:  PREDISPATCH → PRICE table;  P5MIN → REGIONSOLUTION
# Demand:     PREDISPATCH → REGIONSUM;     P5MIN → REGIONSOLUTION
```

**Purpose:** benchmark GridCast against AEMO's own forecasts. Correct nominal→actual run times
(nemseer provides `generate_runtimes`). Returns pandas/xarray. This powers the "forecast
convergence" and GridCast-vs-AEMO error comparison — the project's headline differentiator.

### 3.3 FALLBACK: AEMO aggregated Price & Demand CSVs

```
# URL template
https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_REGION1.csv
# Example
https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_202505_VIC1.csv
```

Columns: `REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE`. 30-minute trading intervals.
Coverage from Dec 1998 (NSW/QLD/VIC/SA), TAS from May 2005.

**⚠️ GOTCHA:** AEMO is making URL-infrastructure changes during 2026 (NEMweb HTTP endpoint
decommission ~7 Apr 2026; base-URL migration ~30 Apr 2026). **Validate this path against a live
request at build time.** Use nemosis as the resilient primary; treat the aggregated CSV as a
fallback only.

### 3.4 AVOID: MMSDM bulk archive
`https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/...` — monthly zips are 8–10 GB
each; individual table zips can exceed 1 GB. **Do NOT use for the core build.** nemosis/nemseer
fetch only the needed tables.

### 3.5 WEATHER: Open-Meteo (free, no API key, CC-BY-4.0)

```
# Historical (training) — ERA5 reanalysis from 1940, hourly
https://archive-api.open-meteo.com/v1/archive

# Historical Forecast API (LEAKAGE-SAFE training, from ~2021) — the weather FORECAST that
# would have been available at the time, not hindsight reanalysis. PREFER THIS for training.
https://historical-forecast-api.open-meteo.com/v1/forecast

# Forecast (live dashboard) — up to 16 days
https://api.open-meteo.com/v1/forecast
```

**Melbourne coords:** lat ≈ −37.81, lon ≈ 144.96
**Variables:** `temperature_2m`, `relative_humidity_2m`, `cloud_cover`, `shortwave_radiation`
(solar — drives rooftop PV), `wind_speed_10m`.
**No auth** for non-commercial use. JSON / CSV / XLSX.

**Leakage note:** for training, use the Historical Forecast API (forecast-as-known) rather than
ERA5 reanalysis, so the model only sees weather information that would genuinely have been
available at forecast time.

### 3.6 BOM
No clean free historical API. Cite BOM Climate Data Online as the authoritative station source,
but use Open-Meteo (which ingests BOM model output) for the pipeline.

---

## 4. TECH STACK & LIBRARY VERSIONS (current ~mid-2026 — pin these)

```
# AEMO data
nemosis                   # latest PyPI
nemseer                   # 1.0.7+

# Nixtla forecasting suite (recommended core)
statsforecast==2.0.3      # AutoARIMA (20x faster than pmdarima), AutoETS
mlforecast                # 3.x — LightGBM workhorse + built-in conformal prediction
neuralforecast==3.1.*     # NHITS, TFT, TiDE (exogenous + quantile/probabilistic)
utilsforecast             # 0.2.x — shared evaluation + plotting

# Alternative / complement
darts                     # 3.x — unified fit/predict, backtesting, probabilistic, SHAP

# ML + data
lightgbm
scikit-learn
pandas
polars                    # fast dataframe ops
pyarrow                   # parquet
numpy
mapie                     # conformal prediction intervals
holidays                  # Victorian public holidays

# UI / deployment
streamlit
plotly                    # interactive charts with confidence bands

# Utilities
requests
python-dotenv
joblib
```

**Python version:** 3.12.
**⚠️ Pin exact versions at build time and re-resolve compatibility** — neuralforecast/PyTorch and
numba/numpy version conflicts are common. Validate the install before proceeding.

---

## 5. REPOSITORY STRUCTURE

```
gridcast/
├── README.md                       ← THE most important file — see Section 11
├── ARCHITECTURE.md                 ← architecture + data flow diagram + design decisions
├── METHODOLOGY.md                  ← forecasting methodology, backtesting, metrics rationale
├── LICENSE                         ← MIT
├── .gitignore                      ← .env, data/cache/ (large parquet), __pycache__/, *.ckpt
├── .env.example                    ← (no keys needed; document optional config)
├── requirements.txt                ← pinned versions (Section 4)
├── config.yaml                     ← all settings: region, date ranges, horizons, model params,
│                                      feature flags, MASE thresholds
│
├── src/
│   ├── __init__.py
│   ├── config.py                   ← loads config.yaml, exposes typed settings
│   ├── data/
│   │   ├── fetch_actuals.py        ← nemosis: price, demand, rooftop PV → parquet cache
│   │   ├── fetch_aemo_forecasts.py ← nemseer: P5MIN, PREDISPATCH → parquet cache
│   │   ├── fetch_weather.py        ← Open-Meteo historical + forecast → parquet
│   │   └── build_dataset.py        ← join actuals + weather, AEST handling, VIC1 filter
│   ├── features/
│   │   ├── calendar_features.py    ← hour, half-hour, DoW, month, VIC holidays, school terms
│   │   ├── lag_features.py         ← lags (1, 48, 336), rolling mean/std/max
│   │   ├── weather_features.py     ← HDD/CDD (base 18°C), lagged temp, solar, temp×hour
│   │   └── build_features.py       ← assemble the full feature matrix (leakage-safe)
│   ├── models/
│   │   ├── baselines.py            ← naive, seasonal-naive (MASE denominator)
│   │   ├── classical.py            ← statsforecast AutoARIMA/SARIMAX, AutoETS
│   │   ├── ml_lightgbm.py          ← mlforecast LightGBM (the workhorse)
│   │   ├── deep_learning.py        ← neuralforecast NHITS/TFT/TiDE (feature-flagged)
│   │   └── conformal.py            ← prediction intervals (mlforecast conformal / MAPIE)
│   ├── evaluation/
│   │   ├── backtest.py             ← walk-forward CV (expanding/sliding), no leakage
│   │   ├── metrics.py              ← MASE, MAE, RMSE, pinball loss, coverage
│   │   ├── spike_classification.py ← P(RRP > $300): precision/recall/F1, PR-AUC
│   │   └── aemo_benchmark.py       ← GridCast vs AEMO official forecast comparison
│   └── utils/
│       ├── aest.py                 ← AEST (UTC+10, no DST) timestamp handling — CRITICAL
│       └── logging_config.py
│
├── models/                         ← COMMITTED trained model artifacts (joblib / .ckpt)
│   ├── demand_lightgbm.joblib
│   └── ...
│
├── data/
│   ├── cache/                      ← gitignored parquet cache (nemosis/nemseer/weather)
│   └── forecasts/                  ← committed latest forecast outputs (for the live dashboard)
│
├── notebooks/                      ← EDA + experimentation (cleaned, narrative)
│   ├── 01_eda_demand.ipynb
│   ├── 02_eda_price.ipynb
│   └── 03_model_comparison.ipynb
│
├── app/
│   └── streamlit_app.py            ← the deployed dashboard (5 tabs)
│
├── results/
│   ├── leaderboard.md              ← model × MASE × MAE × pinball table
│   ├── backtest_results.json
│   └── aemo_benchmark_results.json
│
├── assets/
│   ├── demo.gif                    ← animated forecast walkthrough
│   ├── architecture.png            ← or Mermaid in README
│   └── screenshots/
│
├── tests/
│   ├── test_aest.py                ← timezone correctness (critical)
│   ├── test_features.py            ← no leakage, correct lags
│   ├── test_metrics.py             ← MASE/pinball correctness
│   └── test_backtest.py            ← walk-forward integrity
│
└── .github/
    └── workflows/
        ├── ci.yml                  ← lint + unit tests on PR
        └── daily_refresh.yml       ← cron: fetch data → regenerate forecasts → commit
```

---

## 6. DATA PIPELINE — DETAILED SPECS

### 6.1 `utils/aest.py` (CRITICAL — write and test this FIRST)
```python
"""
AEST timestamp handling. AEMO timestamps are UTC+10, FIXED year-round (NO daylight saving),
and interval-ENDING.
- Provide a fixed UTC+10 timezone constant (NEVER 'Australia/Sydney')
- Functions to parse SETTLEMENTDATE as interval-ending AEST
- Helper to align weather (which Open-Meteo returns in a requested timezone) to AEST
- Unit-tested: a summer timestamp must NOT shift by an hour
"""
```

### 6.2 `data/fetch_actuals.py`
```python
"""
Fetch historical actuals via nemosis.
- DISPATCHPRICE (RRP), DISPATCHREGIONSUM (TOTALDEMAND), ROOFTOP_PV_ACTUAL
- Filter REGIONID == 'VIC1', INTERVENTION == 0
- Cache as typed parquet (cache_compiler)
- Idempotent: skip already-cached ranges
- Default range: 2017-01-01 to present (configurable in config.yaml)
"""
```

### 6.3 `data/fetch_aemo_forecasts.py`
```python
"""
Fetch AEMO official forecasts via nemseer (the differentiator).
- P5MIN (REGIONSOLUTION) + PREDISPATCH (PRICE, REGIONSUM)
- Use generate_runtimes to correct nominal→actual run times
- Extract VIC1 price + demand forecasts aligned to their target intervals
- Cache as parquet
"""
```

### 6.4 `data/fetch_weather.py`
```python
"""
Fetch weather via Open-Meteo (no API key).
- Training: Historical Forecast API (leakage-safe — forecast-as-known), from ~2021
- Also fetch ERA5 archive for the pre-2021 period (document the regime difference)
- Live: Forecast API (16-day) for the dashboard
- Melbourne lat=-37.81, lon=144.96
- Variables: temperature_2m, relative_humidity_2m, cloud_cover, shortwave_radiation, wind_speed_10m
- Request in AEST; align to interval-ending convention
"""
```

### 6.5 `data/build_dataset.py`
```python
"""
Join actuals + weather into a single modelling dataset.
- Resample/align to a consistent interval (30-min for the core; document 5-min option)
- Handle the Oct-2021 5-minute settlement transition explicitly
- Output a tidy long-format frame (unique_id, ds, y, exogenous columns) compatible with Nixtla
"""
```

---

## 7. FEATURE ENGINEERING — DETAILED SPECS

```python
# calendar_features.py
#   hour, half_hour_of_day, day_of_week, month, is_weekend
#   is_vic_public_holiday (holidays package, prov='VIC')
#   is_school_term (Victorian school term dates)

# lag_features.py
#   lags: 1 (previous interval), 48 (1 day at 30-min), 336 (1 week at 30-min)
#   rolling mean/std/max over trailing windows
#   ALL lags computed leakage-safe (only past data)

# weather_features.py
#   temperature_2m (raw)
#   HDD = max(18 - temp, 0); CDD = max(temp - 18, 0)   # base 18°C
#   lagged temperature (thermal inertia — buildings respond slowly)
#   shortwave_radiation (solar — drives rooftop PV → lowers operational demand)
#   temperature × hour interaction

# build_features.py
#   Assemble full matrix; fit any scalers on TRAIN ONLY; document feature list
```

---

## 8. MODELLING — DETAILED SPECS

### 8.1 Progression (build in order; each must beat the previous)

```python
# 1. baselines.py — naive (last value), seasonal-naive (same half-hour yesterday / last week)
#    These DEFINE the MASE denominator. Everything must beat seasonal-naive (MASE < 1).

# 2. classical.py — statsforecast AutoARIMA / SARIMAX (temperature exogenous), AutoETS
#    Fast (AutoARIMA ~20x faster than pmdarima).

# 3. ml_lightgbm.py — mlforecast LightGBM (THE WORKHORSE)
#    Lags + rolling + calendar + weather exogenous. Best accuracy/effort trade-off.
#    Most defensible in interviews.

# 4. deep_learning.py — neuralforecast NHITS, TFT, TiDE (feature-flagged, default off)
#    All support exogenous + quantile/probabilistic outputs.
#    Frame as: "did DL beat well-tuned LightGBM?" An honest negative result is a senior signal.
```

### 8.2 Probabilistic forecasting (`conformal.py`) — CRITICAL for price
```python
"""
Point forecasts alone are weak for spiky prices. Produce CALIBRATED prediction intervals.
- Option A: quantile loss in neuralforecast/mlforecast (predict multiple quantiles)
- Option B: conformal prediction (mlforecast has built-in; MAPIE is the dedicated library)
- Report pinball/quantile loss AND interval coverage (does the 80% interval contain the actual
  80% of the time?)
- Conformal prediction + quantile regression averaging are the 2024-25 state of the art in
  electricity-price forecasting.
"""
```

### 8.3 Targets & horizons
- **Demand (core):** next-day 48 half-hourly intervals. Tractable, weather-driven.
- **Price (stretch):** same horizon, probabilistic.
- **Spike classification (stretch):** P(RRP > $300/MWh in next interval). Threshold well-supported
  — NEM prices above $300/MWh rose from 0.4% to 1.8% of all prices between 2020 and 2024.

---

## 9. BACKTESTING & EVALUATION — DETAILED SPECS

### 9.1 Backtesting (`backtest.py`)
```python
"""
Walk-forward (time-series) cross-validation. NEVER a random split.
- Expanding window OR sliding window; re-forecast at each origin
- Leakage prevention:
    * use Historical Forecast weather (forecast-as-known)
    * lag all features
    * fit scalers on train only
- Use Nixtla cross_validation utilities / Darts backtest
"""
```

### 9.2 Metrics (`metrics.py`)
```python
"""
- MASE (PRIMARY — scale-free, vs seasonal-naive; < 1 means real skill). THIS is the headline.
- MAE, RMSE
- Pinball loss + interval coverage (probabilistic)
- sMAPE (use cautiously)
- DO NOT use MAPE for price (undefined/unstable with zero and negative prices) — flag explicitly
"""
```

### 9.3 Spike classification (`spike_classification.py`)
```python
"""
P(RRP > $300/MWh). Class-imbalanced.
- Metrics: precision, recall, F1, PR-AUC (NOT plain accuracy)
"""
```

### 9.4 AEMO benchmark (`aemo_benchmark.py`) — THE differentiator
```python
"""
Compare GridCast forecasts against AEMO's official PREDISPATCH/P5MIN forecasts (from nemseer),
on the SAME target intervals, using the SAME metrics.
- Produce a comparison table + chart: GridCast vs AEMO error by horizon
- Report honestly where AEMO wins (expected at very short horizons)
- This answers: "Can a graduate match the market operator's own forecast?"
"""
```

### 9.5 Leaderboard (`results/leaderboard.md`)
A markdown table: model × MASE × MAE × pinball loss × coverage. Example format:

| Model | MASE | MAE (MW) | RMSE (MW) | Pinball | 80% Coverage |
|---|---|---|---|---|---|
| Seasonal-naive | 1.00 | — | — | — | — |
| AutoARIMA (SARIMAX) | 0.74 | — | — | — | — |
| LightGBM (mlforecast) | 0.52 | — | — | — | 0.81 |
| NHITS | 0.55 | — | — | — | 0.83 |

*(Claude Code: produce REAL measured numbers — illustrative format only.)*

---

## 10. DEPLOYMENT — STREAMLIT COMMUNITY CLOUD

### 10.1 Configuration
```
- Platform: Streamlit Community Cloud (free, one-click GitHub deploy)
- Free-tier resources are shared (~0.078–2 CPU cores, ~690 MB–2.7 GB RAM, up to ~50 GB storage)
- Apps SLEEP after ~12h of no traffic and wake on visit → warm before interviews
- Entry: app/streamlit_app.py
- Alternative: Hugging Face Spaces (Docker + Streamlit; 2 vCPU / 16 GB RAM; sleeps after 48h)
  → use if RAM/model size exceeds Streamlit's tighter limits
```

### 10.2 Model artifacts
```
- Train OFFLINE, serialize (joblib for LightGBM; neuralforecast .save() for DL), COMMIT to repo
- Load at inference. DO NOT retrain in the cloud app.
- Keep artifacts small.
```

### 10.3 Dashboard tabs (`app/streamlit_app.py`)
```python
"""
Tab 1 — Historical Actuals: VIC1 demand + price time series, date-range selector
Tab 2 — Forecast: next-day forecast with prediction-interval bands (Plotly confidence bands)
Tab 3 — Backtest & Leaderboard: model comparison table + backtest error charts
Tab 4 — GridCast vs AEMO: GridCast forecast overlaid on AEMO official forecast + error-by-horizon
Tab 5 — Scenario What-If: shift temperature ±X°C, show forecast demand response
- All charts Plotly (interactive, confidence bands)
- Pre-load committed forecasts from data/forecasts/ (do NOT recompute in-app)
"""
```

### 10.4 Daily refresh (`.github/workflows/daily_refresh.yml`)
```yaml
# Cron (daily):
# 1. Fetch latest AEMO actuals (nemosis) + AEMO forecasts (nemseer) + Open-Meteo forecast
# 2. Regenerate GridCast forecasts using committed models
# 3. Commit updated data/forecasts/ + results/ so the dashboard stays current
# Keep within free GitHub Actions minutes.
```

---

## 11. README.md — COMPLETE SPECIFICATION

```markdown
# ⚡ GridCast — Australian NEM Electricity Demand & Price Forecasting (VIC1)

> An end-to-end time-series forecasting system for the Australian National Electricity Market —
> forecasting Victorian operational demand and spot price with rigorous backtesting, calibrated
> probabilistic intervals, and benchmarking against AEMO's own official forecasts. Live demo below.

## 🎯 Live Demo
| Asset | Link |
|---|---|
| 📈 Live Forecasting Dashboard | [Open Dashboard](STREAMLIT_URL) |
| 🏆 Model Leaderboard | [View Results](results/leaderboard.md) |
| 📊 GridCast vs AEMO Benchmark | [View Comparison](results/aemo_benchmark_results.json) |

![GridCast Forecast Demo](assets/demo.gif)

## 📐 Architecture
[Mermaid diagram — data pipeline → features → modelling → backtest → deployment]

## 🎯 What Makes This Different
- ✅ MASE < 1 — beats a strong seasonal-naive baseline (proof of real forecasting skill)
- ✅ Benchmarked against AEMO's OWN official forecasts (P5MIN / PREDISPATCH via nemseer)
- ✅ Calibrated probabilistic forecasts (prediction intervals with measured coverage)
- ✅ Rigorous walk-forward backtesting with explicit leakage prevention
- ✅ Model leaderboard: baseline → classical → LightGBM → deep learning
- ✅ Daily auto-refresh via GitHub Actions — the dashboard stays live and current
- ✅ 100% free infrastructure (AEMO + Open-Meteo data, Streamlit Cloud)

## 🏆 Results
[the leaderboard.md table + the GridCast-vs-AEMO chart]
> Honest note: AEMO's own forecasts win at very short horizons (as expected — they have
> dispatch-level information); GridCast is competitive at [X]-hour-ahead and beyond.

## 🛠️ Tech Stack
[table: nemosis, nemseer, statsforecast, mlforecast, neuralforecast, LightGBM, MAPIE,
 Open-Meteo, Plotly, Streamlit]

## 🌏 Domain Notes (why this is credible)
[AEST/interval-ending handling, 5-min settlement, negative prices, operational vs underlying
 demand, market price cap — demonstrates insider knowledge]

## 🚀 Getting Started
[clone, pip install, fetch data, train, run dashboard locally]

## 🧪 Reproducing the Backtest
[python -m src.evaluation.backtest → regenerates leaderboard]

## 📚 Methodology
[link to METHODOLOGY.md]

## 🎓 About
Built by Karthik Reddy Tanigundala — RMIT Master of Data Science. Builds on prior production
experience in Australian/NZ retail energy analytics (ReAmped Energy, Electric Kiwi).
LinkedIn: linkedin.com/in/karthik-reddy-datascience
GitHub: github.com/KarthikReddy-Tanigundala
```

---

## 12. BUILD ORDER (Claude Code Execution Sequence)

Follow this exact sequence. Validate each phase before proceeding.

```
PHASE 1 — FOUNDATION
  1.1  Create full repo structure (Section 5)
  1.2  Write .gitignore, LICENSE, requirements.txt (pinned), config.yaml
  1.3  Write src/config.py + utils/logging_config.py
  1.4  Write AND TEST utils/aest.py FIRST (timezone correctness is critical)
       → Validate: test_aest.py passes; summer timestamp does NOT shift

PHASE 2 — DATA PIPELINE (validate the install works before modelling)
  2.1  Write fetch_actuals.py (nemosis) → cache VIC1 price + demand + rooftop PV
       → Validate: parquet cache populated, VIC1 filter correct, INTERVENTION=0
  2.2  Write fetch_aemo_forecasts.py (nemseer) → cache P5MIN + PREDISPATCH
       → Validate: AEMO forecasts retrieved, run times corrected
  2.3  Write fetch_weather.py (Open-Meteo) → historical forecast + ERA5 + live forecast
       → Validate: Melbourne weather retrieved, aligned to AEST
  2.4  Write build_dataset.py → joined tidy dataset (Nixtla long format)
       → Validate: handles 5-min transition, no NaN gaps in core range

PHASE 3 — FEATURES
  3.1  Write calendar_features.py, lag_features.py, weather_features.py
  3.2  Write build_features.py (leakage-safe)
       → Validate: test_features.py passes (no leakage, correct lags)

PHASE 4 — MODELLING
  4.1  Write baselines.py (naive, seasonal-naive) → establish MASE denominator
  4.2  Write classical.py (AutoARIMA/SARIMAX, AutoETS)
  4.3  Write ml_lightgbm.py (mlforecast — the workhorse)
       → Validate: demand LightGBM achieves MASE < 0.6 vs seasonal-naive
       → IF NOT: fix features/leakage BEFORE proceeding (do not continue with a broken core)
  4.4  Write conformal.py (calibrated prediction intervals)
  4.5  Write deep_learning.py (neuralforecast NHITS/TFT/TiDE — feature-flagged)

PHASE 5 — EVALUATION
  5.1  Write metrics.py (MASE, MAE, RMSE, pinball, coverage) + test_metrics.py
  5.2  Write backtest.py (walk-forward, leakage-safe) + test_backtest.py
  5.3  Write spike_classification.py (P(RRP>$300), PR-AUC)
  5.4  Write aemo_benchmark.py (GridCast vs AEMO official) — THE differentiator
  5.5  Generate results/leaderboard.md + backtest_results.json + aemo_benchmark_results.json
       → Validate: leaderboard shows the progression beating baseline; AEMO comparison complete

PHASE 6 — DEPLOYMENT
  6.1  Train final models, serialize to models/, commit artifacts
  6.2  Generate latest forecasts to data/forecasts/
  6.3  Write app/streamlit_app.py (5 tabs)
       → Validate locally: streamlit run app/streamlit_app.py, all tabs load
  6.4  Write .github/workflows/daily_refresh.yml

PHASE 7 — CI/CD + DOCS
  7.1  Write .github/workflows/ci.yml (lint + tests)
  7.2  Write ARCHITECTURE.md, METHODOLOGY.md
  7.3  Clean notebooks (EDA + model comparison, narrative)
  7.4  Write README.md (full spec, Section 11)

PHASE 8 — FINAL POLISH
  8.1  Record demo.gif (forecast with intervals + scenario what-if)
  8.2  Create architecture diagram (Mermaid in README or assets/architecture.png)
  8.3  Final verification: every link works, leaderboard current, all tests pass
  8.4  Deploy to Streamlit Cloud, confirm live URL, warm before demos
```

---

## 13. QUALITY STANDARDS — NON-NEGOTIABLE

### Code quality
- Type hints + docstrings on all functions; logging (not print)
- All settings (region, ranges, horizons, params, thresholds) in config.yaml — not hardcoded
- Config-driven, tested, reproducible — NOT a single notebook

### Forecasting rigour
- MASE is the headline metric — everything compared to seasonal-naive
- Real measured numbers in the leaderboard — never fabricated
- Walk-forward backtesting only; no random splits; explicit leakage prevention
- Probabilistic forecasts report calibrated coverage, not just point error
- Honest reporting where AEMO's forecasts beat GridCast

### Domain correctness (CRITICAL)
- AEST (UTC+10, no DST, interval-ending) handled correctly and unit-tested
- 5-minute settlement transition (Oct 2021) handled explicitly
- Negative prices tolerated (no naive log transform)
- MAPE NOT used for price (flagged in README)
- Operational demand (not underlying) forecast for `TOTALDEMAND`

### Security / reproducibility
- No secrets needed; .gitignore excludes large parquet cache + checkpoints
- requirements.txt pinned; install validated before modelling

---

## 14. INTERVIEW DEFENSIBILITY MAP

| Component | Likely question | Answer (verbatim) |
|---|---|---|
| MASE | "Why MASE, not MAPE or RMSE?" | "MASE is scale-free and benchmarked against a seasonal-naive forecast — under 1 means I'm genuinely beating the naive baseline. MAPE is useless for electricity price because prices cross zero and go negative, which makes percentage error undefined. RMSE doesn't tell you whether you've beaten a trivial model." |
| AEMO benchmark | "How do you know your forecast is any good?" | "I benchmarked it against AEMO's own official pre-dispatch and 5-minute forecasts using the nemseer package. So I'm not just comparing to a naive baseline — I'm comparing to the market operator's production forecast. They win at very short horizons because they have dispatch-level information, but GridCast is competitive further out. I report that honestly." |
| Probabilistic | "Why prediction intervals?" | "Electricity prices are extremely spiky — they can hit the $20,000 market cap or go negative from excess solar. A point forecast is almost meaningless there. I produce calibrated intervals using conformal prediction and report coverage — does my 80% interval actually contain the outcome 80% of the time? That's what a trader or risk team actually needs." |
| Leakage | "How did you avoid lookahead bias?" | "Three ways: I used Open-Meteo's Historical Forecast API so the model only sees the weather forecast that would have been available, not hindsight reanalysis; all features are lagged; and I fit scalers on the training fold only. Backtesting is walk-forward — I re-forecast at each origin, never a random split." |
| Timezone | "Any gotchas with the data?" | "The big one is timezone. AEMO timestamps are AEST — UTC+10, fixed year-round with no daylight saving — and they're interval-ending. If you naively use 'Australia/Sydney', every summer interval shifts by an hour and your weather joins silently break. I handled it with a fixed UTC+10 zone and unit-tested it." |
| Negative prices | "Anything unusual about the target?" | "Negative prices — about 15% of NEM prices in 2024, up from 3.5% in 2020, driven by excess rooftop and utility solar. That rules out a naive log transform, and it's why I forecast operational demand, which excludes rooftop PV, separately from the behind-the-meter solar effect." |
| LightGBM vs DL | "Why LightGBM over deep learning?" | "LightGBM with good lag and weather features gave the best accuracy-to-effort ratio and it's fast and interpretable. I did try NHITS and TFT from neuralforecast — on a single series they didn't clearly beat a well-tuned LightGBM, which is a common and honest finding. I'd reach for deep learning if I were forecasting many series jointly." |

---

## 15. FINAL DELIVERABLES CHECKLIST

- [ ] GitHub repo: public, clean history
- [ ] README: architecture diagram + forecast GIF + leaderboard + AEMO benchmark + live link in first 25 lines
- [ ] Live Streamlit dashboard: 5 tabs working, forecasts with intervals display
- [ ] Demand model: MASE < 0.6 vs seasonal-naive (validated)
- [ ] nemseer AEMO-forecast benchmark: complete, honest comparison committed
- [ ] Calibrated prediction intervals: coverage reported
- [ ] Walk-forward backtest: leakage-safe, reproducible
- [ ] Model leaderboard: baseline → classical → LightGBM → DL with REAL numbers
- [ ] Price + spike classification (stretch): PR-AUC reported
- [ ] AEST handling: unit-tested (test_aest.py)
- [ ] Committed model artifacts + latest forecasts
- [ ] GitHub Actions: CI tests + daily refresh workflow
- [ ] METHODOLOGY.md + ARCHITECTURE.md written
- [ ] Streamlit app warmed before any interview
- [ ] Interview defensibility: Karthik can speak 60+ seconds on every component

---

*CLAUDE.md v1.0 — GridCast Project*
*Generated for Karthik Reddy Tanigundala — RMIT Master of Data Science 2026*
*Target: Melbourne Data Scientist / Data Analyst / energy-sector roles, AUD $90k–$160k band*
*Supporting Project 3 of 5 — builds on real ReAmped Energy / Electric Kiwi domain experience*

---

## 16. ERRATA & BUILD DECISIONS (v1.1, 2026-10-20) — override §1–15 where they conflict

Each item was verified against live sources or measured on the 2023 validation year at build time.
Each is also an interview talking point.

| # | Spec said | Reality / decision | Why |
|---|---|---|---|
| 1 | Benchmark via `nemseer` 1.0.7 | **Own PREDISPATCH reader** (`data/fetch_aemo_forecasts.py`) | nemseer (last release Nov 2023, `requires_python <3.12`) scrapes the old `PUBLIC_DVD_*` archive names. AEMO renamed MMSDM files to `PUBLIC_ARCHIVE#<TABLE>#ALL#FILE01#…` from Aug 2024, so `nemseer.get_tables()` returns `[]` for every month we need. Reading the same AEMO files directly is ~150 lines. |
| 2 | nemosis primary for actuals | **AEMO monthly Price & Demand CSV primary**; nemosis kept as alternative + cross-check (`fetch_actuals --validate YYYY-MM`) | Same TOTALDEMAND/RRP for VIC1, ~100x smaller than MMSDM table zips; the current-month file updates daily. A same-day top-up from AEMO's dashboard feed gives actuals to 12:00. |
| 3 | NEMweb URL migration in 2026 | Verified 2026-10-20: MMSDM archive live to `MMSDM_2026_09`; CSV URL returns 200 | No change needed; fetchers log and degrade gracefully on 404. |
| 4 | Library versions (§4) | statsforecast 2.1.1, mlforecast 1.1.0, lightgbm 4.7.0, neuralforecast 3.3.0 (optional), pandas 2.3.3, numpy 2.5.4, Python 3.12 | Pinned in `requirements.txt`. |
| 5 | History 2017–2025 | **History 2019-01 → today; evaluation window 2024-02-01 → 2026-09-29 (972 daily origins)** | Open-Meteo's Previous-Runs API (the forecast issued the day before) covers temperature from Jan 2022 and every other variable from Feb 2024 only. So from Feb 2024, every weather input in the headline backtest is a genuine day-before forecast (asserted in code). |
| 6 | "Use the Historical Forecast API (leakage-safe)" | **Previous-Runs `*_previous_day1`** for day-ahead weather; Historical Forecast / ERA5 only as fallback for training history | The Historical Forecast API stitches the first hours of each model run, so it is close to an analysis, not a day-ahead forecast. `weather_regime` records which source fed each interval. |
| 7 | 5-min core / handle 2021 transition | **30-minute modelling grain**; post-Oct-2021 prices = mean of six 5-min RRPs (unit-tested) | Day-ahead is a half-hourly decision problem; same grain as AEMO PD. |
| 8 | Lags 1, 48, 336 | **Lags 72, 96, 144, 336, 672 (all ≥ the 72-step horizon)** | With lag 1 the recursive forecast fed on its own errors (lag1 + lag2 = 97% of gain) and degraded from ~230 to ~560 MW across tomorrow. Lags ≥ horizon make it a **direct** day-ahead model. Validation 2023: MASE 0.564 → 0.517. |
| 9 | — | **`trend_years` feature, all-history training** | Rooftop-PV growth shifts the midday level every year. Validation 2023: trend 0.472 vs 2-year window 0.504 vs neither 0.517. |
| 10 | SARIMAX with AutoARIMA (season 48) | **MSTL (48 + 336) + ARIMA with temperature/HDD/CDD regressors** ("MSTL-ARIMAX") | Seasonal AutoARIMA at lag 48 measured 175 s per fit (vs 2 s); MSTL deseasonalising is the standard multi-seasonal treatment. |
| 11 | Demand → price coupling features | **Tested, off** | Price model with our demand forecast as input: validation MASE 0.705 vs 0.705 without. Simpler model kept; one flag re-enables it. |
| 12 | Conformal intervals | **Rolling split-conformal + adaptive conformal inference (γ = 0.02)** | Plain rolling conformal under-covered on 2023 validation (80% → 0.738); ACI → 0.783 demand / 0.803 price. |
| 13 | AEMO comparator unspecified | **First PD run covering all of tomorrow, published ≈12:33 AEST** | Earlier runs stop at 04:00 tomorrow. AEMO gets ~30 min more data; stated in every results table. |
| 14 | Commit raw data daily | **Partitioned raw parquet; daily job commits forecasts only, raw snapshot monthly** | Re-committing ~16 MB of parquet per day would bloat git history by GBs per year. |
| 15 | Record-low demand 1,594 MW (31 Dec 2023) | Superseded: **1,244 MW on 27 Dec 2025 (13:00)** in our data | Rooftop PV keeps pushing midday operational demand down. |
| 16 | MPC trajectory | Only the current MPC ($20,300) is kept in config, for chart context | Avoids hardcoding unverified future values. |

**Protocol discipline.** Every design choice above (lags, trend, training window, price objective, interval method) was chosen
on the **2023 validation year** (`backtest --validation`, logged in `results/validation.json`) and the evaluation window was scored once.
