# GridCast — Architecture

```mermaid
flowchart LR
    subgraph Sources["Public sources (no API keys)"]
        A1[AEMO Price & Demand CSV<br/>monthly, VIC1]
        A2[AEMO dashboard feed<br/>same-day 5-min actuals]
        A3[AEMO MMSDM archive<br/>PREDISPATCH runs]
        A4[NEMweb Current<br/>latest PD run]
        W1[Open-Meteo<br/>ERA5 + Historical Forecast]
        W2[Open-Meteo Previous Runs<br/>forecast issued the day before]
        W3[Open-Meteo Forecast<br/>live 16-day]
    end
    subgraph Raw["data/raw (committed, partitioned parquet)"]
        R1[actuals/vic1_actuals_YYYY]
        R2[weather/weather_hourly_YYYY]
        R3[predispatch/aemo_predispatch_vic1_YYYYMM]
    end
    A1 & A2 --> R1
    W1 & W2 --> R2
    A3 & A4 --> R3
    R1 & R2 --> D[build_dataset<br/>30-min, AEST interval-ending,<br/>forecast-as-known weather]
    D --> F[features<br/>calendar · weather · lags ≥ 72]
    F --> M[models<br/>baselines · MSTL/ETS/ARIMAX · LightGBM · NHITS]
    M --> BT[walk-forward backtest<br/>+ adaptive conformal intervals]
    BT --> LB[results/leaderboard.md]
    BT & R3 --> AB[aemo_benchmark<br/>results/aemo_benchmark.md]
    M --> T[train → models/*.joblib]
    T & W3 & R3 --> P[predict → data/forecasts/]
    P & LB & AB --> APP[Streamlit dashboard<br/>reads committed files only]
```

## Layers

| Layer | Module(s) | Contract |
|---|---|---|
| Time handling | `utils/aest.py` | Every timestamp is naive **AEST (UTC+10, no DST), interval-ending**. Conversions only at the edges. Unit-tested (summer stamps must not shift). |
| Ingestion | `data/fetch_actuals.py`, `fetch_weather.py`, `fetch_aemo_forecasts.py` | Idempotent and incremental; raw data stored as yearly/monthly parquet partitions; only changed partitions are rewritten. |
| Dataset | `data/build_dataset.py` | One 30-minute table, Nixtla long format (`unique_id, ds, y…`), gap-checked; `weather_regime` records whether each interval's weather was a day-before forecast. |
| Features | `features/*` | Exogenous features (calendar + weather) only use information available at forecast time; target lags are all ≥ the 72-step horizon (direct day-ahead). Leakage is tested. |
| Models | `models/*` | Baselines define the MASE bar; classical models via statsforecast; LightGBM via mlforecast; NHITS optional. |
| Evaluation | `evaluation/*` | Walk-forward only, with a lookahead guard; design chosen on a 2023 validation year; evaluation window scored once. |
| Serving | `train.py`, `predict.py`, `app/` | Models trained offline and committed; the app never fetches data or retrains. |

## Data flow & refresh

| Job | When | Does | Commits |
|---|---|---|---|
| `ci.yml` | PR / push | ruff, pytest, rebuild dataset from committed raw data, smoke backtest | nothing |
| `daily_refresh.yml` | 02:35 UTC = **12:35 AEST** daily | fetch (incremental) → build → predict | `data/forecasts/` (+ live weather file) |
| `daily_refresh.yml` | 3rd of each month / on demand | as above + retrain | + `data/raw/` partitions and `models/` |

Why 12:35 AEST: AEMO's dashboard feed then holds today's actuals to 12:00, and AEMO's 12:30 PREDISPATCH
run is published — so the live forecast follows exactly the backtest protocol and is directly
comparable with AEMO's.

## Design decisions

- [ADR-001 — Direct day-ahead LightGBM (lags ≥ horizon)](docs/adr/ADR-001-direct-day-ahead.md)
- [ADR-002 — Own PREDISPATCH reader instead of nemseer](docs/adr/ADR-002-predispatch-reader.md)
- [ADR-003 — Adaptive conformal intervals](docs/adr/ADR-003-adaptive-conformal.md)
- [ADR-004 — 30-minute grain and day-before weather](docs/adr/ADR-004-grain-and-weather.md)
- Build errata: [CLAUDE.md §16](CLAUDE.md)

## Limitations

- **Single region, single weather point.** Melbourne weather stands in for all of Victoria; regional
  heat (e.g. northern Victoria) is invisible to the model.
- **Operational demand only.** Rooftop PV is not modelled explicitly; its growth is absorbed by the
  trend feature and day-before solar radiation.
- **Price is hard.** Spot price depends on bids, outages and interconnector limits that are not public at
  forecast time; GridCast's price model sees calendar and weather only. AEMO's PREDISPATCH has that
  information — the benchmark shows what it is worth.
- **Preliminary same-day actuals.** The dashboard feed's values can be revised; the monthly CSV supersedes them.
- **Weather history regimes.** Before Feb 2024 some training weather is reanalysis/near-analysis rather
  than a day-before forecast; the evaluation window is entirely day-before forecasts.
