# GridCast task runner (POSIX make). Windows users: see make.ps1.
PY ?= python

.PHONY: help setup setup-dl fetch fetch-aemo build backtest benchmark spikes dl train predict refresh app test lint all

help:       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-11s %s\n", $$1, $$2}'

setup:      ## Create venv + install pinned deps (editable)
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .

setup-dl:   ## Optional deep-learning extras (neuralforecast + CPU torch)
	$(PY) -m pip install -r requirements-dl.txt

fetch:      ## AEMO actuals + Open-Meteo weather -> data/raw (incremental)
	$(PY) -m gridcast.data.fetch_actuals && $(PY) -m gridcast.data.fetch_weather

fetch-aemo: ## AEMO PREDISPATCH forecasts (nemseer) -> data/raw/aemo_predispatch_vic1.parquet
	$(PY) -m gridcast.data.fetch_aemo_forecasts

build:      ## Join actuals + weather -> data/processed/vic1_30min.parquet
	$(PY) -m gridcast.data.build_dataset

backtest:   ## Walk-forward backtest -> results/leaderboard.md
	$(PY) -m gridcast.evaluation.backtest

benchmark:  ## GridCast vs AEMO PREDISPATCH -> results/aemo_benchmark_results.json
	$(PY) -m gridcast.evaluation.aemo_benchmark

spikes:     ## P(RRP > $$300) classifier backtest
	$(PY) -m gridcast.evaluation.spike_classification

dl:         ## Deep-learning comparison (needs setup-dl)
	$(PY) -m gridcast.models.deep_learning

train:      ## Fit final models -> models/*.joblib
	$(PY) -m gridcast.train

predict:    ## Next-day forecast -> data/forecasts/
	$(PY) -m gridcast.predict

refresh:    ## Daily job: fetch -> build -> predict
	$(MAKE) fetch build predict

app:        ## Run the Streamlit dashboard
	$(PY) -m streamlit run app/streamlit_app.py

test:       ## pytest
	$(PY) -m pytest

lint:       ## ruff check + format check
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

all: fetch build backtest benchmark train predict ## Full pipeline
