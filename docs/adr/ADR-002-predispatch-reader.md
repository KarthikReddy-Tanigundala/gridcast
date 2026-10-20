# ADR-002 — Read AEMO PREDISPATCH directly instead of via nemseer

**Status:** accepted · 2026-10-20

## Context
The spec named `nemseer` for AEMO's official forecasts. At build time nemseer 1.0.7 (Nov 2023) declares
`requires_python <3.12`. It still imports on 3.12, but `nemseer.get_tables(2025, 9, "PREDISPATCH")`
returns `[]`. From Aug 2024 AEMO renamed MMSDM archive files to
`PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01#YYYYMM010000.zip`, and nemseer's table discovery does not
recognise the new names.

## Decision
Write `data/fetch_aemo_forecasts.py`, a streaming reader for AEMO's MMS CSV format (`C`/`I`/`D` rows).
- Archive: `PREDISP_ALL_DATA` monthly zips for `PREDISPATCHREGIONSUM` (TOTALDEMAND) and
  `PREDISPATCHPRICE` (RRP). It keeps VIC1 and `INTERVENTION = 0`.
- Live: the newest `PUBLIC_PREDISPATCHIS_*.zip` from NEMweb `Reports/Current`.
- Run time = `LASTCHANGED`, the actual publication time. This is a more honest "as-known-at" than a
  nominal run label.

The comparator is the first run that covers all of tomorrow. Runs published before ~12:30 stop at
04:00 tomorrow, so in practice that is the run published ≈12:33 AEST. Days where that run was more
than an hour late are dropped.

## Consequences
- About 150 lines of code with no fragile dependency, unit-tested on an inline MMS sample.
- 12 months of VIC1 runs (17.5k runs, ~1M rows) are kept as monthly parquet partitions.
- If AEMO changes the format again, a single parser needs updating.
