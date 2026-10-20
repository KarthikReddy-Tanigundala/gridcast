"""AEMO's own official forecasts: PREDISPATCH demand + price for VIC1 (the differentiator, §1.3).

Why not ``nemseer`` (errata §16): nemseer 1.0.7 (last release Nov 2023) discovers tables by scraping
the MMSDM archive's old ``PUBLIC_DVD_*`` file names. AEMO switched the archive to
``PUBLIC_ARCHIVE#<TABLE>#ALL#FILE01#<YYYYMM>010000.zip`` from Aug 2024, so ``nemseer.get_tables``
returns nothing for any month we need. This module reads the same AEMO files directly:

* archive: ``PREDISP_ALL_DATA`` monthly zips for ``PREDISPATCHREGIONSUM`` (``TOTALDEMAND``) and
  ``PREDISPATCHPRICE`` (``RRP``) — every 30-minute PD run, filtered to VIC1 / INTERVENTION = 0.
* live: the newest ``PUBLIC_PREDISPATCHIS_*.zip`` in NEMweb ``Reports/Current`` (daily refresh).

Run time = ``LASTCHANGED`` (when the run was published) — the honest "as-known-at" time. AEMO's
first PD run that covers the whole of the next calendar day is published at ~12:33 AEST; earlier runs
stop at 04:00 tomorrow. That 12:33 run is the comparator for GridCast's 12:00 forecast (AEMO gets
~30 minutes more information — stated in the results).

Output (committed): ``data/raw/aemo_predispatch_vic1.parquet`` with
``run_time, ds, aemo_demand_mw, aemo_rrp`` (naive AEST, ``ds`` interval-ending).
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from gridcast import config
from gridcast.data.http import download, get
from gridcast.data.storage import read_partitions, write_partitions
from gridcast.utils.aest import now_aest, parse_aemo_timestamp
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

ARCHIVE_URL = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m:02d}/"
    "MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_ARCHIVE%23{table}%23ALL%23FILE01%23{y}{m:02d}010000.zip"
)
CURRENT_DIR = "https://nemweb.com.au/Reports/Current/PredispatchIS_Reports/"
CACHE = config.DATA_CACHE / "predispatch"
OUT_DIR = config.DATA_RAW / "predispatch"  # aemo_predispatch_vic1_<YYYYMM>.parquet (by run month)
PREFIX = "aemo_predispatch_vic1"


def load() -> pd.DataFrame | None:
    return read_partitions(OUT_DIR, PREFIX)


# (archive table, record name inside the CSV, value column we keep, output column)
TABLES = {
    "demand": ("PREDISPATCHREGIONSUM", "REGION_SOLUTION", "TOTALDEMAND", "aemo_demand_mw"),
    "price": ("PREDISPATCHPRICE", "REGION_PRICES", "RRP", "aemo_rrp"),
}
KEEP = ["PREDISPATCHSEQNO", "REGIONID", "INTERVENTION", "LASTCHANGED", "DATETIME"]


def iter_mms_records(lines: Iterable[str], record: str, region: str) -> Iterator[dict[str, str]]:
    """Stream rows of one record type from an AEMO MMS CSV (``C``/``I``/``D`` row format).

    ``I`` rows carry the column names for the following ``D`` rows of that record. Rows for other
    regions are skipped by a cheap substring test before CSV parsing (files are 100+ MB).
    """
    header: list[str] | None = None
    tag = f",{record},"
    needle = f",{region},"
    for line in lines:
        if tag not in line[:40]:
            continue
        if line.startswith("I,"):
            header = next(csv.reader([line]))
        elif line.startswith("D,") and header is not None and needle in line:
            yield dict(zip(header, next(csv.reader([line])), strict=False))


def records_to_frame(rows: Iterable[dict[str, str]], value_col: str, out_col: str) -> pd.DataFrame:
    df = pd.DataFrame.from_records(list(rows), columns=[*KEEP, value_col])
    if df.empty:
        return pd.DataFrame(columns=["seqno", "run_time", "ds", out_col])
    df = df[pd.to_numeric(df["INTERVENTION"]) == 0]
    return pd.DataFrame(
        {
            "seqno": df["PREDISPATCHSEQNO"].astype("int64"),
            "run_time": parse_aemo_timestamp(df["LASTCHANGED"]),
            "ds": parse_aemo_timestamp(df["DATETIME"]),
            out_col: pd.to_numeric(df[value_col]).astype("float32"),
        }
    )


def read_zip(path: Path, record: str, value_col: str, out_col: str, region: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        frames = []
        for name in z.namelist():
            with z.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                frames.append(
                    records_to_frame(iter_mms_records(text, record, region), value_col, out_col)
                )
    return pd.concat(frames, ignore_index=True)


def _combine(demand: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    df = demand.merge(price.drop(columns="run_time"), on=["seqno", "ds"], how="outer")
    # A run's publish time is shared by all its rows; fill from either table.
    df["run_time"] = df.groupby("seqno")["run_time"].transform("min")
    return df.dropna(subset=["run_time"])


def fetch_month(month: pd.Timestamp, region: str) -> pd.DataFrame | None:
    parts = {}
    for key, (table, record, col, out) in TABLES.items():
        url = ARCHIVE_URL.format(y=month.year, m=month.month, table=table)
        path = download(url, CACHE / f"{table}_{month:%Y%m}.zip")
        if path is None:
            logger.warning("%s %s not in the MMSDM archive yet", table, f"{month:%Y-%m}")
            return None
        parts[key] = read_zip(path, record, col, out, region)
    df = _combine(parts["demand"], parts["price"])
    logger.info(
        "PREDISPATCH %s: %d runs, %d rows", f"{month:%Y-%m}", df["seqno"].nunique(), len(df)
    )
    return df


def fetch_latest_live(region: str) -> pd.DataFrame:
    """Newest PD run from NEMweb ``Reports/Current`` (kept ~2 days by AEMO)."""
    listing = get(CURRENT_DIR).text
    names = sorted(set(re.findall(r"PUBLIC_PREDISPATCHIS_\d+_\d+\.zip", listing)))
    if not names:
        raise RuntimeError("no PREDISPATCHIS reports listed on NEMweb")
    frames = []
    # The newest run, plus the newest run published at/after 12:30 today (the benchmark comparator).
    for name in _pick_live_runs(names):
        path = download(CURRENT_DIR + name, CACHE / "current" / name)
        d = read_zip(path, *TABLES["demand"][1:], region)
        p = read_zip(path, *TABLES["price"][1:], region)
        frames.append(_combine(d, p))
    return pd.concat(frames, ignore_index=True)


def _pick_live_runs(names: list[str]) -> list[str]:
    """Latest file, plus the first run published at/after 12:30 on each day listed."""
    picked = {names[-1]}
    by_day: dict[str, str] = {}
    for n in names:
        stamp = n.split("_")[2]  # YYYYMMDDHHMM = interval the run is for
        day, hhmm = stamp[:8], stamp[8:]
        if hhmm >= "1300" and day not in by_day:  # run for the 13:00 interval = published ~12:33
            by_day[day] = n
    picked.update(by_day.values())
    return sorted(picked)


def run(*, months: int | None = None, live: bool = True) -> pd.DataFrame:
    s = config.get_settings()
    months = months or s.aemo_benchmark.months
    last = pd.Timestamp(s.data.history_end) - pd.offsets.MonthBegin(1)
    frames = []
    existing = load()
    have: set = set()
    if existing is not None:
        frames.append(existing)
        # A month counts as archived only if it holds (nearly) every half-hourly run; months with just
        # a few live runs still get their archive file once AEMO publishes it.
        runs = existing.groupby(existing["run_time"].dt.to_period("M"))["seqno"].nunique()
        have = set(runs[runs > 1300].index)
    for m in pd.date_range(last - pd.DateOffset(months=months - 1), last, freq="MS"):
        if m.to_period("M") in have:
            continue
        df = fetch_month(m, s.region)
        if df is not None:
            frames.append(df)
    if live:
        try:
            frames.append(fetch_latest_live(s.region))
        except Exception as exc:  # live feed is best-effort; the archive is the benchmark
            logger.warning("Live PREDISPATCH fetch failed: %s", exc)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["seqno", "ds"], keep="last")
    df = df.sort_values(["run_time", "ds"]).reset_index(drop=True)
    write_partitions(df, OUT_DIR, PREFIX, ts_col="run_time", freq="M")
    logger.info(
        "PREDISPATCH: %d runs, %d rows, runs %s → %s",
        df["seqno"].nunique(),
        len(df),
        df["run_time"].min(),
        df["run_time"].max(),
    )
    return df


MAX_COMPARATOR_DELAY = pd.Timedelta(hours=1)


def comparator_runs(
    pd_df: pd.DataFrame,
    origin_hour: int = 12,
    origin_minute: int = 0,
    max_delay: pd.Timedelta = MAX_COMPARATOR_DELAY,
) -> pd.DataFrame:
    """For each day, the first PD run published after the GridCast origin that covers all of
    tomorrow. Returns ``origin, run_time, ds, aemo_demand_mw, aemo_rrp`` for tomorrow's intervals.
    Days whose first full-coverage run is published more than ``max_delay`` after the origin (a
    missing 12:30 run) are dropped rather than compared against a later, better-informed run."""
    df = pd_df.copy()
    df["origin"] = df["run_time"].dt.normalize() + pd.Timedelta(
        hours=origin_hour, minutes=origin_minute
    )
    tomorrow_end = df["origin"].dt.normalize() + pd.Timedelta(days=2)
    df = df[
        (df["run_time"] >= df["origin"])
        & (df["run_time"] <= df["origin"] + max_delay)
        & (df["ds"] > tomorrow_end - pd.Timedelta(days=1))
        & (df["ds"] <= tomorrow_end)
    ]
    cover = df.groupby("seqno")["ds"].transform("size")
    df = df[cover == 48]
    first = df.groupby("origin")["run_time"].transform("min")
    return df[df["run_time"] == first].drop(columns="seqno").reset_index(drop=True)


def now_month() -> pd.Timestamp:
    return now_aest().normalize().replace(day=1)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--no-live", action="store_true")
    a = p.parse_args()
    run(months=a.months, live=not a.no_live)


if __name__ == "__main__":
    main()
