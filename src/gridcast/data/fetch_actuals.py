"""Fetch VIC1 actuals: operational demand (TOTALDEMAND), spot price (RRP) and rooftop PV.

Sources (CLAUDE.md §3, errata §16):

* ``aemo_csv`` (default) — AEMO's monthly *Aggregated Price and Demand* files, one small CSV per
  region-month (``PRICE_AND_DEMAND_YYYYMM_VIC1.csv``). They hold exactly the dispatch
  ``TOTALDEMAND`` and ``RRP`` for VIC1 (intervention pricing already excluded), are ~100x smaller
  than the MMSDM table zips, and the current month is refreshed daily — ideal for the daily job.
* ``nemosis`` — ``DISPATCHREGIONSUM`` + ``DISPATCHPRICE`` from the MMSDM archive, filtered to
  ``REGIONID == 'VIC1'`` and ``INTERVENTION == 0``. Kept as the resilient alternative and used to
  cross-validate the CSV path (``--validate YYYY-MM``).
* Rooftop PV (``ROOFTOP_PV_ACTUAL``, half-hourly, via nemosis) feeds the operational-vs-underlying
  demand analysis. It is optional: a failure there never blocks the core pipeline.

Output (committed): ``data/raw/vic1_actuals_30min.parquet`` with columns
``ds`` (naive AEST, interval-ending), ``demand_mw``, ``rrp``, ``n_sub_intervals``, ``settlement``.
Under 5-minute settlement (from 1 Oct 2021) the 30-minute price is the mean of six dispatch prices
and demand is the mean of six dispatch demands; before it, the file is already 30-minute trading data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from gridcast import config
from gridcast.data.http import download
from gridcast.utils.aest import five_min_to_interval, now_aest, parse_aemo_timestamp
from gridcast.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

CSV_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
CACHE = config.DATA_CACHE / "aemo_price_demand"
OUT = config.DATA_RAW / "vic1_actuals_30min.parquet"
ROOFTOP_OUT = config.DATA_RAW / "vic1_rooftop_pv_30min.parquet"


def month_starts(start: date, end_exclusive: date) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end_exclusive) - pd.Timedelta(days=1), freq="MS"))


def parse_price_demand_csv(raw: pd.DataFrame, region: str, minutes: int = 30) -> pd.DataFrame:
    """Normalise one AEMO price-and-demand file (5- or 30-minute) to the 30-minute grid."""
    df = raw[raw["REGION"] == region].copy()
    df["ds"] = parse_aemo_timestamp(df["SETTLEMENTDATE"])
    df = df.rename(columns={"TOTALDEMAND": "demand_mw", "RRP": "rrp"})[["ds", "demand_mw", "rrp"]]
    df = df.drop_duplicates("ds").sort_values("ds")
    step = df["ds"].diff().dropna().mode()
    if step.empty or step.iloc[0] >= pd.Timedelta(minutes=minutes):
        df["n_sub_intervals"] = 1
        df["settlement"] = "30min"
        return df.reset_index(drop=True)
    out = five_min_to_interval(df, ["demand_mw", "rrp"], minutes=minutes)
    out["settlement"] = "5min"
    return out


def fetch_month(month: pd.Timestamp, region: str, *, refresh: bool) -> pd.DataFrame | None:
    ym = month.strftime("%Y%m")
    path = download(CSV_URL.format(ym=ym, region=region), CACHE / f"{ym}_{region}.csv", overwrite=refresh)
    if path is None:
        logger.warning("No AEMO file for %s yet", ym)
        return None
    return parse_price_demand_csv(pd.read_csv(path), region)


def fetch_csv(start: date, end_exclusive: date, region: str, *, refresh_from: pd.Timestamp | None) -> pd.DataFrame:
    frames = []
    for m in month_starts(start, end_exclusive):
        # Re-download months that may still be changing (the current and the previous one).
        refresh = refresh_from is not None and m >= refresh_from
        frame = fetch_month(m, region, refresh=refresh)
        if frame is not None:
            frames.append(frame)
            logger.info("%s: %d intervals (%s)", m.strftime("%Y-%m"), len(frame), frame["settlement"].iloc[0])
    return pd.concat(frames, ignore_index=True)


def fetch_nemosis(start: date, end_exclusive: date, region: str) -> pd.DataFrame:
    """Alternative path: MMSDM tables via nemosis, INTERVENTION == 0, VIC1 only."""
    from nemosis import dynamic_data_compiler

    cache = config.DATA_CACHE / "nemosis"
    cache.mkdir(parents=True, exist_ok=True)
    s = pd.Timestamp(start).strftime("%Y/%m/%d %H:%M:%S")
    e = pd.Timestamp(end_exclusive).strftime("%Y/%m/%d %H:%M:%S")
    flt = {"filter_cols": ["REGIONID", "INTERVENTION"], "filter_values": ([region], [0])}
    demand = dynamic_data_compiler(s, e, "DISPATCHREGIONSUM", str(cache), select_columns=["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "TOTALDEMAND"], fformat="parquet", **flt)
    price = dynamic_data_compiler(s, e, "DISPATCHPRICE", str(cache), select_columns=["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"], fformat="parquet", **flt)
    df = demand.merge(price, on=["SETTLEMENTDATE", "REGIONID", "INTERVENTION"])
    df = df.rename(columns={"REGIONID": "REGION"})
    return parse_price_demand_csv(df, region)


def fetch_rooftop_pv(start: date, end_exclusive: date, region: str) -> pd.DataFrame | None:
    """Half-hourly rooftop PV actuals (MW). Prefers the ``MEASUREMENT`` estimate type."""
    try:
        from nemosis import dynamic_data_compiler

        cache = config.DATA_CACHE / "nemosis"
        cache.mkdir(parents=True, exist_ok=True)
        df = dynamic_data_compiler(
            pd.Timestamp(start).strftime("%Y/%m/%d %H:%M:%S"),
            pd.Timestamp(end_exclusive).strftime("%Y/%m/%d %H:%M:%S"),
            "ROOFTOP_PV_ACTUAL", str(cache), fformat="parquet",
            filter_cols=["REGIONID"], filter_values=([region],),
        )
    except Exception as exc:  # optional enrichment — never block the core pipeline
        logger.warning("Rooftop PV fetch skipped: %s", exc)
        return None
    df["ds"] = parse_aemo_timestamp(df["INTERVAL_DATETIME"])
    df["POWER"] = pd.to_numeric(df["POWER"], errors="coerce")
    # One value per interval: prefer satellite/measurement-based actuals over the daily estimate.
    rank = {"MEASUREMENT": 0, "SATELLITE": 1, "DAILY": 2}
    df["rank"] = df["TYPE"].map(rank).fillna(9) if "TYPE" in df else 0
    df = df.sort_values(["ds", "rank"]).drop_duplicates("ds")
    return df[["ds"]].assign(rooftop_pv_mw=df["POWER"].to_numpy()).reset_index(drop=True)


def validate_against_nemosis(month: str, region: str) -> pd.DataFrame:
    """Cross-check one month of the CSV path against nemosis (VIC1, INTERVENTION=0)."""
    m = pd.Timestamp(month + "-01")
    end = (m + pd.offsets.MonthBegin(1)).date()
    csv = fetch_csv(m.date(), end, region, refresh_from=None)
    nem = fetch_nemosis(m.date(), end, region)
    both = csv.merge(nem, on="ds", suffixes=("_csv", "_nemosis"))
    diff = {
        "intervals_compared": len(both),
        "demand_max_abs_diff": float((both["demand_mw_csv"] - both["demand_mw_nemosis"]).abs().max()),
        "rrp_max_abs_diff": float((both["rrp_csv"] - both["rrp_nemosis"]).abs().max()),
    }
    logger.info("CSV vs nemosis %s: %s", month, diff)
    return both


def run(source: str | None = None, *, full_refresh: bool = False, with_rooftop: bool = True) -> pd.DataFrame:
    s = config.get_settings()
    source = source or s.data.source
    today = now_aest()
    end_exclusive = max(s.data.history_end, (today + pd.Timedelta(days=1)).date())
    refresh_from = None if full_refresh else (today.normalize().replace(day=1) - pd.offsets.MonthBegin(1))
    if full_refresh:
        refresh_from = pd.Timestamp(s.data.history_start)

    if source == "nemosis":
        df = fetch_nemosis(s.data.history_start, end_exclusive, s.region)
    elif source == "aemo_csv":
        df = fetch_csv(s.data.history_start, end_exclusive, s.region, refresh_from=refresh_from)
    else:
        raise ValueError(f"unknown source {source!r}")

    df = df[df["ds"] > pd.Timestamp(s.data.history_start)].drop_duplicates("ds").sort_values("ds")
    # Only keep complete 30-minute intervals (a partial final interval would bias the mean).
    df = df[(df["settlement"] == "30min") | (df["n_sub_intervals"] == 6)].reset_index(drop=True)
    df["n_sub_intervals"] = df["n_sub_intervals"].astype("int8")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    logger.info("Wrote %s: %d rows, %s → %s", OUT.name, len(df), df["ds"].min(), df["ds"].max())

    if with_rooftop:
        start = s.data.history_start
        if ROOFTOP_OUT.exists() and not full_refresh:
            existing = pd.read_parquet(ROOFTOP_OUT)
            start = (existing["ds"].max() - pd.Timedelta(days=35)).date()
        else:
            existing = None
        pv = fetch_rooftop_pv(start, s.data.history_end, s.region)
        if pv is not None and len(pv):
            if existing is not None:
                pv = pd.concat([existing, pv]).drop_duplicates("ds", keep="last")
            pv.sort_values("ds").to_parquet(ROOFTOP_OUT, index=False)
            logger.info("Wrote %s: %d rows", ROOFTOP_OUT.name, len(pv))
    return df


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["aemo_csv", "nemosis"], default=None)
    p.add_argument("--full-refresh", action="store_true", help="re-download every month")
    p.add_argument("--no-rooftop", action="store_true")
    p.add_argument("--validate", metavar="YYYY-MM", help="cross-check one month CSV vs nemosis")
    a = p.parse_args()
    if a.validate:
        validate_against_nemosis(a.validate, config.get_settings().region)
        return
    run(a.source, full_refresh=a.full_refresh, with_rooftop=not a.no_rooftop)


if __name__ == "__main__":
    main()
