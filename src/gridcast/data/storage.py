"""Partitioned parquet storage for the committed raw extracts.

Raw data is split into yearly (or monthly) files so an incremental fetch rewrites only the
partition(s) that actually changed. That keeps git history small: a refresh touches the current
year's file (~0.4 MB) instead of re-committing the whole multi-year table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _key(ts: pd.Series, freq: str) -> pd.Series:
    return ts.dt.strftime("%Y" if freq == "Y" else "%Y%m")


def write_partitions(
    df: pd.DataFrame, directory: Path, prefix: str, ts_col: str = "ds", freq: str = "Y"
) -> list[Path]:
    """Write ``df`` as ``<directory>/<prefix>_<YYYY|YYYYMM>.parquet``; skip unchanged partitions."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for key, part in df.groupby(_key(df[ts_col], freq), sort=True):
        path = directory / f"{prefix}_{key}.parquet"
        part = part.sort_values(ts_col).reset_index(drop=True)
        if path.exists():
            old = pd.read_parquet(path)
            if old.shape == part.shape and old.equals(part):
                continue
        part.to_parquet(path, index=False)
        written.append(path)
    logger.info(
        "%s: wrote %d partition(s) %s", directory.name, len(written), [p.name for p in written]
    )
    return written


def read_partitions(directory: Path, prefix: str) -> pd.DataFrame | None:
    files = sorted(directory.glob(f"{prefix}_*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
