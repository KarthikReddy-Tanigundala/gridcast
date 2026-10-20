"""Shared polite HTTP helpers (retries, User-Agent, on-disk caching)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CONTACT = os.getenv("GRIDCAST_CONTACT", "").strip()
USER_AGENT = "GridCast/0.1 (portfolio forecasting research" + (
    f"; {_CONTACT})" if _CONTACT else ")"
)
# AEMO's CDN rejects some non-browser agents; a browser-compatible prefix keeps the CSV path working.
HEADERS = {"User-Agent": f"Mozilla/5.0 (compatible; {USER_AGENT})"}


def get(
    url: str, *, params: dict | None = None, timeout: int = 120, retries: int = 4
) -> requests.Response:
    """GET with exponential backoff on network errors, 429 and 5xx."""
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == retries:
                raise
            logger.warning(
                "GET %s failed (%s) — retry %d/%d in %.0fs", url, exc, attempt, retries, delay
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def download(url: str, dest: Path, *, overwrite: bool = False) -> Path | None:
    """Download ``url`` to ``dest`` (cached). Returns None on a 404 (file not published yet)."""
    if dest.exists() and not overwrite:
        return dest
    r = get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest
