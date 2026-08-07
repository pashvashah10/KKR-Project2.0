"""NOAA NCEI Access API adapter --- GHCN-Daily station observations.

This is the primary production source named in the build plan (Part 5). It is
written against the real endpoint and cached to disk. In an environment with no
egress it raises `SourceUnavailable`, and the pipeline falls back to the
surrogate with the provenance flag set accordingly.

Station observations are preferred over reanalysis for pricing because the
contract settles on a gauge, not on a grid cell. Basis risk is the difference
between the two and it is priced explicitly in `pricing.py` --- so the ingest
records `station_distance_km` rather than quietly hiding it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ..config import Location
from .base import DailyRecord
from .gapfill import assemble_daily

__all__ = ["NoaaSource", "SourceUnavailable"]

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
DATA_TYPES = ("TMAX", "TMIN", "PRCP", "SNOW", "AWND")


class SourceUnavailable(RuntimeError):
    """Raised when the upstream service cannot be reached or returns nothing."""


class NoaaSource:
    name = "noaa-ghcnd"

    def __init__(self, cache_dir: Path | str = ".cache/noaa", timeout: int = 90, chunk_years: int = 20):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.chunk_years = chunk_years

    # ------------------------------------------------------------------
    def fetch(self, location: Location, start_year: int, end_year: int) -> DailyRecord:
        rows: list[dict] = []
        # NCEI throttles very long single requests; chunking also means a partial
        # cache survives an interrupted pull.
        for lo in range(start_year, end_year + 1, self.chunk_years):
            hi = min(lo + self.chunk_years - 1, end_year)
            rows.extend(self._fetch_chunk(location.station_id, lo, hi))

        if not rows:
            raise SourceUnavailable(f"NCEI returned no rows for {location.station_id}")

        return self._to_record(location, rows, start_year, end_year)

    # ------------------------------------------------------------------
    def _fetch_chunk(self, station: str, lo: int, hi: int) -> list[dict]:
        cache = self.cache_dir / f"{station}_{lo}_{hi}.json"
        if cache.exists():
            return json.loads(cache.read_text())

        query = urllib.parse.urlencode(
            {
                "dataset": "daily-summaries",
                "stations": station,
                "startDate": f"{lo}-01-01",
                "endDate": f"{hi}-12-31",
                "dataTypes": ",".join(DATA_TYPES),
                "format": "json",
                "units": "metric",
                "includeAttributes": "false",
            }
        )
        url = f"{BASE_URL}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "downside-weather-risk/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise SourceUnavailable(f"NCEI unreachable for {station} {lo}-{hi}: {exc}") from exc

        try:
            rows = json.loads(payload) if payload.strip() else []
        except json.JSONDecodeError as exc:
            raise SourceUnavailable(f"NCEI returned non-JSON for {station}: {exc}") from exc

        cache.write_text(json.dumps(rows))
        return rows

    # ------------------------------------------------------------------
    @staticmethod
    def _to_record(location: Location, rows: list[dict], start_year: int, end_year: int) -> DailyRecord:
        # With units=metric the service already returns degC and mm; values still
        # arrive as strings and missing days are simply absent from the payload.
        by_date: dict[str, dict[str, float]] = {}
        for row in rows:
            date = row.get("DATE")
            if not date:
                continue
            slot = by_date.setdefault(date, {})
            for key in DATA_TYPES:
                raw = row.get(key)
                if raw in (None, ""):
                    continue
                try:
                    slot[key] = float(raw)
                except ValueError:
                    continue

        observed = {
            "tmax_c": {d: v["TMAX"] for d, v in by_date.items() if "TMAX" in v},
            "tmin_c": {d: v["TMIN"] for d, v in by_date.items() if "TMIN" in v},
            "precip_mm": {d: v["PRCP"] for d, v in by_date.items() if "PRCP" in v},
            "snow_mm": {d: v["SNOW"] for d, v in by_date.items() if "SNOW" in v},
            # With units=metric the service returns AWND already in m/s (verified
            # against the raw payload: Boston mid-January reads 4.5-9.8, which is
            # daily *mean* wind, not tenths). This is the canonical definition of
            # `wind_ms` across the engine --- GHCN-Daily AWND is what a contract
            # would actually settle on.
            "wind_ms": {d: v["AWND"] for d, v in by_date.items() if "AWND" in v},
        }
        return assemble_daily(
            location=location,
            observed=observed,
            start_year=start_year,
            end_year=end_year,
            provenance="noaa-ghcnd",
        )


def coverage_report(record: DailyRecord) -> dict:
    """Share of days present per variable. Worth surfacing before trusting a fit."""
    out = {}
    for var in ("tmax_c", "tmin_c", "precip_mm", "snow_mm", "wind_ms"):
        vals = record.get(var)
        out[var] = float(np.isfinite(vals).mean())
    return out
