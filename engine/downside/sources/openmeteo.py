"""Open-Meteo ERA5 archive adapter --- gridded reanalysis at exact coordinates.

Complements the station feed. ERA5 has no gaps and resolves the actual insured
coordinate rather than a gauge some distance away, which makes it the better
input for the *loss* regression. Station data stays the reference for anything
that will **settle** a contract.

ERA5 begins in 1940. For a full 100-year window the pipeline splices the station
record before 1940 onto reanalysis after it, or falls back to the surrogate.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import Location
from .base import DailyRecord
from .gapfill import assemble_daily
from .noaa import SourceUnavailable

__all__ = ["OpenMeteoSource", "ERA5_START_YEAR"]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_START_YEAR = 1940

FIELDS = {
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "precipitation_sum": "precip_mm",
    "snowfall_sum": "snow_mm",
    "wind_speed_10m_max": "wind_ms",
}


class OpenMeteoSource:
    name = "era5-openmeteo"

    def __init__(self, cache_dir: Path | str = ".cache/era5", timeout: int = 120, chunk_years: int = 25):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.chunk_years = chunk_years

    def fetch(self, location: Location, start_year: int, end_year: int) -> DailyRecord:
        lo_year = max(start_year, ERA5_START_YEAR)
        if lo_year > end_year:
            raise SourceUnavailable(f"ERA5 has no coverage before {ERA5_START_YEAR}")

        merged: dict[str, dict[str, float]] = {v: {} for v in FIELDS.values()}
        for lo in range(lo_year, end_year + 1, self.chunk_years):
            hi = min(lo + self.chunk_years - 1, end_year)
            payload = self._fetch_chunk(location, lo, hi)
            daily = payload.get("daily") or {}
            times = daily.get("time") or []
            for api_field, our_field in FIELDS.items():
                series = daily.get(api_field)
                if not series:
                    continue
                for date, value in zip(times, series):
                    if value is not None:
                        merged[our_field][date] = float(value)

        if not any(merged.values()):
            raise SourceUnavailable("Open-Meteo returned no usable series")

        # Open-Meteo reports snowfall in cm and wind in km/h by default.
        merged["snow_mm"] = {d: v * 10.0 for d, v in merged["snow_mm"].items()}
        merged["wind_ms"] = {d: v / 3.6 for d, v in merged["wind_ms"].items()}

        return assemble_daily(
            location=location,
            observed=merged,
            start_year=start_year,
            end_year=end_year,
            provenance="era5-openmeteo",
        )

    def _fetch_chunk(self, location: Location, lo: int, hi: int) -> dict:
        cache = self.cache_dir / f"{location.id}_{lo}_{hi}.json"
        if cache.exists():
            return json.loads(cache.read_text())

        query = urllib.parse.urlencode(
            {
                "latitude": f"{location.lat:.4f}",
                "longitude": f"{location.lon:.4f}",
                "start_date": f"{lo}-01-01",
                "end_date": f"{hi}-12-31",
                "daily": ",".join(FIELDS),
                "timezone": "UTC",
            }
        )
        req = urllib.request.Request(
            f"{BASE_URL}?{query}", headers={"User-Agent": "downside-weather-risk/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise SourceUnavailable(f"Open-Meteo unreachable for {location.id}: {exc}") from exc

        cache.write_text(json.dumps(payload))
        return payload
