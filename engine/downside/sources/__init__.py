"""Weather data adapters.

`resolve_source` implements the production preference order and degrades cleanly
rather than failing the run: NOAA station data first (it is what a parametric
contract settles on), ERA5 reanalysis second (gap-free, exact coordinates), the
calibrated surrogate last. Whatever it returns carries an honest `provenance`
string that travels all the way to the dashboard.
"""

from __future__ import annotations

import logging

from ..config import Location
from .base import VARIABLES, DailyRecord, WeatherSource
from .noaa import NoaaSource, SourceUnavailable
from .openmeteo import OpenMeteoSource
from .synthetic import SyntheticSource

__all__ = [
    "DailyRecord",
    "WeatherSource",
    "VARIABLES",
    "NoaaSource",
    "OpenMeteoSource",
    "SyntheticSource",
    "SourceUnavailable",
    "resolve_source",
    "load_record",
]

log = logging.getLogger(__name__)


def resolve_source(prefer: str = "auto") -> list[WeatherSource]:
    if prefer == "noaa":
        return [NoaaSource()]
    if prefer == "era5":
        return [OpenMeteoSource()]
    if prefer == "surrogate":
        return [SyntheticSource()]
    return [NoaaSource(), OpenMeteoSource(), SyntheticSource()]


def load_record(location: Location, start_year: int, end_year: int, prefer: str = "auto") -> DailyRecord:
    errors: list[str] = []
    for source in resolve_source(prefer):
        try:
            record = source.fetch(location, start_year, end_year)
        except (SourceUnavailable, ValueError, OSError) as exc:
            errors.append(f"{source.name}: {exc}")
            log.info("source %s unavailable for %s (%s)", source.name, location.id, exc)
            continue
        if errors:
            log.info("using %s for %s after %d fallback(s)", source.name, location.id, len(errors))
        return record
    raise RuntimeError(f"no source could supply {location.id}: " + "; ".join(errors))
