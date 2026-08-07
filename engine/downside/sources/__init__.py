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


def load_record(
    location: Location,
    start_year: int,
    end_year: int,
    prefer: str = "auto",
    patch_missing: bool = True,
) -> DailyRecord:
    """Fetch the best available record, falling back through the source chain.

    With `patch_missing`, any variable the winning source did not actually
    measure is refilled from the surrogate rather than left as gap-filled
    constants. Jackson Hole's station, for instance, carries a century of
    temperature and precipitation and no wind at all; without this the wind
    series is 36,525 identical zeros and a wind contract prices at zero premium.
    Every patched field is recorded in `variable_source` so the substitution
    travels with the data.
    """
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
        if patch_missing and source.name != SyntheticSource.name:
            record = _patch_unmeasured(location, record, start_year, end_year)
        return record
    raise RuntimeError(f"no source could supply {location.id}: " + "; ".join(errors))


def _patch_unmeasured(
    location: Location, record: DailyRecord, start_year: int, end_year: int
) -> DailyRecord:
    missing = [v for v in VARIABLES if not record.usable(v)]
    if not missing:
        return record

    log.info("patching unmeasured variables for %s: %s", location.id, ", ".join(missing))
    donor = SyntheticSource().fetch(location, start_year, end_year)
    for var in missing:
        setattr(record, var, donor.get(var))
        record.variable_source[var] = f"{SyntheticSource.name} (not measured at station)"
    return record
