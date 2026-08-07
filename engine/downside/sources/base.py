"""The daily-record container every source produces and every model consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

__all__ = ["DailyRecord", "WeatherSource", "VARIABLES"]

VARIABLES = ("tmax_c", "tmin_c", "precip_mm", "snow_mm", "wind_ms")


@dataclass(slots=True)
class DailyRecord:
    """A gap-free daily series for one coordinate.

    `year` and `doy` are carried explicitly rather than derived on demand because
    every design matrix in the engine is built from them and they get reused
    dozens of times per fit.
    """

    location_id: str
    lat: float
    lon: float
    elevation_m: float
    provenance: str
    dates: np.ndarray  # datetime64[D]
    year: np.ndarray  # int
    doy: np.ndarray  # int, 1-366
    tmax_c: np.ndarray
    tmin_c: np.ndarray
    precip_mm: np.ndarray
    snow_mm: np.ndarray
    wind_ms: np.ndarray
    #: Fraction of days actually observed, per variable, before gap filling.
    #: Carried on the record because a variable the station never measured must
    #: not be mistaken for a variable that measured zero --- see `usable`.
    coverage: dict = field(default_factory=dict)
    #: Per-variable origin, e.g. {"wind_ms": "surrogate"} when a field was
    #: substituted because the primary source did not carry it.
    variable_source: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_years(self) -> int:
        return int(self.year.max() - self.year.min() + 1)

    @property
    def decimal_year(self) -> np.ndarray:
        """Year plus fraction-of-year. The time axis for every trend term."""
        length = np.where(_is_leap(self.year), 366.0, 365.0)
        return self.year + (self.doy - 1) / length

    #: Below this share of observed days a variable is treated as not measured.
    MIN_COVERAGE = 0.20

    def get(self, variable: str) -> np.ndarray:
        if variable not in VARIABLES:
            raise KeyError(f"unknown variable {variable!r}; expected one of {VARIABLES}")
        return getattr(self, variable)

    def usable(self, variable: str) -> bool:
        """Whether this variable was actually measured often enough to model.

        Many GHCN stations record temperature and precipitation for a century but
        never record wind. Gap filling turns those absent days into numbers, and
        for a variable that is absent *everywhere* the filled series collapses to
        a constant --- Jackson Hole came back as exactly 0.00 m/s for all 36,525
        days. Left unchecked that prices a wind contract at zero premium with no
        warning, so anything that consumes a variable is expected to check here
        first.
        """
        return self.coverage.get(variable, 1.0) >= self.MIN_COVERAGE

    def subset_years(self, lo: int, hi: int) -> "DailyRecord":
        mask = (self.year >= lo) & (self.year <= hi)
        return DailyRecord(
            location_id=self.location_id,
            lat=self.lat,
            lon=self.lon,
            elevation_m=self.elevation_m,
            provenance=self.provenance,
            dates=self.dates[mask],
            year=self.year[mask],
            doy=self.doy[mask],
            tmax_c=self.tmax_c[mask],
            tmin_c=self.tmin_c[mask],
            precip_mm=self.precip_mm[mask],
            snow_mm=self.snow_mm[mask],
            wind_ms=self.wind_ms[mask],
        )

    def annual_mean(self, variable: str) -> tuple[np.ndarray, np.ndarray]:
        """Calendar-year means. Returns (years, values)."""
        vals = self.get(variable)
        years = np.unique(self.year)
        out = np.array([vals[self.year == y].mean() for y in years])
        return years.astype(float), out

    def annual_total(self, variable: str) -> tuple[np.ndarray, np.ndarray]:
        vals = self.get(variable)
        years = np.unique(self.year)
        out = np.array([vals[self.year == y].sum() for y in years])
        return years.astype(float), out


def _is_leap(year: np.ndarray) -> np.ndarray:
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


class WeatherSource(Protocol):
    """Every ingest adapter satisfies this, so swapping data is a one-line change."""

    name: str

    def fetch(self, location, start_year: int, end_year: int) -> DailyRecord: ...
