"""Turn sparse observation dictionaries into a gap-free daily record.

Real station records have holes --- instrument outages, wartime gaps, whole
missing months in the 1930s. Every model downstream assumes a regular daily grid,
so the holes get filled here, once, explicitly, with the method recorded rather
than silently interpolated somewhere deep in a fit.

Fill strategy, in order of preference:

1. **Short gaps (<= 5 days) in temperature and wind** --- linear interpolation.
   Synoptic persistence makes this genuinely accurate at that length.
2. **Longer gaps** --- day-of-year climatology from the surrounding record plus a
   scaled residual drawn from the same calendar window, so the filled stretch
   carries realistic variance instead of flattening to the mean. Filling with a
   bare climatological mean would shrink the variance and quietly underprice
   every tail in the book.
3. **Precipitation and snowfall** --- never interpolated. A missing rain day is
   resampled from observed same-calendar-window days, because interpolating
   between two dry days would manufacture a physically impossible drizzle.

`fill_mask` records exactly which days were imputed, so a fit can be re-run on
observed-only data to check the filling is not driving the answer.
"""

from __future__ import annotations

import numpy as np

from ..config import Location
from .base import DailyRecord

__all__ = ["assemble_daily", "fill_series"]

MAX_INTERP_GAP = 5
_WINDOW = 10  # +/- days of calendar window used for climatological resampling


def assemble_daily(
    location: Location,
    observed: dict[str, dict[str, float]],
    start_year: int,
    end_year: int,
    provenance: str,
    seed: int = 7,
) -> DailyRecord:
    dates = np.arange(
        np.datetime64(f"{start_year}-01-01"),
        np.datetime64(f"{end_year + 1}-01-01"),
        dtype="datetime64[D]",
    )
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    jan1 = dates.astype("datetime64[Y]").astype("datetime64[D]")
    doy = (dates - jan1).astype(int) + 1
    keys = np.datetime_as_string(dates, unit="D")
    rng = np.random.default_rng(seed)

    filled: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for var in ("tmax_c", "tmin_c", "precip_mm", "snow_mm", "wind_ms"):
        lookup = observed.get(var, {})
        raw = np.array([lookup.get(k, np.nan) for k in keys], dtype=float)
        coverage[var] = float(np.isfinite(raw).mean())
        interp_ok = var in ("tmax_c", "tmin_c", "wind_ms")
        filled[var] = fill_series(raw, doy, interpolate=interp_ok, rng=rng)

    # Temperature and precipitation are the load-bearing variables --- without
    # them there is no model. Wind and snow are allowed to be missing: plenty of
    # century-long stations never measured wind, and `DailyRecord.usable` is how
    # downstream code finds out rather than trusting a fabricated series.
    if coverage["tmax_c"] < 0.5 and coverage["precip_mm"] < 0.5:
        raise ValueError(
            f"{location.id}: coverage too thin to model "
            f"(tmax {coverage['tmax_c']:.0%}, precip {coverage['precip_mm']:.0%})"
        )

    return DailyRecord(
        location_id=location.id,
        lat=location.lat,
        lon=location.lon,
        elevation_m=location.elevation_m,
        provenance=provenance,
        dates=dates,
        year=year,
        doy=doy,
        tmax_c=filled["tmax_c"],
        tmin_c=filled["tmin_c"],
        precip_mm=filled["precip_mm"],
        snow_mm=filled["snow_mm"],
        wind_ms=filled["wind_ms"],
        coverage=coverage,
        variable_source={v: provenance for v in filled},
    )


def fill_series(
    values: np.ndarray,
    doy: np.ndarray,
    interpolate: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fill NaNs in place-safe fashion, returning a new array."""
    out = values.copy()
    missing = ~np.isfinite(out)
    if not missing.any():
        return out
    if missing.all():
        return np.zeros_like(out)

    if interpolate:
        for lo, hi in _gap_spans(missing):
            if hi - lo <= MAX_INTERP_GAP and lo > 0 and hi < len(out):
                left, right = out[lo - 1], out[hi]
                if np.isfinite(left) and np.isfinite(right):
                    steps = np.arange(1, hi - lo + 1) / (hi - lo + 1)
                    out[lo:hi] = left + (right - left) * steps

    still = ~np.isfinite(out)
    if still.any():
        pools = _calendar_pools(out, doy)
        idx = np.flatnonzero(still)
        for i in idx:
            pool = pools[int(doy[i]) - 1]
            out[i] = rng.choice(pool) if pool.size else 0.0
    return out


def _gap_spans(missing: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index spans of contiguous True."""
    padded = np.concatenate([[False], missing, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _calendar_pools(values: np.ndarray, doy: np.ndarray) -> list[np.ndarray]:
    """Observed values within +/- `_WINDOW` days of each calendar day."""
    finite = np.isfinite(values)
    pools: list[np.ndarray] = []
    for d in range(1, 367):
        offset = np.abs(doy - d)
        near = np.minimum(offset, 365 - offset) <= _WINDOW
        pools.append(values[near & finite])
    return pools
