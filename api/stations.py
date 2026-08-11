"""Match a customer's coordinates to the GHCN-Daily station a contract can settle on.

Until this module existed, a customer site was fitted with `station_id=""`, so the
NOAA adapter asked for an empty station, got nothing, and fell through to ERA5
reanalysis. Worse, `station_distance_km` stayed at zero, and

    pricing.py   load_basis = BASIS_PER_KM * location.station_distance_km * theo

so **basis risk was charged at exactly $0.00 on every customer contract**. Basis
is not a rounding error in this business: a gauge eighteen miles from the venue
is a materially different bet from one in the car park, and the difference is
where a parametric book quietly loses money. Pricing it requires knowing the
distance, which requires knowing the station, which is what this does.

Two files from NCEI make it tractable:

| File | Contents |
|---|---|
| `ghcnd-stations.txt`  | `ID lat lon elev NAME` for ~128k stations |
| `ghcnd-inventory.txt` | `ID lat lon ELEMENT firstyear lastyear`, one row per element |

The inventory is the important one. It says *which variables each station
actually carries and over what span*, so candidates are filtered on record length
and element coverage before a single observation is fetched. That matters after
Jackson Hole: a station present in the list is not a station that measured wind.

### Nearest is the wrong objective

The obvious implementation returns the closest station. It is wrong here. This
model fits a century — a seasonal cycle, a warming response regressed on global
forcing, a block bootstrap and tail fits. A complete 1900-2025 record 40 km away
is worth far more to that fit than a patchy 20-year record 5 km away, and the
5 km station cannot settle a 2050 contract at all if it stopped reporting in
2003.

So candidates are scored on **record length and completeness, discounted by
distance and by elevation difference** (`_score`), and distance enters as a soft
penalty rather than a hard sort key.

### Elevation is the second half of basis risk

Horizontal distance alone understates how different two places can be. A gauge
9 km from a ski resort but 800 m below it is measuring a different climate, not
a nearby one: lapse rate alone puts roughly 5 °C between them, the rain/snow
line sits between them for much of the season, and a snow contract settled on
the valley gauge will pay when the mountain did not need it and stay silent when
it did. That is not basis risk at the margin; it is the wrong variable.

`elev_delta_m` therefore enters the denominator alongside distance at
0.05 km-equivalent per metre — 100 m of vertical costs the same as 5 km of
horizontal — and a delta past `ELEVATION_WARNING_M` both halves the score and
raises `elevation_warning`, which the configure page shows as a microclimate
notice rather than swallowing.

Where nothing qualifies at all the function returns `None` and the caller falls
back to ERA5 — a reanalysis grid cell has its own basis risk, and saying so on
screen is better than inventing a station.
"""

from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "StationMatch", "nearest_station", "load_index", "build_cache",
    "DataIngestionError", "StationIndexUnavailable", "CACHE_PATH",
]

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "ghcn"
#: The pre-processed bundle. Present => zero network dependency at boot.
CACHE_PATH = CACHE_DIR / "stations_parsed.npz"

STATIONS_FILE = "ghcnd-stations.txt"
INVENTORY_FILE = "ghcnd-inventory.txt"

#: Elements a site fit needs before it can be priced at all.
CORE_ELEMENTS = ("TMAX", "TMIN", "PRCP")
#: Elements worth having but never required --- most century stations lack AWND.
OPTIONAL_ELEMENTS = ("SNOW", "SNWD", "AWND")
TRACKED = CORE_ELEMENTS + OPTIONAL_ELEMENTS

#: A station that stopped reporting cannot settle a contract written today.
DEFAULT_MIN_LAST_YEAR = 2018
DEFAULT_MIN_YEARS = 60
DEFAULT_MAX_KM = 120.0

#: Vertical metres are converted to horizontal kilometres at this rate before
#: entering the distance penalty: 100 m of elevation difference costs the same
#: as 5 km of horizontal separation.
ELEVATION_KM_PER_M = 0.05
#: Past this, the station is measuring a different climate rather than a nearby
#: one. Score is halved and the match is flagged for the UI.
ELEVATION_WARNING_M = 300.0
ELEVATION_PENALTY = 0.5

#: Floor on the denominator so a station in the car park does not divide by zero.
MIN_DENOMINATOR_KM = 0.5

#: Live download budget. Short on purpose: a cold cache must not turn into a
#: three-minute page load. Past this we fall back rather than wait.
FETCH_TIMEOUT_S = 10

EARTH_RADIUS_KM = 6371.0088


class DataIngestionError(RuntimeError):
    """The station index is neither cached nor reachable.

    Carries the fix in its message rather than making the reader go looking:
    the offline build script exists precisely for this case.
    """


#: Retained name --- `DataIngestionError` is what this always meant.
StationIndexUnavailable = DataIngestionError


@dataclass(frozen=True, slots=True)
class StationMatch:
    station_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    distance_km: float
    first_year: int
    last_year: int
    elements: tuple[str, ...]
    score: float
    #: Signed: station elevation minus venue elevation, in metres. Negative
    #: means the gauge is below the venue, which is the common and dangerous
    #: case in mountain terrain. `None` when the venue's elevation was not
    #: supplied --- which is not the same as "they match".
    elevation_delta_m: float | None = None
    elevation_warning: bool = False

    @property
    def years(self) -> int:
        return self.last_year - self.first_year + 1

    @property
    def microclimate_note(self) -> str | None:
        """Plain-language warning for the configure page, or `None`."""
        if not self.elevation_warning or self.elevation_delta_m is None:
            return None
        delta = self.elevation_delta_m
        direction = "below" if delta < 0 else "above"
        # Environmental lapse rate, ~6.5 C/km, as the concrete consequence.
        lapse = abs(delta) * 0.0065
        return (
            f"This gauge sits about {abs(delta):,.0f} m {direction} your venue. "
            f"That is roughly {lapse:.1f} °C of lapse-rate difference before any "
            f"other effect, so it is measuring a different microclimate rather than "
            f"a nearby one. Expect a materially wider basis than the "
            f"{self.distance_km:.0f} km separation suggests, and treat snow, freeze "
            f"and rain/snow-line triggers with particular care."
        )

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "elevation_m": round(self.elevation_m, 1),
            "distance_km": round(self.distance_km, 1),
            "elevation_delta_m": (
                round(self.elevation_delta_m, 1) if self.elevation_delta_m is not None else None
            ),
            "elevation_warning": self.elevation_warning,
            "first_year": self.first_year,
            "last_year": self.last_year,
            "years": self.years,
            "elements": list(self.elements),
            "score": round(self.score, 3),
        }


# ----------------------------------------------------------------------
# Download and cache
# ----------------------------------------------------------------------


def _fetch(name: str, timeout: int = FETCH_TIMEOUT_S) -> Path:
    """Download an inventory file once. Subsequent calls read the cache.

    The timeout is the *connect and first-byte* budget, not the whole transfer:
    these files are 11 MB and 35 MB, and a slow-but-working link should be
    allowed to finish rather than be cut off mid-stream.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = f"{BASE_URL}/{name}"
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "downside-weather-risk/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise DataIngestionError(
            f"Could not fetch {url}: {exc}\n"
            f"No pre-built station bundle was found at {CACHE_PATH} either.\n"
            f"Build one on a machine with network access and commit it:\n"
            f"    python -m api.scripts.build_station_cache"
        ) from exc
    tmp.replace(dest)
    return dest


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StationIndex:
    """Column arrays over every station that carries the core elements.

    One row per station, not per element: the per-element spans are collapsed
    during parsing into `first`/`last` (intersection over the core elements, so
    the span is one over which the fit actually has all three variables) and a
    bitmask of which tracked elements are present.
    """

    ids: np.ndarray  # (n,) '<U11'
    names: np.ndarray  # (n,) '<U30'
    lat: np.ndarray  # (n,) float64
    lon: np.ndarray
    elevation: np.ndarray
    first: np.ndarray  # (n,) int16 --- first year with all core elements
    last: np.ndarray  # (n,) int16 --- last  year with all core elements
    mask: np.ndarray  # (n,) uint8  --- bit i set if TRACKED[i] present

    def __len__(self) -> int:
        return int(self.ids.size)


def _parse_inventory(path: Path) -> dict[str, tuple[int, int, int]]:
    """`{station_id: (first_year, last_year, element_mask)}` over tracked elements.

    `first`/`last` are the *intersection* of the core elements' spans. A station
    that measured rainfall from 1890 but temperature only from 1975 has 75 years
    of rainfall and no century of anything this model can fit, and the
    intersection is what says so.
    """
    bit = {name: 1 << i for i, name in enumerate(TRACKED)}
    core_bits = sum(bit[e] for e in CORE_ELEMENTS)

    core_first: dict[str, int] = {}
    core_last: dict[str, int] = {}
    masks: dict[str, int] = {}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            element = line[31:35].strip()
            b = bit.get(element)
            if b is None:
                continue
            sid = line[0:11]
            masks[sid] = masks.get(sid, 0) | b
            if b & core_bits:
                lo, hi = int(line[36:40]), int(line[41:45])
                # Intersection: latest start, earliest end.
                core_first[sid] = max(core_first.get(sid, -9999), lo)
                core_last[sid] = min(core_last.get(sid, 9999), hi)

    out: dict[str, tuple[int, int, int]] = {}
    for sid, m in masks.items():
        if m & core_bits != core_bits:
            continue  # missing at least one of TMAX/TMIN/PRCP entirely
        out[sid] = (core_first[sid], core_last[sid], m)
    return out


def _parse_stations(path: Path, keep: dict[str, tuple[int, int, int]]) -> StationIndex:
    ids: list[str] = []
    names: list[str] = []
    lat: list[float] = []
    lon: list[float] = []
    elev: list[float] = []
    first: list[int] = []
    last: list[int] = []
    mask: list[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sid = line[0:11]
            entry = keep.get(sid)
            if entry is None:
                continue
            try:
                la, lo_, el = float(line[12:20]), float(line[21:30]), float(line[31:37])
            except ValueError:
                continue
            if el < -900:  # GHCN missing-elevation sentinel
                el = 0.0
            f, l, m = entry
            ids.append(sid)
            names.append(line[41:71].strip())
            lat.append(la)
            lon.append(lo_)
            elev.append(el)
            first.append(f)
            last.append(l)
            mask.append(m)

    return StationIndex(
        ids=np.array(ids, dtype="<U11"),
        names=np.array(names, dtype="<U30"),
        lat=np.asarray(lat, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
        elevation=np.asarray(elev, dtype=np.float64),
        first=np.asarray(first, dtype=np.int16),
        last=np.asarray(last, dtype=np.int16),
        mask=np.asarray(mask, dtype=np.uint8),
    )


_INDEX: StationIndex | None = None
_LOCK = threading.RLock()

_FIELDS = ("ids", "names", "lat", "lon", "elevation", "first", "last", "mask")


def _read_bundle(path: Path) -> StationIndex:
    z = np.load(path, allow_pickle=False)
    return StationIndex(**{f: z[f] for f in _FIELDS})


def _write_bundle(path: Path, idx: StationIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a handle, not a path: `savez_compressed` helpfully appends
    # `.npz` to any filename lacking it, which turns a `.part` temp name into a
    # file the subsequent rename cannot find.
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **{f: getattr(idx, f) for f in _FIELDS})
    tmp.replace(path)


def build_cache(dest: Path = CACHE_PATH, keep_raw: bool = False) -> StationIndex:
    """Download, parse and serialise the station index. The offline build step.

    Separated from `load_index` on purpose. Building needs network and takes
    several seconds over ~800k inventory rows; loading needs neither, and the
    two should not be entangled in a request path. `api.scripts.
    build_station_cache` is the CLI wrapper.
    """
    inv = _parse_inventory(_fetch(INVENTORY_FILE))
    idx = _parse_stations(_fetch(STATIONS_FILE), inv)
    _write_bundle(dest, idx)
    log.info("station bundle written to %s: %d stations", dest, len(idx))
    if not keep_raw:
        # The 46 MB of source text is not needed once parsed.
        for name in (INVENTORY_FILE, STATIONS_FILE):
            (CACHE_DIR / name).unlink(missing_ok=True)
    return idx


def load_index(refresh: bool = False) -> StationIndex:
    """The parsed station index, memoised for the process.

    Resolution order, cheapest first:

    1. **In-memory** --- already loaded this process.
    2. **`stations_parsed.npz`** --- the shipped bundle. No network, a few
       milliseconds, and the only path that runs in a sealed environment.
    3. **Live download from NOAA** --- last resort, several seconds, and it
       writes the bundle so it never happens twice.

    Raises `DataIngestionError` with the build command in the message when all
    three fail. Callers in a request path should catch it and degrade to
    reanalysis rather than propagate a 500.
    """
    global _INDEX
    with _LOCK:
        if _INDEX is not None and not refresh:
            return _INDEX

        if CACHE_PATH.exists() and not refresh:
            _INDEX = _read_bundle(CACHE_PATH)
            log.info("station bundle loaded: %d stations", len(_INDEX))
            return _INDEX

        _INDEX = build_cache()
        return _INDEX


def warm_index() -> None:
    """Load the index in the background, tolerating failure.

    Called at server startup so the first customer does not pay for a cold
    cache, and swallowing the error so a NOAA outage cannot stop the process
    from booting --- the storefront works without it, quotes just fall back to
    reanalysis and say so.
    """

    def _run() -> None:
        try:
            load_index()
        except DataIngestionError as exc:
            log.warning("station index unavailable at startup: %s", exc)

    threading.Thread(target=_run, name="station-index-warm", daemon=True).start()


# ----------------------------------------------------------------------
# Geometry and scoring
# ----------------------------------------------------------------------


def haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p1, l1 = np.radians(lat), np.radians(lon)
    dp, dl = p1 - p0, l1 - l0
    a = np.sin(dp / 2.0) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _completeness(last_year: np.ndarray, mask: np.ndarray, now: int) -> np.ndarray:
    """How much of a usable record a station really offers, in (0, ~1.15].

    Two things the raw year count does not capture:

    * **currency** --- how close the record runs to today. A station that stopped
      in 2019 leaves a small gap to bridge; one that stopped in 2005 has a
      structural break sitting inside the settlement basis.
    * **coverage** --- a small bonus for optional elements, so a station that also
      measured snow and wind wins ties. Deliberately small: it must never
      outweigh record length, and after Jackson Hole we know that a station
      lacking an element is better rejected downstream than quietly filled.
    """
    lag = np.clip(now - last_year, 0, 30)
    currency = np.maximum(1.0 - (lag / 30.0) ** 1.5, 0.0)

    n_optional = np.zeros(mask.shape, dtype=np.float64)
    for i, name in enumerate(TRACKED):
        if name in OPTIONAL_ELEMENTS:
            n_optional += ((mask >> i) & 1).astype(np.float64)

    return currency * (1.0 + 0.05 * n_optional)


def _score(
    years: np.ndarray,
    last_year: np.ndarray,
    mask: np.ndarray,
    km: np.ndarray,
    elev_delta_m: np.ndarray | None,
    now: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank candidates. Returns `(score, elevation_warning)`.

        score = (record_years * completeness) / (distance_km + 0.05 * |elev_delta_m|)

    Record length in the numerator, separation in the denominator, where
    "separation" is horizontal and vertical combined. The denominator is floored
    at `MIN_DENOMINATOR_KM` so a gauge in the car park does not divide by zero
    and win by an infinite margin --- at that range the numerator should be
    deciding anyway.

    Past `ELEVATION_WARNING_M` the score is halved outright. That is a blunt
    instrument and it is meant to be: at 300 m the lapse rate alone is ~2 °C,
    which is larger than a century of warming, and no amount of record length
    makes a gauge in the valley a good settlement point for a contract on the
    mountain.
    """
    completeness = _completeness(last_year, mask, now)
    separation = np.asarray(km, dtype=np.float64).copy()

    warn = np.zeros(separation.shape, dtype=bool)
    if elev_delta_m is not None:
        delta = np.abs(np.asarray(elev_delta_m, dtype=np.float64))
        separation = separation + ELEVATION_KM_PER_M * delta
        warn = delta > ELEVATION_WARNING_M

    score = (years.astype(np.float64) * completeness) / np.maximum(
        separation, MIN_DENOMINATOR_KM
    )
    score = np.where(warn, score * ELEVATION_PENALTY, score)
    return score, warn


def _elements_of(mask: int) -> tuple[str, ...]:
    return tuple(name for i, name in enumerate(TRACKED) if (mask >> i) & 1)


def nearest_station(
    lat: float,
    lon: float,
    site_elevation_m: float | None = None,
    *,
    need: tuple[str, ...] = CORE_ELEMENTS,
    min_years: int = DEFAULT_MIN_YEARS,
    max_km: float = DEFAULT_MAX_KM,
    min_last_year: int = DEFAULT_MIN_LAST_YEAR,
    now: int = 2025,
) -> StationMatch | None:
    """Best GHCN-Daily station for settling a contract at these coordinates.

    Pass `site_elevation_m` wherever it is known. Without it the match is made
    on horizontal distance alone, which is the failure mode this function exists
    to avoid in mountain terrain.

    Returns `None` — never a bad match — when no station within `max_km` carries
    the required elements over at least `min_years` and still reports. The
    caller is expected to fall back to reanalysis and say so on screen.
    """
    try:
        idx = load_index()
    except DataIngestionError as exc:
        log.warning("station index unavailable, falling back to reanalysis: %s", exc)
        return None

    if len(idx) == 0:
        return None

    # Cheap bounding box before the trig. At 150 km this drops ~128k rows to a
    # few hundred, so the haversine cost stops mattering.
    dlat = max_km / 111.32
    dlon = max_km / max(111.32 * np.cos(np.radians(lat)), 1e-6)
    box = (
        (np.abs(idx.lat - lat) <= dlat)
        & (np.abs(((idx.lon - lon + 180.0) % 360.0) - 180.0) <= dlon)
    )
    cand = np.flatnonzero(box)
    if cand.size == 0:
        return None

    km = haversine_km(lat, lon, idx.lat[cand], idx.lon[cand])
    years = (idx.last[cand].astype(np.int32) - idx.first[cand].astype(np.int32)) + 1

    need_mask = 0
    for element in need:
        if element in TRACKED:
            need_mask |= 1 << TRACKED.index(element)

    ok = (
        (km <= max_km)
        & (years >= min_years)
        & (idx.last[cand] >= min_last_year)
        & ((idx.mask[cand] & need_mask) == need_mask)
    )
    if not ok.any():
        return None

    cand, km, years = cand[ok], km[ok], years[ok]

    # Signed, station minus venue: negative means the gauge is below the site.
    delta = None
    if site_elevation_m is not None:
        delta = idx.elevation[cand] - float(site_elevation_m)

    scores, warn = _score(years, idx.last[cand].astype(np.int32), idx.mask[cand], km, delta, now)
    best = int(np.argmax(scores))
    i = int(cand[best])

    return StationMatch(
        station_id=str(idx.ids[i]),
        name=str(idx.names[i]),
        lat=float(idx.lat[i]),
        lon=float(idx.lon[i]),
        elevation_m=float(idx.elevation[i]),
        distance_km=float(km[best]),
        first_year=int(idx.first[i]),
        last_year=int(idx.last[i]),
        elements=_elements_of(int(idx.mask[i])),
        score=float(scores[best]),
        elevation_delta_m=float(delta[best]) if delta is not None else None,
        elevation_warning=bool(warn[best]),
    )
