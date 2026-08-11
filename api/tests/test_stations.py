"""Tests for GHCN station matching.

These are the tests that guard a live pricing bug. Before `api/stations.py`
existed, `build_location` set `station_id=""`, so every customer venue silently
fell back to ERA5 reanalysis and `pricing.load_basis` --- which is
`BASIS_PER_KM * station_distance_km * theo` --- charged exactly zero for basis
risk on every contract sold.

The elevation cases matter for the same reason at one remove: a match that is
horizontally close but vertically wrong produces a *small* basis load for a
*large* basis, which is worse than no load at all because it looks priced.
"""

from __future__ import annotations

import numpy as np
import pytest

from api import stations


@pytest.fixture(scope="module")
def index():
    try:
        return stations.load_index()
    except stations.DataIngestionError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"station bundle unavailable: {exc}")


# ----------------------------------------------------------------------
# The bundle
# ----------------------------------------------------------------------


def test_bundle_loads_without_network(index):
    """The shipped `.npz` is the whole point: boot must not need NOAA."""
    assert stations.CACHE_PATH.exists(), "run `python -m api.scripts.build_station_cache`"
    assert len(index) > 20_000
    assert index.lat.shape == index.lon.shape == index.elevation.shape == index.ids.shape


def test_bundle_round_trips(tmp_path, index):
    dest = tmp_path / "stations_parsed.npz"
    stations._write_bundle(dest, index)
    again = stations._read_bundle(dest)
    assert len(again) == len(index)
    assert np.array_equal(again.ids, index.ids)
    assert np.allclose(again.elevation, index.elevation)


def test_every_station_carries_the_core_elements(index):
    """The inventory filter is what stops a wind contract pricing off nothing."""
    core_bits = sum(1 << stations.TRACKED.index(e) for e in stations.CORE_ELEMENTS)
    assert np.all((index.mask & core_bits) == core_bits)


def test_elevation_is_parsed_and_sane(index):
    assert index.elevation.min() > -500.0
    assert index.elevation.max() < 6000.0
    # GHCN's -999.9 missing sentinel must have been mapped, not carried through.
    assert not np.any(index.elevation < -900.0)


# ----------------------------------------------------------------------
# Matching known venues
# ----------------------------------------------------------------------

#: The eight demo venues, with real elevations.
VENUES = [
    ("Boston waterfront", 42.3601, -71.0589, 6.0),
    ("Vail CO", 39.6403, -106.3742, 2470.0),
    ("Jackson Hole WY", 43.4799, -110.7624, 1900.0),
    ("Charleston SC", 32.7765, -79.9311, 6.0),
    ("Napa CA", 38.2975, -122.2869, 5.0),
    ("Scottsdale AZ", 33.4942, -111.9261, 380.0),
    ("Traverse City MI", 44.7631, -85.6206, 180.0),
    ("Asheville NC", 35.5951, -82.5515, 650.0),
]


@pytest.mark.parametrize("name,lat,lon,elev", VENUES)
def test_demo_venues_all_resolve(index, name, lat, lon, elev):
    match = stations.nearest_station(lat, lon, elev)
    assert match is not None, f"{name} found no qualifying station"
    assert match.distance_km <= stations.DEFAULT_MAX_KM
    assert match.years >= stations.DEFAULT_MIN_YEARS
    assert match.last_year >= stations.DEFAULT_MIN_LAST_YEAR
    assert set(stations.CORE_ELEMENTS) <= set(match.elements)


def test_boston_lands_on_the_expected_station(index):
    """Boston has an obvious right answer, so it is the canary for the scoring."""
    match = stations.nearest_station(42.3601, -71.0589, 6.0)
    assert match is not None
    assert match.station_id == "USW00014739"
    assert match.distance_km < 10.0
    assert match.years > 80


def test_mid_ocean_returns_none(index):
    """No station is better than a bad one --- the caller falls back to ERA5."""
    assert stations.nearest_station(25.0, -160.0, 0.0) is None
    assert stations.nearest_station(-45.0, -120.0, 0.0) is None


def test_no_match_when_requirement_cannot_be_met(index):
    """A 200-year minimum should eliminate almost everywhere."""
    assert stations.nearest_station(42.3601, -71.0589, 6.0, min_years=250) is None


def test_required_elements_are_enforced(index):
    """Most century stations never measured wind; asking must not fabricate one."""
    match = stations.nearest_station(42.3601, -71.0589, 6.0, need=("TMAX", "TMIN", "PRCP", "AWND"))
    if match is not None:
        assert "AWND" in match.elements


# ----------------------------------------------------------------------
# Scoring: length beats proximity, and elevation beats both
# ----------------------------------------------------------------------


def _score_one(years, last_year, km, elev_delta, mask=0b111, now=2025):
    score, warn = stations._score(
        np.array([years]), np.array([last_year]), np.array([mask], dtype=np.uint8),
        np.array([km], dtype=float),
        None if elev_delta is None else np.array([elev_delta], dtype=float),
        now,
    )
    return float(score[0]), bool(warn[0])


def test_patchy_records_are_excluded_before_scoring(index):
    """A short record never competes on score --- it is gated out first.

    Worth being precise about, because the score alone does *not* prefer a
    complete far station to a patchy near one: it is `years / distance`, so a
    20-year record 5 km away outscores a 120-year record 40 km away four to
    three. What protects the fit is the hard `min_years` floor, which removes
    the short record from the candidate set entirely.

    The division of labour is deliberate. Record length is a *requirement* (the
    forcing regression needs decades to identify), not a preference to be traded
    off; once that requirement is met, extra distance is real basis risk and
    proximity should win.
    """
    short, _ = _score_one(years=20, last_year=2025, km=5.0, elev_delta=0.0)
    long_far, _ = _score_one(years=120, last_year=2025, km=40.0, elev_delta=0.0)
    assert short > long_far, "score is proximity-weighted; the gate is what protects us"

    # ...and the gate is what makes that harmless.
    assert stations.DEFAULT_MIN_YEARS >= 60
    match = stations.nearest_station(42.3601, -71.0589, 6.0)
    assert match is not None and match.years >= stations.DEFAULT_MIN_YEARS


def test_length_wins_at_equal_distance():
    """Among qualifying stations, more record is strictly better."""
    longer, _ = _score_one(years=120, last_year=2025, km=20.0, elev_delta=0.0)
    shorter, _ = _score_one(years=65, last_year=2025, km=20.0, elev_delta=0.0)
    assert longer > shorter


def test_a_dead_record_scores_below_a_live_one_at_equal_distance():
    live, _ = _score_one(years=100, last_year=2025, km=10.0, elev_delta=0.0)
    dead, _ = _score_one(years=100, last_year=1998, km=10.0, elev_delta=0.0)
    assert live > dead


def test_elevation_difference_is_penalised():
    """100 m of vertical costs the same as 5 km of horizontal."""
    level, _ = _score_one(years=100, last_year=2025, km=10.0, elev_delta=0.0)
    offset, _ = _score_one(years=100, last_year=2025, km=10.0, elev_delta=100.0)
    assert offset < level
    equivalent, _ = _score_one(years=100, last_year=2025, km=15.0, elev_delta=0.0)
    assert offset == pytest.approx(equivalent, rel=1e-9)


def test_elevation_penalty_and_warning_trip_past_the_threshold():
    below, warn_below = _score_one(years=100, last_year=2025, km=10.0, elev_delta=299.0)
    above, warn_above = _score_one(years=100, last_year=2025, km=10.0, elev_delta=301.0)
    assert not warn_below and warn_above
    # A 2 m change in delta must not halve the score by itself --- the cliff is
    # the deliberate 50% penalty, not a continuity bug.
    assert above < below * 0.55


def test_penalty_is_symmetric_in_sign():
    """A gauge 400 m below is as wrong as one 400 m above."""
    lo, w1 = _score_one(years=100, last_year=2025, km=10.0, elev_delta=-400.0)
    hi, w2 = _score_one(years=100, last_year=2025, km=10.0, elev_delta=400.0)
    assert lo == pytest.approx(hi) and w1 and w2


def test_denominator_is_floored():
    """A station in the car park must not score infinity."""
    score, _ = _score_one(years=100, last_year=2025, km=0.0, elev_delta=0.0)
    assert np.isfinite(score) and score > 0


def test_high_altitude_venue_rejects_the_valley_airport(index):
    """The real-world case the elevation term exists for.

    Mammoth Lakes sits at ~2,400 m. The nearest long-record station by
    horizontal distance is Bishop Airport at ~1,250 m --- 1,150 m below the
    venue, roughly 7.5 C of lapse rate, and on the wrong side of the rain/snow
    line for most of the season. Settling a ski contract there is not a small
    basis, it is the wrong variable.
    """
    naive = stations.nearest_station(37.6485, -118.9721)
    aware = stations.nearest_station(37.6485, -118.9721, 2400.0)

    assert naive is not None and aware is not None
    assert aware.station_id != naive.station_id
    assert abs(aware.elevation_m - 2400.0) < abs(naive.elevation_m - 2400.0)
    assert abs(aware.elevation_delta_m) < 400.0


def test_match_reports_elevation_delta_only_when_it_can(index):
    """`None` means unknown, not zero. Conflating them hides the warning."""
    without = stations.nearest_station(39.6403, -106.3742)
    assert without is not None and without.elevation_delta_m is None
    assert without.elevation_warning is False

    with_elev = stations.nearest_station(39.6403, -106.3742, 2470.0)
    assert with_elev is not None and with_elev.elevation_delta_m is not None


def test_microclimate_note_appears_exactly_when_flagged(index):
    """Sea-level venue, mountain gauge: the note must fire and name the lapse."""
    match = stations.nearest_station(39.6403, -106.3742, 200.0)
    assert match is not None
    assert match.elevation_warning is True
    note = match.microclimate_note
    assert note and "microclimate" in note and "°C" in note

    fine = stations.nearest_station(42.3601, -71.0589, 6.0)
    assert fine is not None and fine.microclimate_note is None


def test_to_dict_is_json_safe(index):
    import json

    match = stations.nearest_station(42.3601, -71.0589, 6.0)
    assert match is not None
    payload = json.loads(json.dumps(match.to_dict()))
    assert payload["station_id"] == match.station_id
    assert payload["elevation_warning"] is False
