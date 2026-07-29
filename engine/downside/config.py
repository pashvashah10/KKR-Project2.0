"""Locations, perils, scenarios and the physical constants the engine runs on.

Climate normals below are approximate 1991-2020 published station normals, held
in metric. They exist to *calibrate the offline surrogate record* (see
`sources/synthetic.py`) and to sanity-check real ingests. When the NOAA / ERA5
adapters are live, the fitted climatology comes from observations and these
numbers become nothing more than a smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HISTORY_START_YEAR = 1926
HISTORY_END_YEAR = 2025
PROJECTION_END_YEAR = 2125

#: Reference year that trend terms are centred on. Centring kills most of the
#: collinearity between the intercept and the trend columns.
TREND_EPOCH = 1975.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def in_to_mm(inches: float) -> float:
    return inches * 25.4


#: Daily *maximum* sustained wind runs well above the daily *mean*. The engine
#: models the maximum, because that is what stops a lift or a rigging crew, but
#: published normals quote the mean --- so the mean is scaled by this ratio.
WIND_MAX_TO_MEAN = 1.78


@dataclass(frozen=True, slots=True)
class ClimateNormals:
    """Monthly 1991-2020 normals, metric. Twelve entries each, January first.

    `wind_ms` is the monthly mean 10-m wind speed in m/s (not the daily maximum);
    `wet_days` counts days at or above 0.01 in / 0.254 mm.
    """

    tmax_c: tuple[float, ...]
    tmin_c: tuple[float, ...]
    precip_mm: tuple[float, ...]
    wet_days: tuple[float, ...]
    wind_ms: tuple[float, ...]

    @property
    def wind_max_ms(self) -> tuple[float, ...]:
        return tuple(v * WIND_MAX_TO_MEAN for v in self.wind_ms)

    @staticmethod
    def from_imperial(
        tmax_f: list[float],
        tmin_f: list[float],
        precip_in: list[float],
        wet_days: list[float],
        wind_ms: list[float],
    ) -> "ClimateNormals":
        """Temperatures in F and precipitation in inches are converted; wind is
        already metric, so it passes through untouched."""
        return ClimateNormals(
            tmax_c=tuple(f_to_c(v) for v in tmax_f),
            tmin_c=tuple(f_to_c(v) for v in tmin_f),
            precip_mm=tuple(in_to_mm(v) for v in precip_in),
            wet_days=tuple(wet_days),
            wind_ms=tuple(float(v) for v in wind_ms),
        )


@dataclass(frozen=True, slots=True)
class Location:
    id: str
    name: str
    region: str
    lat: float
    lon: float
    elevation_m: float
    station_id: str
    station_distance_km: float
    vertical: str
    normals: ClimateNormals
    #: Local warming per unit of global warming. Continental interiors and high
    #: latitudes run hot (>1); maritime and the US southeast "warming hole" run
    #: cool (<1). Estimated from the observed record at build time --- this is the
    #: prior the estimate shrinks toward.
    amplification_prior: float
    #: Century-scale observed warming (degC/100yr) used to seed the surrogate.
    observed_trend_c_per_century: float
    #: Fractional change in mean daily precipitation intensity per degC of local
    #: warming. Clausius-Clapeyron gives ~7%/degC for saturation vapour pressure;
    #: observed daily-extreme scaling clusters near that, mean totals lower.
    precip_intensity_scaling: float = 0.062
    season: tuple[int, int] = (1, 12)


def _n(tmax_f, tmin_f, precip_in, wet_days, wind_ms) -> ClimateNormals:
    return ClimateNormals.from_imperial(tmax_f, tmin_f, precip_in, wet_days, wind_ms)


LOCATIONS: tuple[Location, ...] = (
    Location(
        id="vail-co",
        name="Vail, Colorado",
        region="Southern Rockies",
        lat=39.6403,
        lon=-106.3742,
        elevation_m=2476.0,
        station_id="USC00058575",
        station_distance_km=4.2,
        vertical="Ski resort",
        amplification_prior=1.42,
        observed_trend_c_per_century=1.85,
        precip_intensity_scaling=0.055,
        season=(11, 4),
        normals=_n(
            [30, 33, 40, 49, 60, 71, 76, 74, 67, 55, 40, 31],
            [3, 5, 13, 22, 30, 37, 43, 42, 35, 25, 15, 4],
            [2.8, 2.6, 2.6, 2.2, 1.9, 1.3, 1.9, 2.1, 1.9, 1.9, 2.4, 2.8],
            [11, 10, 11, 10, 10, 7, 11, 12, 9, 8, 9, 11],
            [6.5, 6.7, 7.3, 7.6, 7.0, 6.6, 6.0, 5.8, 6.0, 6.2, 6.4, 6.4],
        ),
    ),
    Location(
        id="jackson-wy",
        name="Jackson Hole, Wyoming",
        region="Northern Rockies",
        lat=43.4799,
        lon=-110.7624,
        elevation_m=1900.0,
        station_id="USC00484910",
        station_distance_km=6.8,
        vertical="Ski resort",
        amplification_prior=1.48,
        observed_trend_c_per_century=1.92,
        precip_intensity_scaling=0.055,
        season=(11, 4),
        normals=_n(
            [26, 31, 42, 52, 63, 73, 82, 81, 71, 57, 39, 27],
            [3, 5, 15, 25, 32, 38, 42, 40, 33, 25, 16, 4],
            [2.4, 1.7, 1.6, 1.5, 2.0, 1.7, 1.3, 1.4, 1.6, 1.4, 2.1, 2.5],
            [12, 10, 10, 9, 11, 9, 7, 7, 8, 8, 10, 12],
            [4.0, 4.2, 4.8, 5.2, 5.0, 4.8, 4.4, 4.3, 4.3, 4.4, 4.2, 4.0],
        ),
    ),
    Location(
        id="napa-ca",
        name="Napa Valley, California",
        region="California Coastal",
        lat=38.2975,
        lon=-122.2869,
        elevation_m=18.0,
        station_id="USW00093227",
        station_distance_km=9.5,
        vertical="Winery & events",
        amplification_prior=0.96,
        observed_trend_c_per_century=1.32,
        precip_intensity_scaling=0.070,
        season=(4, 10),
        normals=_n(
            [58, 62, 66, 71, 77, 83, 87, 87, 85, 77, 65, 57],
            [38, 41, 43, 45, 49, 53, 55, 55, 53, 48, 42, 37],
            [4.8, 4.5, 3.2, 1.5, 0.7, 0.2, 0.03, 0.05, 0.2, 1.2, 2.7, 4.5],
            [10, 9, 8, 5, 3, 1, 0.3, 0.3, 1, 3, 7, 10],
            [3.0, 3.3, 3.6, 3.8, 3.9, 3.9, 3.7, 3.5, 3.2, 3.0, 2.9, 3.0],
        ),
    ),
    Location(
        id="scottsdale-az",
        name="Scottsdale, Arizona",
        region="Desert Southwest",
        lat=33.4942,
        lon=-111.9261,
        elevation_m=387.0,
        station_id="USW00023183",
        station_distance_km=17.9,
        vertical="Golf resort",
        amplification_prior=1.31,
        observed_trend_c_per_century=2.05,
        precip_intensity_scaling=0.048,
        season=(10, 5),
        normals=_n(
            [67, 71, 77, 85, 95, 105, 107, 106, 101, 89, 76, 66],
            [46, 49, 53, 60, 69, 78, 84, 84, 78, 66, 54, 46],
            [0.9, 0.9, 1.0, 0.3, 0.1, 0.02, 1.0, 1.0, 0.7, 0.6, 0.7, 0.9],
            [4, 4, 4, 2, 1, 0.3, 4, 5, 3, 3, 3, 4],
            [2.4, 2.7, 3.1, 3.4, 3.5, 3.5, 3.3, 3.1, 2.9, 2.6, 2.4, 2.3],
        ),
    ),
    Location(
        id="orlando-fl",
        name="Orlando, Florida",
        region="Southeast Peninsula",
        lat=28.5383,
        lon=-81.3792,
        elevation_m=25.0,
        station_id="USW00012815",
        station_distance_km=11.2,
        vertical="Outdoor attraction",
        amplification_prior=0.82,
        observed_trend_c_per_century=0.95,
        precip_intensity_scaling=0.075,
        season=(1, 12),
        normals=_n(
            [72, 75, 80, 85, 90, 92, 92, 92, 90, 85, 79, 74],
            [50, 53, 57, 61, 67, 73, 74, 75, 73, 66, 59, 53],
            [2.4, 2.4, 3.4, 2.5, 3.4, 7.6, 7.3, 7.1, 6.0, 3.3, 2.1, 2.3],
            [7, 7, 7, 5, 8, 15, 17, 17, 13, 8, 6, 6],
            [3.9, 4.1, 4.3, 4.2, 3.9, 3.4, 3.2, 3.1, 3.5, 3.8, 3.8, 3.8],
        ),
    ),
    Location(
        id="austin-tx",
        name="Austin, Texas",
        region="Southern Plains",
        lat=30.2672,
        lon=-97.7431,
        elevation_m=149.0,
        station_id="USW00013904",
        station_distance_km=13.4,
        vertical="Festival grounds",
        amplification_prior=1.04,
        observed_trend_c_per_century=1.28,
        precip_intensity_scaling=0.068,
        season=(3, 11),
        normals=_n(
            [62, 66, 73, 80, 86, 92, 96, 97, 91, 82, 71, 63],
            [42, 45, 52, 59, 67, 72, 74, 74, 70, 60, 50, 43],
            [2.6, 2.0, 2.9, 2.3, 5.0, 3.7, 2.0, 2.7, 3.0, 4.0, 2.9, 2.7],
            [7, 7, 7, 6, 8, 7, 5, 5, 6, 7, 7, 7],
            [4.3, 4.5, 4.9, 4.8, 4.3, 4.0, 3.6, 3.4, 3.5, 3.7, 4.0, 4.1],
        ),
    ),
    Location(
        id="asheville-nc",
        name="Asheville, North Carolina",
        region="Southern Appalachia",
        lat=35.5951,
        lon=-82.5515,
        elevation_m=650.0,
        station_id="USW00003812",
        station_distance_km=8.1,
        vertical="Campground group",
        amplification_prior=0.78,
        observed_trend_c_per_century=0.72,
        precip_intensity_scaling=0.072,
        season=(4, 10),
        normals=_n(
            [47, 51, 59, 68, 75, 82, 85, 84, 78, 68, 58, 49],
            [27, 29, 35, 43, 52, 60, 64, 63, 57, 46, 36, 30],
            [3.5, 3.5, 4.2, 3.6, 4.2, 4.4, 4.5, 4.4, 3.9, 3.3, 3.6, 3.6],
            [10, 10, 11, 10, 12, 12, 13, 12, 9, 8, 9, 10],
            [3.6, 3.8, 4.1, 4.0, 3.3, 2.9, 2.8, 2.6, 2.7, 2.9, 3.2, 3.5],
        ),
    ),
    Location(
        id="boston-ma",
        name="Boston Harbor, Massachusetts",
        region="New England Coastal",
        lat=42.3601,
        lon=-71.0589,
        elevation_m=6.0,
        station_id="USW00014739",
        station_distance_km=5.4,
        vertical="Waterfront venue",
        amplification_prior=1.12,
        observed_trend_c_per_century=1.55,
        precip_intensity_scaling=0.078,
        season=(5, 10),
        normals=_n(
            [36.8, 39.0, 45.1, 55.9, 66.1, 75.7, 81.3, 79.7, 72.5, 61.4, 51.2, 42.1],
            [23.7, 25.9, 32.0, 41.2, 50.5, 60.4, 66.8, 65.8, 58.7, 47.7, 39.0, 29.6],
            [3.4, 3.3, 4.3, 3.7, 3.3, 3.8, 3.4, 3.4, 3.6, 4.2, 3.9, 4.0],
            [11, 10, 12, 11, 11, 10, 9, 9, 9, 9, 11, 11],
            [5.9, 5.9, 6.0, 5.7, 5.2, 4.8, 4.6, 4.5, 4.7, 5.1, 5.5, 5.8],
        ),
    ),
)

LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}


# --------------------------------------------------------------------------
# Emission scenarios
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    """A forcing pathway, described by its effective radiative forcing curve.

    Forcing values are approximate CMIP6 SSP effective radiative forcing relative
    to 1750, in W/m^2, at 2020 / 2050 / 2100 / 2125. The engine turns these into
    a temperature response through a two-box energy balance model rather than
    hard-coding a warming number, so the response has the right *shape*: fast
    surface adjustment plus a slow deep-ocean drag.
    """

    id: str
    label: str
    short: str
    description: str
    weight: float  # prior probability mass used to blend scenarios
    forcing_nodes: tuple[tuple[float, float], ...]
    color_slot: int

    def forcing(self, years: np.ndarray) -> np.ndarray:
        yr = np.array([n[0] for n in self.forcing_nodes], dtype=float)
        fv = np.array([n[1] for n in self.forcing_nodes], dtype=float)
        return np.interp(np.asarray(years, dtype=float), yr, fv)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="ssp126",
        label="SSP1-2.6 — rapid decarbonisation",
        short="SSP1-2.6",
        description="Net-zero around 2075. Forcing peaks mid-century then declines.",
        weight=0.20,
        forcing_nodes=((1850, 0.0), (1950, 0.35), (2000, 1.75), (2020, 2.65), (2050, 3.05), (2075, 2.85), (2100, 2.60), (2125, 2.45)),
        color_slot=3,
    ),
    Scenario(
        id="ssp245",
        label="SSP2-4.5 — current policy drift",
        short="SSP2-4.5",
        description="Roughly where stated policy lands. The engine's default.",
        weight=0.45,
        forcing_nodes=((1850, 0.0), (1950, 0.35), (2000, 1.75), (2020, 2.65), (2050, 3.95), (2075, 4.55), (2100, 4.85), (2125, 5.00)),
        color_slot=1,
    ),
    Scenario(
        id="ssp370",
        label="SSP3-7.0 — regional rivalry",
        short="SSP3-7.0",
        description="Weak coordination, high land-use emissions, no decline this century.",
        weight=0.25,
        forcing_nodes=((1850, 0.0), (1950, 0.35), (2000, 1.75), (2020, 2.65), (2050, 4.55), (2075, 6.05), (2100, 7.05), (2125, 7.85)),
        color_slot=4,
    ),
    Scenario(
        id="ssp585",
        label="SSP5-8.5 — fossil-fuelled growth",
        short="SSP5-8.5",
        description="High end. Retained as a tail-pricing stress case, not a central view.",
        weight=0.10,
        forcing_nodes=((1850, 0.0), (1950, 0.35), (2000, 1.75), (2020, 2.65), (2050, 5.10), (2075, 7.20), (2100, 8.55), (2125, 9.60)),
        color_slot=8,
    ),
)

SCENARIOS_BY_ID: dict[str, Scenario] = {s.id: s for s in SCENARIOS}
DEFAULT_SCENARIO = "ssp245"


# --------------------------------------------------------------------------
# Two-box energy balance parameters
# --------------------------------------------------------------------------

#: Equilibrium climate sensitivity, degC per doubling of CO2. IPCC AR6 assessed
#: likely range 2.5-4.0 with a best estimate of 3.0.
ECS_C = 3.0
ECS_SIGMA = 0.55
#: Forcing from a CO2 doubling, W/m^2.
F2X = 3.93
#: Upper-box (surface + mixed layer) heat capacity, W yr /m^2/K.
C_UPPER = 7.3
#: Deep-ocean box heat capacity.
C_DEEP = 106.0
#: Deep-ocean exchange coefficient, W/m^2/K.
GAMMA = 0.73
#: Efficacy of deep-ocean heat uptake.
EFFICACY = 1.28


# --------------------------------------------------------------------------
# Perils
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Peril:
    """A parametric trigger definition.

    `variable` names a simulated daily field; `statistic` says how days inside the
    accumulation window are collapsed before comparison to the threshold.
    """

    id: str
    label: str
    variable: str  # precip_mm | tmax_c | tmin_c | wind_ms | snow_mm
    statistic: str  # daily | window_sum | window_max | window_min | consecutive
    comparator: str  # ge | le
    threshold: float
    unit: str
    window_days: int = 1
    description: str = ""
    applies_to: tuple[str, ...] = field(default=())

    def evaluate(self, series: np.ndarray) -> np.ndarray:
        """Reduce a (paths, days) array to the per-path statistic being tested."""
        if self.statistic == "daily":
            return series
        if self.statistic == "window_sum":
            return _rolling(series, self.window_days, np.sum)
        if self.statistic == "window_max":
            return _rolling(series, self.window_days, np.max)
        if self.statistic == "window_min":
            return _rolling(series, self.window_days, np.min)
        if self.statistic == "consecutive":
            # Longest run meeting the threshold, compared later against window_days.
            hit = series >= self.threshold if self.comparator == "ge" else series <= self.threshold
            return _longest_run(hit)
        raise ValueError(f"unknown statistic {self.statistic!r}")

    def triggered(self, series: np.ndarray) -> np.ndarray:
        stat = self.evaluate(series)
        if self.statistic == "consecutive":
            return stat >= self.window_days
        return stat >= self.threshold if self.comparator == "ge" else stat <= self.threshold


def _rolling(series: np.ndarray, window: int, fn) -> np.ndarray:
    if window <= 1:
        return series
    n = series.shape[-1]
    if n < window:
        return fn(series, axis=-1, keepdims=True)
    windows = np.lib.stride_tricks.sliding_window_view(series, window, axis=-1)
    return fn(windows, axis=-1)


def _longest_run(hit: np.ndarray) -> np.ndarray:
    """Longest consecutive True run per row, vectorised over paths."""
    padded = np.concatenate([np.zeros((*hit.shape[:-1], 1), dtype=int), hit.astype(int)], axis=-1)
    runs = np.zeros_like(padded)
    for i in range(1, padded.shape[-1]):
        runs[..., i] = (runs[..., i - 1] + 1) * padded[..., i]
    return runs.max(axis=-1)


PERILS: tuple[Peril, ...] = (
    Peril(
        id="rain-day",
        label="Rain day",
        variable="precip_mm",
        statistic="daily",
        comparator="ge",
        threshold=in_to_mm(0.35),
        unit="mm",
        description="A single day clearing 0.35 in. The point where outdoor attendance measurably drops.",
        applies_to=("Festival grounds", "Campground group", "Golf resort", "Waterfront venue", "Winery & events", "Outdoor attraction"),
    ),
    Peril(
        id="washout",
        label="Washout day",
        variable="precip_mm",
        statistic="daily",
        comparator="ge",
        threshold=in_to_mm(0.80),
        unit="mm",
        description="0.80 in in a day. Past this the day is generally lost, not just degraded.",
        applies_to=("Festival grounds", "Waterfront venue", "Outdoor attraction", "Winery & events"),
    ),
    Peril(
        id="wet-weekend",
        label="Wet weekend",
        variable="precip_mm",
        statistic="window_sum",
        comparator="ge",
        threshold=in_to_mm(1.25),
        unit="mm",
        window_days=3,
        description="1.25 in across a rolling 3-day window. Prices a whole event weekend, not one day.",
        applies_to=("Festival grounds", "Campground group", "Waterfront venue"),
    ),
    Peril(
        id="extreme-heat",
        label="Extreme heat",
        variable="tmax_c",
        statistic="daily",
        comparator="ge",
        threshold=f_to_c(95.0),
        unit="degC",
        description="Daily maximum at or above 95F. The Bible's point about nonlinearity applies here hard.",
        applies_to=("Golf resort", "Festival grounds", "Outdoor attraction", "Winery & events"),
    ),
    Peril(
        id="heat-wave",
        label="Heat wave",
        variable="tmax_c",
        statistic="consecutive",
        comparator="ge",
        threshold=f_to_c(100.0),
        unit="degC",
        window_days=3,
        description="Three consecutive days at or above 100F. Autocorrelation is the whole story here.",
        applies_to=("Golf resort", "Festival grounds", "Outdoor attraction"),
    ),
    Peril(
        id="hard-freeze",
        label="Hard freeze",
        variable="tmin_c",
        statistic="daily",
        comparator="le",
        threshold=f_to_c(28.0),
        unit="degC",
        description="Minimum at or below 28F. Bud-break frost risk for vineyards and orchards.",
        applies_to=("Winery & events",),
    ),
    Peril(
        id="high-wind",
        label="High wind",
        variable="wind_ms",
        statistic="daily",
        comparator="ge",
        threshold=13.4,
        unit="m/s",
        description="Sustained wind at or above 30 mph. Lifts, rigging and marine ops all stop.",
        applies_to=("Ski resort", "Waterfront venue", "Festival grounds"),
    ),
    Peril(
        id="snow-drought",
        label="Snow drought",
        variable="snow_mm",
        statistic="window_sum",
        comparator="le",
        threshold=in_to_mm(24.0),
        unit="mm",
        window_days=30,
        description="Under 24 in of snowfall in a rolling 30-day window. The existential risk for a ski operator.",
        applies_to=("Ski resort",),
    ),
    Peril(
        id="warm-snowline",
        label="Warm snowline",
        variable="tmax_c",
        statistic="daily",
        comparator="ge",
        threshold=4.0,
        unit="degC",
        description="Base-area max above 4C, when falling precipitation arrives as rain and snowmaking stops.",
        applies_to=("Ski resort",),
    ),
)

PERILS_BY_ID: dict[str, Peril] = {p.id: p for p in PERILS}


def perils_for(location: Location) -> list[Peril]:
    return [p for p in PERILS if not p.applies_to or location.vertical in p.applies_to]
