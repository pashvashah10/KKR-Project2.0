"""The complete fitted model for one location, and the simulator that runs it.

`SiteModel.fit` estimates every component from the daily record; `SiteModel.
simulate` generates synthetic daily weather for any future year under any
scenario. Everything the pricing layer needs comes out of `simulate`.

Structure of the joint model
----------------------------
Weather variables are strongly dependent and the dependence is what prices the
product, so the simulator is built around a **shared synoptic factor** rather
than drawing each variable independently:

    u_t          AR(1) Gaussian latent --- "what the weather system is doing"
    tmax, tmin   marginal from the spliced-tail model, coupled to u_t by a
                 Gaussian copula (preserves the fitted marginal *and* the
                 fitted autocorrelation)
    wet_t        two-state Markov chain, with day-of-year, warming and the
                 current temperature anomaly all as covariates
    amount_t     Gamma GLM mean, times a spliced Gamma/GPD multiplier
    snow_t       precipitation partitioned by temperature, using the empirical
                 ratio distribution from this station's own record
    wind_t       log-normal-ish marginal on its own AR(1) latent, correlated
                 with u_t

Two modelling choices worth flagging, because they change prices materially:

* **Temperature enters the precipitation model as a covariate.** Rain days are
  not thermally neutral --- they are cool in summer and mild in winter. Drawing
  precipitation independently of temperature would make combined triggers ("hot
  *and* dry", "cold *and* wet") wrong in both directions.
* **Autocorrelation is carried through the copula, not bolted on.** Consecutive-
  day triggers such as the 3-day heat wave are priced off run lengths, and an
  independent-day simulator underprices them by a wide margin --- roughly an
  order of magnitude for a 3-day run at rho = 0.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .climatology import BASE_HI, BASE_LO, ClimatologyFit, DesignSpec, fit_climatology
from .config import HISTORY_END_YEAR, Location, Scenario
from .forcing import TwoBoxResponse, global_temperature_path
from .glm import GLMFit, fit_gamma_log, fit_logistic
from .projection import ProjectionParams, build_params
from .sources.base import DailyRecord
from .tails import GPDFit, SplicedTail, fit_gpd

__all__ = ["SiteModel", "SimulatedWeather"]

WET_THRESHOLD_MM = 0.254  # 0.01 in, the standard gauge reporting threshold


# ----------------------------------------------------------------------
# Design helpers for the precipitation GLMs
# ----------------------------------------------------------------------


def _precip_design(
    doy: np.ndarray,
    g: np.ndarray,
    prev_wet: np.ndarray | None,
    z_temp: np.ndarray | None,
    n_harmonics: int = 3,
) -> tuple[np.ndarray, list[str]]:
    doy = np.asarray(doy, dtype=float)
    g = np.broadcast_to(np.asarray(g, dtype=float), doy.shape)
    phase = 2.0 * np.pi * doy / 365.25

    cols = [np.ones_like(doy)]
    names = ["intercept"]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * phase))
        names.append(f"cos{k}")
        cols.append(np.sin(k * phase))
        names.append(f"sin{k}")
    cols.append(g)
    names.append("global")
    cols.append(g * np.cos(phase))
    names.append("global_cos1")
    cols.append(g * np.sin(phase))
    names.append("global_sin1")
    if prev_wet is not None:
        cols.append(np.asarray(prev_wet, dtype=float))
        names.append("prev_wet")
    if z_temp is not None:
        zt = np.asarray(z_temp, dtype=float)
        cols.append(zt)
        names.append("z_temp")
        cols.append(zt * np.cos(phase))
        names.append("z_temp_cos1")
        cols.append(zt * np.sin(phase))
        names.append("z_temp_sin1")
    return np.column_stack(cols), names


# ----------------------------------------------------------------------


@dataclass(slots=True)
class SimulatedWeather:
    """(n_paths, n_days) arrays for one simulated year/window."""

    year: int
    scenario: str
    doy: np.ndarray
    tmax_c: np.ndarray
    tmin_c: np.ndarray
    precip_mm: np.ndarray
    snow_mm: np.ndarray
    wind_ms: np.ndarray

    def get(self, variable: str) -> np.ndarray:
        return getattr(self, variable)

    @property
    def n_paths(self) -> int:
        return self.tmax_c.shape[0]


@dataclass(slots=True)
class SiteModel:
    location: Location
    record: DailyRecord
    response: TwoBoxResponse = field(repr=False)

    tmax: ClimatologyFit = field(repr=False)
    tmin: ClimatologyFit = field(repr=False)
    log_wind: ClimatologyFit = field(repr=False)

    tmax_tail: SplicedTail = field(repr=False)
    tmin_tail: SplicedTail = field(repr=False)
    wind_tail: SplicedTail = field(repr=False)

    occurrence: GLMFit = field(repr=False)
    intensity: GLMFit = field(repr=False)
    intensity_tail: GPDFit = field(repr=False)
    intensity_body: np.ndarray = field(repr=False)

    params: ProjectionParams = field(repr=False)
    #: Correlation between the tmax and tmin standardised anomalies.
    tmax_tmin_corr: float = 0.0
    #: Correlation between the temperature latent and the wind latent.
    temp_wind_corr: float = 0.0
    #: Empirical snow/precip ratio samples, bucketed by mean temperature.
    snow_ratio_bins: np.ndarray = field(repr=False, default=None)
    snow_ratio_samples: list = field(repr=False, default_factory=list)
    provenance: str = "unknown"

    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        location: Location,
        record: DailyRecord,
        scenario: Scenario,
        n_boot: int = 200,
        spec: DesignSpec | None = None,
    ) -> "SiteModel":
        response = global_temperature_path(scenario, 1850, 2130)
        spec = spec or DesignSpec()

        tmax = fit_climatology(record, response, "tmax_c", spec, n_boot=n_boot)
        tmin = fit_climatology(record, response, "tmin_c", spec, n_boot=0)

        # Wind is positive and right-skewed; fit in logs so the Gaussian
        # machinery applies, then exponentiate back at simulation time.
        log_wind_record = _with_variable(record, "wind_ms", np.log(np.clip(record.wind_ms, 0.05, None)))
        log_wind = fit_climatology(log_wind_record, response, "wind_ms", spec, n_boot=0)

        tmax_tail = SplicedTail.from_residuals(tmax.z_residuals)
        tmin_tail = SplicedTail.from_residuals(tmin.z_residuals)
        wind_tail = SplicedTail.from_residuals(log_wind.z_residuals)

        g = response.baseline_shift(record.decimal_year, BASE_LO, BASE_HI)
        wet = (record.precip_mm >= WET_THRESHOLD_MM).astype(float)
        prev_wet = np.concatenate([[wet[0]], wet[:-1]])
        z_temp = tmax.z_residuals

        X_occ, occ_names = _precip_design(record.doy, g, prev_wet, z_temp)
        occurrence = fit_logistic(X_occ, wet, occ_names)

        wet_mask = wet > 0.5
        X_int, int_names = _precip_design(
            record.doy[wet_mask], g[wet_mask], None, z_temp[wet_mask]
        )
        amounts = record.precip_mm[wet_mask]
        intensity = fit_gamma_log(X_int, amounts, int_names)

        # The multiplier `amount / fitted_mean` is what actually gets sampled.
        # Modelling the ratio rather than the raw amount separates "how wet is
        # this time of year" (the GLM) from "how extreme was this particular
        # storm" (the tail), so the tail can be fitted on all wet days at once
        # instead of season by season.
        mu = intensity.predict(X_int)
        ratio = np.clip(amounts / np.clip(mu, 1e-6, None), 1e-6, None)
        intensity_tail = fit_gpd(ratio, threshold_quantile=0.96)
        body = np.sort(ratio[ratio <= intensity_tail.threshold])

        # Snow partition, straight from this station's own record.
        snow_bins, snow_samples = _fit_snow_ratio(record)

        corr_tt = float(np.corrcoef(tmax.z_residuals, tmin.z_residuals)[0, 1])
        corr_tw = float(np.corrcoef(tmax.z_residuals, log_wind.z_residuals)[0, 1])

        return cls(
            location=location,
            record=record,
            response=response,
            tmax=tmax,
            tmin=tmin,
            log_wind=log_wind,
            tmax_tail=tmax_tail,
            tmin_tail=tmin_tail,
            wind_tail=wind_tail,
            occurrence=occurrence,
            intensity=intensity,
            intensity_tail=intensity_tail,
            intensity_body=body,
            params=build_params(tmax, location),
            tmax_tmin_corr=corr_tt,
            temp_wind_corr=corr_tw,
            snow_ratio_bins=snow_bins,
            snow_ratio_samples=snow_samples,
            provenance=record.provenance,
        )

    # ------------------------------------------------------------------
    def simulate(
        self,
        year: int,
        doy_window: np.ndarray,
        scenario: Scenario,
        n_paths: int = 4000,
        rng: np.random.Generator | None = None,
        parameter_uncertainty: bool = True,
    ) -> SimulatedWeather:
        """Simulate `n_paths` independent realisations of the window in `year`.

        With `parameter_uncertainty=True` the amplification is redrawn per path
        from its posterior, so the returned spread contains estimation
        uncertainty as well as weather noise. That is the number that should
        drive a price: quoting off weather noise alone would systematically
        understate the risk being warehoused.
        """
        rng = rng or np.random.default_rng(0)
        doy = np.asarray(doy_window, dtype=float)
        n_days = len(doy)

        g_scalar = float(
            global_temperature_path(scenario, 1850, 2130).baseline_shift(
                np.array([float(year) + 0.5]), BASE_LO, BASE_HI
            )[0]
        )

        # Per-path amplification multiplier. The fitted mean already contains the
        # central amplification, so this scales the warming increment only.
        if parameter_uncertainty and self.params.amplification_sd > 0:
            amp_draw = rng.normal(
                self.params.amplification, self.params.amplification_sd, size=n_paths
            )
            amp_scale = np.clip(amp_draw / max(self.params.amplification, 1e-6), 0.1, 3.0)
        else:
            amp_scale = np.ones(n_paths)

        g = g_scalar * amp_scale[:, None] * np.ones((1, n_days))
        frozen_year = np.full(n_days, float(HISTORY_END_YEAR))

        # --- latent synoptic factor --------------------------------------
        rho_t = float(np.clip(self.tmax.ar1, 0.0, 0.95))
        u = _ar1_gaussian(n_paths, n_days, rho_t, rng)
        e_min = _ar1_gaussian(n_paths, n_days, rho_t * 0.9, rng)
        e_wind = _ar1_gaussian(n_paths, n_days, float(np.clip(self.log_wind.ar1, 0.0, 0.95)), rng)

        a = float(np.clip(self.tmax_tmin_corr, -0.99, 0.99))
        lat_tmin = a * u + np.sqrt(max(1.0 - a * a, 1e-9)) * e_min
        b = float(np.clip(self.temp_wind_corr, -0.99, 0.99))
        lat_wind = b * u + np.sqrt(max(1.0 - b * b, 1e-9)) * e_wind

        # --- temperature --------------------------------------------------
        z_tmax = _copula_to_marginal(u, self.tmax_tail)
        z_tmin = _copula_to_marginal(lat_tmin, self.tmin_tail)

        mu_tmax = self.tmax.predict_mean(_tile(doy, n_paths), g, _tile(frozen_year, n_paths))
        sd_tmax = self.tmax.predict_sd(_tile(doy, n_paths), g)
        mu_tmin = self.tmin.predict_mean(_tile(doy, n_paths), g, _tile(frozen_year, n_paths))
        sd_tmin = self.tmin.predict_sd(_tile(doy, n_paths), g)

        tmax = mu_tmax + sd_tmax * z_tmax
        tmin = mu_tmin + sd_tmin * z_tmin
        # Enforce tmin <= tmax without inventing spread.
        gap = tmax - tmin
        bad = gap < 0.5
        mid = 0.5 * (tmax + tmin)
        tmax = np.where(bad, mid + 0.25, tmax)
        tmin = np.where(bad, mid - 0.25, tmin)

        # --- precipitation -------------------------------------------------
        precip = self._simulate_precip(doy, g, z_tmax, rng)

        # --- snow ------------------------------------------------------------
        snow = self._simulate_snow(precip, tmax, tmin, rng)

        # --- wind -------------------------------------------------------------
        z_wind = _copula_to_marginal(lat_wind, self.wind_tail)
        mu_lw = self.log_wind.predict_mean(_tile(doy, n_paths), g, _tile(frozen_year, n_paths))
        sd_lw = self.log_wind.predict_sd(_tile(doy, n_paths), g)
        wind = np.exp(np.clip(mu_lw + sd_lw * z_wind, -3.0, 4.5))

        return SimulatedWeather(
            year=year,
            scenario=scenario.id,
            doy=doy,
            tmax_c=tmax,
            tmin_c=tmin,
            precip_mm=precip,
            snow_mm=snow,
            wind_ms=wind,
        )

    # ------------------------------------------------------------------
    def _simulate_precip(
        self, doy: np.ndarray, g: np.ndarray, z_temp: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_paths, n_days = z_temp.shape
        wet = np.zeros((n_paths, n_days), dtype=bool)

        # The chain is sequential in time but fully vectorised across paths.
        prev = np.zeros(n_paths)
        # Seed yesterday's state from the unconditional wet rate for this day.
        for t in range(n_days):
            X, _ = _precip_design(
                np.full(n_paths, doy[t]), g[:, t], prev, z_temp[:, t]
            )
            p = self.occurrence.predict(X)
            draw = rng.random(n_paths) < p
            wet[:, t] = draw
            prev = draw.astype(float)

        amounts = np.zeros((n_paths, n_days))
        idx = np.flatnonzero(wet.ravel())
        if idx.size:
            flat_doy = np.tile(doy, (n_paths, 1)).ravel()[idx]
            flat_g = g.ravel()[idx]
            flat_z = z_temp.ravel()[idx]
            X, _ = _precip_design(flat_doy, flat_g, None, flat_z)
            mu = self.intensity.predict(X)
            mult = _sample_spliced_ratio(self.intensity_body, self.intensity_tail, idx.size, rng)
            flat = amounts.ravel()
            flat[idx] = mu * mult
            amounts = flat.reshape(n_paths, n_days)

        return np.clip(amounts, 0.0, None)

    def _simulate_snow(
        self, precip: np.ndarray, tmax: np.ndarray, tmin: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.snow_ratio_bins is None or not self.snow_ratio_samples:
            return np.zeros_like(precip)
        tmean = 0.5 * (tmax + tmin)
        out = np.zeros_like(precip)
        wet = precip > 0.2
        if not wet.any():
            return out

        n_amount = len(SNOW_AMOUNT_EDGES) - 1
        n_temp = len(self.snow_ratio_bins) - 1
        ti = np.clip(np.digitize(tmean[wet], self.snow_ratio_bins) - 1, 0, n_temp - 1)
        ai = np.clip(np.digitize(precip[wet], SNOW_AMOUNT_EDGES) - 1, 0, n_amount - 1)
        flat = ti * n_amount + ai

        ratios = np.zeros(flat.size)
        for b in np.unique(flat):
            pool = self.snow_ratio_samples[int(b)]
            sel = flat == b
            ratios[sel] = rng.choice(pool, size=int(sel.sum())) if len(pool) else 0.0
        out[wet] = precip[wet] * ratios
        return np.clip(out, 0.0, None)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _tile(arr: np.ndarray, n_paths: int) -> np.ndarray:
    return np.tile(np.asarray(arr, dtype=float), (n_paths, 1))


def _ar1_gaussian(n_paths: int, n_days: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) Gaussian paths, unit marginal variance."""
    out = np.empty((n_paths, n_days))
    eps = rng.standard_normal((n_paths, n_days))
    scale = np.sqrt(max(1.0 - rho * rho, 1e-9))
    out[:, 0] = eps[:, 0]
    for t in range(1, n_days):
        out[:, t] = rho * out[:, t - 1] + scale * eps[:, t]
    return out


def _copula_to_marginal(latent: np.ndarray, tail: SplicedTail) -> np.ndarray:
    """Gaussian copula: push an AR(1) normal through to the spliced marginal.

    The rank correlation is preserved almost exactly and the marginal becomes
    the fitted spliced distribution --- so the simulated series has both the
    right extremes and the right persistence. Drawing the marginal directly
    would lose the autocorrelation; using the AR(1) normal directly would lose
    the tail.
    """
    u = stats.norm.cdf(latent)
    return _spliced_ppf(tail, np.clip(u, 1e-9, 1 - 1e-9))


def _spliced_ppf(tail: SplicedTail, u: np.ndarray) -> np.ndarray:
    out = np.empty_like(u)
    hi_rate = tail.upper.exceedance_rate
    lo_rate = tail.lower.exceedance_rate if tail.lower is not None else 0.0

    is_hi = u > 1.0 - hi_rate
    is_lo = u < lo_rate
    is_body = ~(is_hi | is_lo)

    if is_body.any():
        ub = (u[is_body] - lo_rate) / max(1.0 - lo_rate - hi_rate, 1e-12)
        idx = np.clip((ub * len(tail.sorted_body)).astype(int), 0, len(tail.sorted_body) - 1)
        out[is_body] = tail.sorted_body[idx]
    if is_hi.any():
        p = (1.0 - u[is_hi]) / max(hi_rate, 1e-12)
        out[is_hi] = tail.upper.threshold + stats.genpareto.isf(
            np.clip(p, 1e-12, 1.0), tail.upper.shape, scale=tail.upper.scale
        )
    if is_lo.any() and tail.lower is not None:
        p = u[is_lo] / max(lo_rate, 1e-12)
        out[is_lo] = -(
            -tail.lower.threshold
            + stats.genpareto.isf(np.clip(p, 1e-12, 1.0), tail.lower.shape, scale=tail.lower.scale)
        )
    return out


def _sample_spliced_ratio(
    body: np.ndarray, tail: GPDFit, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw the intensity multiplier: empirical body, GPD above the threshold."""
    u = rng.random(size)
    out = np.empty(size)
    rate = tail.exceedance_rate
    is_hi = u > 1.0 - rate
    is_body = ~is_hi
    if is_body.any() and body.size:
        ub = u[is_body] / max(1.0 - rate, 1e-12)
        idx = np.clip((ub * body.size).astype(int), 0, body.size - 1)
        out[is_body] = body[idx]
    elif is_body.any():
        out[is_body] = 1.0
    if is_hi.any():
        p = (1.0 - u[is_hi]) / max(rate, 1e-12)
        out[is_hi] = tail.threshold + stats.genpareto.isf(
            np.clip(p, 1e-12, 1.0), tail.shape, scale=tail.scale
        )
    return np.clip(out, 0.0, None)


#: Precipitation-magnitude bin edges (mm) for the snow-ratio lookup.
SNOW_AMOUNT_EDGES = np.array([0.2, 3.0, 8.0, 20.0, np.inf])


def _fit_snow_ratio(record: DailyRecord, n_bins: int = 14) -> tuple[np.ndarray, list]:
    """Empirical snow/precip ratio, binned by mean temperature *and* by amount.

    Non-parametric on purpose. The relationship is sharply nonlinear around
    freezing and varies by site (elevation, typical storm type), so a fitted
    functional form would describe it worse than the station's own history.

    The second axis is not optional. Snow ratio and precipitation amount are
    strongly anticorrelated in reality: 20:1 powder falls from cold, dry, small
    events, while big liquid totals arrive on warm advection and land as dense
    3:1 snow. A lookup on temperature alone is free to pair a cold-bin 20:1 ratio
    with a tail precipitation draw, which produced a simulated 1745 mm daily
    snowfall at Vail against 634 mm in the record --- and since snow-drought
    contracts settle on accumulated depth, that error goes straight into the
    price.
    """
    precip = record.precip_mm
    wet = precip > 0.2
    if not wet.any():
        return np.array([]), []
    tmean = 0.5 * (record.tmax_c + record.tmin_c)
    ratio = np.zeros_like(precip)
    ratio[wet] = record.snow_mm[wet] / precip[wet]

    temp_bins = np.linspace(-20.0, 12.0, n_bins + 1)
    n_amount = len(SNOW_AMOUNT_EDGES) - 1
    samples: list[np.ndarray] = []
    for i in range(n_bins):
        in_temp = wet & (tmean >= temp_bins[i]) & (tmean < temp_bins[i + 1])
        for j in range(n_amount):
            sel = in_temp & (precip >= SNOW_AMOUNT_EDGES[j]) & (precip < SNOW_AMOUNT_EDGES[j + 1])
            vals = ratio[sel]
            if vals.size < 8:
                # Back off to the temperature bin as a whole, then to "no snow".
                vals = ratio[in_temp] if in_temp.sum() >= 8 else np.array([0.0])
            samples.append(np.clip(vals, 0.0, 40.0))
    return temp_bins, samples


def _with_variable(record: DailyRecord, variable: str, values: np.ndarray) -> DailyRecord:
    """Shallow copy of a record with one variable replaced."""
    return DailyRecord(
        location_id=record.location_id,
        lat=record.lat,
        lon=record.lon,
        elevation_m=record.elevation_m,
        provenance=record.provenance,
        dates=record.dates,
        year=record.year,
        doy=record.doy,
        tmax_c=values if variable == "tmax_c" else record.tmax_c,
        tmin_c=values if variable == "tmin_c" else record.tmin_c,
        precip_mm=values if variable == "precip_mm" else record.precip_mm,
        snow_mm=values if variable == "snow_mm" else record.snow_mm,
        wind_ms=values if variable == "wind_ms" else record.wind_ms,
    )
