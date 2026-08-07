"""A calibrated 100-year daily record for offline work.

Why this exists
---------------
The engine is built against NOAA GHCN-Daily and ERA5 (see `noaa.py`,
`openmeteo.py`). Where those endpoints are unreachable --- a sandbox with no
egress, an offline demo, a CI run --- the pipeline still needs a hundred years of
daily weather to fit against.

This module produces that record. It is **not** a real observation set and every
artefact it feeds is labelled `provenance="surrogate"` so the distinction never
gets lost. What it is: a structurally faithful stand-in, built forward from
published monthly normals and a physical forcing path, carrying the features that
actually make this estimation problem hard:

* seasonal cycle from the station's own monthly normals, as a Fourier series
* warming driven by the two-box response to historical forcing --- so it has the
  real *shape*, including the 1940-1975 aerosol plateau and the volcanic notches,
  not a straight line
* seasonally asymmetric warming (winters warm faster at continental sites)
* heteroskedastic daily anomalies (winter variance well above summer)
* skewed temperature anomalies --- cold outbreaks are sharper than warm ones
* AR(1) day-to-day persistence, so heat waves and wet spells cluster
* interannual regime variability with an ENSO-like quasi-period, which is what
  makes trend detection genuinely hard rather than trivially easy
* precipitation as occurrence (Markov chain) times intensity (gamma with a heavy
  Pareto tail), with intensity scaling on local temperature at the
  Clausius-Clapeyron rate

The estimators downstream get no privileged access to any of these parameters.
They rediscover them from the daily series, which is the point --- the surrogate
exercises the algorithm rather than short-circuiting it.
"""

from __future__ import annotations

import numpy as np

from ..config import Location, SCENARIOS_BY_ID
from ..forcing import global_temperature_path
from .base import DailyRecord

__all__ = ["SyntheticSource"]

#: Day-of-year at the centre of each month (non-leap).
MONTH_CENTRES = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5])


def _harmonic_design(doy: np.ndarray, n_harmonics: int) -> np.ndarray:
    """[1, cos(1), sin(1), ..., cos(k), sin(k)] on an annual period."""
    phase = 2.0 * np.pi * np.asarray(doy, dtype=float) / 365.25
    cols = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * phase))
        cols.append(np.sin(k * phase))
    return np.column_stack(cols)


def _fit_monthly_harmonics(monthly: tuple[float, ...], n_harmonics: int = 3) -> np.ndarray:
    """Least-squares Fourier coefficients through the twelve monthly normals."""
    X = _harmonic_design(MONTH_CENTRES, n_harmonics)
    y = np.asarray(monthly, dtype=float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _evaluate(coefs: np.ndarray, doy: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    return _harmonic_design(doy, n_harmonics) @ coefs


_MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_OF_DOY = np.repeat(np.arange(12), _MONTH_LENGTHS)
_MONTH_OF_DOY = np.concatenate([_MONTH_OF_DOY, [11]])  # pad doy 366


def _month_index(doy: np.ndarray) -> np.ndarray:
    return _MONTH_OF_DOY[np.clip(np.asarray(doy, dtype=int) - 1, 0, 365)]


def _daily_from_monthly(
    monthly: tuple[float, ...] | np.ndarray,
    doy: np.ndarray,
    link: str = "log",
    n_harmonics: int = 3,
    iterations: int = 6,
) -> np.ndarray:
    """Smooth day-of-year curve whose *monthly means reproduce the normals exactly*.

    A plain Fourier fit through the twelve month-centre values does not do this.
    Smoothing shaves the peaks, so the implied annual total comes out low ---
    badly so where the seasonal cycle is sharp, like the Orlando summer wet
    season, where a 2-harmonic fit lost 15% of annual rainfall.

    Fixing it matters beyond cosmetics: every exceedance probability and every
    premium in the book is an integral over this curve, so a 15% shortfall in the
    wet season is a 15% error in the rain-day price.

    So: fit, measure the realised monthly means, rescale the targets by the
    shortfall, refit. Converges in a handful of passes and keeps the curve smooth
    because the correction is applied to the fit inputs, not to the output days.
    """
    target = np.asarray(monthly, dtype=float)
    grid = np.arange(1, 367)
    grid_month = _month_index(grid)

    def curve_from(vals: np.ndarray, at: np.ndarray) -> np.ndarray:
        if link == "log":
            coefs = _fit_monthly_harmonics(tuple(np.log(np.clip(vals, 1e-6, None))), n_harmonics)
            return np.exp(_evaluate(coefs, at, n_harmonics))
        if link == "logit":
            p = np.clip(vals, 1e-4, 1 - 1e-4)
            coefs = _fit_monthly_harmonics(tuple(np.log(p / (1 - p))), n_harmonics)
            return 1.0 / (1.0 + np.exp(-_evaluate(coefs, at, n_harmonics)))
        coefs = _fit_monthly_harmonics(tuple(vals), n_harmonics)
        return _evaluate(coefs, at, n_harmonics)

    work = target.copy()
    for _ in range(iterations):
        full = curve_from(work, grid)
        realised = np.array([full[grid_month == m].mean() for m in range(12)])
        if link == "identity":
            # Temperatures cross zero, so the correction has to be additive ---
            # a ratio correction diverges the moment a monthly normal sits near
            # 0 degC, which it does at every mountain site in midwinter.
            work = work + (target - realised)
        else:
            work = work * target / np.clip(realised, 1e-9, None)
            if link == "logit":
                work = np.clip(work, 1e-4, 0.98)

    return curve_from(work, np.asarray(doy))


class SyntheticSource:
    """Generates the surrogate record. Deterministic given `(location, seed)`."""

    name = "surrogate"

    def __init__(self, seed: int = 20260728):
        self.seed = seed

    # ------------------------------------------------------------------
    def fetch(self, location: Location, start_year: int, end_year: int) -> DailyRecord:
        rng = np.random.default_rng(self.seed + (abs(hash(location.id)) % 100_000))

        dates = np.arange(
            np.datetime64(f"{start_year}-01-01"),
            np.datetime64(f"{end_year + 1}-01-01"),
            dtype="datetime64[D]",
        )
        year = dates.astype("datetime64[Y]").astype(int) + 1970
        jan1 = dates.astype("datetime64[Y]").astype("datetime64[D]")
        doy = (dates - jan1).astype(int) + 1
        n = len(dates)

        # -- seasonal climatology from published normals -----------------
        seas_tmax = _daily_from_monthly(location.normals.tmax_c, doy, link="identity")
        seas_tmin = _daily_from_monthly(location.normals.tmin_c, doy, link="identity")
        seas_wind = np.clip(_daily_from_monthly(location.normals.wind_ms, doy, link="log"), 0.4, None)

        # -- forcing-driven warming --------------------------------------
        # Normals describe 1991-2020, so the warming signal is expressed relative
        # to that window and the generated series reproduces it by construction.
        response = global_temperature_path(SCENARIOS_BY_ID["ssp245"], 1850, end_year + 1)
        dec_year = year + (doy - 1) / 365.25
        global_anom = response.baseline_shift(dec_year, 1991, 2020)

        # Scale so the site's century-scale warming matches its observed value.
        span = response.baseline_shift(np.array([2025.0]), 1991, 2020)[0] - response.baseline_shift(
            np.array([1925.0]), 1991, 2020
        )[0]
        amp = location.observed_trend_c_per_century / max(span, 1e-6)
        local_warm = amp * global_anom

        # Seasonal asymmetry: cold-season warming runs faster inland. Expressed
        # as a multiplier peaking in January and troughing in July.
        season_phase = np.cos(2.0 * np.pi * (doy - 15.0) / 365.25)
        asym = 1.0 + 0.30 * season_phase * (location.amplification_prior - 0.8)
        warm_tmax = local_warm * asym
        # Minimums warm faster than maximums almost everywhere --- reduced diurnal
        # range is one of the most robust signals in the observed record.
        warm_tmin = local_warm * asym * 1.18

        # -- interannual regime variability (ENSO-like) -------------------
        regime = self._regime_series(start_year, end_year, rng)
        regime_daily = np.interp(dec_year, np.arange(start_year, end_year + 1) + 0.5, regime)
        # Teleconnection loading: strongest in winter, sign varies by region.
        tele_t = self._temp_teleconnection(location)
        tele_p = self._precip_teleconnection(location)
        regime_t = tele_t * regime_daily * (0.6 + 0.4 * season_phase)
        regime_p = tele_p * regime_daily * (0.6 + 0.4 * season_phase)

        # -- daily temperature anomalies ----------------------------------
        # Heteroskedastic: winter spread is materially wider than summer.
        continentality = 0.55 + 0.45 * min(abs(location.lat) / 45.0, 1.4)
        sd_tmax = continentality * (3.05 + 1.35 * season_phase)
        sd_tmin = continentality * (2.70 + 1.15 * season_phase)
        sd_tmax = np.clip(sd_tmax, 1.1, None)
        sd_tmin = np.clip(sd_tmin, 1.0, None)

        rho = 0.74  # day-to-day persistence of the synoptic anomaly
        shared = self._ar1_skewed(n, rho, skew=-0.45, rng=rng)
        idio_max = self._ar1_skewed(n, rho * 0.85, skew=-0.25, rng=rng)
        idio_min = self._ar1_skewed(n, rho * 0.85, skew=-0.35, rng=rng)

        anom_tmax = sd_tmax * (0.82 * shared + 0.57 * idio_max)
        anom_tmin = sd_tmin * (0.82 * shared + 0.57 * idio_min)

        tmax = seas_tmax + warm_tmax + regime_t + anom_tmax
        tmin = seas_tmin + warm_tmin + regime_t * 0.9 + anom_tmin
        # Physical ordering must hold; compress rather than swap so the diurnal
        # range stays positive without inventing spurious spread.
        gap = tmax - tmin
        too_small = gap < 1.0
        mid = 0.5 * (tmax + tmin)
        tmax = np.where(too_small, mid + 0.5, tmax)
        tmin = np.where(too_small, mid - 0.5, tmin)

        # -- precipitation -------------------------------------------------
        precip = self._precipitation(location, doy, dec_year, local_warm, regime_p, shared, rng)

        # -- snowfall --------------------------------------------------------
        snow = self._snowfall(precip, tmax, tmin)

        # -- wind -------------------------------------------------------------
        wind = self._wind(seas_wind, shared, rng)

        return DailyRecord(
            location_id=location.id,
            lat=location.lat,
            lon=location.lon,
            elevation_m=location.elevation_m,
            provenance="surrogate",
            dates=dates,
            year=year,
            doy=doy,
            tmax_c=tmax,
            tmin_c=tmin,
            precip_mm=precip,
            snow_mm=snow,
            wind_ms=wind,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _regime_series(start_year: int, end_year: int, rng: np.random.Generator) -> np.ndarray:
        """AR(2) with complex roots --- a damped oscillation near a 4.2-year period.

        This is what gives the record realistic low-frequency variance. Without
        it, a 100-year trend would be trivially detectable and every uncertainty
        band in the dashboard would be dishonestly narrow.
        """
        n = end_year - start_year + 1
        period, damping = 4.2, 0.86
        phi1 = 2.0 * damping * np.cos(2.0 * np.pi / period)
        phi2 = -(damping**2)
        x = np.zeros(n + 60)
        eps = rng.standard_normal(n + 60)
        for t in range(2, len(x)):
            x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]
        x = x[60:]
        return x / max(x.std(), 1e-9)

    @staticmethod
    def _temp_teleconnection(location: Location) -> float:
        """degC of winter temperature response per unit regime index."""
        table = {
            "Southern Rockies": 0.42,
            "Northern Rockies": 0.55,
            "California Coastal": 0.34,
            "Desert Southwest": 0.38,
            "Southeast Peninsula": -0.30,
            "Southern Plains": -0.26,
            "Southern Appalachia": -0.22,
            "New England Coastal": 0.30,
        }
        return table.get(location.region, 0.30)

    @staticmethod
    def _precip_teleconnection(location: Location) -> float:
        """Fractional precipitation response per unit regime index."""
        table = {
            "Southern Rockies": -0.10,
            "Northern Rockies": -0.16,
            "California Coastal": 0.20,
            "Desert Southwest": 0.24,
            "Southeast Peninsula": 0.18,
            "Southern Plains": 0.22,
            "Southern Appalachia": 0.10,
            "New England Coastal": 0.06,
        }
        return table.get(location.region, 0.10)

    @staticmethod
    def _ar1_skewed(n: int, rho: float, skew: float, rng: np.random.Generator) -> np.ndarray:
        """Unit-variance AR(1) driven by skewed innovations.

        Skew-normal innovations via the standard bivariate construction. Cold
        outbreaks in the real record overshoot further than warm spells do, and a
        symmetric generator would understate the cold tail.
        """
        delta = np.clip(skew, -0.95, 0.95)
        u0 = rng.standard_normal(n)
        u1 = rng.standard_normal(n)
        z = delta * np.abs(u0) + np.sqrt(max(1.0 - delta**2, 1e-9)) * u1
        z = (z - z.mean()) / max(z.std(), 1e-9)

        out = np.empty(n)
        scale = np.sqrt(max(1.0 - rho**2, 1e-9))
        out[0] = z[0]
        for t in range(1, n):
            out[t] = rho * out[t - 1] + scale * z[t]
        return out

    def _precipitation(
        self,
        location: Location,
        doy: np.ndarray,
        dec_year: np.ndarray,
        local_warm: np.ndarray,
        regime_p: np.ndarray,
        synoptic: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Occurrence x intensity, the standard Richardson weather-generator form."""
        n = len(doy)
        norm = location.normals

        # Target wet-day probability per day of year, from monthly wet-day counts.
        days_in_month = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        p_wet_month = np.clip(np.asarray(norm.wet_days) / days_in_month, 0.01, 0.95)
        p_wet = _daily_from_monthly(p_wet_month, doy, link="logit")

        # Mean intensity on wet days, from monthly totals / wet days.
        mean_int_month = np.asarray(norm.precip_mm) / np.maximum(np.asarray(norm.wet_days), 0.3)
        mean_int = _daily_from_monthly(np.clip(mean_int_month, 0.4, None), doy, link="log")

        # Regime and warming modulation. Intensity scales with local warming at
        # roughly the Clausius-Clapeyron rate; occurrence barely moves.
        #
        # The two regime multipliers are applied to factors that then multiply
        # together, so E[(1+a*r)(1+b*r)] = 1 + a*b*Var(r) --- a positive bias on
        # long-run totals. It is taken out here, using the *realised* variance of
        # `regime_p`: the underlying regime index is standardised, but regime_p
        # has already been scaled by a teleconnection loading well below one, so
        # assuming unit variance over-corrects and pulls annual totals ~13% under
        # the published normals.
        a_occ, b_int = 0.45, 0.35
        regime_bias = 1.0 + a_occ * b_int * float(np.var(regime_p))
        cc = 1.0 + location.precip_intensity_scaling * local_warm
        p_wet = np.clip(p_wet * (1.0 + a_occ * regime_p), 0.005, 0.96)
        mean_int = mean_int * cc * (1.0 + b_int * regime_p) / regime_bias

        # Two-state Markov occurrence. Persistence ratio r > 1 makes wet days
        # cluster into spells, matching observed run-length statistics.
        #
        # Occurrence is generated twice. The realised wet-day frequency drifts
        # from the target because the chain is clipped, tilted by the synoptic
        # driver and modulated by the regime index, and in dry months the logit
        # curve overshoots. One correction pass measures the drift per calendar
        # month and rescales the target before regenerating.
        tilt = -0.10 * synoptic
        month = _month_index(doy)
        target_rate = np.clip(np.asarray(norm.wet_days) / days_in_month, 0.005, 0.95)
        calib = (dec_year >= 1991) & (dec_year <= 2021)

        wet = self._markov_occurrence(p_wet, tilt, rng.spawn(1)[0])
        for m in range(12):
            sel = calib & (month == m)
            if sel.sum() < 30:
                continue
            realised = wet[sel].mean()
            if realised > 1e-4:
                p_wet[month == m] *= target_rate[m] / realised
        p_wet = np.clip(p_wet, 0.002, 0.97)
        wet = self._markov_occurrence(p_wet, tilt, rng.spawn(1)[0])

        # Intensity: gamma body with a heavy-tailed component. A pure gamma
        # underprices cloudbursts, which is exactly the peril being sold.
        shape = 0.72
        amounts = np.zeros(n)
        idx = np.flatnonzero(wet)
        if idx.size:
            scale = mean_int[idx] / shape
            body = rng.gamma(shape, scale)
            heavy_mask = rng.random(idx.size) < 0.045
            # Pareto multiplier on the heavy fraction, tail index ~2.6.
            heavy_mult = (1.0 - rng.random(idx.size)) ** (-1.0 / 2.6)
            amounts[idx] = body * np.where(heavy_mask, np.clip(heavy_mult, 1.0, 14.0), 1.0)

        # Final per-month scaling so the 1991-2020 totals reproduce the normals.
        # A single multiplicative factor per month leaves the shape of the
        # intensity distribution --- and therefore the fitted tail index --- alone.
        for m in range(12):
            sel = calib & (month == m)
            if sel.sum() < 30:
                continue
            realised = amounts[sel].sum() / (sel.sum() / days_in_month[m])
            if realised > 1e-6:
                amounts[month == m] *= float(norm.precip_mm[m] / realised)

        return np.round(np.clip(amounts, 0.0, None), 2)

    @staticmethod
    def _markov_occurrence(p_wet: np.ndarray, tilt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Two-state chain whose stationary wet probability is `p_wet`.

        With p_ww = p*r and p_dw = p(1-p_ww)/(1-p), the stationary distribution
        works out to exactly p for any persistence ratio r, so spell clustering
        can be tuned without disturbing the wet-day frequency.
        """
        r = 1.85
        n = len(p_wet)
        u = rng.random(n)
        wet = np.zeros(n, dtype=bool)
        prev = bool(u[0] < p_wet[0])
        wet[0] = prev
        for t in range(1, n):
            p = p_wet[t]
            p_ww = min(p * r, 0.97)
            p_dw = min(p * (1.0 - p_ww) / max(1.0 - p, 1e-6), 0.97)
            thresh = min(max((p_ww if prev else p_dw) + tilt[t], 0.001), 0.985)
            prev = bool(u[t] < thresh)
            wet[t] = prev
        return wet

    @staticmethod
    def _snowfall(precip: np.ndarray, tmax: np.ndarray, tmin: np.ndarray) -> np.ndarray:
        """Liquid-equivalent to snow depth, using a temperature-dependent ratio.

        Snow fraction ramps across the wet-bulb-ish band between -1C and +3C mean
        temperature; the depth ratio rises as it gets colder (dry powder) and
        collapses near freezing (heavy wet snow).

        The ratio is additionally damped at high precipitation amounts. Very high
        ratios and very large liquid totals do not co-occur in reality --- big
        totals come from warm advection carrying moisture, which produces dense
        snow. Multiplying an unconstrained 20:1 ratio by a tail precipitation
        draw would otherwise manufacture single-day snowfalls beyond any observed
        record and blow out the snow-drought pricing.
        """
        tmean = 0.5 * (tmax + tmin)
        frac = np.clip((1.5 - tmean) / 3.5, 0.0, 1.0)
        ratio = np.clip(8.0 + 0.85 * (0.0 - tmean), 6.0, 20.0)
        ratio = ratio * np.clip(1.0 - 0.006 * precip, 0.42, 1.0)
        return np.round(precip * frac * ratio, 1)

    @staticmethod
    def _wind(seasonal: np.ndarray, synoptic: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Weibull-distributed daily mean wind with synoptic persistence.

        Built by mapping a persistent gaussian through the Weibull quantile
        function, so the marginal is Weibull(k~2) while day-to-day correlation
        survives. Independent daily wind would badly underprice multi-day
        shutdowns.
        """
        n = len(seasonal)
        z = np.empty(n)
        rho = 0.62
        eps = rng.standard_normal(n)
        z[0] = eps[0]
        s = np.sqrt(max(1.0 - rho**2, 1e-9))
        for t in range(1, n):
            z[t] = rho * z[t - 1] + s * eps[t]
        # Blend in the shared synoptic driver: windy days track pressure systems.
        z = 0.75 * z + 0.25 * synoptic
        from scipy import stats as _st

        u = np.clip(_st.norm.cdf(z), 1e-6, 1 - 1e-6)
        k = 2.05
        lam = seasonal / 0.8862  # mean of Weibull(k=2) is lam * Gamma(1+1/k)
        return np.round(lam * (-np.log(1.0 - u)) ** (1.0 / k), 2)
