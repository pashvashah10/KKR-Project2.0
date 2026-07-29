"""Forward projection to 2125, and an honest account of what we don't know.

Projecting is the easy half. The hard half --- and the half that decides whether
a 100-year price is defensible --- is saying how wrong the projection could be,
and *why*.

Uncertainty here is not one number. It is three sources with completely
different behaviour over time, and conflating them produces a band that is
wrong at both ends of the horizon (Hawkins & Sutton's decomposition):

1. **Internal variability.** Weather is noisy. Even with the climate perfectly
   known, any single future decade differs from its own expectation. This
   component is roughly *constant* in absolute terms, so as a share of total
   uncertainty it dominates the first ~20 years and then fades. It is why a
   one-season contract is mostly a bet on noise.

2. **Parameter uncertainty.** We estimated the amplification from 100 years of
   one station and it carries a real standard error --- the block-bootstrap one
   from `resample.py`, not the flattering classical figure. This grows roughly
   linearly with the warming signal.

3. **Scenario uncertainty.** Which emissions path the world takes. Near-zero at
   short horizons (the paths have barely separated by 2035) and *dominant* by
   2100, when SSP1-2.6 and SSP5-8.5 differ by several degrees.

The practical consequence for pricing: a 2027 contract is priced almost entirely
off internal variability, so the historical distribution is nearly sufficient.
A 2075 contract is priced off scenario spread, and quoting it from historical
frequencies is simply wrong. The dashboard shows the crossover explicitly,
because it is the clearest single argument for why this product needs a model
rather than a lookup table.

One deliberate restriction: the station `drift` coefficient is **frozen at its
2025 value and never extrapolated**. Drift absorbs urbanisation and siting
artefacts. Those are real and belong in the fit of the past; projecting a car
park's growth for a further century is not climate science, and left unfrozen it
is the single largest source of nonsense in a long extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .climatology import BASE_HI, BASE_LO, ClimatologyFit
from .config import ECS_C, ECS_SIGMA, HISTORY_END_YEAR, SCENARIOS, Scenario, Location
from .forcing import global_temperature_path, shrink_amplification

__all__ = [
    "ProjectionParams",
    "ProjectedYear",
    "project_variable",
    "uncertainty_decomposition",
    "scenario_responses",
]

#: Cached two-box responses, keyed by (scenario id, ECS). Building one is cheap
#: but it happens inside every Monte Carlo draw, so it is worth memoising.
_RESPONSE_CACHE: dict[tuple[str, float], object] = {}


def scenario_responses(ecs: float = ECS_C) -> dict[str, object]:
    out = {}
    for sc in SCENARIOS:
        key = (sc.id, round(ecs, 4))
        if key not in _RESPONSE_CACHE:
            _RESPONSE_CACHE[key] = global_temperature_path(sc, 1850, 2130, ecs=ecs)
        out[sc.id] = _RESPONSE_CACHE[key]
    return out


@dataclass(slots=True)
class ProjectionParams:
    """Everything needed to evaluate the model at an arbitrary future year."""

    amplification: float
    amplification_sd: float
    #: Seasonal amplification coefficients, evaluated per day-of-year at use time.
    fit: ClimatologyFit = field(repr=False)
    location: Location = field(repr=False)
    ecs_mean: float = ECS_C
    ecs_sd: float = ECS_SIGMA
    freeze_drift_year: float = float(HISTORY_END_YEAR)

    def global_anomaly(self, years: np.ndarray, scenario: Scenario, ecs: float | None = None) -> np.ndarray:
        resp = _RESPONSE_CACHE.get((scenario.id, round(ecs or self.ecs_mean, 4)))
        if resp is None:
            resp = global_temperature_path(scenario, 1850, 2130, ecs=ecs or self.ecs_mean)
            _RESPONSE_CACHE[(scenario.id, round(ecs or self.ecs_mean, 4))] = resp
        return resp.baseline_shift(years, BASE_LO, BASE_HI)


@dataclass(slots=True)
class ProjectedYear:
    year: int
    scenario: str
    #: Mean of the variable over the requested day window.
    mean: float
    #: Standard deviation of the daily value (the weather spread, not the
    #: uncertainty in the mean).
    sd: float
    #: Total uncertainty in the *expected* value, and its three components.
    sigma_internal: float
    sigma_parameter: float
    sigma_scenario: float

    @property
    def sigma_total(self) -> float:
        return float(
            np.sqrt(self.sigma_internal**2 + self.sigma_parameter**2 + self.sigma_scenario**2)
        )

    def band(self, k: float = 1.96) -> tuple[float, float]:
        s = self.sigma_total
        return self.mean - k * s, self.mean + k * s


def build_params(fit: ClimatologyFit, location: Location) -> ProjectionParams:
    """Shrink the fitted amplification toward its physical prior.

    A single station over a single century does not pin the amplification ratio
    down --- the bootstrap standard errors run 0.15-0.35. Shrinking toward the
    regional prior (`forcing.shrink_amplification`, a conjugate normal update)
    keeps a lucky run of hot summers from being extrapolated for a century.
    """
    amp, amp_sd = shrink_amplification(
        estimate=fit.amplification,
        std_error=fit.amplification_se,
        prior_mean=location.amplification_prior,
    )
    return ProjectionParams(
        amplification=amp,
        amplification_sd=amp_sd,
        fit=fit,
        location=location,
    )


def _seasonal_amp_scale(fit: ClimatologyFit, doy: np.ndarray) -> np.ndarray:
    """Seasonal amplification renormalised to a mean of 1 over the window.

    The fitted seasonal-amplification curve is applied as a *shape*, with the
    overall level supplied by the shrunk scalar amplification. Keeping the two
    separate means the shrinkage acts on the level without flattening the
    seasonal structure, which is the part the pricing actually needs (a January
    degree and a July degree are not the same degree).
    """
    curve = fit.seasonal_amplification(doy)
    level = float(np.mean(curve))
    if abs(level) < 1e-9:
        return np.ones_like(curve)
    return curve / level


def project_variable(
    params: ProjectionParams,
    years: np.ndarray,
    doy_window: np.ndarray,
    scenario: Scenario,
    all_scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> list[ProjectedYear]:
    """Project the window-mean of the fitted variable across `years`.

    Returns one `ProjectedYear` per year, each carrying the decomposed
    uncertainty on the expected value.
    """
    fit = params.fit
    doy_window = np.asarray(doy_window, dtype=float)
    amp_shape = _seasonal_amp_scale(fit, doy_window)
    weighted_shape = float(np.mean(amp_shape))

    # Baseline: the fitted mean over the window with G = 0 and the drift frozen.
    frozen = np.full_like(doy_window, params.freeze_drift_year)
    base = float(np.mean(fit.predict_mean(doy_window, np.zeros_like(doy_window), frozen)))

    # Internal variability of the window mean. Daily values inside a window are
    # autocorrelated, so the effective count is well below the day count ---
    # using the raw n would understate this component by a factor of ~2.
    daily_sd = float(np.mean(fit.predict_sd(doy_window, np.zeros_like(doy_window))))
    rho = float(np.clip(fit.ar1, 0.0, 0.95))
    n_days = max(len(doy_window), 1)
    n_eff = n_days * (1.0 - rho) / (1.0 + rho)
    sigma_internal = daily_sd / np.sqrt(max(n_eff, 1.0))

    years = np.asarray(years, dtype=float)
    out: list[ProjectedYear] = []
    for yr in years:
        g = float(params.global_anomaly(np.array([yr]), scenario)[0])
        mean = base + params.amplification * weighted_shape * g

        sigma_param = abs(g * weighted_shape) * params.amplification_sd

        # Scenario spread: prior-weighted standard deviation of the projected
        # mean across the full scenario set at this year.
        means, weights = [], []
        for sc in all_scenarios:
            g_sc = float(params.global_anomaly(np.array([yr]), sc)[0])
            means.append(base + params.amplification * weighted_shape * g_sc)
            weights.append(sc.weight)
        means_arr = np.array(means)
        w = np.array(weights) / np.sum(weights)
        centre = float(np.sum(w * means_arr))
        sigma_scen = float(np.sqrt(np.sum(w * (means_arr - centre) ** 2)))

        # Climate sensitivity uncertainty rides on top of the scenario spread:
        # the same forcing path gives different warming for different ECS.
        g_hi = float(params.global_anomaly(np.array([yr]), scenario, ecs=params.ecs_mean + params.ecs_sd)[0])
        g_lo = float(params.global_anomaly(np.array([yr]), scenario, ecs=max(params.ecs_mean - params.ecs_sd, 1.2))[0])
        sigma_ecs = abs(g_hi - g_lo) / 2.0 * params.amplification * abs(weighted_shape)
        sigma_param = float(np.sqrt(sigma_param**2 + sigma_ecs**2))

        sd_daily = float(np.mean(fit.predict_sd(doy_window, np.full_like(doy_window, g))))

        out.append(
            ProjectedYear(
                year=int(yr),
                scenario=scenario.id,
                mean=float(mean),
                sd=sd_daily,
                sigma_internal=float(sigma_internal),
                sigma_parameter=sigma_param,
                sigma_scenario=sigma_scen,
            )
        )
    return out


def uncertainty_decomposition(projected: list[ProjectedYear]) -> list[dict]:
    """Fractional share of total variance from each source, per year.

    This is the series behind the dashboard's clearest chart: internal
    variability owning the near term, scenario choice owning the far term, and a
    visible crossover between them.
    """
    rows = []
    for p in projected:
        v_int = p.sigma_internal**2
        v_par = p.sigma_parameter**2
        v_scn = p.sigma_scenario**2
        total = v_int + v_par + v_scn
        if total <= 0:
            rows.append(
                {"year": p.year, "internal": 1.0, "parameter": 0.0, "scenario": 0.0, "sigma_total": 0.0}
            )
            continue
        rows.append(
            {
                "year": p.year,
                "internal": float(v_int / total),
                "parameter": float(v_par / total),
                "scenario": float(v_scn / total),
                "sigma_total": float(np.sqrt(total)),
            }
        )
    return rows
