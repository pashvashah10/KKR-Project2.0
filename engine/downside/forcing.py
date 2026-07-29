"""Global forcing to local temperature response.

Two things happen in this module.

1. A **two-box energy balance model** turns an effective-radiative-forcing path
   into a global mean temperature path. This matters more than it sounds: a
   naive linear extrapolation of the observed 100-year trend gets the next
   century badly wrong, because the response to forcing is not linear in time.
   It has a fast surface component and a slow ocean component, so warming
   accelerates while forcing rises and then keeps creeping after forcing flattens
   (the "committed warming" tail that SSP1-2.6 makes visible).

       C_u dT_u/dt = F(t) - lambda*T_u - efficacy*gamma*(T_u - T_d)
       C_d dT_d/dt = gamma*(T_u - T_d)
       lambda      = F_2x / ECS

2. A **regional amplification factor** maps global warming onto the specific
   coordinate. This is estimated from the site's own record by regressing local
   annual mean temperature on modelled global mean temperature, then shrunk
   toward a physical prior --- a hundred years of one noisy station is not enough
   to pin the ratio down on its own, and unshrunk estimates are wild.

Volcanic forcing is included over the historical window. It is not decoration:
Agung / El Chichon / Pinatubo put real multi-year cooling notches in the record,
and a trend model that ignores them attributes that cooling to the trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import C_DEEP, C_UPPER, ECS_C, EFFICACY, F2X, GAMMA, Scenario

__all__ = [
    "TwoBoxResponse",
    "run_two_box",
    "historical_forcing",
    "scenario_forcing",
    "global_temperature_path",
    "shrink_amplification",
]

#: (peak year, peak forcing anomaly W/m^2, e-folding decay years) for the major
#: stratospheric eruptions inside the analysis window.
VOLCANIC_EVENTS: tuple[tuple[float, float, float], ...] = (
    (1883.0, -2.6, 2.0),   # Krakatoa
    (1902.0, -1.4, 1.6),   # Santa Maria
    (1912.0, -1.1, 1.4),   # Novarupta
    (1963.5, -1.5, 1.8),   # Agung
    (1982.4, -1.6, 1.7),   # El Chichon
    (1991.6, -2.4, 2.1),   # Pinatubo
)


@dataclass(slots=True)
class TwoBoxResponse:
    years: np.ndarray
    forcing: np.ndarray
    t_upper: np.ndarray  # global mean surface temperature anomaly, degC
    t_deep: np.ndarray
    ecs: float

    def anomaly_at(self, year: float | np.ndarray) -> np.ndarray:
        return np.interp(year, self.years, self.t_upper)

    def baseline_shift(self, target_year: float | np.ndarray, base_lo: float, base_hi: float) -> np.ndarray:
        """Warming at `target_year` relative to the mean over [base_lo, base_hi]."""
        mask = (self.years >= base_lo) & (self.years <= base_hi)
        base = float(self.t_upper[mask].mean()) if mask.any() else 0.0
        return self.anomaly_at(target_year) - base


def volcanic_forcing(years: np.ndarray) -> np.ndarray:
    """Sum of exponentially-decaying aerosol pulses."""
    years = np.asarray(years, dtype=float)
    out = np.zeros_like(years)
    for peak_year, peak, decay in VOLCANIC_EVENTS:
        dt = years - peak_year
        pulse = np.where(dt >= 0, peak * np.exp(-dt / decay), 0.0)
        # Short ramp-in so the notch is not a step function.
        ramp = np.where((dt < 0) & (dt > -0.6), peak * (1.0 + dt / 0.6), 0.0)
        out += pulse + ramp
    return out


def historical_forcing(years: np.ndarray) -> np.ndarray:
    """Well-mixed GHG + aerosol effective radiative forcing, 1850 onward.

    Nodes approximate the AR6 total anthropogenic ERF series. Volcanic pulses are
    layered on top.
    """
    nodes = np.array(
        [
            [1850, 0.00],
            [1900, 0.20],
            [1920, 0.32],
            [1940, 0.52],
            [1950, 0.60],
            [1960, 0.72],
            [1970, 0.85],
            [1980, 1.15],
            [1990, 1.50],
            [2000, 1.75],
            [2010, 2.20],
            [2020, 2.65],
        ]
    )
    years = np.asarray(years, dtype=float)
    ghg = np.interp(years, nodes[:, 0], nodes[:, 1])
    return ghg + volcanic_forcing(years)


def scenario_forcing(scenario: Scenario, years: np.ndarray) -> np.ndarray:
    """Historical forcing spliced into the scenario pathway at 2020."""
    years = np.asarray(years, dtype=float)
    hist = historical_forcing(years)
    fut = scenario.forcing(years)
    # Smooth handover across 2015-2030 so there is no kink at the splice.
    w = np.clip((years - 2015.0) / 15.0, 0.0, 1.0)
    return (1.0 - w) * hist + w * (fut + volcanic_forcing(years) * (1.0 - w))


def run_two_box(
    years: np.ndarray,
    forcing: np.ndarray,
    ecs: float = ECS_C,
    c_upper: float = C_UPPER,
    c_deep: float = C_DEEP,
    gamma: float = GAMMA,
    efficacy: float = EFFICACY,
) -> TwoBoxResponse:
    """Integrate the two-box model on an annual grid (forward Euler, sub-stepped).

    `c_upper` is small enough that a 1-year step is marginal for stability, so we
    sub-step by quarters.
    """
    years = np.asarray(years, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    lam = F2X / ecs

    n = len(years)
    t_u = np.zeros(n)
    t_d = np.zeros(n)
    sub = 4
    for i in range(1, n):
        dt = (years[i] - years[i - 1]) / sub
        u, d = t_u[i - 1], t_d[i - 1]
        for k in range(sub):
            f = forcing[i - 1] + (forcing[i] - forcing[i - 1]) * (k / sub)
            du = (f - lam * u - efficacy * gamma * (u - d)) / c_upper
            dd = (gamma * (u - d)) / c_deep
            u += du * dt
            d += dd * dt
        t_u[i], t_d[i] = u, d

    return TwoBoxResponse(years=years, forcing=forcing, t_upper=t_u, t_deep=t_d, ecs=ecs)


def global_temperature_path(
    scenario: Scenario,
    start_year: int = 1850,
    end_year: int = 2125,
    ecs: float = ECS_C,
) -> TwoBoxResponse:
    years = np.arange(start_year, end_year + 1, dtype=float)
    return run_two_box(years, scenario_forcing(scenario, years), ecs=ecs)


def shrink_amplification(
    estimate: float,
    std_error: float,
    prior_mean: float,
    prior_sd: float = 0.22,
) -> tuple[float, float]:
    """Precision-weighted blend of the observed amplification and its prior.

    Straight Bayesian conjugate update for a normal mean with known variance
    (Quant Bible section 2.1). One station over one century gives a genuinely
    noisy ratio; the shrinkage keeps a lucky run of hot summers from projecting
    an implausible amplification out to 2125.
    """
    if not np.isfinite(estimate) or std_error <= 0:
        return prior_mean, prior_sd
    w_obs = 1.0 / (std_error**2)
    w_pri = 1.0 / (prior_sd**2)
    post_mean = (w_obs * estimate + w_pri * prior_mean) / (w_obs + w_pri)
    post_sd = float(np.sqrt(1.0 / (w_obs + w_pri)))
    # Amplification below zero is not physical for a continental site.
    return float(np.clip(post_mean, 0.25, 3.0)), post_sd
