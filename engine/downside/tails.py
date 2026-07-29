"""Extreme value models --- where the money is.

Everything a weather-protection product pays on lives in the tail. A Gaussian
body fitted to daily anomalies describes the middle of the distribution well and
then understates the frequency of the events that actually trigger payouts,
sometimes by an order of magnitude. Pricing off a Gaussian tail is the single
most reliable way to go bust in this business.

Two complementary tools, both standard extreme value theory:

**Peaks over threshold (GPD).** By the Pickands-Balkema-de Haan theorem, the
distribution of exceedances above a high threshold converges to the Generalised
Pareto regardless of the parent distribution. Fitting it gives a shape parameter
`xi` that governs how fat the tail is:

    xi > 0   heavy tail, unbounded --- daily precipitation lives here
    xi = 0   exponential tail
    xi < 0   bounded tail with a finite upper limit --- daily temperature,
             which is capped by the physics of air masses, lives here

**Block maxima (GEV).** Fitted to annual maxima. Less statistically efficient
than POT because it throws away all but one observation per year, but it maps
directly onto the return periods operators and reinsurers actually talk in
("the 1-in-50-year heat event"), so it is worth carrying for the interface even
where POT drives the pricing.

The distribution used downstream is a **splice**: empirical below the threshold,
GPD above it. That keeps the well-observed body honest and the sparse tail
smooth and extrapolable, without the discontinuity a naive switch would create.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = ["GPDFit", "GEVFit", "fit_gpd", "fit_gev", "SplicedTail", "mean_residual_life"]


@dataclass(slots=True)
class GPDFit:
    threshold: float
    shape: float  # xi
    scale: float  # sigma
    n_exceedances: int
    n_total: int
    #: P(X > threshold), the rate the conditional GPD is scaled by.
    exceedance_rate: float
    converged: bool
    shape_se: float = 0.0
    scale_se: float = 0.0

    def survival(self, x: np.ndarray) -> np.ndarray:
        """Unconditional P(X > x) for x above the threshold."""
        x = np.asarray(x, dtype=float)
        excess = np.clip(x - self.threshold, 0.0, None)
        return self.exceedance_rate * stats.genpareto.sf(excess, self.shape, scale=self.scale)

    def return_level(self, return_period: float, per_year: float = 365.25) -> float:
        """Level exceeded once every `return_period` years on average."""
        target = 1.0 / (return_period * per_year)
        if target >= self.exceedance_rate:
            return float(self.threshold)
        p = target / self.exceedance_rate
        return float(self.threshold + stats.genpareto.isf(p, self.shape, scale=self.scale))

    @property
    def upper_bound(self) -> float | None:
        """Finite endpoint when xi < 0 --- the physical ceiling the fit implies."""
        if self.shape >= -1e-6:
            return None
        return float(self.threshold - self.scale / self.shape)


@dataclass(slots=True)
class GEVFit:
    shape: float
    loc: float
    scale: float
    n_blocks: int
    converged: bool

    def return_level(self, return_period: float) -> float:
        return float(stats.genextreme.isf(1.0 / return_period, self.shape, loc=self.loc, scale=self.scale))

    def return_period(self, level: float) -> float:
        p = stats.genextreme.sf(level, self.shape, loc=self.loc, scale=self.scale)
        return float(1.0 / max(p, 1e-12))


def mean_residual_life(x: np.ndarray, quantiles: np.ndarray | None = None) -> list[dict]:
    """Mean excess over a grid of candidate thresholds.

    GPD validity requires a threshold high enough for the asymptotic result to
    bite. Above such a threshold the mean excess is linear in the threshold, so
    the point where this plot straightens is the diagnostic for threshold choice.
    Exposed so the dashboard can show *why* a threshold was picked rather than
    asserting one.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if quantiles is None:
        quantiles = np.linspace(0.80, 0.995, 30)
    out = []
    for q in quantiles:
        u = float(np.quantile(x, q))
        exc = x[x > u] - u
        if exc.size < 12:
            continue
        out.append(
            {
                "quantile": float(q),
                "threshold": u,
                "mean_excess": float(exc.mean()),
                "se": float(exc.std(ddof=1) / np.sqrt(exc.size)),
                "n": int(exc.size),
            }
        )
    return out


def fit_gpd(
    x: np.ndarray,
    threshold_quantile: float = 0.97,
    threshold: float | None = None,
    lower_tail: bool = False,
) -> GPDFit:
    """Maximum-likelihood GPD fit to exceedances above a high threshold.

    `lower_tail=True` flips the data so the same machinery fits a *cold* or *dry*
    tail --- hard freezes and snow droughts are lower-tail perils.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if lower_tail:
        x = -x
        if threshold is not None:
            threshold = -threshold

    n_total = x.size
    u = float(np.quantile(x, threshold_quantile)) if threshold is None else float(threshold)
    excess = x[x > u] - u
    n_exc = excess.size

    if n_exc < 25:
        # Too few points for a stable shape parameter. Fall back to an
        # exponential tail (xi = 0), which is the honest default: it neither
        # invents a heavy tail nor imposes a ceiling.
        scale = float(excess.mean()) if n_exc else 1.0
        return GPDFit(
            threshold=-u if lower_tail else u,
            shape=0.0,
            scale=max(scale, 1e-6),
            n_exceedances=n_exc,
            n_total=n_total,
            exceedance_rate=n_exc / max(n_total, 1),
            converged=False,
        )

    shape, scale, converged = _gpd_mle(excess)
    shape_se, scale_se = _gpd_se(excess, shape, scale)

    return GPDFit(
        threshold=-u if lower_tail else u,
        shape=shape,
        scale=scale,
        n_exceedances=n_exc,
        n_total=n_total,
        exceedance_rate=n_exc / max(n_total, 1),
        converged=converged,
        shape_se=shape_se,
        scale_se=scale_se,
    )


def _gpd_mle(excess: np.ndarray) -> tuple[float, float, bool]:
    """Grimshaw's one-dimensional profile likelihood.

    Substituting `theta = xi / sigma` collapses the two-parameter likelihood onto
    one variable. For fixed theta the profile maximum is closed-form:

        xi(theta)    = mean( log(1 + theta * x) )
        sigma(theta) = xi(theta) / theta
        -logL(theta) = n * log(sigma) + n * (1 + xi)

    subject to `1 + theta*x_i > 0` for every exceedance (so `theta > -1/max(x)`)
    and `sigma > 0` (so xi and theta must share a sign).

    Working in one dimension matters. scipy's two-parameter `genpareto.fit`
    routinely wanders into the region where the support constraint is violated
    and returns whatever the optimiser stopped on.

    The negative-shape case needs specific care: when `xi < 0` the distribution
    has a finite endpoint and the maximising theta sits hard against
    `-1/max(x)`. The search is therefore seeded densely just inside that
    boundary rather than trusting a generic bracketing method to find it.
    """
    n = excess.size
    xmax = float(excess.max())
    if xmax <= 0:
        return 0.0, 1e-6, False
    boundary = -1.0 / xmax

    def profile(theta: float) -> tuple[float, float] | None:
        if abs(theta) < 1e-13:
            return None
        w = 1.0 + theta * excess
        if np.any(w <= 1e-12):
            return None
        xi = float(np.mean(np.log(w)))
        if abs(xi) < 1e-12 or xi / theta <= 0:
            return None
        return xi, xi / theta

    def neg_ll(theta: float) -> float:
        got = profile(float(theta))
        if got is None:
            return 1e12
        xi, sigma = got
        if sigma <= 0 or not np.isfinite(sigma):
            return 1e12
        return n * np.log(sigma) + n * (1.0 + xi)

    # Dense just inside the negative boundary (that is where xi < 0 lives),
    # log-spaced on the positive side (xi > 0).
    #
    # The positive arm is scaled by the *mean* excess, not the max. theta is
    # xi/sigma and sigma tracks the mean excess, whereas for a heavy tail the max
    # can be orders of magnitude larger --- scaling by it pushes the whole grid
    # below the optimum and silently returns a too-thin tail.
    mean_excess = float(excess.mean())
    eps = np.logspace(-10, -0.3, 80)
    grid = np.concatenate(
        [
            boundary * (1.0 - eps),
            np.logspace(-6, 1.0, 90) / max(mean_excess, 1e-9),
            np.logspace(-6, 1.5, 40) / xmax,
        ]
    )
    grid = np.unique(grid)
    values = np.array([neg_ll(t) for t in grid])
    best = int(np.argmin(values))
    if values[best] >= 1e11:
        return _gpd_pwm(excess)

    lo, hi = sorted((float(grid[max(best - 1, 0)]), float(grid[min(best + 1, len(grid) - 1)])))
    theta = float(grid[best])
    if hi > lo:
        res = optimize.minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
        if res.success and neg_ll(float(res.x)) <= values[best]:
            theta = float(res.x)

    got = profile(theta)
    if got is None:
        return _gpd_pwm(excess)
    xi, sigma = got
    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        return _gpd_pwm(excess)
    return float(np.clip(xi, -0.75, 0.95)), float(sigma), True


def _gpd_pwm(excess: np.ndarray) -> tuple[float, float, bool]:
    """Probability-weighted moments --- less efficient, but never blows up."""
    n = excess.size
    m = float(excess.mean())
    order = np.sort(excess)
    a1 = float(np.mean(order * (1.0 - (np.arange(1, n + 1) - 0.35) / n)))
    denom = m - 2.0 * a1
    if abs(denom) < 1e-12:
        return 0.0, max(m, 1e-6), False
    xi = 2.0 - m / denom
    sigma = m * (1.0 - xi) if xi < 1 else m
    return float(np.clip(xi, -0.75, 0.95)), float(max(sigma, 1e-6)), False


def _gpd_se(excess: np.ndarray, shape: float, scale: float) -> tuple[float, float]:
    """Asymptotic standard errors from the GPD information matrix.

    Valid for xi > -0.5; outside that the MLE is not asymptotically normal and
    zeros are returned rather than a misleading number.
    """
    n = excess.size
    if shape <= -0.5 + 1e-3 or n < 25:
        return 0.0, 0.0
    cov = (1.0 + shape) / n * np.array([[1.0 + shape, scale], [scale, 2.0 * scale**2]])
    try:
        return float(np.sqrt(max(cov[0, 0], 0.0))), float(np.sqrt(max(cov[1, 1], 0.0)))
    except (ValueError, FloatingPointError):
        return 0.0, 0.0


def fit_gev(block_maxima: np.ndarray) -> GEVFit:
    """GEV fit to annual maxima (or minima, if the caller negates first)."""
    x = np.asarray(block_maxima, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 12:
        return GEVFit(shape=0.0, loc=float(x.mean()) if x.size else 0.0, scale=1.0, n_blocks=x.size, converged=False)
    try:
        shape, loc, scale = stats.genextreme.fit(x)
        if not np.isfinite([shape, loc, scale]).all() or scale <= 0:
            raise ValueError
        return GEVFit(shape=float(shape), loc=float(loc), scale=float(scale), n_blocks=x.size, converged=True)
    except (ValueError, RuntimeError, FloatingPointError):
        # Gumbel fallback (shape = 0), which is the right default for a
        # temperature-like variable when the MLE will not settle.
        loc, scale = stats.gumbel_r.fit(x)
        return GEVFit(shape=0.0, loc=float(loc), scale=float(scale), n_blocks=x.size, converged=False)


@dataclass(slots=True)
class SplicedTail:
    """Empirical body + GPD tails, as one callable distribution.

    Used for the standardised residuals `z`. Sampling from this rather than from
    a normal is what gives the Monte Carlo realistic extremes.
    """

    sorted_body: np.ndarray
    upper: GPDFit
    lower: GPDFit | None = None

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(size)
        out = np.empty(size)

        hi_rate = self.upper.exceedance_rate
        lo_rate = self.lower.exceedance_rate if self.lower is not None else 0.0

        is_hi = u > 1.0 - hi_rate
        is_lo = u < lo_rate
        is_body = ~(is_hi | is_lo)

        if is_body.any():
            # Inverse-CDF draw from the empirical body. `sorted_body` holds only
            # the values *between* the two thresholds, so the uniform maps across
            # the body alone --- indexing the full sorted sample here would let a
            # body draw return a tail value that the GPD arms already account
            # for, double-counting the extremes and inflating the variance.
            ub = (u[is_body] - lo_rate) / max(1.0 - lo_rate - hi_rate, 1e-12)
            idx = np.clip((ub * len(self.sorted_body)).astype(int), 0, len(self.sorted_body) - 1)
            out[is_body] = self.sorted_body[idx]

        if is_hi.any():
            p = (1.0 - u[is_hi]) / max(hi_rate, 1e-12)
            out[is_hi] = self.upper.threshold + stats.genpareto.isf(
                np.clip(p, 1e-12, 1.0), self.upper.shape, scale=self.upper.scale
            )

        if is_lo.any() and self.lower is not None:
            p = u[is_lo] / max(lo_rate, 1e-12)
            # `lower` was fitted on negated data, so undo the flip.
            out[is_lo] = -(
                -self.lower.threshold
                + stats.genpareto.isf(np.clip(p, 1e-12, 1.0), self.lower.shape, scale=self.lower.scale)
            )

        return out

    @classmethod
    def from_residuals(
        cls,
        z: np.ndarray,
        upper_q: float = 0.97,
        lower_q: float = 0.03,
        two_sided: bool = True,
    ) -> "SplicedTail":
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        upper = fit_gpd(z, threshold_quantile=upper_q)
        lower = fit_gpd(z, threshold_quantile=1.0 - lower_q, lower_tail=True) if two_sided else None

        keep = z <= upper.threshold
        if lower is not None:
            keep &= z >= lower.threshold
        body = np.sort(z[keep])
        if body.size == 0:
            body = np.sort(z)
        return cls(sorted_body=body, upper=upper, lower=lower)
