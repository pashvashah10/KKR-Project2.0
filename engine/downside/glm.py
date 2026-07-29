"""Generalised linear models, fitted by iteratively reweighted least squares.

Precipitation cannot be modelled with the Gaussian machinery in `climatology.py`,
for two reasons that are both structural rather than cosmetic:

* **It is zero-inflated.** Most days are dry. Roughly 70% of the mass sits on a
  single atom at zero. No amount of transformation makes that Gaussian.
* **It is strictly positive and right-skewed when it does rain**, with variance
  rising with the mean.

So it is modelled the standard way, as occurrence times intensity:

    P(wet today) ~ Bernoulli, logit link, with yesterday's state as a covariate
    amount | wet  ~ Gamma, log link

The logit occurrence model with a lagged wet indicator is exactly a two-state
Markov chain whose transition probabilities vary with day of year and with
global temperature --- so it can express "wet spells cluster" and "the wet season
is shifting later" at the same time, which a fixed chain cannot.

IRLS is used rather than a generic optimiser because it is the natural algorithm
here: each iteration is a weighted least squares solve against the same design
matrix, reusing `fit_ols` and inheriting its diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import LinearFit, fit_ols, fit_ridge

__all__ = ["GLMFit", "fit_logistic", "fit_gamma_log"]

_MAX_ITER = 60
_TOL = 1e-9


@dataclass(slots=True)
class GLMFit:
    family: str
    beta: np.ndarray
    names: list[str]
    n_obs: int
    deviance: float
    null_deviance: float
    converged: bool
    iterations: int
    se: np.ndarray
    dispersion: float = 1.0

    @property
    def pseudo_r2(self) -> float:
        """McFadden-style deviance ratio."""
        if self.null_deviance <= 0:
            return 0.0
        return float(1.0 - self.deviance / self.null_deviance)

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        eta = self.linear_predictor(X)
        if self.family == "binomial":
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        return np.exp(np.clip(eta, -30.0, 30.0))

    def z_scores(self) -> np.ndarray:
        return np.divide(self.beta, self.se, out=np.zeros_like(self.beta), where=self.se > 0)

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def summary_rows(self) -> list[dict]:
        z = self.z_scores()
        return [
            {
                "name": nm,
                "beta": float(b),
                "se": float(s),
                "z": float(zi),
                "significant": bool(abs(zi) >= 2.0),
            }
            for nm, b, s, zi in zip(self.names, self.beta, self.se, z)
        ]


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str] | None = None,
    ridge_lambda: float = 1e-6,
) -> GLMFit:
    """Bernoulli GLM with a logit link.

    A whisper of ridge is on by default. Seasonal harmonics interacted with a
    lagged state produce near-collinear columns in months that are almost always
    dry or almost always wet, and unpenalised IRLS can diverge there --- the
    classic separation problem.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    names = names or [f"x{i}" for i in range(p)]

    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    converged, it, last = False, 0, None

    for it in range(1, _MAX_ITER + 1):
        eta = np.clip(X @ beta, -35.0, 35.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        last = fit_ridge(X, z, ridge_lambda, names=names, weights=w)
        step = last.beta - beta
        beta = last.beta
        if np.max(np.abs(step)) < _TOL:
            converged = True
            break

    eta = np.clip(X @ beta, -35.0, 35.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    deviance = float(-2.0 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    null_deviance = float(-2.0 * np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))

    w = np.clip(mu * (1 - mu), 1e-9, None)
    se = _glm_se(X, w, ridge_lambda)

    return GLMFit(
        family="binomial",
        beta=beta,
        names=list(names),
        n_obs=n,
        deviance=deviance,
        null_deviance=null_deviance,
        converged=converged,
        iterations=it,
        se=se,
    )


def fit_gamma_log(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str] | None = None,
    ridge_lambda: float = 1e-6,
) -> GLMFit:
    """Gamma GLM with a log link, for rainfall amount on wet days.

    Gamma rather than lognormal-via-OLS because the log-link Gamma models
    `E[y|x]` directly. Regressing `log(y)` instead fits the *median* and needs a
    retransformation correction that is only exact under homoskedasticity ---
    which rainfall violates, since wet-season storms are more variable as well as
    larger. Getting this wrong biases every expected payout low.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y = np.clip(y, 1e-6, None)
    n, p = X.shape
    names = names or [f"x{i}" for i in range(p)]

    beta = np.zeros(p)
    beta[0] = np.log(y.mean())
    converged, it, last = False, 0, None

    for it in range(1, _MAX_ITER + 1):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        # Gamma with log link: W = 1 (variance function mu^2 cancels the
        # squared derivative), z = eta + (y - mu)/mu.
        z = eta + (y - mu) / mu
        w = np.ones_like(y)
        last = fit_ridge(X, z, ridge_lambda, names=names, weights=w)
        step = last.beta - beta
        beta = last.beta
        if np.max(np.abs(step)) < _TOL:
            converged = True
            break

    mu = np.exp(np.clip(X @ beta, -30.0, 30.0))
    deviance = float(2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu))
    mu0 = y.mean()
    null_deviance = float(2.0 * np.sum(-np.log(y / mu0) + (y - mu0) / mu0))
    dispersion = float(np.sum(((y - mu) / mu) ** 2) / max(n - p, 1))

    se = _glm_se(X, np.ones_like(y), ridge_lambda) * np.sqrt(max(dispersion, 1e-9))

    return GLMFit(
        family="gamma",
        beta=beta,
        names=list(names),
        n_obs=n,
        deviance=deviance,
        null_deviance=null_deviance,
        converged=converged,
        iterations=it,
        se=se,
        dispersion=dispersion,
    )


def _glm_se(X: np.ndarray, w: np.ndarray, ridge_lambda: float) -> np.ndarray:
    xtwx = (X * w[:, None]).T @ X + np.eye(X.shape[1]) * ridge_lambda
    try:
        cov = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(xtwx)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))
