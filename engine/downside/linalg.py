"""Linear-model toolkit.

This is the statistical spine of the whole engine. Everything downstream --- the
seasonal climatology, the variance model, the trend extrapolation --- is a linear
model on a cleverly-built design matrix, so it all funnels through here.

The estimators and diagnostics follow the Quant Bible (MIT SBC) sections 4.3
"Regressions" and 4.6 "The Econometrics Perspective":

    beta_hat  = (X'X)^-1 X'y                    closed form OLS
    H         = X (X'X)^-1 X'                   hat / projection matrix
    sigma^2   = RSS / (N - p - 1)               sample variance
    Var(beta) = (X'X)^-1 sigma^2                coefficient covariance
    z_j       = beta_j / (sigma * sqrt(v_j))    per-coefficient t-test
    F         = ((RSS0-RSS1)/(p1-p0)) / (RSS1/(N-p1-1))    group test
    beta_ridge= (X'X + lambda I)^-1 X'y         always invertible

The Bible flags that a nonlinear conditional expectation function guarantees
heteroskedasticity, which is exactly our situation (daily weather variance swings
by season), so `fit` also reports HC1 robust standard errors alongside the
classical ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "LinearFit",
    "fit_ols",
    "fit_ridge",
    "f_test",
    "design_from_blocks",
    "prior_to_penalty",
    "omitted_variable_bias",
]


@dataclass(slots=True)
class LinearFit:
    """A fitted linear model plus the diagnostics needed to defend it."""

    beta: np.ndarray
    names: list[str]
    n_obs: int
    n_params: int
    rss: float
    tss: float
    sigma2: float
    xtx_inv: np.ndarray
    se_classical: np.ndarray
    se_robust: np.ndarray
    ridge_lambda: float = 0.0
    weights: np.ndarray | None = field(default=None, repr=False)

    # --- goodness of fit -------------------------------------------------
    @property
    def r_squared(self) -> float:
        return 0.0 if self.tss <= 0 else float(1.0 - self.rss / self.tss)

    @property
    def adj_r_squared(self) -> float:
        dof = self.n_obs - self.n_params
        if dof <= 0 or self.tss <= 0:
            return 0.0
        return float(1.0 - (self.rss / dof) / (self.tss / (self.n_obs - 1)))

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(self.sigma2, 0.0)))

    # --- inference -------------------------------------------------------
    def z_scores(self, robust: bool = True) -> np.ndarray:
        """Standardised coefficients. |z| > 2 is the Bible's keep/drop threshold."""
        se = self.se_robust if robust else self.se_classical
        return np.divide(self.beta, se, out=np.zeros_like(self.beta), where=se > 0)

    def p_values(self, robust: bool = True) -> np.ndarray:
        dof = max(self.n_obs - self.n_params, 1)
        return 2.0 * stats.t.sf(np.abs(self.z_scores(robust)), df=dof)

    def conf_int(self, level: float = 0.95, robust: bool = True) -> np.ndarray:
        """Returns an (p, 2) array. `beta +/- 2 se` is the 95% interval."""
        dof = max(self.n_obs - self.n_params, 1)
        crit = stats.t.ppf(0.5 + level / 2.0, df=dof)
        se = self.se_robust if robust else self.se_classical
        return np.column_stack([self.beta - crit * se, self.beta + crit * se])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def se(self, name: str, robust: bool = True) -> float:
        arr = self.se_robust if robust else self.se_classical
        return float(arr[self.names.index(name)])

    @property
    def cov_beta(self) -> np.ndarray:
        """Coefficient covariance, used to draw parameter-uncertainty samples."""
        return self.xtx_inv * self.sigma2

    def sample_beta(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `size` coefficient vectors from the asymptotic normal posterior.

        The Bible gives beta_hat -> N(beta, (X'X)^-1 sigma^2). Sampling from it is
        how parameter uncertainty gets propagated into the forward projection
        instead of being quietly ignored.
        """
        cov = _nearest_psd(self.cov_beta)
        return rng.multivariate_normal(self.beta, cov, size=size, method="cholesky")

    def summary_rows(self, robust: bool = True) -> list[dict]:
        z = self.z_scores(robust)
        p = self.p_values(robust)
        ci = self.conf_int(robust=robust)
        se = self.se_robust if robust else self.se_classical
        return [
            {
                "name": nm,
                "beta": float(b),
                "se": float(s),
                "z": float(zi),
                "p": float(pi),
                "lo": float(lo),
                "hi": float(hi),
                "significant": bool(abs(zi) >= 2.0),
            }
            for nm, b, s, zi, pi, (lo, hi) in zip(self.names, self.beta, se, z, p, ci)
        ]


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so a numerically-ragged covariance stays usable."""
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(vals.max(), 0.0) * 1e-12
    vals = np.clip(vals, floor, None)
    return vecs @ np.diag(vals) @ vecs.T


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> LinearFit:
    """Ordinary (or weighted) least squares with classical and HC1 robust SEs.

    Solved through the normal equations with a pseudo-inverse fallback. The Bible
    notes that rank-deficient X leaves beta non-unique while the fitted values
    stay well defined --- the pseudo-inverse picks the minimum-norm beta so the
    fit never blows up on a collinear design.
    """
    return _fit(X, y, names, weights, ridge_lambda=0.0)


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float | np.ndarray,
    names: list[str] | None = None,
    weights: np.ndarray | None = None,
    penalise_intercept: bool = False,
) -> LinearFit:
    """Ridge: `(X'X + Lambda)^-1 X'y`, always nonsingular.

    `ridge_lambda` may be a scalar or a **per-column vector**. The vector form is
    what makes this useful here: it shrinks one specific coefficient toward zero
    while leaving the rest unpenalised.

    That maps onto a proper prior. A ridge penalty of `lambda_j = sigma^2 /
    tau_j^2` is exactly the posterior mode under a `N(0, tau_j^2)` prior on
    `beta_j`, so "I believe this coefficient is small, with scale tau" becomes a
    number rather than a vibe.
    """
    return _fit(
        X,
        y,
        names,
        weights,
        ridge_lambda=ridge_lambda,
        penalise_intercept=penalise_intercept,
    )


def prior_to_penalty(sigma2: float, prior_sd: float) -> float:
    """Ridge weight implied by a Gaussian prior of scale `prior_sd`."""
    return float(sigma2 / max(prior_sd**2, 1e-12))


def omitted_variable_bias(short: LinearFit, long: LinearFit, name: str) -> dict:
    """Quant Bible section 4.6: how much does adding a control move the coefficient?

    `OVB = beta_short - beta_long`. Small OVB means the estimate is robust to the
    control; large OVB means the two regressors are fighting over the same
    variance and the estimate should not be quoted without saying which
    specification produced it.
    """
    b_s, b_l = short.coef(name), long.coef(name)
    se_l = long.se(name)
    return {
        "short": b_s,
        "long": b_l,
        "ovb": b_s - b_l,
        "ovb_in_se": float((b_s - b_l) / se_l) if se_l > 0 else 0.0,
        "robust": bool(se_l > 0 and abs(b_s - b_l) < se_l),
    }


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str] | None,
    weights: np.ndarray | None,
    ridge_lambda: float | np.ndarray,
    penalise_intercept: bool = False,
) -> LinearFit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if names is None:
        names = [f"x{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError(f"{len(names)} names for {p} columns")

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        rw = np.sqrt(w)
        Xw, yw = X * rw[:, None], y * rw
    else:
        w, Xw, yw = None, X, y

    xtx = Xw.T @ Xw
    lam_vec = np.broadcast_to(np.asarray(ridge_lambda, dtype=float), (p,)).astype(float).copy()
    if np.any(lam_vec > 0):
        if not penalise_intercept:
            lam_vec[0] = 0.0
        xtx = xtx + np.diag(lam_vec)

    xty = Xw.T @ yw
    try:
        xtx_inv = np.linalg.inv(xtx)
        beta = xtx_inv @ xty
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ xty

    resid = y - X @ beta
    rw_resid = resid if w is None else resid * np.sqrt(w)
    rss = float(rw_resid @ rw_resid)

    ymean = float(np.average(y, weights=w)) if w is not None else float(y.mean())
    dev = y - ymean
    tss = float(dev @ dev) if w is None else float((w * dev * dev).sum())

    dof = max(n - p, 1)
    sigma2 = rss / dof

    var_classical = np.clip(np.diag(xtx_inv) * sigma2, 0.0, None)
    se_classical = np.sqrt(var_classical)

    # HC1 sandwich: (X'X)^-1 (sum e_i^2 x_i x_i') (X'X)^-1, scaled n/(n-p).
    meat = (Xw * (rw_resid**2)[:, None]).T @ Xw
    var_robust = np.diag(xtx_inv @ meat @ xtx_inv) * (n / dof)
    se_robust = np.sqrt(np.clip(var_robust, 0.0, None))

    return LinearFit(
        beta=beta,
        names=list(names),
        n_obs=n,
        n_params=p,
        rss=rss,
        tss=tss,
        sigma2=sigma2,
        xtx_inv=xtx_inv,
        se_classical=se_classical,
        se_robust=se_robust,
        ridge_lambda=float(np.max(lam_vec)),
        weights=w,
    )


def f_test(restricted: LinearFit, full: LinearFit) -> dict:
    """Joint significance of the coefficients the restricted model drops.

    F = ((RSS0 - RSS1) / (p1 - p0)) / (RSS1 / (N - p1 - 1))

    This is how we answer "is there a trend at all", rather than squinting at a
    dozen individually-marginal seasonal trend terms. The Bible's point is that a
    group can be jointly significant even when no single member clears |z| > 2.
    """
    d_params = full.n_params - restricted.n_params
    dof_resid = full.n_obs - full.n_params
    if d_params <= 0 or dof_resid <= 0:
        return {"f": 0.0, "p": 1.0, "df_num": max(d_params, 0), "df_den": max(dof_resid, 0)}

    num = (restricted.rss - full.rss) / d_params
    den = full.rss / dof_resid
    f = float(num / den) if den > 0 else 0.0
    return {
        "f": f,
        "p": float(stats.f.sf(f, d_params, dof_resid)) if f > 0 else 1.0,
        "df_num": int(d_params),
        "df_den": int(dof_resid),
        "significant": bool(f > 0 and stats.f.sf(f, d_params, dof_resid) < 0.05),
    }


def design_from_blocks(blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, slice]]:
    """Concatenate named column blocks and keep an index of where each landed.

    The slice map is what lets us run the F-test by rebuilding the design without
    one named block (e.g. drop every trend column at once).
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    spans: dict[str, slice] = {}
    cursor = 0
    for block, mat in blocks.items():
        mat = np.atleast_2d(np.asarray(mat, dtype=float))
        if mat.shape[0] == 1 and mat.shape[1] != 1:
            mat = mat.T
        width = mat.shape[1]
        cols.append(mat)
        names.extend([f"{block}[{i}]" if width > 1 else block for i in range(width)])
        spans[block] = slice(cursor, cursor + width)
        cursor += width
    return np.column_stack(cols), names, spans
