"""Block bootstrap for coefficients estimated on autocorrelated daily data.

The Quant Bible lists "no autocorrelation in the residuals" among the core
assumptions of OLS, and notes that violating it invalidates the usual standard
error `SE(beta) = sigma_e / (sqrt(n) * sigma_X)`. Daily weather violates it about
as hard as it can be violated:

* day-to-day residual autocorrelation around 0.70 (synoptic systems last days);
* multi-year regime structure with an ENSO-like quasi-period.

The practical consequence is severe. A 100-year daily record has 36,525 rows, and
the classical formula treats every one as independent evidence about the warming
trend. It is not. The effective sample size for a *trend* coefficient is closer
to the number of independent multi-year epochs --- order 20-30 --- so the
classical standard error understates the true uncertainty by roughly an order of
magnitude.

That is not an academic point. It is the difference between telling an operator
"warming here is 1.28 +/- 0.08 degrees per degree global" (precise, and wrong) and
"1.28 +/- 0.34" (honest, and wide enough to matter for a 100-year price).

The fix used here is a **moving-block bootstrap over whole years**. Years are
resampled in contiguous blocks long enough to carry the regime persistence with
them, the model is refit on each replicate, and the spread of the resulting
coefficients is the standard error. It makes no assumption about the form of the
dependence, which is the reason to prefer it over a parametric HAC correction
with a bandwidth nobody can defend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import fit_ridge

__all__ = ["BlockBootstrap", "block_bootstrap_coefficients", "effective_sample_size"]

#: Block length in years. Long enough to span the ~4-year regime cycle plus its
#: damping tail, short enough that a 100-year record still yields many distinct
#: resamples.
DEFAULT_BLOCK_YEARS = 8


@dataclass(slots=True)
class BlockBootstrap:
    names: list[str]
    replicates: np.ndarray  # (n_boot, p)
    block_years: int
    n_boot: int

    def se(self) -> np.ndarray:
        return self.replicates.std(axis=0, ddof=1)

    def se_of(self, name: str) -> float:
        return float(self.se()[self.names.index(name)])

    def quantiles(self, name: str, qs=(0.025, 0.5, 0.975)) -> np.ndarray:
        return np.quantile(self.replicates[:, self.names.index(name)], qs)

    def inflation_vs(self, classical_se: np.ndarray) -> np.ndarray:
        """Ratio of bootstrap SE to classical SE, per coefficient."""
        return np.divide(
            self.se(), classical_se, out=np.ones_like(classical_se), where=classical_se > 0
        )


def block_bootstrap_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    year: np.ndarray,
    names: list[str],
    penalty: np.ndarray | float = 0.0,
    n_boot: int = 240,
    block_years: int = DEFAULT_BLOCK_YEARS,
    seed: int = 11,
) -> BlockBootstrap:
    """Refit the model on `n_boot` year-block resamples of the record."""
    rng = np.random.default_rng(seed)
    years = np.unique(year)
    n_years = len(years)
    n_blocks = max(int(np.ceil(n_years / block_years)), 1)

    # Precompute row indices per year once; the inner loop is then pure gather.
    rows_for_year = {int(yr): np.flatnonzero(year == yr) for yr in years}

    reps = np.empty((n_boot, X.shape[1]), dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n_years, size=n_blocks)
        picked: list[np.ndarray] = []
        for s in starts:
            # Circular blocks so late years are not under-sampled.
            block = [(s + k) % n_years for k in range(block_years)]
            picked.extend(rows_for_year[int(years[i])] for i in block)
        idx = np.concatenate(picked)
        fit = fit_ridge(X[idx], y[idx], penalty, names=names)
        reps[b] = fit.beta

    return BlockBootstrap(names=list(names), replicates=reps, block_years=block_years, n_boot=n_boot)


def effective_sample_size(residuals: np.ndarray, max_lag: int = 400) -> float:
    """Rough effective N implied by the residual autocorrelation function.

        N_eff = N / (1 + 2 * sum_k rho_k)

    Reported for context on the diagnostics screen --- it makes the size of the
    autocorrelation problem legible rather than leaving it as a footnote.
    """
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    n = len(r)
    if n < 10:
        return float(n)
    denom = float(r @ r)
    if denom <= 0:
        return float(n)

    total = 0.0
    for lag in range(1, min(max_lag, n - 1)):
        rho = float(r[:-lag] @ r[lag:]) / denom
        # Truncate at the first non-positive pair, the standard initial-positive-
        # sequence rule --- beyond it the estimates are mostly noise.
        if rho <= 0:
            break
        total += rho
    return float(n / max(1.0 + 2.0 * total, 1.0))
