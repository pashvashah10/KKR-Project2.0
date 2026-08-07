"""Out-of-sample validation --- does the forecast actually have skill?

Everything else in the engine produces numbers. This module is what says whether
to believe them.

The test is a **walk-forward split**: fit on the early record, predict the years
the model never saw, and score the predictions. That is the only honest way to
evaluate a climate model, because in-sample fit is nearly free --- add enough
harmonics and R^2 climbs while genuine forecast skill falls.

Three scores, each answering a different question:

* **CRPS** (continuous ranked probability score) --- generalises absolute error to
  a whole predicted *distribution*, so a forecast is rewarded for being sharp
  only when it is also right. Compared against climatology; the skill score is
  the fraction of CRPS beaten, so > 0 means the model adds something over "the
  average of the last thirty years".
* **PIT** (probability integral transform) --- where each observation fell in its
  own predicted distribution. If the distributions are honest, PIT values are
  uniform. A U-shaped histogram means the forecast is overconfident (too many
  observations in the tails); a dome means it is underconfident. This is the
  check that catches the failure mode that matters most for pricing, since an
  overconfident model underprices every tail it sells.
* **Reliability** --- of the days assigned a 20% chance of triggering, did close to
  20% trigger? Directly the number a counterparty should ask about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .climatology import BASE_HI, BASE_LO, DesignSpec, fit_climatology
from .forcing import TwoBoxResponse
from .sources.base import DailyRecord

__all__ = [
    "BacktestResult",
    "walk_forward",
    "crps_gaussian",
    "pit_values",
    "reliability_curve",
    "detect_inhomogeneity",
    "assess_quality",
]


@dataclass(slots=True)
class BacktestResult:
    variable: str
    train_years: tuple[int, int]
    test_years: tuple[int, int]
    n_test: int
    rmse: float
    bias: float
    crps_model: float
    crps_climatology: float
    crps_skill: float
    pit_histogram: list = field(default_factory=list)
    pit_uniformity_p: float = 0.0
    reliability: list = field(default_factory=list)
    coverage_90: float = 0.0
    coverage_50: float = 0.0

    @property
    def pit_max_deviation(self) -> float:
        """Largest bin's departure from uniform, as a share of expected count."""
        if not self.pit_histogram:
            return 0.0
        exp = self.pit_histogram[0]["expected"]
        if exp <= 0:
            return 0.0
        return max(abs(b["count"] - exp) / exp for b in self.pit_histogram)

    @property
    def calibrated(self) -> bool:
        """Are the predicted distributions honest enough to price off?

        Judged on **effect size, not a p-value**. With ~11,000 out-of-sample days
        a chi-square test on the PIT histogram rejects uniformity for deviations
        far too small to matter --- it returned p < 1e-6 at every site, which made
        the flag useless rather than informative. What matters for pricing is
        whether the stated intervals hold and whether any PIT bin is badly
        over-populated, so those are the thresholds.
        """
        return (
            abs(self.coverage_90 - 0.90) <= 0.04
            and abs(self.coverage_50 - 0.50) <= 0.06
            and self.pit_max_deviation <= 0.35
        )


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution.

        CRPS = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sigma
    """
    sigma = np.clip(sigma, 1e-9, None)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def pit_values(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return stats.norm.cdf((y - mu) / np.clip(sigma, 1e-9, None))


def reliability_curve(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Forecast probability vs observed frequency, binned."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        sel = (prob >= edges[i]) & (prob < edges[i + 1] if i < n_bins - 1 else prob <= 1.0)
        if sel.sum() < 20:
            continue
        rows.append(
            {
                "forecast": float(prob[sel].mean()),
                "observed": float(outcome[sel].mean()),
                "n": int(sel.sum()),
            }
        )
    return rows


def walk_forward(
    record: DailyRecord,
    response: TwoBoxResponse,
    variable: str = "tmax_c",
    split_year: int = 1995,
    spec: DesignSpec | None = None,
    trigger_threshold: float | None = None,
) -> BacktestResult:
    """Fit through `split_year`, score everything after it.

    The model never sees the test years --- not for the mean, not for the
    variance, not for the tail. `response` is the forcing path, which *is*
    known over the test window; that is the correct setup for evaluating a
    climate model, since the question is whether the response to forcing was
    learned, not whether future emissions can be guessed.
    """
    spec = spec or DesignSpec()
    train = record.year <= split_year
    test = ~train
    if test.sum() < 365:
        raise ValueError(f"only {test.sum()} test days; need at least a year")

    fit = fit_climatology(record, response, variable, spec, mask=train, n_boot=0)

    g = response.baseline_shift(record.decimal_year, BASE_LO, BASE_HI)
    frozen = np.full(test.sum(), float(split_year))
    mu = fit.predict_mean(record.doy[test], g[test], frozen)
    sigma = fit.predict_sd(record.doy[test], g[test])
    y = record.get(variable)[test]

    resid = y - mu
    rmse = float(np.sqrt(np.mean(resid**2)))
    bias = float(np.mean(resid))

    crps_m = float(np.mean(crps_gaussian(y, mu, sigma)))

    # Climatology reference: day-of-year mean and sd from the training years
    # only. This is the "just average the past" benchmark the model must beat.
    clim_mu = np.zeros(test.sum())
    clim_sd = np.zeros(test.sum())
    train_doy = record.doy[train]
    train_y = record.get(variable)[train]
    for i, d in enumerate(record.doy[test]):
        offset = np.abs(train_doy - d)
        near = np.minimum(offset, 365 - offset) <= 7
        vals = train_y[near]
        clim_mu[i] = vals.mean() if vals.size else train_y.mean()
        clim_sd[i] = vals.std() if vals.size > 1 else train_y.std()
    crps_c = float(np.mean(crps_gaussian(y, clim_mu, np.clip(clim_sd, 1e-6, None))))
    skill = float(1.0 - crps_m / crps_c) if crps_c > 0 else 0.0

    pit = pit_values(y, mu, sigma)
    hist, edges = np.histogram(pit, bins=10, range=(0, 1))
    expected = len(pit) / 10.0
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    pit_p = float(stats.chi2.sf(chi2, df=9))

    cov90 = float(np.mean((pit > 0.05) & (pit < 0.95)))
    cov50 = float(np.mean((pit > 0.25) & (pit < 0.75)))

    rel: list[dict] = []
    if trigger_threshold is not None:
        prob = 1.0 - stats.norm.cdf((trigger_threshold - mu) / np.clip(sigma, 1e-9, None))
        rel = reliability_curve(prob, (y >= trigger_threshold).astype(float))

    return BacktestResult(
        variable=variable,
        train_years=(int(record.year.min()), split_year),
        test_years=(split_year + 1, int(record.year.max())),
        n_test=int(test.sum()),
        rmse=rmse,
        bias=bias,
        crps_model=crps_m,
        crps_climatology=crps_c,
        crps_skill=skill,
        pit_histogram=[
            {"bin": round(float(edges[i]), 2), "count": int(hist[i]), "expected": float(expected)}
            for i in range(10)
        ],
        pit_uniformity_p=pit_p,
        reliability=rel,
        coverage_90=cov90,
        coverage_50=cov50,
    )


# ----------------------------------------------------------------------
# Record quality
# ----------------------------------------------------------------------


def detect_inhomogeneity(record: DailyRecord, fit, variable: str = "tmax_c") -> dict:
    """Look for a step change in the record that climate cannot explain.

    Long station records are rarely homogeneous. Instruments get replaced, sites
    get moved, observation times shift. Each leaves a *step* in the series, and a
    trend model has no way to represent one: it absorbs the step into the warming
    coefficient and reports whatever falls out.

    Jackson Hole is the live example --- its 2010s decade mean sits 1.8 degC below
    its 1990s, which is not a climate signal. Fitted naively it produced an
    amplification of -0.35, meaning the model believed the site *cools* as the
    globe warms, and would have projected cooling to 2125.

    The test runs on annual means of the model residuals rather than on the raw
    series, so a genuine forced response --- which the fit already captures --- does
    not register. What remains is a standard two-sample split scan (a Pettitt-style
    change point): for every candidate split year, the standardised difference of
    residual means either side. A large statistic means a discontinuity the
    climate model cannot account for.

    Thresholds are calibrated, not guessed. Across clean records the statistic
    tops out at 2.5 with shifts under 0.3 degC; a planted 2 degC step scores 4.2,
    and the real Jackson Hole record scores 3.2 with a -0.74 degC shift. Hence
    `t > 3.0` and `|shift| > 0.4`.

    **This test alone is a weak detector, by construction.** The fit partially
    absorbs any step before the residuals are formed --- a planted 2 degC step
    pushed the fitted amplification to 3.97 degC per degC and left only 0.62 degC
    in the residuals. So a step shows up mainly as a *non-physical amplification*,
    and `assess_quality` checks that separately. The two together catch what
    neither catches alone; a step around 1 degC evades this one on its own.

    Operational climatology solves this with pairwise homogenisation against
    neighbouring stations. That is the right fix and it is not implemented here,
    so the honest alternative is to detect the problem and refuse to quote
    confidently on that site.
    """
    years = np.unique(record.year)
    y = record.get(variable)
    resid_annual = []
    for yr in years:
        m = record.year == yr
        if m.sum() < 200:
            continue
        g = fit.response.baseline_shift(record.decimal_year[m], BASE_LO, BASE_HI)
        pred = fit.predict_mean(record.doy[m], g, np.full(m.sum(), float(yr)))
        resid_annual.append(float((y[m] - pred).mean()))

    r = np.asarray(resid_annual, dtype=float)
    n = len(r)
    if n < 30:
        return {"detected": False, "statistic": 0.0, "year": None, "shift": 0.0, "n_years": n}

    sd = float(r.std(ddof=1))
    if sd <= 0:
        return {"detected": False, "statistic": 0.0, "year": None, "shift": 0.0, "n_years": n}

    best_t, best_i = 0.0, None
    for i in range(10, n - 10):
        a, b = r[:i], r[i:]
        pooled = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if pooled <= 0:
            continue
        t = abs(float(b.mean() - a.mean()) / pooled)
        if t > best_t:
            best_t, best_i = t, i

    if best_i is None:
        return {"detected": False, "statistic": 0.0, "year": None, "shift": 0.0, "n_years": n}

    valid = np.unique(record.year)[: len(r)] if len(r) == len(years) else years[: len(r)]
    shift = float(r[best_i:].mean() - r[:best_i].mean())
    return {
        "detected": bool(best_t > 3.0 and abs(shift) > 0.4),
        "statistic": round(best_t, 2),
        "year": int(valid[best_i]) if best_i < len(valid) else None,
        "shift": round(shift, 3),
        "n_years": n,
    }


def assess_quality(amplification: float, backtest: BacktestResult, inhomogeneity: dict) -> dict:
    """Gate a site before its numbers are quoted.

    A model can fit, project and price without ever raising, so the failure mode
    here is a confident wrong number rather than a crash. These checks are what
    stand between a bad record and a bad quote.
    """
    problems: list[str] = []

    if not 0.2 <= amplification <= 3.0:
        problems.append(
            f"Amplification of {amplification:.2f} degC per degC is not physical for a "
            "continental site; the warming coefficient is not identified here."
        )
    if abs(backtest.bias) > 0.8:
        problems.append(
            f"Out-of-sample bias of {backtest.bias:+.2f} degC over 30 held-out years."
        )
    if backtest.crps_skill < -0.05:
        problems.append(
            f"Forecast scores {abs(backtest.crps_skill):.0%} worse than plain climatology."
        )
    if inhomogeneity.get("detected"):
        problems.append(
            f"Step change of {inhomogeneity['shift']:+.2f} degC around "
            f"{inhomogeneity['year']} that the climate model cannot explain — "
            "most likely a station move or instrument change, not weather."
        )

    return {
        "usable": not problems,
        "problems": problems,
        "verdict": "ok" if not problems else ("caution" if len(problems) == 1 else "unreliable"),
    }
