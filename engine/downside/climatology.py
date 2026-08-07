"""The mean and variance models --- the core of the 100-year estimation problem.

The central design decision in this module: **local temperature is regressed on
modelled global temperature, not on calendar time.**

That sounds like a detail. It is the single biggest determinant of whether the
forward century is credible.

Regressing on time forces a shape onto the trend --- linear, or whatever
polynomial gets picked --- and the observed record does not have that shape. It
has a 1940-1975 plateau where rising greenhouse forcing was masked by sulphate
aerosols, volcanic notches after Agung, El Chichon and Pinatubo, and a steepening
after 1980. A linear-in-time fit splits the difference across all of it and then
extrapolates that compromise for another century.

Regressing on `G(t)` --- the two-box response to observed forcing --- instead
gives:

* a mean model that reproduces the plateau and the notches without being told
  about them, because they are in `G(t)` already;
* a coefficient with a physical meaning. `lambda` *is* the regional amplification
  factor: degrees of local warming per degree of global warming;
* an extrapolation that requires no assumption about the shape of the future,
  only a choice of emissions scenario, which then supplies its own `G(t)`.

Model, for a daily variable `y` on day-of-year `d` in year `t`:

    mu(d,t) = b0
            + sum_k [a_k cos(2 pi k d/365.25) + b_k sin(2 pi k d/365.25)]   seasonal cycle
            + lambda * G(t)                                                  amplification
            + sum_k G(t) * [c_k cos(...) + d_k sin(...)]                     seasonal amplification
            + delta * (t - epoch)                                            residual drift

The seasonal-amplification block is what lets winter warm faster than summer,
which it demonstrably does at continental sites. The residual drift term absorbs
station-specific artefacts --- urban heat island growth, instrument and siting
changes --- and is deliberately **not** extrapolated forward: see
`projection.py`. Extrapolating a station's move from a field to a car park for a
hundred years is not climate science.

Variance is modelled too, in logs, on the same design. Two reasons. Weather
variance has a strong seasonal cycle, so the OLS homoskedasticity assumption
fails outright (the Quant Bible notes a nonlinear CEF guarantees this). And for
pricing tails, a change in variance can matter more than a change in the mean ---
if summer variance widens, extreme-heat frequency rises even with the mean
pinned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TREND_EPOCH, Scenario
from .forcing import TwoBoxResponse
from .linalg import (
    LinearFit,
    f_test,
    fit_ols,
    fit_ridge,
    omitted_variable_bias,
    prior_to_penalty,
)
from .resample import BlockBootstrap, block_bootstrap_coefficients, effective_sample_size
from .sources.base import DailyRecord

__all__ = [
    "ClimatologyFit",
    "DesignSpec",
    "build_design",
    "fit_climatology",
]

#: Baseline window that `G(t)` anomalies are expressed against.
BASE_LO, BASE_HI = 1961, 1990


@dataclass(frozen=True, slots=True)
class DesignSpec:
    n_seasonal: int = 4
    n_seasonal_trend: int = 2
    include_drift: bool = True
    ridge_lambda: float = 0.0
    #: Prior scale on the residual drift coefficient, degC per century.
    #:
    #: `G(t)` and calendar time are both near-monotonic across 1926-2025, so they
    #: are strongly collinear --- the Quant Bible's multicollinearity case, and it
    #: bites hard here. Left unpenalised the two coefficients trade enormous
    #: offsetting values (one site fitted amplification 1.38 against a drift of
    #: -0.68 degC/century, neither number meaning anything on its own).
    #:
    #: Dropping the drift term is one remedy, but then genuine station artefacts
    #: --- urban heat island growth, a move from grass to tarmac --- get
    #: attributed to climate and extrapolated for a century. Instead the drift
    #: coefficient carries a `N(0, tau^2)` prior: real siting artefacts are
    #: typically well under a tenth of a degree per century, so that is the scale
    #: allowed. The amplification coefficient stays unpenalised.
    #:
    #: Set from the walk-forward backtest rather than by taste. At a loose 0.20
    #: the drift term absorbs genuine warming from the training window and then
    #: gets frozen at projection time, leaving a 0.55 degC out-of-sample cold bias
    #: at Boston. Tightening to 0.05 halves that bias and improves RMSE at both
    #: test sites. Dropping the term entirely scores about the same but gives up
    #: any defence against real siting artefacts, so the tight prior wins.
    drift_prior_sd: float = 0.05


@dataclass(slots=True)
class ClimatologyFit:
    """A fitted mean + variance model for one variable at one location."""

    variable: str
    location_id: str
    spec: DesignSpec
    mean_fit: LinearFit
    var_fit: LinearFit
    #: Amplification: degC local per degC global. From the `G` coefficient.
    amplification: float
    #: Block-bootstrap standard error --- the one to quote. The classical figure
    #: sits in `amplification_se_classical` and is roughly 3x too tight because
    #: OLS treats 36,525 autocorrelated days as 36,525 independent observations.
    amplification_se: float
    amplification_se_classical: float
    #: Joint significance of the whole warming block (F-test).
    trend_f: dict
    #: Joint significance of the seasonal-amplification block.
    seasonal_trend_f: dict
    #: Standardised residuals, z = (y - mu) / sigma. Feeds the tail fit.
    z_residuals: np.ndarray = field(repr=False)
    #: Lag-1 autocorrelation of z. Drives spell clustering in the simulator.
    ar1: float = 0.0
    ar2: float = 0.0
    response: TwoBoxResponse = field(repr=False, default=None)
    drift_per_century: float = 0.0
    drift_se: float = 0.0
    #: How far the amplification moves when the drift control is added.
    #: Quant Bible section 4.6 --- a small OVB is the evidence that the estimate
    #: is not an artefact of the specification.
    amplification_ovb: dict = field(default_factory=dict)
    bootstrap: BlockBootstrap | None = field(repr=False, default=None)
    n_effective: float = 0.0
    se_inflation: float = 1.0

    # ------------------------------------------------------------------
    def predict_mean(self, doy: np.ndarray, g: np.ndarray, dec_year: np.ndarray | None = None) -> np.ndarray:
        doy, g, dec_year, shape = _flatten_inputs(doy, g, dec_year)
        X = _assemble(doy, g, dec_year, self.spec)
        return (X @ self.mean_fit.beta).reshape(shape)

    def predict_sd(self, doy: np.ndarray, g: np.ndarray, dec_year: np.ndarray | None = None) -> np.ndarray:
        doy, g, dec_year, shape = _flatten_inputs(doy, g, dec_year)
        X = _assemble(doy, g, dec_year, self.spec, variance=True)
        log_var = X @ self.var_fit.beta
        # Cap keeps a wild extrapolation from producing an absurd sigma.
        return np.exp(0.5 * np.clip(log_var, -12.0, 8.0)).reshape(shape)

    @property
    def r_squared(self) -> float:
        return self.mean_fit.r_squared

    def seasonal_amplification(self, doy: np.ndarray) -> np.ndarray:
        """Amplification factor as a function of day of year.

        This is the curve that shows a January degree and a July degree are not
        the same degree.
        """
        names = self.mean_fit.names
        beta = self.mean_fit.beta
        out = np.full(np.shape(doy), beta[names.index("global")], dtype=float)
        phase = 2.0 * np.pi * np.asarray(doy, dtype=float) / 365.25
        for k in range(1, self.spec.n_seasonal_trend + 1):
            ci = names.index(f"global_cos{k}")
            si = names.index(f"global_sin{k}")
            out = out + beta[ci] * np.cos(k * phase) + beta[si] * np.sin(k * phase)
        return out


# ----------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------


def _flatten_inputs(
    doy: np.ndarray, g: np.ndarray, dec_year: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[int, ...]]:
    """Broadcast the three inputs to a common shape, then flatten to 1-D.

    Prediction is called both with 1-D day vectors (fitting, diagnostics) and
    with 2-D `(paths, days)` grids (Monte Carlo). Normalising here keeps the
    design-matrix code single-shape.
    """
    doy = np.asarray(doy, dtype=float)
    g = np.asarray(g, dtype=float)
    shape = np.broadcast_shapes(doy.shape, g.shape)
    if dec_year is not None:
        dec_year = np.asarray(dec_year, dtype=float)
        shape = np.broadcast_shapes(shape, dec_year.shape)
        dec_year = np.broadcast_to(dec_year, shape).reshape(-1)
    doy = np.broadcast_to(doy, shape).reshape(-1)
    g = np.broadcast_to(g, shape).reshape(-1)
    return doy, g, dec_year, shape


def _harmonics(doy: np.ndarray, n: int) -> tuple[list[np.ndarray], list[str]]:
    phase = 2.0 * np.pi * np.asarray(doy, dtype=float) / 365.25
    cols, names = [], []
    for k in range(1, n + 1):
        cols.append(np.cos(k * phase))
        names.append(f"cos{k}")
        cols.append(np.sin(k * phase))
        names.append(f"sin{k}")
    return cols, names


def _assemble(
    doy: np.ndarray,
    g: np.ndarray,
    dec_year: np.ndarray | None,
    spec: DesignSpec,
    variance: bool = False,
) -> np.ndarray:
    doy = np.asarray(doy, dtype=float)
    g = np.asarray(g, dtype=float)
    if g.shape != doy.shape:
        g = np.broadcast_to(g, doy.shape)

    cols = [np.ones_like(doy)]
    seas, _ = _harmonics(doy, spec.n_seasonal)
    cols.extend(seas)
    cols.append(g)
    st, _ = _harmonics(doy, spec.n_seasonal_trend)
    cols.extend([g * c for c in st])
    if spec.include_drift and not variance:
        if dec_year is None:
            raise ValueError("drift term requires dec_year")
        cols.append((np.asarray(dec_year, dtype=float) - TREND_EPOCH) / 100.0)
    return np.column_stack(cols)


def _names(spec: DesignSpec, variance: bool = False) -> list[str]:
    names = ["intercept"]
    _, seas = _harmonics(np.array([1.0]), spec.n_seasonal)
    names.extend(seas)
    names.append("global")
    _, st = _harmonics(np.array([1.0]), spec.n_seasonal_trend)
    names.extend([f"global_{s}" for s in st])
    if spec.include_drift and not variance:
        names.append("drift_per_century")
    return names


def build_design(
    record: DailyRecord,
    response: TwoBoxResponse,
    spec: DesignSpec,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Returns (X, names, g) for the record's own time axis."""
    dec_year = record.decimal_year
    g = response.baseline_shift(dec_year, BASE_LO, BASE_HI)
    X = _assemble(record.doy, g, dec_year, spec)
    return X, _names(spec), g


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def fit_climatology(
    record: DailyRecord,
    response: TwoBoxResponse,
    variable: str,
    spec: DesignSpec | None = None,
    mask: np.ndarray | None = None,
    n_boot: int = 200,
) -> ClimatologyFit:
    """Fit the mean and variance models for one variable.

    `mask` restricts the fit to a subset of days --- used by the backtest to hold
    out the last thirty years, and by the precipitation model to fit intensity on
    wet days only.
    """
    spec = spec or DesignSpec()
    y_all = record.get(variable)
    X_all, names, g_all = build_design(record, response, spec)
    dec_year_all = record.decimal_year

    if mask is None:
        mask = np.ones(len(record), dtype=bool)
    X, y = X_all[mask], y_all[mask]

    # Pass 1: unpenalised, purely to get a residual scale for the drift prior.
    pilot = fit_ols(X, y, names=names)

    penalty = np.full(len(names), float(spec.ridge_lambda))
    if "drift_per_century" in names and spec.drift_prior_sd > 0:
        penalty[names.index("drift_per_century")] = prior_to_penalty(
            pilot.sigma2, spec.drift_prior_sd
        )

    mean_fit = (
        fit_ridge(X, y, penalty, names=names) if np.any(penalty > 0) else pilot
    )

    resid = y - X @ mean_fit.beta

    # --- variance model -------------------------------------------------
    # Regress log(e^2) on the same seasonal + forcing design. The -1.2704 offset
    # is E[log(chi2_1)], which debiases log(e^2) as an estimator of log(sigma^2);
    # without it every sigma comes out systematically small and every tail
    # probability comes out systematically low.
    var_names = _names(spec, variance=True)
    Xv = _assemble(record.doy[mask], g_all[mask], None, spec, variance=True)
    log_e2 = np.log(np.clip(resid**2, 1e-8, None)) + 1.2704
    var_fit = fit_ols(Xv, log_e2, names=var_names)

    sigma = np.exp(0.5 * np.clip(Xv @ var_fit.beta, -12.0, 8.0))
    z = resid / np.clip(sigma, 1e-6, None)

    # --- inference on the warming block ---------------------------------
    g_cols = [i for i, n in enumerate(names) if n.startswith("global")]
    keep = [i for i in range(len(names)) if i not in g_cols]
    restricted = fit_ols(X[:, keep], y, names=[names[i] for i in keep])
    trend_f = f_test(restricted, mean_fit)

    st_cols = [i for i, n in enumerate(names) if n.startswith("global_")]
    keep_st = [i for i in range(len(names)) if i not in st_cols]
    restricted_st = fit_ols(X[:, keep_st], y, names=[names[i] for i in keep_st])
    seasonal_trend_f = f_test(restricted_st, mean_fit)

    # --- persistence -----------------------------------------------------
    ar1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if len(z) > 2 else 0.0
    ar2 = float(np.corrcoef(z[:-2], z[2:])[0, 1]) if len(z) > 3 else 0.0

    # --- specification robustness (OVB) ----------------------------------
    drift = drift_se = 0.0
    ovb: dict = {}
    if "drift_per_century" in names:
        drift = mean_fit.coef("drift_per_century")
        drift_se = mean_fit.se("drift_per_century")
        di = names.index("drift_per_century")
        keep_nodrift = [i for i in range(len(names)) if i != di]
        short = fit_ols(X[:, keep_nodrift], y, names=[names[i] for i in keep_nodrift])
        ovb = omitted_variable_bias(short, mean_fit, "global")

    # --- honest uncertainty on the warming coefficient --------------------
    boot = None
    amp_se = mean_fit.se("global")
    inflation = 1.0
    if n_boot > 0:
        boot = block_bootstrap_coefficients(
            X, y, record.year[mask], names, penalty=penalty, n_boot=n_boot
        )
        boot_se = boot.se_of("global")
        if boot_se > 0:
            inflation = boot_se / max(mean_fit.se("global"), 1e-12)
            amp_se = boot_se

    return ClimatologyFit(
        variable=variable,
        location_id=record.location_id,
        spec=spec,
        mean_fit=mean_fit,
        var_fit=var_fit,
        amplification=mean_fit.coef("global"),
        amplification_se=amp_se,
        amplification_se_classical=mean_fit.se("global"),
        trend_f=trend_f,
        seasonal_trend_f=seasonal_trend_f,
        z_residuals=z,
        ar1=ar1,
        ar2=ar2,
        response=response,
        drift_per_century=drift,
        drift_se=drift_se,
        amplification_ovb=ovb,
        bootstrap=boot,
        n_effective=effective_sample_size(resid),
        se_inflation=inflation,
    )
