"""Engine invariants.

These check the estimators against answers that are known independently ---
recovering coefficients planted in synthetic data, closed-form properties of the
distributions, and the arithmetic of the trigger definitions. The point is to
catch a silently wrong number, which is the failure mode that matters in a
pricing system: a broken fit does not raise, it just quotes.

Run with `pytest engine/tests`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from downside.config import F2X, PERILS_BY_ID, Peril  # noqa: E402
from downside.forcing import run_two_box, shrink_amplification  # noqa: E402
from downside.glm import fit_gamma_log, fit_logistic  # noqa: E402
from downside.hedging import analyse_hedge  # noqa: E402
from downside.linalg import f_test, fit_ols, fit_ridge, omitted_variable_bias  # noqa: E402
from downside.backtest import crps_gaussian  # noqa: E402
from downside.resample import block_bootstrap_coefficients  # noqa: E402
from downside.sources.synthetic import SyntheticSource  # noqa: E402
from downside.tails import SplicedTail, fit_gpd  # noqa: E402


# ----------------------------------------------------------------------
# linear models
# ----------------------------------------------------------------------


def test_ols_recovers_planted_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([2.5, -1.25, 0.75])
    y = X @ beta + rng.standard_normal(n) * 0.5

    fit = fit_ols(X, y, names=["a", "b", "c"])
    assert np.allclose(fit.beta, beta, atol=0.05)
    # Every coefficient here is real, so all should clear the |z| > 2 bar.
    assert np.all(np.abs(fit.z_scores()) > 2)
    # Signal variance 1.25^2 + 0.75^2 = 2.125 against noise 0.5^2 = 0.25, so the
    # population R^2 is 2.125 / 2.375 = 0.895. Assert just under it.
    assert fit.r_squared > 0.88


def test_ols_conf_interval_covers_truth():
    rng = np.random.default_rng(1)
    hits = 0
    trials = 60
    for i in range(trials):
        n = 500
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)
        ci = fit_ols(X, y).conf_int(0.95)
        if ci[1, 0] <= 2.0 <= ci[1, 1]:
            hits += 1
    # Nominal 95%; allow sampling slack on 60 trials.
    assert hits >= 51


def test_f_test_detects_a_real_block_and_ignores_noise():
    rng = np.random.default_rng(2)
    n = 3000
    base = np.column_stack([np.ones(n), rng.standard_normal(n)])
    extra = rng.standard_normal((n, 3))

    # Case 1: the extra block genuinely matters.
    y = base @ np.array([1.0, 0.5]) + extra @ np.array([0.4, -0.3, 0.25]) + rng.standard_normal(n)
    full = fit_ols(np.column_stack([base, extra]), y)
    restricted = fit_ols(base, y)
    assert f_test(restricted, full)["significant"]

    # Case 2: the extra block is pure noise.
    y2 = base @ np.array([1.0, 0.5]) + rng.standard_normal(n)
    full2 = fit_ols(np.column_stack([base, extra]), y2)
    restricted2 = fit_ols(base, y2)
    assert not f_test(restricted2, full2)["significant"]


def test_ridge_shrinks_only_the_penalised_column():
    rng = np.random.default_rng(3)
    n = 2000
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 2.0, 2.0]) + rng.standard_normal(n)

    penalty = np.array([0.0, 0.0, 5000.0])
    ridged = fit_ridge(X, y, penalty, names=["a", "b", "c"])
    plain = fit_ols(X, y, names=["a", "b", "c"])

    assert abs(ridged.coef("c")) < abs(plain.coef("c"))
    assert ridged.coef("b") == pytest.approx(plain.coef("b"), abs=0.05)


def test_robust_se_exceeds_classical_under_heteroskedasticity():
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    # Noise scale grows with |x| --- textbook heteroskedasticity.
    y = 1.0 + 0.5 * x + rng.standard_normal(n) * (0.3 + 1.5 * np.abs(x))
    fit = fit_ols(X, y)
    assert fit.se_robust[1] > fit.se_classical[1]


def test_ovb_is_small_when_control_is_orthogonal():
    rng = np.random.default_rng(5)
    n = 3000
    treat = rng.standard_normal(n)
    control = rng.standard_normal(n)  # independent of treat
    y = 1.0 + 0.8 * treat + 0.5 * control + rng.standard_normal(n)

    long = fit_ols(np.column_stack([np.ones(n), treat, control]), y, names=["a", "t", "c"])
    short = fit_ols(np.column_stack([np.ones(n), treat]), y, names=["a", "t"])
    assert omitted_variable_bias(short, long, "t")["robust"]


# ----------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------


def test_block_bootstrap_inflates_se_on_autocorrelated_data():
    """The whole reason `resample.py` exists."""
    rng = np.random.default_rng(6)
    years = np.repeat(np.arange(1950, 2020), 365)
    n = len(years)
    t = (years - 1985) / 100.0

    # Strongly autocorrelated noise: classical SEs will be far too tight.
    eps = np.empty(n)
    eps[0] = rng.standard_normal()
    for i in range(1, n):
        eps[i] = 0.95 * eps[i - 1] + rng.standard_normal() * 0.31

    y = 1.0 + 2.0 * t + eps
    X = np.column_stack([np.ones(n), t])
    fit = fit_ols(X, y, names=["a", "trend"])
    boot = block_bootstrap_coefficients(X, y, years, ["a", "trend"], n_boot=80)

    assert boot.se_of("trend") > 3 * fit.se("trend", robust=False)


# ----------------------------------------------------------------------
# GLMs
# ----------------------------------------------------------------------


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(7)
    n = 20000
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 1.2 * x)))
    y = (rng.random(n) < p).astype(float)

    fit = fit_logistic(X, y, names=["a", "x"])
    assert fit.converged
    assert fit.coef("a") == pytest.approx(-0.5, abs=0.08)
    assert fit.coef("x") == pytest.approx(1.2, abs=0.08)


def test_gamma_log_recovers_the_mean_not_the_median():
    """Log-link Gamma models E[y|x]; regressing log(y) would fit the median."""
    rng = np.random.default_rng(8)
    n = 20000
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    mu = np.exp(1.0 + 0.6 * x)
    y = rng.gamma(shape=2.0, scale=mu / 2.0)

    fit = fit_gamma_log(X, y, names=["a", "x"])
    assert fit.coef("a") == pytest.approx(1.0, abs=0.05)
    assert fit.coef("x") == pytest.approx(0.6, abs=0.05)
    # An OLS fit on log(y) is biased low for the mean by the retransformation gap.
    naive = fit_ols(X, np.log(y), names=["a", "x"])
    assert naive.coef("a") < fit.coef("a")


# ----------------------------------------------------------------------
# extreme value theory
# ----------------------------------------------------------------------


@pytest.mark.parametrize("xi,scale", [(0.25, 2.0), (-0.20, 3.0), (0.0, 1.5), (0.5, 1.0)])
def test_gpd_recovers_shape_across_the_range(xi, scale):
    x = stats.genpareto.rvs(xi, scale=scale, size=40000, random_state=11)
    fit = fit_gpd(x, threshold_quantile=0.95)
    assert fit.converged
    assert fit.shape == pytest.approx(xi, abs=0.06)


def test_gpd_negative_shape_implies_a_finite_ceiling():
    x = stats.genpareto.rvs(-0.3, scale=2.0, size=30000, random_state=12)
    fit = fit_gpd(x, threshold_quantile=0.95)
    assert fit.shape < 0
    assert fit.upper_bound is not None
    # The implied endpoint must sit at or above everything observed.
    assert fit.upper_bound >= x.max() - 1e-6


def test_gpd_return_level_increases_with_period():
    x = stats.genpareto.rvs(0.15, scale=2.0, size=30000, random_state=13)
    fit = fit_gpd(x, threshold_quantile=0.95)
    levels = [fit.return_level(p) for p in (2, 10, 50, 100, 250)]
    assert all(b > a for a, b in zip(levels, levels[1:]))


def test_spliced_tail_preserves_the_marginal():
    """The splice must not distort the body or inflate the variance."""
    rng = np.random.default_rng(14)
    z = rng.standard_normal(60000)
    tail = SplicedTail.from_residuals(z)
    drawn = tail.sample(200000, rng)

    assert drawn.std() == pytest.approx(z.std(), rel=0.05)
    for q in (1, 5, 50, 95, 99):
        assert np.percentile(drawn, q) == pytest.approx(np.percentile(z, q), abs=0.12)


# ----------------------------------------------------------------------
# forcing
# ----------------------------------------------------------------------


def test_two_box_reaches_equilibrium_climate_sensitivity():
    """Held at 2xCO2 forcing forever, warming must approach ECS."""
    ecs = 3.0
    years = np.arange(0, 4000, dtype=float)
    forcing = np.full_like(years, F2X)
    resp = run_two_box(years, forcing, ecs=ecs)
    assert resp.t_upper[-1] == pytest.approx(ecs, rel=0.03)


def test_two_box_response_lags_the_forcing():
    """Ocean thermal inertia: warming at any point sits below its equilibrium."""
    years = np.arange(0, 200, dtype=float)
    forcing = np.full_like(years, F2X)
    resp = run_two_box(years, forcing, ecs=3.0)
    assert resp.t_upper[50] < 3.0
    assert resp.t_upper[50] > resp.t_upper[10]


def test_shrinkage_pulls_a_noisy_estimate_toward_the_prior():
    tight, _ = shrink_amplification(estimate=2.5, std_error=0.05, prior_mean=1.0)
    loose, _ = shrink_amplification(estimate=2.5, std_error=1.00, prior_mean=1.0)
    assert tight > loose  # a precise estimate moves less
    assert 1.0 < loose < 2.5


# ----------------------------------------------------------------------
# peril arithmetic
# ----------------------------------------------------------------------


def test_daily_threshold_counts_correctly():
    p = Peril(id="t", label="t", variable="precip_mm", statistic="daily",
              comparator="ge", threshold=10.0, unit="mm")
    series = np.array([[0.0, 12.0, 9.9, 10.0, 30.0]])
    assert p.triggered(series).sum() == 3


def test_window_sum_uses_a_rolling_window():
    p = Peril(id="w", label="w", variable="precip_mm", statistic="window_sum",
              comparator="ge", threshold=10.0, unit="mm", window_days=3)
    # No single day clears 10, but days 1-3 sum to 12.
    series = np.array([[4.0, 4.0, 4.0, 0.0, 0.0]])
    assert p.triggered(series).sum() == 1


def test_consecutive_requires_an_unbroken_run():
    p = Peril(id="c", label="c", variable="tmax_c", statistic="consecutive",
              comparator="ge", threshold=35.0, unit="degC", window_days=3)
    unbroken = np.array([[36.0, 36.0, 36.0, 20.0]])
    broken = np.array([[36.0, 36.0, 20.0, 36.0]])
    assert p.triggered(unbroken)[0]
    assert not p.triggered(broken)[0]


def test_le_comparator_fires_below_the_threshold():
    p = PERILS_BY_ID["hard-freeze"]
    cold = np.full((1, 3), p.threshold - 1.0)
    warm = np.full((1, 3), p.threshold + 1.0)
    assert p.triggered(cold).all()
    assert not p.triggered(warm).any()


# ----------------------------------------------------------------------
# hedging
# ----------------------------------------------------------------------


def test_min_variance_ratio_equals_cov_over_var():
    rng = np.random.default_rng(15)
    n = 20000
    payout = rng.gamma(2.0, 5000.0, n)
    loss = 0.8 * payout + rng.normal(0, 4000, n)

    res = analyse_hedge(loss, payout, price=1000.0, reserves=200_000,
                        fixed_burn_monthly=10_000)
    expected = np.cov(loss, payout)[0, 1] / np.var(payout, ddof=1)
    assert res.min_variance_ratio == pytest.approx(expected, rel=0.05)
    assert res.basis_risk_share == pytest.approx(1 - res.correlation**2, rel=1e-6)


def test_uncorrelated_payout_gives_no_hedge():
    rng = np.random.default_rng(16)
    n = 8000
    loss = rng.gamma(2.0, 5000.0, n)
    payout = rng.gamma(2.0, 5000.0, n)  # independent
    res = analyse_hedge(loss, payout, price=1000.0, reserves=200_000,
                        fixed_burn_monthly=10_000)
    assert abs(res.min_variance_ratio) < 0.1


def test_thin_reserves_justify_more_hedging_than_deep_ones():
    """The intellectual differentiator, as an assertion.

    Same exposure, same contract, same price --- different cash position, so the
    correct hedge differs. Expected-value reasoning cannot produce this.
    """
    rng = np.random.default_rng(17)
    n = 12000
    payout = rng.gamma(1.5, 20_000.0, n)
    loss = 0.9 * payout + rng.normal(0, 8_000, n)
    price = float(payout.mean() * 1.3)

    thin = analyse_hedge(loss, payout, price, reserves=120_000, fixed_burn_monthly=20_000)
    deep = analyse_hedge(loss, payout, price, reserves=4_000_000, fixed_burn_monthly=20_000)
    assert thin.recommended_fraction >= deep.recommended_fraction


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------


def test_crps_is_minimised_by_the_true_distribution():
    rng = np.random.default_rng(18)
    y = rng.normal(10.0, 2.0, 40000)
    truth = np.mean(crps_gaussian(y, np.full_like(y, 10.0), np.full_like(y, 2.0)))
    biased = np.mean(crps_gaussian(y, np.full_like(y, 11.5), np.full_like(y, 2.0)))
    overconfident = np.mean(crps_gaussian(y, np.full_like(y, 10.0), np.full_like(y, 0.7)))
    underconfident = np.mean(crps_gaussian(y, np.full_like(y, 10.0), np.full_like(y, 5.0)))
    assert truth < biased
    assert truth < overconfident
    assert truth < underconfident


# ----------------------------------------------------------------------
# surrogate calibration
# ----------------------------------------------------------------------


def test_surrogate_reproduces_published_normals():
    from downside.config import LOCATIONS_BY_ID

    loc = LOCATIONS_BY_ID["boston-ma"]
    rec = SyntheticSource().fetch(loc, 1991, 2020)

    total = rec.precip_mm.sum() / 30.0
    assert total == pytest.approx(sum(loc.normals.precip_mm), rel=0.05)

    months = rec.dates.astype("datetime64[M]").astype(int) % 12
    for m in range(12):
        got = rec.tmax_c[months == m].mean()
        assert got == pytest.approx(loc.normals.tmax_c[m], abs=1.0)


def test_surrogate_has_realistic_persistence():
    """Independent days would make every consecutive-day trigger mispriced."""
    from downside.config import LOCATIONS_BY_ID

    rec = SyntheticSource().fetch(LOCATIONS_BY_ID["boston-ma"], 1990, 2020)
    ac = np.corrcoef(rec.tmax_c[:-1], rec.tmax_c[1:])[0, 1]
    assert ac > 0.6

    wet = rec.precip_mm >= 0.254
    p_wet = wet.mean()
    p_wet_given_wet = (wet[1:] & wet[:-1]).sum() / max(wet[:-1].sum(), 1)
    assert p_wet_given_wet > p_wet  # wet spells cluster


def test_surrogate_is_deterministic():
    from downside.config import LOCATIONS_BY_ID

    loc = LOCATIONS_BY_ID["napa-ca"]
    a = SyntheticSource(seed=42).fetch(loc, 1990, 2000)
    b = SyntheticSource(seed=42).fetch(loc, 1990, 2000)
    assert np.array_equal(a.tmax_c, b.tmax_c)
    assert np.array_equal(a.precip_mm, b.precip_mm)
