"""The operator's side: what weather costs them, and how much to hedge.

Three ideas from the build plan drive this module, and all three are places where
the obvious approach is wrong.

**1. Regress the trigger, never ask for it.**
Operators do not know their own thresholds. Ask and you get "rain hurts us";
regress and you find 0.3 in does nothing measurable, 0.8 in destroys the day, and
sustained heat above 95F costs more than any amount of light rain. So the loss
curve is fitted with a **piecewise-linear hinge basis** rather than a single
slope --- a linear fit would average a flat region and a cliff into a slope that
describes neither, and would size every hedge off that fiction.

**2. Hedge contribution margin, not revenue.**
A washed-out day loses its revenue but saves its variable cost. A $40k day with a
30% variable-cost ratio is a $28k loss, not $40k. Hedging revenue overhedges by
exactly the variable-cost ratio, every single time.

**3. Hedge to survival, not to expected value.**
Every hedge has negative expected value --- the premium contains the seller's risk
load, which is precisely what `pricing.py` computes. Judged on expected dollars,
the correct hedge is always zero. That is the wrong objective. Businesses
maximise survival and compounding, so the criterion here is **expected log
wealth** subject to a ruin constraint.

The consequence is the thing worth putting on screen: two identical businesses
with identical weather exposure should hedge *different amounts* if their cash
positions differ. A $28k loss against $60k of reserves is potentially existential;
against $2M it is a rounding error. Nobody in this market frames it this way, and
the optimiser here makes it a number rather than an argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .linalg import LinearFit, fit_ols
from .sources.base import DailyRecord

__all__ = [
    "LossCurve",
    "fit_loss_curve",
    "HedgeAnalysis",
    "analyse_hedge",
    "synthetic_revenue",
]


# ----------------------------------------------------------------------
# Loss curve
# ----------------------------------------------------------------------


@dataclass(slots=True)
class LossCurve:
    """Fitted sensitivity of daily contribution margin to one weather variable."""

    variable: str
    knots: np.ndarray
    fit: LinearFit = field(repr=False)
    grid: np.ndarray = field(repr=False)
    margin_at: np.ndarray = field(repr=False)
    baseline_margin: float = 0.0
    r_squared: float = 0.0
    #: Where the curve is steepest --- the empirical trigger, discovered not asked.
    breakpoint: float = 0.0
    #: Marginal margin lost per unit of the weather variable, on each segment.
    segment_slopes: list = field(default_factory=list)

    def loss_at(self, values: np.ndarray) -> np.ndarray:
        """Margin lost relative to the benign baseline, floored at zero."""
        interp = np.interp(np.asarray(values, dtype=float), self.grid, self.margin_at)
        return np.clip(self.baseline_margin - interp, 0.0, None)

    def curve_rows(self) -> list[dict]:
        return [
            {"x": float(x), "margin": float(m), "loss": float(max(self.baseline_margin - m, 0.0))}
            for x, m in zip(self.grid, self.margin_at)
        ]


def _hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear (ReLU) basis: [x, (x-k1)+, (x-k2)+, ...].

    Continuous by construction, and each coefficient reads as the *change* in
    slope at that knot --- so the fitted output can be read directly as "nothing
    happens until here, then it falls off a cliff", which is the shape these
    curves actually have.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    hinges = np.clip(x - np.asarray(knots, dtype=float).reshape(1, -1), 0.0, None)
    return np.column_stack([x, hinges])


def fit_loss_curve(
    record: DailyRecord,
    revenue: np.ndarray,
    variable: str = "precip_mm",
    variable_cost_ratio: float = 0.30,
    n_knots: int = 5,
    mask: np.ndarray | None = None,
) -> LossCurve:
    """Regress daily contribution margin on a weather variable.

    Controls for day-of-week and seasonality, because both are confounded with
    weather --- summer is both busier and hotter, and attributing the seasonal
    revenue peak to temperature would invert the sign of the heat sensitivity.
    This is the Quant Bible's conditional independence assumption doing real work:
    the weather coefficient is only causal once the things that move with weather
    *and* with revenue are held fixed.
    """
    if mask is None:
        mask = np.ones(len(record), dtype=bool)

    margin = np.asarray(revenue, dtype=float)[mask] * (1.0 - variable_cost_ratio)
    x = record.get(variable)[mask]
    doy = record.doy[mask].astype(float)
    dow = (record.dates[mask].astype("datetime64[D]").astype(int) + 4) % 7

    # Knots at quantiles of the *active* range. For precipitation that means
    # quantiles of wet days: with 70% dry days, unconditional quantiles would
    # stack every knot at zero and resolve nothing.
    active = x[x > (0.2 if variable == "precip_mm" else -np.inf)]
    if active.size < 50:
        active = x
    qs = np.linspace(0.15, 0.95, n_knots)
    knots = np.unique(np.quantile(active, qs))

    hinge = _hinge_basis(x, knots)
    phase = 2.0 * np.pi * doy / 365.25
    seasonal = np.column_stack(
        [np.cos(k * phase) for k in (1, 2, 3)] + [np.sin(k * phase) for k in (1, 2, 3)]
    )
    dow_dummies = np.column_stack([(dow == d).astype(float) for d in range(1, 7)])

    X = np.column_stack([np.ones_like(x), hinge, seasonal, dow_dummies])
    names = (
        ["intercept", f"{variable}_slope"]
        + [f"hinge_{i}" for i in range(len(knots))]
        + [f"cos{k}" for k in (1, 2, 3)]
        + [f"sin{k}" for k in (1, 2, 3)]
        + [f"dow_{d}" for d in range(1, 7)]
    )
    fit = fit_ols(X, margin, names=names)

    # Evaluate the curve holding season and weekday at their averages, so the
    # displayed shape is the weather effect alone.
    grid = np.linspace(float(np.min(x)), float(np.quantile(x, 0.999)), 120)
    hinge_grid = _hinge_basis(grid, knots)
    mean_seasonal = seasonal.mean(axis=0)
    mean_dow = dow_dummies.mean(axis=0)
    Xg = np.column_stack(
        [
            np.ones_like(grid),
            hinge_grid,
            np.tile(mean_seasonal, (len(grid), 1)),
            np.tile(mean_dow, (len(grid), 1)),
        ]
    )
    margin_at = Xg @ fit.beta

    baseline = float(margin_at[0]) if variable == "precip_mm" else float(np.max(margin_at))

    slopes = np.diff(margin_at) / np.diff(grid)
    breakpoint = float(grid[int(np.argmin(slopes))]) if slopes.size else 0.0

    seg = []
    edges = np.concatenate([[grid[0]], knots, [grid[-1]]])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (grid >= lo) & (grid <= hi)
        if sel.sum() > 2:
            s = np.polyfit(grid[sel], margin_at[sel], 1)[0]
            seg.append({"from": float(lo), "to": float(hi), "slope": float(s)})

    return LossCurve(
        variable=variable,
        knots=knots,
        fit=fit,
        grid=grid,
        margin_at=margin_at,
        baseline_margin=baseline,
        r_squared=fit.r_squared,
        breakpoint=breakpoint,
        segment_slopes=seg,
    )


# ----------------------------------------------------------------------
# Hedge sizing
# ----------------------------------------------------------------------


@dataclass(slots=True)
class HedgeAnalysis:
    #: Cov(L, X) / Var(X). The variance-minimising number of contracts.
    min_variance_ratio: float
    correlation: float
    sigma_loss: float
    sigma_payout: float
    #: Hedge fraction maximising expected log wealth --- the one to act on.
    kelly_fraction: float
    #: Fraction at which marginal ruin reduction stops paying for its premium.
    recommended_fraction: float
    unhedged_ruin: float
    hedged_ruin: float
    unhedged_cvar: float
    hedged_cvar: float
    expected_cost: float
    basis_risk_share: float
    frontier: list = field(default_factory=list)


def analyse_hedge(
    loss: np.ndarray,
    payout: np.ndarray,
    price: float,
    reserves: float,
    fixed_burn_monthly: float,
    months_of_runway_required: float = 3.0,
    max_fraction: float = 2.0,
    n_grid: int = 41,
) -> HedgeAnalysis:
    """Size the hedge against ruin risk, not against expected value.

    `loss` and `payout` are paired samples from the same simulated paths --- that
    pairing is essential, since the whole question is how well the contract
    covers *this* operator's loss, not how the two behave separately.

    `price` is the premium per unit of contract (the offer from `pricing.py`).
    """
    loss = np.asarray(loss, dtype=float)
    payout = np.asarray(payout, dtype=float)
    n = min(len(loss), len(payout))
    loss, payout = loss[:n], payout[:n]

    sigma_l = float(loss.std(ddof=1))
    sigma_x = float(payout.std(ddof=1))
    corr = float(np.corrcoef(loss, payout)[0, 1]) if sigma_l > 0 and sigma_x > 0 else 0.0
    h_star = float(corr * sigma_l / sigma_x) if sigma_x > 0 else 0.0

    # Ruin threshold: the cash floor below which the business cannot fund its
    # fixed burn through the off-season.
    floor = months_of_runway_required * fixed_burn_monthly

    # Hedge fraction f is expressed as a multiple of the min-variance ratio, so
    # f = 1 means "fully hedged in the variance-minimising sense".
    fractions = np.linspace(0.0, max_fraction, n_grid)
    frontier = []
    best_log, best_f = -np.inf, 0.0

    for f in fractions:
        q = f * h_star
        wealth = reserves - loss + q * payout - q * price
        ruin = float((wealth < floor).mean())
        cvar = float(-np.mean(np.sort(wealth)[: max(int(0.05 * n), 1)]))
        # Log utility needs a strictly positive argument; anything at or below
        # the floor is treated as ruin and given a large finite penalty rather
        # than -inf, so the optimiser still sees a usable gradient.
        safe = wealth - floor
        util = np.where(safe > 1.0, np.log(np.clip(safe, 1.0, None)), -12.0)
        exp_log = float(util.mean())
        frontier.append(
            {
                "fraction": float(f),
                "contracts": float(q),
                "premium": float(q * price),
                "ruin": ruin,
                "expected_log_wealth": exp_log,
                "expected_wealth": float(wealth.mean()),
                "cvar95": cvar,
                "sd_wealth": float(wealth.std(ddof=1)),
            }
        )
        if exp_log > best_log:
            best_log, best_f = exp_log, float(f)

    # Recommended size: walk out along the frontier while each additional step
    # still buys a meaningful reduction in ruin probability. Past that point the
    # operator is paying risk premium for noise.
    unhedged_ruin = frontier[0]["ruin"]
    recommended = 0.0
    for row in frontier[1:]:
        gain = unhedged_ruin - row["ruin"]
        if gain <= 0.001 and row["fraction"] > 0.25:
            break
        recommended = row["fraction"]
        if row["fraction"] >= best_f:
            break
    recommended = min(recommended, best_f) if best_f > 0 else 0.0

    at_rec = min(range(len(frontier)), key=lambda i: abs(frontier[i]["fraction"] - recommended))

    return HedgeAnalysis(
        min_variance_ratio=h_star,
        correlation=corr,
        sigma_loss=sigma_l,
        sigma_payout=sigma_x,
        kelly_fraction=best_f,
        recommended_fraction=recommended,
        unhedged_ruin=unhedged_ruin,
        hedged_ruin=frontier[at_rec]["ruin"],
        unhedged_cvar=frontier[0]["cvar95"],
        hedged_cvar=frontier[at_rec]["cvar95"],
        expected_cost=frontier[at_rec]["premium"],
        # 1 - rho^2 is the share of loss variance the contract cannot reach ---
        # the basis risk that no amount of notional will hedge away.
        basis_risk_share=float(1.0 - corr**2),
        frontier=frontier,
    )


# ----------------------------------------------------------------------
# Demonstration revenue
# ----------------------------------------------------------------------


def synthetic_revenue(
    record: DailyRecord,
    vertical: str,
    seed: int = 5,
) -> np.ndarray:
    """Plausible daily revenue with a known nonlinear weather response.

    **This is placeholder data.** Real daily revenue is the single most valuable
    input the platform takes, and nothing here substitutes for it --- the loss
    curve is only as real as the revenue behind it.

    Its purpose is to exercise `fit_loss_curve`: the response built in here is
    deliberately flat-then-cliff, so the fitted hinge basis can be checked
    against a known answer rather than admired for producing a smooth line.
    """
    rng = np.random.default_rng(seed + (abs(hash(vertical)) % 9973))
    n = len(record)
    doy = record.doy.astype(float)
    phase = 2.0 * np.pi * doy / 365.25

    profile = {
        "Ski resort": (140_000.0, -0.85, 0.20),
        "Golf resort": (48_000.0, 0.35, 0.55),
        "Festival grounds": (95_000.0, 0.55, 0.30),
        "Winery & events": (36_000.0, 0.60, 0.35),
        "Campground group": (28_000.0, 0.70, 0.25),
        "Waterfront venue": (62_000.0, 0.65, 0.30),
        "Outdoor attraction": (110_000.0, 0.50, 0.40),
    }
    base, season_sign, weekend_lift = profile.get(vertical, (50_000.0, 0.5, 0.35))

    seasonal = 1.0 + season_sign * 0.55 * np.cos(phase - (0.0 if season_sign < 0 else np.pi))
    dow = (record.dates.astype("datetime64[D]").astype(int) + 4) % 7
    weekend = 1.0 + weekend_lift * np.isin(dow, [5, 6])

    # Flat until the threshold, then a cliff --- the shape the regression must find.
    rain = record.precip_mm
    rain_hit = np.clip((rain - 8.0) / 14.0, 0.0, 1.0) * 0.62 + np.clip((rain - 2.0) / 30.0, 0.0, 0.10)
    heat = np.clip((record.tmax_c - 35.0) / 8.0, 0.0, 1.0) * 0.40
    cold = np.clip((2.0 - record.tmax_c) / 12.0, 0.0, 1.0) * 0.18
    wind = np.clip((record.wind_ms - 12.0) / 8.0, 0.0, 1.0) * 0.25

    if vertical == "Ski resort":
        # Snow-driven: revenue tracks accumulated depth, and warmth is the peril.
        rolling = np.convolve(record.snow_mm, np.ones(21), mode="same")
        cover = np.clip(rolling / (21 * 18.0), 0.0, 1.6)
        weather = np.clip(0.30 + 0.70 * cover, 0.0, 1.25) * (1.0 - wind)
        closed = ~np.isin(record.doy, np.concatenate([np.arange(1, 121), np.arange(305, 367)]))
        weather = np.where(closed, 0.0, weather)
    else:
        weather = np.clip(1.0 - rain_hit - heat - cold - wind, 0.0, 1.25)

    noise = rng.lognormal(0.0, 0.16, size=n)
    growth = 1.0 + 0.012 * (record.year - record.year.min())
    return np.clip(base * seasonal * weekend * weather * noise * growth, 0.0, None)
