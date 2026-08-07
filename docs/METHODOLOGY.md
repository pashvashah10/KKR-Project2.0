# Methodology

How a century of daily weather becomes a price, and what each step assumes.

Statistical toolkit follows the MIT Sloan Business Club *Quant Bible* — §4.3
regression, §4.6 econometrics, §6.2 market making — applied to weather rather
than to equities.

---

## 1. Ingest

Daily observations at a station: `TMAX`, `TMIN`, `PRCP`, `SNOW`, `AWND`.

Station data is preferred over reanalysis because a parametric contract settles
on a gauge, not on a grid cell. The distance between the gauge and the insured
site is **basis risk**, and it is priced explicitly rather than assumed away.

Gaps are filled once, explicitly, with the method recorded:

- temperature and wind gaps ≤ 5 days — linear interpolation (synoptic
  persistence makes this accurate at that length)
- longer gaps — resampled from the same calendar window, so the filled stretch
  keeps realistic variance rather than flattening to the mean
- precipitation and snow — never interpolated; interpolating between two dry
  days manufactures a physically impossible drizzle

**Per-variable coverage is tracked.** A variable the station never measured must
not be mistaken for one that measured zero.

---

## 2. Mean model

For daily variable `y` on day-of-year `d` in year `t`:

```
mu(d,t) = b0
        + Σ_k [a_k cos(2πk d/365.25) + b_k sin(2πk d/365.25)]     seasonal cycle
        + λ · G(t)                                                 amplification
        + Σ_k G(t) · [c_k cos(·) + d_k sin(·)]                     seasonal amplification
        + δ · (t − t0)                                             station drift
```

`G(t)` is global mean temperature anomaly from a two-box energy balance model
driven by observed forcing.

**Why not regress on time.** The observed record is not linear in time. It has an
aerosol plateau, volcanic notches, and a post-1980 steepening. Fitting against
time forces a shape the data does not have. Fitting against `G(t)` reproduces
those features automatically, and gives `λ` a physical meaning: degrees of local
warming per degree of global warming.

**Seasonal amplification** is what lets winter warm faster than summer, which it
demonstrably does at continental sites. The F-test on this block is significant
at every site tested.

**Station drift** absorbs urbanisation and siting artefacts. It is fitted with a
tight `N(0, 0.05²)` prior and **frozen at 2025 — never extrapolated**. Projecting
a car park's growth for a century is not climate science.

Estimated by OLS, `β̂ = (XᵀX)⁻¹Xᵀy`, with a per-column ridge penalty on the drift
term where `λ_j = σ²/τ_j²` — the posterior mode under that prior.

---

## 3. Variance model

Residuals are heteroskedastic: winter spread far exceeds summer. `log(e²)` is
regressed on the same design, with the `E[log χ²₁] = −1.2704` offset applied to
debias it. Without that correction every σ comes out small and every tail
probability comes out low.

Variance *trend* matters as much as mean trend for tail pricing: if summer
variance widens, extreme-heat frequency rises even with the mean pinned.

---

## 4. Standard errors

Classical OLS standard errors are **wrong here by roughly 3×**.

Daily weather has lag-1 residual autocorrelation ≈ 0.70 plus multi-year regime
structure. OLS treats 36,525 days as 36,525 independent observations. The
effective sample size for a *trend* coefficient is closer to the number of
independent multi-year epochs — order 20–30.

Corrected with a **moving-block bootstrap over whole years** (8-year circular
blocks, 160–240 replicates), which assumes nothing about the form of the
dependence. Interval coverage was checked against known ground truth: 8/8.

HC1 robust SEs are also reported, but they fix heteroskedasticity, not
autocorrelation, and are not sufficient on their own.

---

## 5. Tails

Peaks-over-threshold, above the 95th–97th percentile. By Pickands–Balkema–de
Haan the exceedance distribution converges to Generalised Pareto regardless of
the parent.

Fitted by **Grimshaw's one-dimensional profile likelihood** — substituting
`θ = ξ/σ` collapses the two-parameter likelihood onto one variable with a
closed-form profile. scipy's generic two-parameter fit routinely violates the
support constraint and returns whatever the optimiser stopped on.

Recovered shapes on synthetic GPD data across ξ ∈ [−0.4, +0.8] are accurate to
within ~0.04. On real data: rainfall ξ > 0 (heavy, unbounded), temperature
residuals ξ < 0 (bounded, as the physics requires).

The sampling distribution is a **splice**: empirical below the threshold, GPD
above. The body holds only between-threshold values — indexing the full sorted
sample lets a body draw return a tail value the GPD arm already accounts for,
double-counting extremes and inflating variance by ~13%.

---

## 6. Simulation

A shared **synoptic latent factor** drives everything, because weather variables
are dependent and the dependence is what prices the product:

| Field | Construction |
|---|---|
| `u_t` | AR(1) Gaussian latent |
| tmax, tmin | spliced marginal via Gaussian copula on `u_t` |
| wet/dry | two-state Markov chain; covariates = day-of-year, `G(t)`, temperature anomaly |
| amount | Gamma GLM mean × spliced Gamma/GPD multiplier |
| snow | precipitation partitioned by the station's own empirical ratio distribution |
| wind | log-normal marginal on its own AR(1), correlated with `u_t` |

**Autocorrelation travels through the copula**, so the marginal *and* the
persistence are both preserved. Consecutive-day triggers — a 3-day heat wave —
are priced off run lengths, and an independent-day simulator underprices them by
about an order of magnitude at ρ = 0.7.

**Temperature enters the precipitation model as a covariate.** Rain days are cool
in summer and mild in winter. Drawing precipitation independently would get
combined triggers ("hot and dry", "cold and wet") wrong in both directions.

**Snow ratio is binned on temperature *and* amount.** They are strongly
anticorrelated: 20:1 powder falls from cold dry events, while big liquid totals
arrive on warm advection as dense 3:1 snow. Binning on temperature alone paired a
cold-bin ratio with a tail precipitation draw and produced a 1745 mm daily
snowfall at Vail against 634 mm in the record.

---

## 7. Projection

Forcing pathway → two-box EBM → `G(t)` → local response through the shrunk
amplification.

Amplification is shrunk toward a regional physical prior by conjugate normal
update. One station over one century gives bootstrap SEs of 0.15–0.35; shrinkage
stops a lucky run of hot summers being extrapolated for a century.

### Uncertainty decomposition

| Source | Behaviour | Dominates |
|---|---|---|
| Internal variability | roughly constant in absolute terms | first ~20 years |
| Parameter | grows with the warming signal | middle |
| Scenario | near zero short, large long | after ~2050 |

Measured crossover at the sites tested: internal variability holds ~75–80% of
variance in 2026; scenario holds ~74–82% by 2125.

This is the argument for modelling rather than looking up a historical
frequency, and it is shown on the dashboard for exactly that reason.

---

## 8. Pricing

Market-making framing (*Quant Bible* §6.2):

```
theo        = E[payout] under the contract-year distribution
half_width  = f(model uncertainty)           uncertain quantity → wider market
skew        = f(inventory)                   give up edge to flatten the book
```

Loads, each charged for a named reason so a quote can be defended line by line:

| Load | Basis |
|---|---|
| Capital | `0.16 · (CVaR₉₉ − E) + 0.10 · σ` — payouts are wildly skewed; EV alone doesn't pay for the capital behind the tail |
| Model | `1.25 · σ_scenario` — spread of theo across pathways |
| Basis | `0.0045 · km · theo` — gauge-to-site distance |
| Expense | `0.045 · limit · √(days/90)` — origination and settlement |

Contracts use an **attachment point** above the typical season, so they cover a
genuinely bad year rather than the ordinary friction a business already absorbs.
Without one, trigger probability approaches 100% and rate-on-line approaches
90% — that is a transfer, not insurance.

---

## 9. Hedging

Loss basis is **contribution margin**, not revenue. A washed-out $40k day with a
30% variable-cost ratio is a $28k loss. Hedging revenue overhedges by exactly
the variable-cost ratio.

Sensitivity is fitted with a **piecewise-linear hinge basis**, controlling for
day-of-week and seasonality (both confounded with weather — summer is busier
*and* hotter). Validated against a known 8 mm generating threshold: recovered at
9.4 mm with the correct flat → steep → flat segment structure.

Sizing:

```
h*   = ρ · (σ_loss / σ_payout)                  min-variance ratio
f*   = argmax E[log(W − loss + q·payout − q·price)]   subject to a ruin floor
```

Walk out along the frontier while each step still buys a meaningful reduction in
ruin probability; past that the operator is paying risk premium for noise.

`1 − ρ²` is the share of loss variance no amount of notional can hedge — the
basis risk, reported rather than buried.

---

## 10. Validation

Walk-forward: fit through 1995, score 1996–2025, which the model never sees.

| Metric | Question |
|---|---|
| CRPS vs climatology | does it beat "average the past"? |
| PIT histogram | is the predicted spread honest? U-shaped = overconfident |
| Reliability | of days given 20% odds, did ~20% trigger? |
| Interval coverage | does the 90% band contain 90%? |

Results on real station data: interval coverage 92–93% against a nominal 90%
(slightly *under*confident, the safe direction for a seller), reliability close
to the diagonal, CRPS skill in the low single digits, residual cold bias
0.2–0.3 °C.

The bias is real and is reported. It drove a model change: at a loose drift
prior the drift term absorbs genuine training-window warming and is then frozen
at projection time, leaving a 0.55 °C cold bias at Boston. Tightening the prior
to 0.05 halved it.

For daily temperature at 30-year lead, climatology is a very strong baseline.
The value here is in the trend, the tails and the full distribution, not in
daily point forecasts.
