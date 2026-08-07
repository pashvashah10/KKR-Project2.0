# Downside — weather risk analytics

Sensible Weather protects the guest. This protects the business.

A hundred years of daily weather at a coordinate, fitted; a hundred years
projected forward under four emissions pathways; and a price on the contracts
that sit on top.

```
api/        FastAPI service --- what a customer actually talks to
engine/     Python analytics engine --- the models and the math
web/        Customer app and the internal terminal
docs/       Methodology
```

---

## How a business uses this

```bash
pip install -r api/requirements.txt -r engine/requirements.txt
python3 web/build.py all
python3 -m uvicorn api.main:app --port 8000    # then open localhost:8000
```

The landing page takes coordinates, fits a century of weather at that point,
and returns an exposure, suggested terms and a price. Nothing is precomputed.

Everything it does is a public endpoint --- see `api/README.md`, or `/docs` for
the interactive reference:

| | |
|---|---|
| `POST /v1/accounts` | sign up, receive an API key |
| `POST /v1/sites` | add a venue at any coordinates (async, returns a job) |
| `POST /v1/sites/{id}/revenue` | upload daily revenue CSV |
| `GET /v1/sites/{id}/exposure` | your loss curve and expected loss |
| `GET /v1/sites/{id}/suggested-terms` | terms proposed from your own exposure |
| `POST /v1/sites/{id}/quote` | price a contract |
| `POST /v1/sites/{id}/hedge` | size it against your balance sheet |

Fitting takes 1--3 minutes and happens once per site; the model is cached, so
every quote afterwards returns in milliseconds.

---

## What it does

**Fits a century.** Daily station records from NOAA GHCN-Daily, decomposed into a
seasonal cycle, a warming response, and heteroskedastic noise with fitted tails.

**Projects a century.** Four SSP forcing pathways through a two-box energy
balance model, scaled by the station's own estimated regional amplification.

**Prices the result.** Monte Carlo simulation of the fitted generator produces
trigger probabilities and payout distributions, which become a two-sided quote
with named risk loads.

**Sizes the hedge.** For the operator: an empirical loss curve regressed from
daily revenue, and hedge sizing against ruin probability rather than expected
value.

---

## The design decisions that matter

Most of what makes this defensible is a handful of choices where the obvious
approach is wrong. Each is documented at the top of the module that implements
it.

### Local temperature regresses on global temperature, not on time

`climatology.py`

Fitting a trend against calendar time forces a shape onto the past that the
record does not have — the 1940–75 aerosol plateau, the volcanic notches after
Agung, El Chichón and Pinatubo, the post-1980 steepening. A linear fit averages
all of it and then extrapolates the average for another century.

Regressing on `G(t)` — the two-box response to observed forcing — reproduces
those features without being told they exist, and the coefficient *is* the
regional amplification factor: degrees local per degree global. Extrapolation
then needs no assumption about the shape of the future, only a choice of
scenario.

### The standard errors are wrong by 3× unless you fix them

`resample.py`

Daily weather has lag-1 residual autocorrelation around 0.70 plus multi-year
regime structure. Classical OLS treats 36,525 days as 36,525 independent
observations, so it reports the warming coefficient about three times more
precisely than the data supports.

A year-block bootstrap gives the honest figure. This is not pedantry — it is the
difference between quoting "1.28 ± 0.08 °C per °C" and "1.28 ± 0.27", and the
second one changes every long-dated price.

### Uncertainty is three things, not one

`projection.py`

Internal variability dominates the first ~20 years and is roughly constant in
absolute terms. Scenario uncertainty is near zero at short horizons and dominant
by 2100. Parameter uncertainty sits between them.

Conflating them produces a band that is wrong at both ends. The practical
consequence: a 2027 contract is priced almost entirely off weather noise, so
history is nearly sufficient; a 2075 contract is priced off scenario spread, and
quoting it from historical frequencies is simply wrong.

### Tails get extreme value theory, not a Gaussian

`tails.py`

Everything a weather product pays on lives in the tail. A Gaussian fitted to the
body understates trigger frequency badly. Peaks-over-threshold GPD fits give a
shape parameter that comes out positive for rainfall (heavy, unbounded) and
negative for temperature (bounded, as the physics requires).

### The trigger is regressed, never asked

`hedging.py`

Operators do not know their own thresholds. Ask and you get "rain hurts us".
Regress with a piecewise-linear hinge basis and you find the curve is flat until
about 0.3 in, then falls off a cliff. A single-slope fit averages the flat part
and the cliff into a line that describes neither.

### Hedge to survival, not to expected value

`hedging.py`

Every hedge loses money in expected dollars — the premium contains the seller's
risk load. Judged on expected value the correct hedge is always zero.

Businesses maximise survival and compounding, so sizing runs on expected log
wealth subject to a ruin constraint. The consequence worth putting in front of a
customer: two identical businesses with identical exposure should hedge
*different amounts* if their cash positions differ.

---

## Data

| Source | Role |
|---|---|
| NOAA NCEI / GHCN-Daily | Primary. Station observations — what a contract settles on. |
| Open-Meteo / ERA5 | Fallback. Gap-free, exact coordinates, 1940 onward. |
| Calibrated surrogate | Last resort. Structurally faithful, **not observations**. |

Whatever answers carries an honest `provenance` string that travels to the
dashboard and is displayed there.

**Network access.** The NOAA and Open-Meteo hosts must be reachable. In a Claude
Code cloud environment that means setting **Network access** to **Custom** and
allowing `www.ncei.noaa.gov` and `archive-api.open-meteo.com`, with the default
package-manager list left enabled.

### Per-variable coverage

Many century-long stations record temperature and precipitation and never record
wind. Gap filling turns absent days into numbers, and for a variable absent
*everywhere* the filled series collapses to a constant — Jackson Hole returned
36,525 identical zeros for wind, which would price a wind contract at zero
premium.

Records therefore carry per-variable coverage, `DailyRecord.usable()` gates on
it, unmeasured fields are refilled from the surrogate and flagged, and the
dashboard omits perils whose settlement variable was never measured rather than
showing a fabricated frequency.

---

## Honest limitations

**Daily revenue is placeholder.** `hedging.synthetic_revenue` exists to exercise
the loss-curve regression against a known answer. Every loss curve, hedge ratio
and ruin probability is only as real as the revenue behind it, and real daily
revenue at daily granularity is the single most valuable input this platform
takes.

**Out-of-sample skill against climatology is small.** Walk-forward validation
(fit to 1995, scored on 1996–2025) gives CRPS skill in the low single digits and
a residual cold bias of 0.2–0.3 °C. For *daily temperature at 30-year lead*
climatology is a very strong baseline, and beating it at all is meaningful — but
the value here is in the trend, the tails and the distribution, not in daily
point forecasts. The dashboard shows the backtest rather than hiding it.

**Amplification is weakly identified at some sites.** `G(t)` and calendar time
are near-collinear over a century, so the warming coefficient and any station
drift trade against each other. The drift coefficient carries a tight prior, the
OVB shift is reported, and where the estimate is specification-sensitive the
model diagnostics say so.

**Kalshi is not wired in.** Contract pricing here is model-based. Live contract
prices, backtested settlement and basis against traded instruments are the next
step.

---

## Layout

| Module | Contents |
|---|---|
| `linalg.py` | OLS, ridge with per-column penalties, t/F tests, HC1 robust SEs, OVB |
| `glm.py` | Logistic and Gamma GLMs by IRLS, for precipitation |
| `resample.py` | Year-block bootstrap, effective sample size |
| `forcing.py` | Two-box energy balance model, SSP pathways, volcanic forcing |
| `climatology.py` | Seasonal + forcing mean model, heteroskedastic variance model |
| `tails.py` | GPD peaks-over-threshold, GEV block maxima, spliced distributions |
| `site.py` | Full per-site model and the Monte Carlo weather generator |
| `projection.py` | 100-year forward, three-way uncertainty decomposition |
| `pricing.py` | Theoretical value, risk loads, two-sided quotes |
| `hedging.py` | Loss curves, min-variance ratio, ruin-adjusted sizing |
| `backtest.py` | Walk-forward validation, CRPS, PIT, reliability |

Reference for the statistical toolkit: MIT Sloan Business Club *Quant Bible*,
§4.3 (regression), §4.6 (econometrics), §6.2 (market making).
