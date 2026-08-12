# Downside — weather risk analytics

Sensible Weather protects the guest. This protects the business.

A hundred years of daily weather at a coordinate, fitted; a hundred years
projected forward under four emissions pathways; and a price on the contracts
that sit on top.

```
api/        FastAPI service --- the JSON API and the storefront
engine/     Python analytics engine --- the models and the math
web/        Templates, static assets, and the internal terminal
docs/       Methodology
DESIGN.md   The storefront's design specification
```

---

## How a business uses this

### In a browser, nothing installed (GitHub Codespaces)

Open the repository on GitHub → **Code ▾** → **Codespaces** → **Create codespace**.
When the editor loads, type one command in the terminal:

```bash
./run.sh
```

Then open the **PORTS** tab and click the globe icon on port 8000.

The container deliberately runs **no setup commands of its own** — `run.sh`
installs what it needs. Two earlier versions tried to automate this and hung the
codespace on "Setting up" instead, because a devcontainer waits for its lifecycle
hooks to exit and a web server never does. One visible command beats a spinner.

### The local way

```bash
./run.sh
```

That is the whole thing. It installs dependencies if they are missing, generates
and persists a signing key, fits the eight example venues in the background, and
serves on `localhost:8000`. Safe to re-run; `./run.sh --no-seed` skips the fitting.

Seeding needs internet and takes 5–20 minutes depending on how fast NCEI
responds. Station matching does not — that bundle is committed.

<details>
<summary>Or step by step</summary>

```bash
pip install -r api/requirements.txt -r engine/requirements.txt
export DOWNSIDE_SECRET=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
python3 -m api.scripts.build_station_cache   # only to refresh the station bundle
python3 -m api.seed                          # eight example venues
python3 -m uvicorn api.main:app --port 8000
```
</details>

### The storefront

A multi-page shop with real URLs, a cart and a checkout.

| Route | |
|---|---|
| `/` | The wedge, the four services, the honest limits |
| `/services`, `/services/{slug}` | Catalogue and detail |
| `/configure/{slug}` | Pick a venue, see the station match, get a price |
| `/cart`, `/checkout`, `/orders/{id}` | Order and invoice |
| **`/venues/{id}`** | **The report — what an Exposure Report actually buys** |
| `/account` | Venues, orders, subscriptions, API key |
| `/dashboard` | The analytics terminal (the eight example venues) |

Four things are for sale. Three are fixed-price and fit the venue *after*
purchase, as fulfilment. The fourth --- Parametric Cover --- needs the fit before
a price exists, so `/configure` starts an asynchronous job and polls it, showing
the staged progress the fitter emits, with already-fitted example venues offered
as the instant path.

**Checkout takes no payment and collects no card details.** It records a real
order, opens real subscriptions, starts real fulfilment, and says plainly on
screen that no money moved. `checkout.StripeProvider` documents the three
integration points and refuses rather than silently falling back to the mock.

**We are not an insurer, a broker or an MGA.** Parametric pricing is indicative
and subject to underwriting; every page that shows a price says so, and the
invoice is headed *"Indicative Parametric Risk Estimate — Not a Binding Policy
Contract"*.

### The report

`/venues/{id}` is what an Exposure Report buys, and it is the page the rest of
the platform exists to produce: the century fitted against the record, the
projection under four pathways, the uncertainty split three ways, the peril
frequencies with their pathway spread, the GPD tail with return levels, the loss
curve with its regressed breakpoint, and the walk-forward backtest — shown
whether or not it flatters the model.

Charts are server-rendered SVG (`api/charts.py`). No charting library and no
JavaScript: the report should render in a print dialogue and with scripting off,
because it is the deliverable rather than a dashboard.

### The API

Everything the storefront does is also a public endpoint --- see `api/README.md`,
or `/docs` for the interactive reference:

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

### The settlement station is chosen on record length, not on proximity

`api/stations.py`

Until this existed, a customer venue was fitted with `station_id=""`, fell
through to ERA5 reanalysis, and carried `station_distance_km = 0` — so
`pricing.load_basis`, which is `BASIS_PER_KM × station_distance_km × theo`,
charged **exactly nothing for basis risk on every contract sold**.

Matching is not `min(distance)`. A complete 1900–2025 record 40 km away is worth
more to a century fit than a patchy 20-year record 5 km away, and the near one
cannot settle a 2050 contract at all if it stopped reporting in 2003. Record
length is a hard gate; distance is the tiebreak.

Elevation is the half everyone forgets. A gauge 9 km away but 800 m below the
venue is separated by ~5 °C of lapse rate and sits on the wrong side of the
rain/snow line for much of the season — it is measuring a different climate, not
a nearby one. Vertical metres therefore enter both the match and the price at
0.05 km-equivalent each, and a gap past 300 m raises a microclimate warning on
the configure page instead of being quietly averaged away.

The effect is not theoretical: matching Mammoth Lakes on distance alone selects
Bishop Airport, 1,150 m below the resort. With elevation, it selects Bodie,
151 m off.

The index is a committed 889 KB numpy bundle covering 38,412 stations, built by
`python -m api.scripts.build_station_cache`. The server boots and matches with no
network at all; NOAA being down cannot take the storefront with it.

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

**No payment is taken and no cover is bound.** Checkout is a mock provider with a
documented Stripe seam; no card fields exist anywhere in the application.
Parametric Cover produces an indicative quote — issuing a policy needs a
licensed carrier or MGA on the paper, and we are not one.

**Aggregate contracts only cover per-day triggers.** A rolling-window peril such
as snow drought does not produce independent triggering days, so it cannot be
expressed in the aggregate structure priced here. Where that empties a venue's
peril list, `/configure` says which filter did it rather than reporting
"nothing triggers often enough" — those are different facts and only one of them
is about the weather.

**The station index is a snapshot.** Stations open and close; the bundle is
rebuilt deliberately, never silently at runtime, because a re-download that
changes which gauge a live contract settles on is not a thing that should happen
on its own.

**Mail delivery is not wired up, but the link is real.** A venue gets a signed
resume link the moment its fit starts. The link authenticates on its own, so it
opens the finished venue from a phone that has never seen the session — which is
what makes "you can close this tab" true. `web.SmtpSender` is a documented stub
that refuses rather than degrading silently, so `ConsoleSender` logs the link
and the page shows it on screen to bookmark. No page claims an email was sent.

Set **`DOWNSIDE_SECRET`** in any real deployment. Without it the signing key is
generated per process, so sessions and resume links do not survive a restart;
the server warns at startup and shortens the link's advertised lifetime to a day
rather than promising a week it cannot keep.

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

| API module | Contents |
|---|---|
| `stations.py` | GHCN index, terrain-aware nearest-station matching |
| `security.py` | The app signing key, and the signed resume links minted from it |
| `charts.py` | Server-rendered SVG for the report — no charting library, no JS |
| `catalog.py` | The four products, as code |
| `checkout.py` | Cart, orders, mock payment, Stripe seam |
| `web.py` | Storefront routes, async fit jobs, session cart |
| `service.py` | Fitting, exposure, pricing, hedging for one site |
| `store.py` | SQLite persistence and additive migrations |
| `seed.py` | The eight example venues |

Reference for the statistical toolkit: MIT Sloan Business Club *Quant Bible*,
§4.3 (regression), §4.6 (econometrics), §6.2 (market making).
