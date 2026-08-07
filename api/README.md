# Downside API

What a customer actually does:

```bash
# 1. Sign up. The key is shown once.
curl -X POST $API/v1/accounts -H 'content-type: application/json' \
  -d '{"name":"Harbourside Events","email":"ops@harbourside.example"}'

# 2. Add a venue. Coordinates, not a city --- basis risk lives in that gap.
curl -X POST $API/v1/sites -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{
    "name":"Charleston Waterfront Park","lat":32.7796,"lon":-79.9245,
    "vertical":"Waterfront venue","season_start_month":3,"season_end_month":11,
    "variable_cost_ratio":0.32}'
# -> 202 Accepted, with a job to poll. Fitting takes 1-3 minutes, once per site.

# 3. Watch it fit.
curl $API/v1/jobs/$JOB -H "authorization: Bearer $KEY"

# 4. Upload daily revenue. This is what makes the loss curve yours.
curl -X POST $API/v1/sites/$SITE/revenue -H "authorization: Bearer $KEY" \
  -F file=@revenue.csv          # columns: date,revenue

# 5. Your exposure.
curl $API/v1/sites/$SITE/exposure -H "authorization: Bearer $KEY"

# 6. Terms we suggest, from your own exposure.
curl $API/v1/sites/$SITE/suggested-terms -H "authorization: Bearer $KEY"

# 7. A price.
curl -X POST $API/v1/sites/$SITE/quote -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{
    "peril_id":"rain-day","year":2027,"payout_per_day":73800,
    "limit":590400,"attachment_days":35}'

# 8. How much to buy, given your balance sheet.
curl -X POST $API/v1/sites/$SITE/hedge -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{
    "peril_id":"rain-day","year":2027,"payout_per_day":73800,"limit":590400,
    "attachment_days":35,"reserves":900000,"monthly_burn":180000}'
```

Interactive docs at `/docs`.

## Running it

```bash
pip install -r api/requirements.txt
python3 -m uvicorn api.main:app --port 8000
```

SQLite at `api/data/downside.db`, fitted models pickled to `api/data/models/`.
The schema ports to Postgres unchanged when there is a reason to move.

## Why fitting is asynchronous

One site is a century of daily observations, a mean and variance model, a
160-replicate block bootstrap, tail fits and a Monte Carlo grid: 80-140 seconds.
Nobody waits that long on a request and nobody should pay to recompute it per
page view. `POST /v1/sites` returns `202` with a job; everything afterwards
reads the cached artefact and returns in milliseconds.

## What is not here yet

- **Nearest-station lookup.** A customer's coordinates currently resolve to ERA5
  reanalysis, which is gap-free and exact but is not what a contract settles on.
  Matching to the nearest long-record GHCN station, and reporting the distance as
  basis risk, is the next piece.
- **Payments and policy issuance.** This prices and sizes; it does not bind.
- **Rate limiting and key rotation.**
