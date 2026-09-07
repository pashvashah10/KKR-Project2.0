"""Downside API.

What a customer actually does, in order:

    POST /v1/accounts                    sign up, receive an API key
    POST /v1/sites                       add a venue: coordinates, season, vertical
    GET  /v1/jobs/{id}                   watch the fit run (80-140s, once per site)
    POST /v1/sites/{id}/revenue          upload daily revenue CSV
    GET  /v1/sites/{id}/exposure         their loss curve and expected annual loss
    GET  /v1/sites/{id}/outlook          projection to 2125 under four pathways
    GET  /v1/sites/{id}/perils           trigger frequencies over time
    POST /v1/sites/{id}/quote            price a contract
    POST /v1/sites/{id}/hedge            size it against their balance sheet

Auth is a bearer API key issued at signup. That is the right level of ceremony
for design partners and paid pilots; it is not OAuth and does not pretend to be.

The one structural constraint worth understanding: **fitting is asynchronous.**
A century of daily observations, a bootstrap and a Monte Carlo grid take a
minute or two, so `POST /v1/sites` returns `202` with a job to poll. Everything
after that reads a cached artefact and returns immediately.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator
from starlette.middleware.sessions import SessionMiddleware

from . import demo, security, service, stations, store
from . import web as storefront

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("downside.api")

app = FastAPI(
    title="Downside",
    version="0.1.0",
    description="Weather risk analytics for operators: exposure, outlook, pricing, hedging.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The storefront cart lives in this cookie, and magic links are signed with the
# same key --- resolved once in `security` so the two cannot drift apart. Set
# DOWNSIDE_SECRET in any deployment where a session or a resume link should
# survive a restart.
app.add_middleware(
    SessionMiddleware,
    secret_key=security.app_secret(),
    session_cookie="downside_session",
    same_site="lax",
    max_age=60 * 60 * 24 * 30,
)

WEB = service.Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")

# One fit at a time per worker keeps memory predictable; each fit is already
# internally vectorised and the container has four cores.
FITTERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fit")

# Anonymous exposure checks get their own single worker rather than sharing the
# pool above. Without the split, a handful of visitors clicking the free check
# occupies both fitting threads and a paying customer's venue sits in a queue
# behind strangers --- the cheapest possible way to make the funnel damage the
# product it feeds.
PREVIEWERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")

storefront.configure_executor(
    lambda site_id, job_id: FITTERS.submit(service.fit_site, site_id, job_id),
    lambda site_id, job_id: PREVIEWERS.submit(service.fit_site, site_id, job_id, True),
)
app.include_router(storefront.router)


@app.on_event("startup")
def _startup() -> None:
    store.init()
    log.info("database ready at %s", store.DB_PATH)
    security.warn_if_ephemeral()
    # A hosted instance starts with a blank disk. Restoring the committed
    # snapshot means it serves real fitted venues in seconds instead of needing
    # a 15-minute NOAA fit before anything is worth looking at.
    demo.restore_if_empty()
    # Load the GHCN bundle off the request path. Failure is logged, not fatal:
    # without it, venues fall back to reanalysis and say so, but the storefront
    # still boots.
    stations.warm_index()


# ----------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------


def current_account(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Provide your API key as 'Authorization: Bearer <key>'.")
    account = store.account_by_key(authorization.split(" ", 1)[1].strip())
    if account is None:
        raise HTTPException(401, "That API key is not recognised.")
    return account


Account = Annotated[dict, Depends(current_account)]


def owned_site(site_id: str, account: dict) -> dict:
    site = store.get_site(site_id, account_id=account["id"])
    if site is None:
        raise HTTPException(404, "No such site on this account.")
    return site


def require_ready(site: dict) -> dict:
    if site["status"] == "failed":
        job = store.latest_job(site["id"])
        raise HTTPException(409, f"Fitting failed for this site: {(job or {}).get('error')}")
    if site["status"] != "ready":
        raise HTTPException(
            409,
            "This site is still being fitted. Poll the job from POST /v1/sites, "
            "or GET /v1/sites/{id} for its status.",
        )
    return site


# ----------------------------------------------------------------------
# schemas
# ----------------------------------------------------------------------

VERTICALS = [
    "Ski resort", "Golf resort", "Festival grounds", "Winery & events",
    "Campground group", "Waterfront venue", "Outdoor attraction",
]


class AccountIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr


class SiteIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-90, le=90, description="Decimal degrees. Use the venue, not the town.")
    lon: float = Field(ge=-180, le=180)
    elevation_m: float | None = Field(default=None, ge=-500, le=6000)
    vertical: str = Field(default="Outdoor attraction")
    season_start_month: int = Field(ge=1, le=12, default=1)
    season_end_month: int = Field(ge=1, le=12, default=12)
    variable_cost_ratio: float = Field(
        default=0.30, ge=0, le=0.95,
        description="Share of revenue that is variable cost. Loss is contribution "
                    "margin, not revenue; hedging revenue overhedges by exactly this.",
    )
    reserves: float | None = Field(default=None, ge=0)
    monthly_burn: float | None = Field(default=None, ge=0)

    @field_validator("vertical")
    @classmethod
    def known_vertical(cls, v: str) -> str:
        if v not in VERTICALS:
            raise ValueError(f"vertical must be one of: {', '.join(VERTICALS)}")
        return v


class QuoteIn(BaseModel):
    peril_id: str
    year: int = Field(ge=2026, le=2125)
    payout_per_day: float = Field(gt=0)
    limit: float = Field(gt=0)
    attachment_days: int = Field(ge=0, default=0)
    scenario_id: str = Field(default="ssp245")


class HedgeIn(QuoteIn):
    reserves: float = Field(gt=0)
    monthly_burn: float = Field(gt=0)


# ----------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", "verticals": VERTICALS}


@app.get("/v1/perils")
def perils() -> dict:
    """The trigger definitions available to price against."""
    from downside.config import PERILS

    return {
        "perils": [
            {
                "id": p.id, "label": p.label, "variable": p.variable,
                "statistic": p.statistic, "comparator": p.comparator,
                "threshold": round(p.threshold, 3), "unit": p.unit,
                "window_days": p.window_days, "description": p.description,
                "applies_to": list(p.applies_to),
            }
            for p in PERILS
        ]
    }


@app.post("/v1/accounts", status_code=201)
def signup(body: AccountIn) -> dict:
    existing = store.account_by_email(body.email)
    if existing:
        raise HTTPException(409, "An account already exists for that email.")
    account = store.create_account(body.name, body.email)
    return {
        "id": account["id"],
        "name": account["name"],
        "email": account["email"],
        "api_key": account["api_key"],
        "note": "Store this key. It is shown once and is the only credential.",
    }


@app.get("/v1/me")
def me(account: Account) -> dict:
    return {"id": account["id"], "name": account["name"], "email": account["email"]}


@app.post("/v1/sites", status_code=202)
def create_site(body: SiteIn, account: Account) -> dict:
    site = store.create_site(
        account["id"],
        name=body.name, lat=body.lat, lon=body.lon, elevation_m=body.elevation_m,
        vertical=body.vertical, season_start_month=body.season_start_month,
        season_end_month=body.season_end_month,
        variable_cost_ratio=body.variable_cost_ratio,
        reserves=body.reserves, monthly_burn=body.monthly_burn,
    )
    job = store.create_job(site["id"], "fit")
    FITTERS.submit(service.fit_site, site["id"], job["id"])
    return {
        "site": {k: site[k] for k in ("id", "name", "lat", "lon", "vertical", "status")},
        "job": {"id": job["id"], "status": "queued"},
        "poll": f"/v1/jobs/{job['id']}",
        "note": "Fitting a century of daily weather takes 1-3 minutes and happens once.",
    }


@app.get("/v1/sites")
def list_sites(account: Account) -> dict:
    sites = store.list_sites(account["id"])
    for s in sites:
        s["quality"] = json.loads(s["quality"]) if s["quality"] else None
    return {"sites": sites}


@app.get("/v1/sites/{site_id}")
def get_site(site_id: str, account: Account) -> dict:
    site = owned_site(site_id, account)
    site["quality"] = json.loads(site["quality"]) if site["quality"] else None
    job = store.latest_job(site_id)
    return {"site": site, "job": job, "revenue": store.revenue_summary(site_id)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, account: Account) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    owned_site(job["site_id"], account)
    return job


@app.post("/v1/sites/{site_id}/revenue")
async def upload_revenue(site_id: str, file: UploadFile, account: Account) -> dict:
    """Upload daily revenue as CSV with `date,revenue` columns.

    This is the single most valuable input the platform takes. Without it the
    loss curve is fitted against placeholder revenue and describes a plausible
    business rather than yours.
    """
    site = owned_site(site_id, account)
    raw = (await file.read()).decode("utf-8-sig", errors="replace")

    rows: list[tuple[str, float]] = []
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        raise HTTPException(400, "That file has no header row. Expected columns: date,revenue")

    lowered = {c.lower().strip(): c for c in reader.fieldnames}
    date_col = next((lowered[c] for c in ("date", "day", "dt") if c in lowered), None)
    rev_col = next(
        (lowered[c] for c in ("revenue", "amount", "sales", "gross") if c in lowered), None
    )
    if not date_col or not rev_col:
        raise HTTPException(
            400,
            f"Could not find date and revenue columns. Saw: {', '.join(reader.fieldnames)}. "
            "Expected something like: date,revenue",
        )

    bad = 0
    for row in reader:
        day = (row.get(date_col) or "").strip()[:10]
        try:
            amount = float(str(row.get(rev_col, "")).replace(",", "").replace("$", "").strip())
        except ValueError:
            bad += 1
            continue
        if len(day) == 10 and day[4] == "-" and amount >= 0:
            rows.append((day, amount))
        else:
            bad += 1

    if len(rows) < 180:
        raise HTTPException(
            400,
            f"Only {len(rows)} usable daily rows. At least 180 are needed to fit a loss curve, "
            "and two to three years is where it becomes reliable.",
        )

    store.put_revenue(site_id, rows)
    return {
        "accepted": len(rows),
        "rejected": bad,
        "summary": store.revenue_summary(site_id),
        "note": "Re-fetch /exposure to see the loss curve refitted against this.",
    }


@app.get("/v1/sites/{site_id}/exposure")
def exposure(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    try:
        return service.exposure(site_id)
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/v1/sites/{site_id}/outlook")
def outlook(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    payload = store.get_analysis(site_id, "outlook")
    if payload is None:
        raise HTTPException(409, "Outlook not computed for this site yet.")
    return payload


@app.get("/v1/sites/{site_id}/history")
def history(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    return store.get_analysis(site_id, "history") or {}


@app.get("/v1/sites/{site_id}/perils")
def site_perils(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    return store.get_analysis(site_id, "perils") or {}


@app.get("/v1/sites/{site_id}/tail")
def tail(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    return store.get_analysis(site_id, "tail") or {}


@app.get("/v1/sites/{site_id}/diagnostics")
def diagnostics(site_id: str, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    return store.get_analysis(site_id, "diagnostics") or {}


@app.get("/v1/sites/{site_id}/suggested-terms")
def suggested_terms(site_id: str, account: Account, year: int = 2027) -> dict:
    """Contract terms proposed from this site's own exposure.

    Saves the customer guessing an attachment point, which is the parameter they
    are most likely to get wrong and the one that decides whether the product is
    insurance or a financing arrangement.
    """
    require_ready(owned_site(site_id, account))
    try:
        return service.suggest_contract(site_id, year)
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/v1/sites/{site_id}/quote")
def quote(site_id: str, body: QuoteIn, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    try:
        return service.quote(
            site_id, body.peril_id, body.year, body.payout_per_day,
            body.limit, body.attachment_days, body.scenario_id,
        )
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/v1/sites/{site_id}/quotes")
def quote_history(site_id: str, account: Account) -> dict:
    owned_site(site_id, account)
    return {"quotes": store.list_quotes(site_id)}


@app.post("/v1/sites/{site_id}/hedge")
def hedge(site_id: str, body: HedgeIn, account: Account) -> dict:
    require_ready(owned_site(site_id, account))
    try:
        return service.hedge(
            site_id, body.peril_id, body.year, body.payout_per_day, body.limit,
            body.attachment_days, body.reserves, body.monthly_burn,
        )
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc


# ----------------------------------------------------------------------
# The analytics terminal
#
# `/` and every storefront page now come from `web.router`. The terminal stays a
# prebuilt single file because it is also published as a standalone artifact
# under a CSP that blocks external requests, so its fonts and canvas have to be
# inlined --- which is exactly what the storefront should *not* do on every page
# view, and why the two build paths differ.
# ----------------------------------------------------------------------


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> Response:
    page = WEB / "dist-dashboard.html"
    if page.exists():
        return FileResponse(page)
    return Response(
        "Run `python3 web/build.py dashboard` to compile the terminal.",
        media_type="text/plain",
    )
