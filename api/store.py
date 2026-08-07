"""Persistence.

SQLite rather than Postgres, deliberately, for this stage: it is a single file,
needs no service to run, and the schema below ports to Postgres unchanged when
there is a reason to move. The access pattern here is a handful of writes per
customer and cached reads, which SQLite handles for far longer than most people
expect.

The important design point is not the database. It is that **a fitted model is
expensive and must be cached**. Fitting one site takes 80-140 seconds: a century
of daily observations fetched, a mean and variance model estimated, a 200-replicate
block bootstrap, tail fits, and a Monte Carlo grid. Nobody waits that long on a
web request, and nobody should pay to recompute it on every page view. So a fit
happens once, asynchronously, and everything downstream reads the cached artefact.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "downside.db"
MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    api_key      TEXT NOT NULL UNIQUE,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sites (
    id                 TEXT PRIMARY KEY,
    account_id         TEXT NOT NULL REFERENCES accounts(id),
    name               TEXT NOT NULL,
    lat                REAL NOT NULL,
    lon                REAL NOT NULL,
    elevation_m        REAL,
    vertical           TEXT NOT NULL,
    season_start_month INTEGER NOT NULL,
    season_end_month   INTEGER NOT NULL,
    variable_cost_ratio REAL NOT NULL DEFAULT 0.30,
    reserves           REAL,
    monthly_burn       REAL,
    status             TEXT NOT NULL DEFAULT 'pending',
    provenance         TEXT,
    station_id         TEXT,
    station_km         REAL,
    quality            TEXT,
    created_at         REAL NOT NULL,
    fitted_at          REAL
);
CREATE INDEX IF NOT EXISTS idx_sites_account ON sites(account_id);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    site_id     TEXT NOT NULL REFERENCES sites(id),
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    step        TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_site ON jobs(site_id);

-- Cached analysis artefacts, one row per (site, kind).
CREATE TABLE IF NOT EXISTS analyses (
    site_id    TEXT NOT NULL REFERENCES sites(id),
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (site_id, kind)
);

CREATE TABLE IF NOT EXISTS revenue (
    site_id  TEXT NOT NULL REFERENCES sites(id),
    day      TEXT NOT NULL,
    amount   REAL NOT NULL,
    PRIMARY KEY (site_id, day)
);

CREATE TABLE IF NOT EXISTS quotes (
    id          TEXT PRIMARY KEY,
    site_id     TEXT NOT NULL REFERENCES sites(id),
    peril_id    TEXT NOT NULL,
    year        INTEGER NOT NULL,
    request     TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quotes_site ON quotes(site_id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL lets the background fitting worker write while requests read.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


# ----------------------------------------------------------------------
# accounts
# ----------------------------------------------------------------------


def create_account(name: str, email: str) -> dict:
    account = {
        "id": new_id("acct"),
        "name": name,
        "email": email.lower().strip(),
        "api_key": "dsk_" + secrets.token_urlsafe(24),
        "created_at": time.time(),
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO accounts (id,name,email,api_key,created_at) VALUES (?,?,?,?,?)",
            tuple(account[k] for k in ("id", "name", "email", "api_key", "created_at")),
        )
    return account


def account_by_key(api_key: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE api_key = ?", (api_key,)).fetchone()
    return dict(row) if row else None


def account_by_email(email: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    return dict(row) if row else None


# ----------------------------------------------------------------------
# sites
# ----------------------------------------------------------------------


def create_site(account_id: str, **fields: Any) -> dict:
    site = {
        "id": new_id("site"),
        "account_id": account_id,
        "status": "pending",
        "created_at": time.time(),
        **fields,
    }
    cols = [
        "id", "account_id", "name", "lat", "lon", "elevation_m", "vertical",
        "season_start_month", "season_end_month", "variable_cost_ratio",
        "reserves", "monthly_burn", "status", "created_at",
    ]
    with connect() as conn:
        conn.execute(
            f"INSERT INTO sites ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(site.get(c) for c in cols),
        )
    return site


def get_site(site_id: str, account_id: str | None = None) -> dict | None:
    q = "SELECT * FROM sites WHERE id = ?"
    args: tuple = (site_id,)
    if account_id:
        q += " AND account_id = ?"
        args += (account_id,)
    with connect() as conn:
        row = conn.execute(q, args).fetchone()
    return dict(row) if row else None


def list_sites(account_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sites WHERE account_id = ? ORDER BY created_at DESC", (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_site(site_id: str, **fields: Any) -> None:
    if not fields:
        return
    sets = ",".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE sites SET {sets} WHERE id = ?", (*fields.values(), site_id))


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------


def create_job(site_id: str, kind: str) -> dict:
    job = {
        "id": new_id("job"),
        "site_id": site_id,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "step": "queued",
        "created_at": time.time(),
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id,site_id,kind,status,progress,step,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (job["id"], site_id, kind, "queued", 0.0, "queued", job["created_at"]),
        )
    return job


def update_job(job_id: str, **fields: Any) -> None:
    sets = ",".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))


def get_job(job_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def latest_job(site_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE site_id = ? ORDER BY created_at DESC LIMIT 1", (site_id,)
        ).fetchone()
    return dict(row) if row else None


# ----------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------


def put_analysis(site_id: str, kind: str, payload: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO analyses (site_id,kind,payload,created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(site_id,kind) DO UPDATE SET payload=excluded.payload, "
            "created_at=excluded.created_at",
            (site_id, kind, json.dumps(payload), time.time()),
        )


def get_analysis(site_id: str, kind: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT payload FROM analyses WHERE site_id = ? AND kind = ?", (site_id, kind)
        ).fetchone()
    return json.loads(row["payload"]) if row else None


# ----------------------------------------------------------------------
# revenue
# ----------------------------------------------------------------------


def put_revenue(site_id: str, rows: list[tuple[str, float]]) -> int:
    with connect() as conn:
        conn.executemany(
            "INSERT INTO revenue (site_id,day,amount) VALUES (?,?,?) "
            "ON CONFLICT(site_id,day) DO UPDATE SET amount=excluded.amount",
            [(site_id, d, a) for d, a in rows],
        )
    return len(rows)


def get_revenue(site_id: str) -> list[tuple[str, float]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT day, amount FROM revenue WHERE site_id = ? ORDER BY day", (site_id,)
        ).fetchall()
    return [(r["day"], r["amount"]) for r in rows]


def revenue_summary(site_id: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, MIN(day) lo, MAX(day) hi, AVG(amount) avg "
            "FROM revenue WHERE site_id = ?",
            (site_id,),
        ).fetchone()
    return {"days": row["n"], "from": row["lo"], "to": row["hi"], "mean_daily": row["avg"]}


# ----------------------------------------------------------------------
# quotes
# ----------------------------------------------------------------------


def save_quote(site_id: str, peril_id: str, year: int, request: dict, response: dict) -> str:
    qid = new_id("qt")
    with connect() as conn:
        conn.execute(
            "INSERT INTO quotes (id,site_id,peril_id,year,request,response,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (qid, site_id, peril_id, year, json.dumps(request), json.dumps(response), time.time()),
        )
    return qid


def list_quotes(site_id: str, limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM quotes WHERE site_id = ? ORDER BY created_at DESC LIMIT ?",
            (site_id, limit),
        ).fetchall()
    return [
        {**dict(r), "request": json.loads(r["request"]), "response": json.loads(r["response"])}
        for r in rows
    ]
