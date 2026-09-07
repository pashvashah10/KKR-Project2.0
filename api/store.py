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
    station_name       TEXT,
    station_first_year INTEGER,
    station_last_year  INTEGER,
    station_elevation_m       REAL,
    station_elevation_delta_m REAL,
    station_elevation_warning INTEGER NOT NULL DEFAULT 0,
    is_example         INTEGER NOT NULL DEFAULT 0,
    contact_email      TEXT,
    -- Signed link back to this venue, minted when the fit starts. Stored rather
    -- than regenerated so the address in the notification and the one on screen
    -- are the same string.
    resume_token       TEXT,
    notified_at        REAL,
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

-- ----------------------------------------------------------------------
-- Commerce
--
-- Money is stored in integer cents everywhere below. Floats do not belong in
-- an order total: 0.1 + 0.2 is not 0.3, and a storefront that rounds
-- differently on the cart page and the invoice is a support ticket waiting to
-- happen. Conversion to display currency happens once, at the template.
-- ----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
    id               TEXT PRIMARY KEY,
    account_id       TEXT NOT NULL REFERENCES accounts(id),
    status           TEXT NOT NULL,          -- pending | paid | cancelled
    currency         TEXT NOT NULL DEFAULT 'USD',
    subtotal_cents   INTEGER NOT NULL,
    tax_cents        INTEGER NOT NULL DEFAULT 0,
    total_cents      INTEGER NOT NULL,
    invoice_number   TEXT NOT NULL UNIQUE,
    billing_name     TEXT,
    billing_email    TEXT,
    billing_company  TEXT,
    billing_address  TEXT,
    payment_provider TEXT NOT NULL DEFAULT 'mock',
    payment_reference TEXT,
    created_at       REAL NOT NULL,
    paid_at          REAL
);
CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id);

-- `product_name` and `unit_price_cents` are snapshotted from the catalogue at
-- purchase, never joined back to it. A price change next quarter must not
-- silently rewrite what a customer was charged last quarter.
CREATE TABLE IF NOT EXISTS order_items (
    id                TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL REFERENCES orders(id),
    product_slug      TEXT NOT NULL,
    product_name      TEXT NOT NULL,
    site_id           TEXT REFERENCES sites(id),
    config            TEXT NOT NULL DEFAULT '{}',
    unit_price_cents  INTEGER NOT NULL,
    quantity          INTEGER NOT NULL DEFAULT 1,
    line_total_cents  INTEGER NOT NULL,
    fulfilment_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id);

CREATE TABLE IF NOT EXISTS subscriptions (
    id                 TEXT PRIMARY KEY,
    account_id         TEXT NOT NULL REFERENCES accounts(id),
    site_id            TEXT REFERENCES sites(id),
    order_id           TEXT REFERENCES orders(id),
    product_slug       TEXT NOT NULL,
    status             TEXT NOT NULL,        -- active | cancelled
    period             TEXT NOT NULL,        -- month | year
    price_cents        INTEGER NOT NULL,
    started_at         REAL NOT NULL,
    current_period_end REAL NOT NULL,
    cancelled_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_subs_account ON subscriptions(account_id);

-- Forward bookings: what is already on the books for a future day.
--
-- Separate from `revenue`, which is the *historical* series the loss curve is
-- fitted against. Conflating them would be a category error --- one is evidence,
-- the other is exposure --- and the join key is the only thing they share.
CREATE TABLE IF NOT EXISTS bookings (
    site_id  TEXT NOT NULL REFERENCES sites(id),
    day      TEXT NOT NULL,
    revenue  REAL NOT NULL,
    covers   INTEGER,
    source   TEXT NOT NULL DEFAULT 'upload',
    PRIMARY KEY (site_id, day)
);

-- Anonymous exposure checks. Two jobs: rate limiting, and the lead list.
--
-- The IP is stored as a salted hash, never in the clear. It is needed to count
-- requests per hour and for nothing else, so keeping the address itself would be
-- collecting a piece of personal data with no use for it.
CREATE TABLE IF NOT EXISTS checks (
    id         TEXT PRIMARY KEY,
    ip_hash    TEXT NOT NULL,
    site_id    TEXT REFERENCES sites(id),
    job_id     TEXT,
    email      TEXT,
    lat        REAL,
    lon        REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checks_ip ON checks(ip_hash, created_at);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

#: Columns added after the first release. `CREATE TABLE IF NOT EXISTS` is a
#: no-op on an existing table, so new columns need an explicit additive step.
#: Additive only, and idempotent --- there is no down-migration and no rewrite.
MIGRATIONS = [
    ("sites", "station_name", "TEXT"),
    ("sites", "station_first_year", "INTEGER"),
    ("sites", "station_last_year", "INTEGER"),
    ("sites", "station_elevation_m", "REAL"),
    ("sites", "station_elevation_delta_m", "REAL"),
    ("sites", "station_elevation_warning", "INTEGER NOT NULL DEFAULT 0"),
    ("sites", "is_example", "INTEGER NOT NULL DEFAULT 0"),
    ("sites", "contact_email", "TEXT"),
    ("sites", "resume_token", "TEXT"),
    ("sites", "notified_at", "REAL"),
]


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
        for table, column, decl in MIGRATIONS:
            have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


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
        "reserves", "monthly_burn", "status", "created_at", "is_example",
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


def list_example_sites() -> list[dict]:
    """Venues seeded and fitted at build time, offered to visitors as examples.

    A fit takes 80-140 seconds. Without these, the first thing a visitor to the
    storefront would experience is a two-minute wait before seeing any number at
    all --- so the configure page offers real, already-fitted venues that price
    in milliseconds, and adding your own coordinates is the second option rather
    than the only one.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sites WHERE is_example = 1 AND status = 'ready' ORDER BY name"
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


# ----------------------------------------------------------------------
# orders
# ----------------------------------------------------------------------


def next_invoice_number(conn: sqlite3.Connection) -> str:
    """Sequential, gapless invoice numbers.

    Deliberately not `max(id) + 1` and not a random token. Invoice numbers are
    an accounting artefact: they must be unique, ordered, and not reused, and a
    concurrent checkout must not be able to mint the same one twice. The
    `UPDATE ... RETURNING` is atomic inside the caller's transaction, and the
    UNIQUE constraint on `orders.invoice_number` is the backstop if it ever
    were not.
    """
    conn.execute(
        "INSERT INTO counters (name, value) VALUES ('invoice', 1000) "
        "ON CONFLICT(name) DO NOTHING"
    )
    row = conn.execute(
        "UPDATE counters SET value = value + 1 WHERE name = 'invoice' RETURNING value"
    ).fetchone()
    return f"DW-{row['value']}"


def create_order(account_id: str, items: list[dict], billing: dict, tax_cents: int = 0) -> dict:
    """Write an order and its lines in one transaction.

    Each `item` carries its own `unit_price_cents` and `product_name` --- taken
    from the catalogue by the caller and frozen here. Nothing downstream reads
    a price back out of the catalogue.
    """
    order_id = new_id("ord")
    subtotal = sum(int(i["unit_price_cents"]) * int(i.get("quantity", 1)) for i in items)
    total = subtotal + int(tax_cents)
    now = time.time()

    with connect() as conn:
        invoice = next_invoice_number(conn)
        conn.execute(
            "INSERT INTO orders (id,account_id,status,currency,subtotal_cents,tax_cents,"
            "total_cents,invoice_number,billing_name,billing_email,billing_company,"
            "billing_address,payment_provider,payment_reference,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                order_id, account_id, "pending", "USD", subtotal, int(tax_cents), total,
                invoice, billing.get("name"), billing.get("email"), billing.get("company"),
                billing.get("address"), billing.get("provider", "mock"), None, now,
            ),
        )
        for item in items:
            qty = int(item.get("quantity", 1))
            unit = int(item["unit_price_cents"])
            conn.execute(
                "INSERT INTO order_items (id,order_id,product_slug,product_name,site_id,"
                "config,unit_price_cents,quantity,line_total_cents,fulfilment_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id("oi"), order_id, item["product_slug"], item["product_name"],
                    item.get("site_id"), json.dumps(item.get("config") or {}),
                    unit, qty, unit * qty, "pending",
                ),
            )

    return get_order(order_id)  # type: ignore[return-value]


def get_order(order_id: str, account_id: str | None = None) -> dict | None:
    q = "SELECT * FROM orders WHERE id = ?"
    args: tuple = (order_id,)
    if account_id:
        q += " AND account_id = ?"
        args += (account_id,)
    with connect() as conn:
        row = conn.execute(q, args).fetchone()
        if row is None:
            return None
        items = conn.execute(
            "SELECT * FROM order_items WHERE order_id = ? ORDER BY rowid", (order_id,)
        ).fetchall()
    order = dict(row)
    order["items"] = [{**dict(i), "config": json.loads(i["config"])} for i in items]
    return order


def list_orders(account_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE account_id = ? ORDER BY created_at DESC", (account_id,)
        ).fetchall()
        out = []
        for r in rows:
            items = conn.execute(
                "SELECT * FROM order_items WHERE order_id = ? ORDER BY rowid", (r["id"],)
            ).fetchall()
            o = dict(r)
            o["items"] = [{**dict(i), "config": json.loads(i["config"])} for i in items]
            out.append(o)
    return out


def mark_order_paid(order_id: str, reference: str, provider: str = "mock") -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE orders SET status='paid', paid_at=?, payment_reference=?, "
            "payment_provider=? WHERE id = ?",
            (time.time(), reference, provider, order_id),
        )


def update_item_fulfilment(item_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE order_items SET fulfilment_status = ? WHERE id = ?", (status, item_id)
        )


def get_order_item(item_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM order_items WHERE id = ?", (item_id,)).fetchone()
    return {**dict(row), "config": json.loads(row["config"])} if row else None


def attach_item_site(item_id: str, site_id: str) -> None:
    """Point a paid line at a venue.

    A report can be bought before its venue exists --- someone orders three and
    adds the sites afterwards. Without this the line sits at `awaiting-venue`
    for ever and the customer has paid for something the product cannot deliver.
    """
    with connect() as conn:
        conn.execute(
            "UPDATE order_items SET site_id = ? WHERE id = ?", (site_id, item_id)
        )


# ----------------------------------------------------------------------
# subscriptions
# ----------------------------------------------------------------------


def create_subscription(
    account_id: str,
    order_id: str,
    product_slug: str,
    price_cents: int,
    period: str = "month",
    site_id: str | None = None,
) -> dict:
    now = time.time()
    span = 365 * 86400.0 if period == "year" else 30 * 86400.0
    sub = {
        "id": new_id("sub"),
        "account_id": account_id,
        "site_id": site_id,
        "order_id": order_id,
        "product_slug": product_slug,
        "status": "active",
        "period": period,
        "price_cents": int(price_cents),
        "started_at": now,
        "current_period_end": now + span,
        "cancelled_at": None,
    }
    cols = list(sub)
    with connect() as conn:
        conn.execute(
            f"INSERT INTO subscriptions ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            tuple(sub[c] for c in cols),
        )
    return sub


def list_subscriptions(account_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM subscriptions WHERE account_id = ? ORDER BY started_at DESC",
            (account_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_subscription(sub_id: str, account_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE subscriptions SET status='cancelled', cancelled_at=? "
            "WHERE id = ? AND account_id = ? AND status = 'active'",
            (time.time(), sub_id, account_id),
        )
    return cur.rowcount > 0


# ----------------------------------------------------------------------
# bookings
# ----------------------------------------------------------------------


def put_bookings(site_id: str, rows: list[tuple[str, float, int | None]]) -> int:
    """Upsert forward bookings. `rows` is `(day, revenue, covers)`."""
    with connect() as conn:
        conn.executemany(
            "INSERT INTO bookings (site_id,day,revenue,covers) VALUES (?,?,?,?) "
            "ON CONFLICT(site_id,day) DO UPDATE SET "
            "revenue=excluded.revenue, covers=excluded.covers",
            [(site_id, d, r, c) for d, r, c in rows],
        )
    return len(rows)


def bookings_window(site_id: str, first_day: str, last_day: str) -> dict[str, float]:
    """`{day: revenue}` over an inclusive ISO date range.

    A dict rather than a list because the caller walks a contiguous calendar and
    needs to distinguish "nothing booked" from "no row for that day" --- both are
    zero exposure, but only one of them is worth telling the customer about.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT day, revenue FROM bookings "
            "WHERE site_id = ? AND day >= ? AND day <= ? ORDER BY day",
            (site_id, first_day, last_day),
        ).fetchall()
    return {r["day"]: float(r["revenue"]) for r in rows}


def bookings_summary(site_id: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, MIN(day) lo, MAX(day) hi, SUM(revenue) total "
            "FROM bookings WHERE site_id = ?",
            (site_id,),
        ).fetchone()
    return {"days": row["n"], "from": row["lo"], "to": row["hi"], "total": row["total"] or 0.0}


# ----------------------------------------------------------------------
# anonymous checks
# ----------------------------------------------------------------------


def record_check(ip_hash: str, lat: float, lon: float, site_id: str, job_id: str) -> str:
    check_id = new_id("chk")
    with connect() as conn:
        conn.execute(
            "INSERT INTO checks (id,ip_hash,site_id,job_id,lat,lon,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (check_id, ip_hash, site_id, job_id, lat, lon, time.time()),
        )
    return check_id


def checks_since(ip_hash: str, since: float) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n FROM checks WHERE ip_hash = ? AND created_at >= ?",
            (ip_hash, since),
        ).fetchone()
    return int(row["n"])


def get_check(check_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM checks WHERE id = ?", (check_id,)).fetchone()
    return dict(row) if row else None


def check_by_job(job_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM checks WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def claim_check(check_id: str, email: str) -> None:
    """Attach an email to a check --- the lead, captured when the result unlocks."""
    with connect() as conn:
        conn.execute(
            "UPDATE checks SET email = ? WHERE id = ?", (email.lower().strip(), check_id)
        )
