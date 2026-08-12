"""Tests for the venue report and for resolving an order that has no venue yet.

Both exist because of the same discovery: the storefront could sell an Exposure
Report and then had nowhere to deliver it. `/dashboard` is the prebuilt
eight-venue demo terminal with no picker for a customer's own sites, `/account`
linked to nothing, and a line bought before its venue existed sat at
`awaiting-venue` for ever.

A product that takes money for a report it cannot open is the most serious kind
of gap here, so these are acceptance tests rather than unit tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import catalog, charts, checkout


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from api import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path / "models")
    store.init()
    return store


@pytest.fixture()
def app(db):
    from concurrent.futures import Future

    from api import main, web

    started: list[tuple[str, str]] = []

    def fake_submit(site_id: str, job_id: str):
        started.append((site_id, job_id))
        f: Future = Future()
        f.set_result(None)
        return f

    web.configure_executor(fake_submit)
    main.app.state.started = started
    return main.app


@pytest.fixture()
def client(app):
    return TestClient(app)


def make_site(db, account_id, name="Orchard Hill", status="ready", example=0):
    row = db.create_site(
        account_id, name=name, lat=42.443, lon=-76.502, elevation_m=250.0,
        vertical="Outdoor attraction", season_start_month=4, season_end_month=10,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=example,
    )
    db.update_site(row["id"], status=status)
    return db.get_site(row["id"])


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------


def test_charts_return_empty_string_rather_than_crashing_on_thin_data():
    """A venue with a truncated record must not 500 the report."""
    assert charts.century_chart({}) == ""
    assert charts.century_chart({"observed": [{"year": 2000, "tmax": 10}]}) == ""
    assert charts.projection_chart({}) == ""
    assert charts.uncertainty_chart({"uncertainty": []}) == ""
    assert charts.loss_curve_chart({"curve": []}) == ""
    assert charts.sparkbar([]) == ""


def test_century_chart_draws_both_series():
    history = {
        "observed": [{"year": 1926 + i, "tmax": 20 + i * 0.02} for i in range(100)],
        "fitted": [{"year": 1926 + i, "tmax": 20.1 + i * 0.02} for i in range(100)],
    }
    svg = charts.century_chart(history)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<path") == 2
    assert "var(--ember)" in svg and "var(--muted)" in svg


def test_charts_use_only_tokens_never_literal_colours():
    """A literal hex would look right in one theme and wrong in the other."""
    import re

    history = {
        "observed": [{"year": 2000 + i, "tmax": 20 + i} for i in range(10)],
        "fitted": [{"year": 2000 + i, "tmax": 20 + i} for i in range(10)],
    }
    outlook = {
        "scenarios": {
            "ssp245": [{"year": 2026 + i, "mean": 20 + i * .1, "lo": 19, "hi": 21}
                       for i in range(20)],
            "ssp585": [{"year": 2026 + i, "mean": 20 + i * .2, "lo": 18, "hi": 23}
                       for i in range(20)],
        },
        "uncertainty": [{"year": 2026 + i, "internal": 1 - i / 40,
                         "parameter": .2, "scenario": i / 40} for i in range(20)],
    }
    for svg in (charts.century_chart(history), charts.projection_chart(outlook),
                charts.uncertainty_chart(outlook), charts.sparkbar([1, 2, 3])):
        assert not re.search(r"#[0-9a-fA-F]{3,6}", svg), "literal colour in an SVG"


def test_uncertainty_areas_stack_to_the_full_height():
    """Shares, not absolute values --- the figure claims to sum to one."""
    outlook = {"uncertainty": [
        {"year": 2026 + i, "internal": 3.0, "parameter": 1.0, "scenario": 0.0}
        for i in range(10)
    ]}
    svg = charts.uncertainty_chart(outlook)
    # Exactly three filled areas. The frame is drawn with <line>, so any extra
    # <path> would mean a stray band.
    assert svg.count("<path") == 3
    assert "var(--s1)" in svg and "var(--s4)" in svg and "var(--s2)" in svg


def test_sparkbar_scales_against_a_shared_maximum():
    """Rows in a table must be comparable to each other, not self-normalised."""
    tall = charts.sparkbar([1, 2, 3], vmax=3)
    short = charts.sparkbar([1, 1, 1], vmax=3)
    assert tall != short


# ----------------------------------------------------------------------
# The report page
# ----------------------------------------------------------------------


def test_an_example_venue_report_is_public(client, db):
    """The seeded venues exist to be looked at."""
    account = db.create_account("Examples", "ex@downside.local")
    site = make_site(db, account["id"], name="Demo Venue", example=1)
    res = client.get(f"/venues/{site['id']}")
    assert res.status_code == 200
    assert "Demo Venue" in res.text


def test_a_customer_venue_is_not_public(app, db):
    account = db.create_account("Acme", "a@acme.test")
    site = make_site(db, account["id"], name="Private Venue")

    stranger = TestClient(app)
    assert stranger.get(f"/venues/{site['id']}").status_code == 404


def test_a_resume_token_opens_the_report_from_a_cold_client(app, db):
    """The report is the thing the link is for."""
    from api import security

    account = db.create_account("Acme", "b@acme.test")
    site = make_site(db, account["id"], name="Tokened Venue")
    token = security.issue_resume_token(site["id"], account["id"], "exposure-report")

    fresh = TestClient(app)
    res = fresh.get(f"/venues/{site['id']}?t={token}")
    assert res.status_code == 200
    assert "Tokened Venue" in res.text


def test_a_missing_venue_is_a_404(client):
    assert client.get("/venues/site_nope").status_code == 404


def test_an_unfitted_venue_shows_progress_not_an_error(client, db):
    account = db.create_account("Acme", "c@acme.test")
    site = make_site(db, account["id"], name="Still Fitting", status="pending", example=1)
    db.create_job(site["id"], "fit")

    res = client.get(f"/venues/{site['id']}")
    assert res.status_code == 200
    assert "Fitting a century" in res.text
    assert "Still Fitting" in res.text


def test_a_failed_venue_says_so_and_offers_a_way_forward(client, db):
    account = db.create_account("Acme", "d@acme.test")
    site = make_site(db, account["id"], name="Broken Venue", status="failed", example=1)
    job = db.create_job(site["id"], "fit")
    db.update_job(job["id"], status="failed", error="SourceUnavailable: no data")

    res = client.get(f"/venues/{site['id']}")
    assert res.status_code == 200
    assert "Fit failed" in res.text
    assert "SourceUnavailable" in res.text
    assert "Try different coordinates" in res.text


# ----------------------------------------------------------------------
# Attaching a venue to a paid line
# ----------------------------------------------------------------------


BILLING = {"name": "Priya Raman", "email": "priya@orchard.test"}


def _order_without_a_venue(db, account):
    cart = checkout.Cart([
        checkout.CartLine(catalog.BY_SLUG["exposure-report"], 1, None, None, {})
    ])
    order = checkout.place_order(account, cart, BILLING)
    checkout.fulfil(order, lambda s, j: None)
    return db.get_order(order["id"])


def test_a_report_bought_without_a_venue_is_awaiting_not_lost(db):
    account = db.create_account("Acme", "e@acme.test")
    order = _order_without_a_venue(db, account)
    assert order["items"][0]["fulfilment_status"] == "awaiting-venue"
    assert order["items"][0]["site_id"] is None


def test_attaching_a_fitted_venue_delivers_the_line(client, db):
    """The fix for the dead end."""
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "Priya", "billing_email": "p@o.test"},
                follow_redirects=False)

    acct = db.account_by_email("p@o.test")
    order = db.list_orders(acct["id"])[0]
    item = order["items"][0]
    assert item["fulfilment_status"] == "awaiting-venue"

    site = make_site(db, acct["id"], name="Orchard Hill")
    res = client.post(
        f"/orders/{order['id']}/items/{item['id']}/venue",
        data={"site_id": site["id"]}, follow_redirects=False,
    )
    assert res.status_code == 303

    again = db.get_order(order["id"])["items"][0]
    assert again["site_id"] == site["id"]
    assert again["fulfilment_status"] == "delivered"


def test_attaching_an_unfitted_venue_starts_a_fit(client, db, app):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "Q", "billing_email": "q@o.test"},
                follow_redirects=False)
    acct = db.account_by_email("q@o.test")
    order = db.list_orders(acct["id"])[0]
    item = order["items"][0]

    site = make_site(db, acct["id"], name="Unfitted", status="pending")
    client.post(f"/orders/{order['id']}/items/{item['id']}/venue",
                data={"site_id": site["id"]}, follow_redirects=False)

    again = db.get_order(order["id"])["items"][0]
    assert again["fulfilment_status"] == "fitting"
    assert app.state.started, "a fit should have been queued"


def test_you_cannot_attach_someone_elses_venue(client, db, app):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "R", "billing_email": "r@o.test"},
                follow_redirects=False)
    acct = db.account_by_email("r@o.test")
    order = db.list_orders(acct["id"])[0]
    item = order["items"][0]

    mallory = db.create_account("Mallory", "m@evil.test")
    theirs = make_site(db, mallory["id"], name="Not Yours")

    res = client.post(f"/orders/{order['id']}/items/{item['id']}/venue",
                      data={"site_id": theirs["id"]}, follow_redirects=False)
    assert res.status_code == 404
    assert db.get_order(order["id"])["items"][0]["site_id"] is None


def test_the_order_page_offers_the_attach_control(client, db):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "S", "billing_email": "s@o.test"},
                follow_redirects=False)
    acct = db.account_by_email("s@o.test")
    order = db.list_orders(acct["id"])[0]
    make_site(db, acct["id"], name="Attachable Venue")

    body = client.get(f"/orders/{order['id']}").text
    assert "Attach venue" in body
    assert "Attachable Venue" in body


def test_a_delivered_line_links_to_its_report(client, db):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "T", "billing_email": "t@o.test"},
                follow_redirects=False)
    acct = db.account_by_email("t@o.test")
    order = db.list_orders(acct["id"])[0]
    item = order["items"][0]
    site = make_site(db, acct["id"], name="Delivered Venue")

    client.post(f"/orders/{order['id']}/items/{item['id']}/venue",
                data={"site_id": site["id"]}, follow_redirects=False)

    body = client.get(f"/orders/{order['id']}").text
    assert f"/venues/{site['id']}" in body
    assert "Open the report" in body


def test_the_account_page_links_to_each_fitted_report(client, db):
    """The other route a customer would reasonably take to find their report."""
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    client.post("/checkout", data={"billing_name": "U", "billing_email": "u@o.test"},
                follow_redirects=False)
    acct = db.account_by_email("u@o.test")
    site = make_site(db, acct["id"], name="Findable Venue")

    body = client.get("/account").text
    assert f"/venues/{site['id']}" in body
    assert "Open the report" in body
