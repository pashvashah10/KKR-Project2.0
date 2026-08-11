"""Tests for the resume link, and a smoke pass over the storefront routes.

The form on `/configure` promises "leave an address and you can close this tab".
Every test below exists because some part of that sentence was previously false:
nothing sent, the link pointed at the wrong service, the link never reached the
screen, and it did not authenticate off the session that minted it.

The one that matters most is `test_link_works_from_a_browser_with_no_cookies`.
Everything else is a supporting detail; that one *is* the feature.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api import security, web


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from api import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path / "models")
    store.init()
    return store


@pytest.fixture()
def account(db):
    return db.create_account("Acme Outdoor", "ops@acme.test")


@pytest.fixture()
def site(db, account):
    # Deliberately not "Sonoma Ridge" --- that is the form's own placeholder, so
    # a name-based presence check would pass on an empty page.
    row = db.create_site(
        account["id"], name="Thornbury Marsh", lat=38.4405, lon=-122.7141, elevation_m=120.0,
        vertical="Winery & events", season_start_month=4, season_end_month=10,
        variable_cost_ratio=0.30, reserves=None, monthly_burn=None, is_example=0,
    )
    db.update_site(row["id"], status="ready")
    return db.get_site(row["id"])


@pytest.fixture()
def app(db, monkeypatch):
    """The real app, with fitting stubbed out.

    `fit_site` is 80-140 seconds of numerics and is not what any of this is
    testing, so the executor records the call and returns a resolved future.
    """
    from concurrent.futures import Future

    from api import main

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


class Recorder:
    """A sender that remembers instead of delivering."""

    def __init__(self, configured: bool = False, explode: bool = False):
        self.configured = configured
        self.explode = explode
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, link: str) -> None:
        if self.explode:
            raise RuntimeError("relay refused the connection")
        self.sent.append((to, subject, link))


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------


def test_token_round_trips():
    token = security.issue_resume_token("site_a", "acct_1", "exposure-report")
    payload = security.read_resume_token(token)
    assert payload == {"site_id": "site_a", "account_id": "acct_1", "slug": "exposure-report"}


def test_token_is_bound_to_its_site():
    """A link to one venue must not unlock another."""
    token = security.issue_resume_token("site_a", "acct_1", "exposure-report")
    assert security.read_resume_token(token, site_id="site_a") is not None
    assert security.read_resume_token(token, site_id="site_b") is None


def test_a_tampered_token_is_rejected():
    token = security.issue_resume_token("site_a", "acct_1", "exposure-report")
    # Flip a character in the payload segment, leaving the shape intact.
    body, _, sig = token.rpartition(".")
    mangled = ("X" + body[1:]) + "." + sig
    assert security.read_resume_token(mangled) is None


def test_an_unsigned_lookalike_is_rejected():
    import base64
    import json

    raw = base64.urlsafe_b64encode(
        json.dumps({"site_id": "site_a", "account_id": "acct_1"}).encode()
    ).decode()
    assert security.read_resume_token(raw) is None


def test_an_expired_token_is_rejected():
    token = security.issue_resume_token("site_a", "acct_1", "exposure-report")
    assert security.read_resume_token(token, max_age=-1) is None


def test_empty_and_missing_tokens_are_rejected():
    assert security.read_resume_token(None) is None
    assert security.read_resume_token("") is None
    assert security.read_resume_token("not-a-token") is None


def test_a_token_signed_with_another_key_is_rejected(monkeypatch):
    """The check that makes DOWNSIDE_SECRET meaningful."""
    from itsdangerous import URLSafeTimedSerializer

    foreign = URLSafeTimedSerializer("some-other-secret", salt="downside-resume-link")
    token = foreign.dumps({"site_id": "site_a", "account_id": "acct_1", "slug": "x"})
    assert security.read_resume_token(token) is None


def test_ephemeral_secret_shortens_the_advertised_lifetime():
    """Do not promise a week we cannot keep on a generated key."""
    if security.SECRET_IS_EPHEMERAL:
        assert security.resume_max_age() == security.EPHEMERAL_MAX_AGE
    else:
        assert security.resume_max_age() == security.RESUME_MAX_AGE


# ----------------------------------------------------------------------
# The link itself
# ----------------------------------------------------------------------


def test_link_uses_the_slug_it_was_configured_with(site):
    """The direct regression test.

    `magic_link` used to hardcode `parametric-cover`, so a visitor who asked to
    be emailed about an Exposure Report was sent to a different product.
    """
    for slug in ("exposure-report", "season-monitor", "portfolio-analytics"):
        link = web.magic_link(site, slug, "tok")
        assert link.startswith(f"/configure/{slug}?site={site['id']}")
        assert "parametric-cover" not in link


def test_link_carries_the_token(site):
    assert web.magic_link(site, "exposure-report", "tok").endswith("&t=tok")


def test_link_without_a_token_is_still_a_valid_path(site):
    link = web.magic_link({**site, "resume_token": None}, "exposure-report")
    assert link == f"/configure/exposure-report?site={site['id']}"
    assert "t=" not in link


# ----------------------------------------------------------------------
# Notification
# ----------------------------------------------------------------------


def _selected(html: str) -> bool:
    """Did the page render a venue as chosen?

    Keys off the sidebar the configure page only emits for a selected venue.
    Matching on the venue's *name* would be unreliable --- the empty form's own
    placeholder text contains a venue-shaped string.
    """
    return "Settlement basis" in html or "Come back to this venue" in html


def _prepare(db, site, slug="exposure-report", email="ops@acme.test"):
    token = security.issue_resume_token(site["id"], site["account_id"], slug)
    db.update_site(site["id"], resume_token=token, contact_email=email)
    return token


def test_notify_sends_once_and_only_once(db, site, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site)

    assert web.notify_if_requested(site["id"]) is True
    assert web.notify_if_requested(site["id"]) is False, "a retry must not send twice"
    assert len(rec.sent) == 1

    to, subject, link = rec.sent[0]
    assert to == "ops@acme.test"
    assert "Thornbury Marsh" in subject
    assert link.startswith("/configure/exposure-report?site=")
    assert db.get_site(site["id"])["notified_at"] is not None


def test_notify_uses_the_slug_from_the_token(db, site, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site, slug="season-monitor")

    web.notify_if_requested(site["id"])
    assert "/configure/season-monitor?" in rec.sent[0][2]


def test_no_address_means_no_send(db, site, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site, email=None)

    assert web.notify_if_requested(site["id"]) is False
    assert rec.sent == []


def test_a_failed_fit_sends_nothing(db, site, monkeypatch):
    """There is nothing worth linking to, and saying otherwise would be a lie."""
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site)
    db.update_site(site["id"], status="failed")

    assert web.notify_if_requested(site["id"]) is False
    assert rec.sent == []


def test_an_unfinished_fit_sends_nothing(db, site, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site)
    db.update_site(site["id"], status="pending")

    assert web.notify_if_requested(site["id"]) is False


def test_a_missing_site_is_not_an_error(db, monkeypatch):
    monkeypatch.setattr(web, "SENDER", Recorder())
    assert web.notify_if_requested("site_does_not_exist") is False


def test_a_sender_that_raises_does_not_propagate(db, site, monkeypatch):
    """A mail fault must never turn a good fit into a failed one."""
    rec = Recorder(explode=True)
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site)

    assert web.notify_if_requested(site["id"]) is False
    # Left unmarked, so a later attempt can still deliver.
    assert db.get_site(site["id"])["notified_at"] is None


def test_smtp_sender_refuses_rather_than_silently_logging(site):
    sender = web.SmtpSender()
    assert sender.configured is False
    with pytest.raises(NotImplementedError):
        sender.send("a@b.test", "subject", "/link")


def test_console_sender_reports_itself_unconfigured():
    """What stops the UI claiming an email went out."""
    assert web.ConsoleSender().configured is False
    assert web.SENDER.configured is False


# ----------------------------------------------------------------------
# Routes --- the acceptance tests
# ----------------------------------------------------------------------


def test_link_works_from_a_browser_with_no_cookies(app, db, site):
    """"You can close this tab" --- the whole point of the feature.

    A cookie-free client is a different device. Without the token the venue is
    invisible; with it, the page renders the venue as selected.
    """
    token = _prepare(db, site)

    stranger = TestClient(app)
    without = stranger.get(f"/configure/exposure-report?site={site['id']}")
    assert without.status_code == 200
    assert not _selected(without.text), "no token, no venue"

    fresh = TestClient(app)
    with_token = fresh.get(f"/configure/exposure-report?site={site['id']}&t={token}")
    assert with_token.status_code == 200
    assert _selected(with_token.text)
    assert "Thornbury Marsh" in with_token.text


def test_a_forged_token_does_not_open_the_venue(app, db, site):
    _prepare(db, site)
    stranger = TestClient(app)
    res = stranger.get(f"/configure/exposure-report?site={site['id']}&t=forged.nonsense")
    assert res.status_code == 200
    assert not _selected(res.text)


def test_a_token_for_another_venue_does_not_open_this_one(app, db, account, site):
    other = db.create_site(
        account["id"], name="Somewhere Else", lat=42.0, lon=-71.0, elevation_m=10.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    token = security.issue_resume_token(other["id"], account["id"], "exposure-report")

    stranger = TestClient(app)
    res = stranger.get(f"/configure/exposure-report?site={site['id']}&t={token}")
    assert not _selected(res.text)


def test_job_status_accepts_the_token_from_a_cold_client(app, db, site):
    token = _prepare(db, site)
    job = db.create_job(site["id"], "fit")
    db.update_job(job["id"], status="succeeded", progress=1.0, step="ready")

    stranger = TestClient(app)
    assert stranger.get(f"/api/v1/jobs/{job['id']}").status_code == 404

    fresh = TestClient(app)
    ok = fresh.get(f"/api/v1/jobs/{job['id']}?t={token}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "succeeded"
    assert body["magic_link"].endswith(f"&t={token}")
    assert body["mail_configured"] is False


def test_starting_a_fit_returns_a_usable_resume_link(app, db):
    client = TestClient(app)
    res = client.post(
        "/configure/exposure-report/venue",
        data={
            "name": "Cold Creek Farm", "lat": "44.0", "lon": "-72.5",
            "elevation_m": "300", "vertical": "Outdoor attraction",
            "season_start_month": "5", "season_end_month": "9",
            "notify_email": "grower@cold.test",
        },
    )
    assert res.status_code == 202
    body = res.json()

    assert body["resume_link"].startswith("http")
    assert "/configure/exposure-report?" in body["resume_link"]
    assert body["mail_configured"] is False
    assert body["notify"] is True

    # The token in the link must actually validate against the new site.
    token = body["resume_link"].split("&t=")[1]
    assert security.read_resume_token(token, site_id=body["site_id"]) is not None

    stored = db.get_site(body["site_id"])
    assert stored["contact_email"] == "grower@cold.test"
    assert stored["resume_token"] == token


def test_a_venue_started_without_an_email_still_gets_a_link(app, db):
    """Nothing to send to is not a reason to have nothing to come back to."""
    client = TestClient(app)
    res = client.post(
        "/configure/season-monitor/venue",
        data={
            "name": "No Email Venue", "lat": "40.0", "lon": "-75.0",
            "vertical": "Outdoor attraction",
            "season_start_month": "1", "season_end_month": "12",
        },
    )
    body = res.json()
    assert body["notify"] is False
    assert "/configure/season-monitor?" in body["resume_link"]
    assert db.get_site(body["site_id"])["contact_email"] is None


def test_the_completion_callback_fires_the_notification(app, db, site, monkeypatch):
    """The wiring that makes any of this happen without a poller."""
    rec = Recorder()
    monkeypatch.setattr(web, "SENDER", rec)
    _prepare(db, site)

    job = db.create_job(site["id"], "fit")
    web.submit_fit(site["id"], job["id"])   # fake executor resolves immediately

    assert len(rec.sent) == 1
    assert rec.sent[0][0] == "ops@acme.test"


def test_the_bookmark_panel_appears_for_the_owner(app, db, site):
    token = _prepare(db, site)
    client = TestClient(app)
    res = client.get(f"/configure/exposure-report?site={site['id']}&t={token}")

    assert "Come back to this venue" in res.text
    assert "not configured" in res.text, "must not imply an email was sent"
    assert token in res.text


def test_no_page_claims_an_email_was_sent_when_none_was(app, db, site):
    token = _prepare(db, site)
    client = TestClient(app)
    body = client.get(f"/configure/exposure-report?site={site['id']}&t={token}").text
    assert "We sent this link" not in body


# ----------------------------------------------------------------------
# Storefront smoke tests
#
# `api/web.py` had no route coverage at all. These are cheap and would have
# caught the Starlette `TemplateResponse` signature change that broke every
# page during the last build.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/", "/services", "/cart", "/account",
        "/services/exposure-report", "/services/season-monitor",
        "/services/parametric-cover", "/services/portfolio-analytics",
        "/configure/exposure-report", "/configure/parametric-cover",
    ],
)
def test_page_renders(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert "<html" in res.text.lower()


def test_unknown_service_is_a_404(client):
    assert client.get("/services/no-such-thing").status_code == 404
    assert client.get("/configure/no-such-thing").status_code == 404


def test_empty_checkout_redirects_to_the_cart(client):
    res = client.get("/checkout", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/cart"


def test_cart_survives_add_update_and_remove(client):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "2"},
                follow_redirects=False)
    assert "$4,800" in client.get("/cart").text

    client.post("/cart/update", data={"index": "0", "quantity": "1"}, follow_redirects=False)
    assert "$2,400" in client.get("/cart").text

    client.post("/cart/remove", data={"index": "0"}, follow_redirects=False)
    assert "Nothing in the cart yet" in client.get("/cart").text


def test_checkout_page_has_no_card_fields(client):
    """Inspect the *inputs*, not the prose.

    The page explains at length that it does not collect a card number, so a
    naive substring search over the body finds the words and fails on copy that
    is doing exactly the right thing.
    """
    import re

    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    body = client.get("/checkout").text

    fields = re.findall(r"<(?:input|select)\b[^>]*>", body, flags=re.I)
    assert fields, "the checkout form should have some fields"
    for field in fields:
        low = field.lower()
        for banned in ("card", "cvv", "cvc", "cc-number", "cc-exp", "credit", "expiry"):
            assert banned not in low, f"card-like field on checkout: {field}"


def test_a_stranger_cannot_read_someone_elses_order(app, db, account):
    from api import catalog, checkout as co

    cart = co.Cart([co.CartLine(catalog.BY_SLUG["exposure-report"], 1, None, None, {})])
    order = co.place_order(account, cart, {"name": "D", "email": "d@acme.test"})

    stranger = TestClient(app)
    assert stranger.get(f"/orders/{order['id']}").status_code == 404


def test_the_disclaimer_is_on_every_price_surface(client):
    client.post("/cart/add", data={"slug": "exposure-report", "quantity": "1"},
                follow_redirects=False)
    needle = "indicative estimates for risk management"
    for path in ("/cart", "/checkout", "/configure/parametric-cover", "/account"):
        assert needle in client.get(path).text, f"{path} is missing the disclaimer"
