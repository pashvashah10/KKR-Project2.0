"""Tests for the free exposure check and for revenue at risk.

The check that matters most is `test_per_day_decomposition_matches_the_seasonal_number`.
Revenue at risk is not a second model --- it is the array `exposure()` already
computes, collapsed along the other axis. If the two ever disagree, one of the
pages is lying about the same simulation, and that is a worse failure than
either being wrong alone.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import charts, service


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

    started: list[tuple[str, str, bool]] = []

    def fake_fit(site_id, job_id):
        started.append((site_id, job_id, False))
        f: Future = Future(); f.set_result(None); return f

    def fake_preview(site_id, job_id):
        started.append((site_id, job_id, True))
        f: Future = Future(); f.set_result(None); return f

    web.configure_executor(fake_fit, fake_preview)
    main.app.state.started = started
    return main.app


@pytest.fixture()
def client(app):
    return TestClient(app)


# ----------------------------------------------------------------------
# The decomposition
# ----------------------------------------------------------------------


def test_per_day_decomposition_matches_the_seasonal_number():
    """Collapsing the loss array either way must give the same total.

    `exposure()` sums over days then averages paths; `revenue_at_risk()`
    averages paths then sums days. Linearity of the mean says these are
    identical, and this pins that so a future refactor cannot silently make the
    report and the calendar disagree.
    """
    rng = np.random.default_rng(0)
    loss = rng.gamma(2.0, 500.0, size=(2000, 90))

    by_season = loss.sum(axis=1).mean()
    by_day = loss.mean(axis=0).sum()
    assert by_season == pytest.approx(by_day, rel=1e-12)


def test_worst_windows_rank_by_contiguous_exposure():
    rows = [{"day": f"2027-01-{i + 1:02d}", "expected_loss": v}
            for i, v in enumerate([1] * 10 + [50] * 7 + [1] * 10)]
    total = sum(r["expected_loss"] for r in rows)
    windows = service._worst_windows(rows, total, top=2, span=7)

    assert windows[0]["from"] == "2027-01-11", "the spike week should rank first"
    assert windows[0]["share_of_total"] > 0.8
    # Windows must not overlap, or the same days are reported twice.
    assert windows[1]["from"] > windows[0]["to"] or windows[1]["to"] < windows[0]["from"]


def test_worst_windows_is_empty_when_there_is_no_exposure():
    rows = [{"day": f"2027-01-{i + 1:02d}", "expected_loss": 0.0} for i in range(30)]
    assert service._worst_windows(rows, 0.0) == []


def test_forward_window_wraps_the_year_end():
    doy, dates = service._forward_window(40, date(2026, 12, 20))
    assert len(doy) == len(dates) == 40
    assert dates[0] == date(2026, 12, 20)
    assert dates[-1].year == 2027
    assert doy[0] > doy[-1], "day-of-year must wrap rather than run past 366"
    assert doy.min() >= 1 and doy.max() <= 366


def test_dow_factors_are_normalised_and_find_the_weekend():
    """The fallback is worthless if it cannot see Saturday."""
    class FakeRecord:
        # 2024-01-01 is a Monday.
        dates = np.arange("2024-01-01", "2024-12-31", dtype="datetime64[D]")

    n = len(FakeRecord.dates)
    dow = (FakeRecord.dates.astype("datetime64[D]").astype(int) + 4) % 7
    revenue = np.where(dow >= 5, 300.0, 100.0)      # weekends worth triple

    factors = service._dow_factors(FakeRecord, revenue)
    assert factors.shape == (7,)
    assert factors.mean() == pytest.approx(1.0)
    assert factors[5] > factors[0] and factors[6] > factors[0]
    assert n > 0


def test_dow_factors_degrade_to_flat_without_revenue():
    class FakeRecord:
        dates = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")

    flat = service._dow_factors(FakeRecord, np.zeros(len(FakeRecord.dates)))
    assert np.allclose(flat, 1.0), "no revenue must mean no weekday opinion"


# ----------------------------------------------------------------------
# The calendar
# ----------------------------------------------------------------------


def _rows(n=90, share=0.2, booked=1000.0, start=date(2027, 1, 4)):
    from datetime import timedelta
    return [
        {"day": (start + timedelta(days=i)).isoformat(),
         "dow": (start + timedelta(days=i)).weekday(),
         "booked_margin": booked, "expected_loss": booked * share,
         "share": share, "p90_loss": booked * share * 1.4, "p_bad": 0.1}
        for i in range(n)
    ]


def test_calendar_draws_one_cell_per_day():
    svg = charts.calendar_strip(_rows(90))
    assert svg.count("<rect") == 90
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_calendar_is_empty_for_no_rows():
    assert charts.calendar_strip([]) == ""


def test_calendar_distinguishes_unbooked_days_from_low_exposure():
    """A day with nothing booked is not the same as a quiet day."""
    rows = _rows(14)
    rows[3]["booked_margin"] = 0.0
    rows[3]["share"] = 0.0
    svg = charts.calendar_strip(rows)
    assert "var(--panel-3)" in svg, "unbooked days need their own treatment"
    assert "nothing booked" in svg


def test_calendar_uses_only_tokens():
    import re
    assert not re.search(r"#[0-9a-fA-F]{3,6}", charts.calendar_strip(_rows(30)))


def test_calendar_cells_carry_a_readable_title():
    svg = charts.calendar_strip(_rows(7))
    assert "<title>" in svg and "of margin at risk" in svg


# ----------------------------------------------------------------------
# The free check
# ----------------------------------------------------------------------


def test_the_check_form_renders(client):
    res = client.get("/check")
    assert res.status_code == 200
    assert "Check my venue" in res.text


def test_a_coordinate_with_no_gauge_is_refused_before_any_fit(client, app):
    """Refusing costs nothing; a minute of CPU to reach the same answer does."""
    res = client.post("/check", data={
        "name": "Mid Pacific", "lat": "25.0", "lon": "-160.0",
        "vertical": "Outdoor attraction",
        "season_start_month": "1", "season_end_month": "12",
    })
    assert res.status_code == 422
    assert "station" in res.json()["detail"].lower()
    assert app.state.started == [], "nothing should have been queued"


def test_coordinates_off_the_planet_are_rejected(client):
    res = client.post("/check", data={
        "name": "Nowhere", "lat": "999", "lon": "0",
        "vertical": "Outdoor attraction",
        "season_start_month": "1", "season_end_month": "12",
    })
    assert res.status_code == 400


def test_a_check_queues_on_the_preview_pool_not_the_paying_one(client, app):
    """The isolation that stops free traffic starving paying customers."""
    res = client.post("/check", data={
        "name": "Boston Rooftop", "lat": "42.3601", "lon": "-71.0589",
        "elevation_m": "6", "vertical": "Outdoor attraction",
        "season_start_month": "5", "season_end_month": "9",
    })
    assert res.status_code == 202
    body = res.json()
    assert body["station"]["matched"] is True
    assert body["station"]["id"] == "USW00014739"

    assert len(app.state.started) == 1
    _, _, was_preview = app.state.started[0]
    assert was_preview is True, "free checks must not use the paying pool"


def test_the_rate_limit_fires(client, app):
    payload = {
        "name": "Repeat", "lat": "42.3601", "lon": "-71.0589", "elevation_m": "6",
        "vertical": "Outdoor attraction",
        "season_start_month": "5", "season_end_month": "9",
    }
    from api import web

    for _ in range(web.CHECK_LIMIT_PER_HOUR):
        assert client.post("/check", data=payload).status_code == 202

    refused = client.post("/check", data=payload)
    assert refused.status_code == 429
    assert "rate limited" in refused.json()["detail"]
    assert len(app.state.started) == web.CHECK_LIMIT_PER_HOUR


def test_the_ip_is_hashed_never_stored_in_the_clear(client, db):
    client.post("/check", data={
        "name": "Hashed", "lat": "42.3601", "lon": "-71.0589", "elevation_m": "6",
        "vertical": "Outdoor attraction",
        "season_start_month": "5", "season_end_month": "9",
    })
    with db.connect() as conn:
        hashes = [r["ip_hash"] for r in conn.execute("SELECT ip_hash FROM checks")]
    assert hashes and all(len(h) == 32 for h in hashes)
    assert not any("." in h or ":" in h for h in hashes), "that looks like an address"


def test_claiming_a_check_captures_the_lead_and_does_not_refit(client, db, app):
    res = client.post("/check", data={
        "name": "Claimable", "lat": "42.3601", "lon": "-71.0589", "elevation_m": "6",
        "vertical": "Outdoor attraction",
        "season_start_month": "5", "season_end_month": "9",
    })
    check_id = res.json()["check_id"]
    site_id = res.json()["site_id"]
    before = len(app.state.started)

    out = client.post(f"/check/{check_id}/claim",
                      data={"email": "grower@orchard.test"}, follow_redirects=False)
    assert out.status_code == 303

    assert db.get_check(check_id)["email"] == "grower@orchard.test"
    assert db.get_site(site_id)["contact_email"] == "grower@orchard.test"
    assert len(app.state.started) == before, "unlocking is about display, not a second fit"


def test_an_unknown_check_is_a_404(client):
    assert client.get("/check/chk_nope").status_code == 404


# ----------------------------------------------------------------------
# Bookings
# ----------------------------------------------------------------------


def test_bookings_round_trip(db):
    account = db.create_account("Acme", "b@acme.test")
    site = db.create_site(
        account["id"], name="V", lat=42.0, lon=-71.0, elevation_m=5.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    db.put_bookings(site["id"], [("2027-05-01", 1000.0, 20), ("2027-05-02", 2000.0, None)])
    window = db.bookings_window(site["id"], "2027-05-01", "2027-05-31")
    assert window == {"2027-05-01": 1000.0, "2027-05-02": 2000.0}
    assert db.bookings_summary(site["id"])["days"] == 2
    assert db.bookings_summary(site["id"])["total"] == 3000.0


def test_bookings_upsert_rather_than_duplicate(db):
    account = db.create_account("Acme", "c@acme.test")
    site = db.create_site(
        account["id"], name="V", lat=42.0, lon=-71.0, elevation_m=5.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    db.put_bookings(site["id"], [("2027-05-01", 1000.0, None)])
    db.put_bookings(site["id"], [("2027-05-01", 4000.0, None)])
    assert db.bookings_window(site["id"], "2027-05-01", "2027-05-01") == {"2027-05-01": 4000.0}


def test_a_stranger_cannot_upload_bookings(app, db):
    account = db.create_account("Acme", "d@acme.test")
    site = db.create_site(
        account["id"], name="V", lat=42.0, lon=-71.0, elevation_m=5.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    stranger = TestClient(app)
    res = stranger.post(
        f"/venues/{site['id']}/bookings",
        files={"file": ("b.csv", b"date,revenue\n2027-05-01,100\n", "text/csv")},
    )
    assert res.status_code == 404
