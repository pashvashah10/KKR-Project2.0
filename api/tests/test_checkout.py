"""Tests for the catalogue, cart, orders and the mock payment seam.

The tests that matter most here are the boring ones. Arithmetic on money and the
immutability of a placed order are the two things a storefront cannot get wrong
and the two things nobody notices are wrong until an invoice disagrees with a
bank statement.
"""

from __future__ import annotations

import importlib

import pytest

from api import catalog, checkout


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A throwaway database per test, so orders cannot leak between them."""
    from api import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path / "models")
    store.init()
    return store


@pytest.fixture()
def account(db):
    return db.create_account("Acme Outdoor", "ops@acme.test")


def cart_of(*lines) -> checkout.Cart:
    """`(slug, quantity)` pairs to a Cart, priced from the catalogue."""
    return checkout.Cart([
        checkout.CartLine(
            product=catalog.BY_SLUG[slug],
            quantity=qty,
            site_id=None,
            site_name=None,
            config=config or {},
        )
        for slug, qty, *rest in lines
        for config in [rest[0] if rest else None]
    ])


# ----------------------------------------------------------------------
# Catalogue integrity
# ----------------------------------------------------------------------


def test_all_four_products_are_listed():
    assert len(catalog.PRODUCTS) == 4
    assert {p.slug for p in catalog.PRODUCTS} == {
        "exposure-report", "season-monitor", "parametric-cover", "portfolio-analytics",
    }


@pytest.mark.parametrize("product", catalog.PRODUCTS, ids=lambda p: p.slug)
def test_every_product_is_completely_specified(product):
    """A half-filled product page is worse than no page."""
    assert product.name and product.tagline and product.summary
    assert product.deliverables and product.how_it_works and product.faq
    assert product.included and product.not_included
    assert product.turnaround
    assert product.accent in {"s1", "s2", "s3", "s4"}
    assert product.pricing_model in {"fixed", "per_unit", "tiered", "quoted"}
    assert product.billing in {"once", "month", "year", "contract"}


@pytest.mark.parametrize("product", catalog.PRODUCTS, ids=lambda p: p.slug)
def test_no_product_copy_implies_we_are_an_insurer(product):
    """Regulatory framing, enforced rather than remembered.

    We are an analytics provider. Copy that says "insurance", "policy" or "buy
    cover" implies a licence we do not hold, so the words are banned outright
    except where they appear inside an explicit disclaimer of exactly that.
    """
    banned = ("buy cover", "purchase policy", "our policy covers", "we insure")
    blob = " ".join([
        product.name, product.tagline, product.summary, product.turnaround,
        *product.deliverables, *product.included,
        *(t for pair in product.how_it_works for t in pair),
        *(t for pair in product.faq for t in pair),
    ]).lower()
    for phrase in banned:
        assert phrase not in blob, f"{product.slug} says {phrase!r}"


def test_parametric_cover_states_it_cannot_bind():
    """The single most important sentence in the catalogue."""
    product = catalog.BY_SLUG["parametric-cover"]
    blob = (" ".join(product.not_included) + " ".join(a for _, a in product.faq)).lower()
    assert "carrier" in blob or "broker" in blob
    assert "indicative" in blob
    assert product.is_quoted and product.price_cents == 0


def test_quoted_product_shows_no_list_price():
    product = catalog.BY_SLUG["parametric-cover"]
    assert "$" not in product.price_label
    assert product.price_label == "Priced by the engine"


# ----------------------------------------------------------------------
# Pricing arithmetic
# ----------------------------------------------------------------------


def test_money_formats_whole_and_partial_amounts():
    assert catalog.money(2_400_00) == "$2,400"
    assert catalog.money(1_234_56) == "$1,234.56"
    assert catalog.money(0) == "$0"
    assert catalog.money(-500_00) == "-$500"


def test_per_unit_price_scales_with_quantity():
    report = catalog.BY_SLUG["exposure-report"]
    assert catalog.price_for(report, 1) == 2_400_00
    assert catalog.price_for(report, 3) == 7_200_00


def test_tiered_pricing_applies_one_band_to_every_unit():
    """Banded, not stacked --- crossing a threshold lowers the price on all."""
    portfolio = catalog.BY_SLUG["portfolio-analytics"]
    assert portfolio.unit_price_cents(5) == 1_800_00
    assert portfolio.unit_price_cents(11) == 1_700_00
    assert portfolio.unit_price_cents(40) == 1_650_00
    assert portfolio.unit_price_cents(11) < portfolio.unit_price_cents(10)


@pytest.mark.parametrize("product", catalog.PRODUCTS, ids=lambda p: p.slug)
def test_totals_never_decrease_with_volume(product):
    """Nobody may lower their bill by buying more.

    This is the check that caught the original bands. At $1,800 for 3-10 and
    $1,450 for 11-30, eleven venues came to $15,950 against ten at $18,000 ---
    a customer with ten venues could save $2,050 by adding an eleventh, and a
    customer with eleven could be talked into twelve by a salesperson who had
    not read the price list. Flat banding caps the threshold discount at
    `1/min_units`; these rates sit inside it.
    """
    if product.is_quoted:
        pytest.skip("no list price to be monotonic in")

    totals = [
        catalog.price_for(product, q)
        for q in range(product.min_quantity, min(product.max_quantity, 60) + 1)
    ]
    for lower, higher in zip(totals, totals[1:], strict=False):
        assert higher > lower, "one more unit must never cost less in total"


def test_tier_bands_are_contiguous_and_ordered():
    for product in catalog.PRODUCTS:
        if not product.tiers:
            continue
        for prev, nxt in zip(product.tiers, product.tiers[1:], strict=False):
            assert prev.max_units is not None
            assert nxt.min_units == prev.max_units + 1, "a gap between bands is unpriceable"
            assert nxt.price_cents <= prev.price_cents, "volume must not cost more"


def test_cart_totals_are_integer_arithmetic():
    cart = cart_of(("exposure-report", 2), ("season-monitor", 3))
    assert cart.subtotal_cents == 2 * 2_400_00 + 3 * 390_00
    assert cart.tax_cents == 0
    assert cart.total_cents == cart.subtotal_cents
    assert isinstance(cart.total_cents, int)
    assert cart.count == 5


def test_cart_separates_recurring_from_one_off():
    cart = cart_of(("exposure-report", 1), ("season-monitor", 2))
    assert cart.has_subscription
    assert cart.recurring_cents == 2 * 390_00
    assert cart.recurring_cents < cart.subtotal_cents


def test_quoted_line_prices_from_its_config_not_the_catalogue():
    cart = cart_of(("parametric-cover", 1, {"premium_cents": 47_350_00}))
    assert cart.subtotal_cents == 47_350_00


# ----------------------------------------------------------------------
# Cart session behaviour
# ----------------------------------------------------------------------


def test_cart_round_trips_through_a_session_dict():
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=2, site_id="site_a", site_name="Vail")
    cart = checkout.read_cart(session)
    assert cart.count == 2
    assert cart.lines[0].site_name == "Vail"


def test_the_cookie_never_stores_a_price():
    """A tampered cookie must not be able to buy anything at its own number."""
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=1)
    raw = session[checkout.CART_KEY][0]
    assert "price" not in json_keys(raw)
    assert "unit_price_cents" not in json_keys(raw)


def json_keys(obj, prefix=""):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= json_keys(v, k)
    return keys


def test_same_product_same_site_merges_rather_than_duplicating():
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=1, site_id="site_a")
    checkout.add_line(session, "exposure-report", quantity=4, site_id="site_a")
    cart = checkout.read_cart(session)
    assert len(cart.lines) == 1 and cart.lines[0].quantity == 4


def test_same_product_different_site_is_a_separate_line():
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=1, site_id="site_a")
    checkout.add_line(session, "exposure-report", quantity=1, site_id="site_b")
    assert len(checkout.read_cart(session).lines) == 2


def test_quantity_is_clamped_to_the_product_bounds():
    session: dict = {}
    checkout.add_line(session, "portfolio-analytics", quantity=1)  # min is 3
    assert checkout.read_cart(session).lines[0].quantity == 3
    checkout.add_line(session, "portfolio-analytics", quantity=99_999)
    assert checkout.read_cart(session).lines[0].quantity == 200


def test_update_to_zero_removes_the_line():
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=2)
    checkout.update_line(session, 0, 0)
    assert not checkout.read_cart(session).lines


def test_removing_and_updating_out_of_range_is_a_no_op():
    session: dict = {}
    checkout.add_line(session, "exposure-report", quantity=1)
    checkout.remove_line(session, 7)
    checkout.update_line(session, 7, 3)
    assert len(checkout.read_cart(session).lines) == 1


def test_an_unknown_slug_in_the_cookie_is_dropped_not_fatal():
    session = {checkout.CART_KEY: [{"slug": "retired-product", "quantity": 1}]}
    assert checkout.read_cart(session).lines == []


# ----------------------------------------------------------------------
# Orders
# ----------------------------------------------------------------------


BILLING = {"name": "Dana Reyes", "email": "dana@acme.test", "company": "Acme", "address": "1 Way"}


def test_placing_an_order_writes_it_and_marks_it_paid(db, account):
    cart = cart_of(("exposure-report", 2))
    order = checkout.place_order(account, cart, BILLING)

    assert order["paid"] is True
    assert order["status"] == "paid"
    assert order["subtotal_cents"] == 4_800_00
    assert order["total_cents"] == 4_800_00
    assert order["invoice_number"].startswith("DW-")
    assert len(order["items"]) == 1
    assert order["items"][0]["line_total_cents"] == 4_800_00


def test_the_mock_provider_says_plainly_that_no_money_moved(db, account):
    order = checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)
    charge = order["charge"]
    assert charge.provider == "mock"
    assert charge.reference.startswith("mock_")
    assert "no payment was taken" in charge.message.lower()
    assert "no money moved" in charge.message.lower()


def test_stripe_provider_refuses_rather_than_silently_mocking(db, account):
    """Falling back to the mock would be the dangerous failure."""
    with pytest.raises(NotImplementedError):
        checkout.place_order(
            account, cart_of(("exposure-report", 1)), BILLING,
            provider=checkout.StripeProvider(),
        )


def test_an_empty_cart_cannot_be_ordered(db, account):
    with pytest.raises(ValueError):
        checkout.place_order(account, checkout.Cart([]), BILLING)


def test_invoice_numbers_are_unique_and_increasing(db, account):
    numbers = [
        checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)["invoice_number"]
        for _ in range(6)
    ]
    assert len(set(numbers)) == 6
    values = [int(n.split("-")[1]) for n in numbers]
    assert values == sorted(values)
    assert values[-1] - values[0] == 5, "sequence must be gapless"


def test_price_is_snapshotted_and_survives_a_catalogue_change(db, account, monkeypatch):
    """The test the whole snapshot design exists for.

    Change the list price after an order is placed; the order must not move.
    """
    order = checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)
    original = order["items"][0]["unit_price_cents"]
    assert original == 2_400_00

    hiked = catalog.Product(**{
        **{f: getattr(catalog.BY_SLUG["exposure-report"], f)
           for f in catalog.Product.__slots__},
        "price_cents": 9_999_00,
    })
    monkeypatch.setitem(catalog.BY_SLUG, "exposure-report", hiked)
    assert catalog.get("exposure-report").price_cents == 9_999_00

    again = db.get_order(order["id"])
    assert again["items"][0]["unit_price_cents"] == original
    assert again["total_cents"] == 2_400_00


def test_product_name_is_snapshotted_too(db, account):
    order = checkout.place_order(account, cart_of(("season-monitor", 1)), BILLING)
    assert order["items"][0]["product_name"] == "Season Monitor"


def test_a_subscription_line_opens_a_subscription(db, account):
    order = checkout.place_order(account, cart_of(("season-monitor", 3)), BILLING)
    subs = db.list_subscriptions(account["id"])

    assert len(subs) == 1
    sub = subs[0]
    assert sub["product_slug"] == "season-monitor"
    assert sub["status"] == "active"
    assert sub["period"] == "month"
    assert sub["price_cents"] == 3 * 390_00
    assert sub["current_period_end"] > sub["started_at"]
    assert sub["order_id"] == order["id"]


def test_a_one_off_line_opens_no_subscription(db, account):
    checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)
    assert db.list_subscriptions(account["id"]) == []


def test_cancelling_a_subscription_is_idempotent(db, account):
    checkout.place_order(account, cart_of(("season-monitor", 1)), BILLING)
    sub = db.list_subscriptions(account["id"])[0]

    assert db.cancel_subscription(sub["id"], account["id"]) is True
    assert db.cancel_subscription(sub["id"], account["id"]) is False
    assert db.list_subscriptions(account["id"])[0]["status"] == "cancelled"


def test_a_subscription_cannot_be_cancelled_by_another_account(db, account):
    checkout.place_order(account, cart_of(("season-monitor", 1)), BILLING)
    sub = db.list_subscriptions(account["id"])[0]
    other = db.create_account("Someone else", "mallory@evil.test")

    assert db.cancel_subscription(sub["id"], other["id"]) is False
    assert db.list_subscriptions(account["id"])[0]["status"] == "active"


def test_orders_are_scoped_to_their_account(db, account):
    order = checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)
    other = db.create_account("Someone else", "mallory2@evil.test")

    assert db.get_order(order["id"], account_id=account["id"]) is not None
    assert db.get_order(order["id"], account_id=other["id"]) is None
    assert db.list_orders(other["id"]) == []


# ----------------------------------------------------------------------
# Fulfilment
# ----------------------------------------------------------------------


def test_cover_lines_are_marked_awaiting_underwriting(db, account):
    cart = cart_of(("parametric-cover", 1, {"premium_cents": 12_000_00}))
    order = checkout.place_order(account, cart, BILLING)

    started: list[tuple[str, str]] = []
    checkout.fulfil(order, lambda s, j: started.append((s, j)))

    assert started == [], "an indicative quote provisions nothing"
    refreshed = db.get_order(order["id"])
    assert refreshed["items"][0]["fulfilment_status"] == "awaiting-underwriting"


def test_a_report_without_a_venue_waits_for_one(db, account):
    order = checkout.place_order(account, cart_of(("exposure-report", 1)), BILLING)
    checkout.fulfil(order, lambda s, j: None)
    assert db.get_order(order["id"])["items"][0]["fulfilment_status"] == "awaiting-venue"


def test_a_report_on_an_unfitted_venue_starts_a_fit(db, account):
    site = db.create_site(
        account["id"], name="Test venue", lat=42.0, lon=-71.0, elevation_m=10.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    cart = checkout.Cart([
        checkout.CartLine(catalog.BY_SLUG["exposure-report"], 1, site["id"], "Test venue", {})
    ])
    order = checkout.place_order(account, cart, BILLING)

    started: list[tuple[str, str]] = []
    jobs = checkout.fulfil(order, lambda s, j: started.append((s, j)))

    assert len(started) == 1 and started[0][0] == site["id"]
    assert len(jobs) == 1
    assert db.get_order(order["id"])["items"][0]["fulfilment_status"] == "fitting"


def test_a_report_on_an_already_fitted_venue_delivers_immediately(db, account):
    site = db.create_site(
        account["id"], name="Fitted venue", lat=42.0, lon=-71.0, elevation_m=10.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )
    db.update_site(site["id"], status="ready")
    cart = checkout.Cart([
        checkout.CartLine(catalog.BY_SLUG["exposure-report"], 1, site["id"], "Fitted venue", {})
    ])
    order = checkout.place_order(account, cart, BILLING)

    started: list = []
    checkout.fulfil(order, lambda s, j: started.append(s))

    assert started == [], "no reason to refit"
    assert db.get_order(order["id"])["items"][0]["fulfilment_status"] == "delivered"


# ----------------------------------------------------------------------
# Invoice view
# ----------------------------------------------------------------------


def test_order_view_formats_money_from_the_snapshot(db, account):
    order = checkout.place_order(account, cart_of(("exposure-report", 2)), BILLING)
    view = checkout.order_view(db.get_order(order["id"]))

    assert view["total"] == "$4,800"
    assert view["items"][0]["unit_price"] == "$2,400"
    assert view["items"][0]["line_total"] == "$4,800"
    assert view["created_iso"]


def test_module_imports_cleanly():
    """Cheap guard against a syntax error shipping in a file with no other test."""
    for name in ("api.catalog", "api.checkout", "api.web", "api.seed", "api.stations"):
        importlib.import_module(name)
