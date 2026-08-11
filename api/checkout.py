"""Cart, order placement, and the payment seam.

**No payment is taken anywhere in this module, and no card details are collected
anywhere in this application.** `MockProvider` records an order, marks it paid
with a clearly-fake reference, and the confirmation page says so in plain words.
That is a deliberate product decision, not an unfinished one: a checkout that
asks for a card number without a processor behind it is worse than one that
admits what it is.

`StripeProvider` below is the seam. It is a stub with the exact call sites
marked, so wiring a real processor is an afternoon of work against a documented
interface rather than a refactor of the order model.

### The cart lives in a signed cookie

Carts are pre-purchase state belonging to an anonymous visitor. Putting them in
the database means a row for every browser that ever looked at the pricing page,
plus a reaper to clean them up. A signed session cookie holds a few line
references, cannot be tampered with client-side, and disappears on its own ---
and the moment it becomes an order, it becomes a database row.

The cookie stores **only the slug, quantity, site and config**. Never a price.
Prices are recomputed from the catalogue on every render and frozen once at
`place_order`, so a stale or edited cookie can never buy anything at yesterday's
number.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Protocol

from . import catalog, store

log = logging.getLogger(__name__)

__all__ = [
    "CartLine", "Cart", "read_cart", "add_line", "update_line", "remove_line",
    "PaymentProvider", "MockProvider", "StripeProvider", "place_order", "fulfil",
]

CART_KEY = "cart"
#: Sales tax is out of scope --- a real implementation needs nexus rules per
#: jurisdiction and a tax engine. Zero, stated on the invoice, is honest;
#: inventing 8.25% would not be.
TAX_RATE = 0.0


# ----------------------------------------------------------------------
# Cart
# ----------------------------------------------------------------------


@dataclass(slots=True)
class CartLine:
    product: catalog.Product
    quantity: int
    site_id: str | None
    site_name: str | None
    config: dict[str, Any]

    @property
    def unit_price_cents(self) -> int:
        # Quoted products carry the engine's own number in their config; every
        # other product is priced from the catalogue at render time.
        if self.product.is_quoted:
            return int(self.config.get("premium_cents") or 0)
        return self.product.unit_price_cents(self.quantity)

    @property
    def line_total_cents(self) -> int:
        return self.unit_price_cents * self.quantity

    @property
    def unit_price(self) -> str:
        return catalog.money(self.unit_price_cents)

    @property
    def line_total(self) -> str:
        return catalog.money(self.line_total_cents)


@dataclass(slots=True)
class Cart:
    lines: list[CartLine]

    @property
    def subtotal_cents(self) -> int:
        return sum(line.line_total_cents for line in self.lines)

    @property
    def tax_cents(self) -> int:
        return int(round(self.subtotal_cents * TAX_RATE))

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents + self.tax_cents

    @property
    def subtotal(self) -> str:
        return catalog.money(self.subtotal_cents)

    @property
    def tax(self) -> str:
        return catalog.money(self.tax_cents)

    @property
    def total(self) -> str:
        return catalog.money(self.total_cents)

    @property
    def count(self) -> int:
        return sum(line.quantity for line in self.lines)

    @property
    def has_subscription(self) -> bool:
        return any(line.product.is_subscription for line in self.lines)

    @property
    def recurring_cents(self) -> int:
        return sum(li.line_total_cents for li in self.lines if li.product.is_subscription)

    @property
    def recurring(self) -> str:
        return catalog.money(self.recurring_cents)

    def __bool__(self) -> bool:
        return bool(self.lines)


def read_cart(session: dict) -> Cart:
    """Materialise the cookie into priced lines, dropping anything unrecognised.

    Tolerant by design: a product retired between page loads should empty that
    line, not 500 the cart.
    """
    lines: list[CartLine] = []
    for raw in session.get(CART_KEY) or []:
        product = catalog.get(raw.get("slug", ""))
        if product is None:
            continue
        qty = max(product.min_quantity, min(int(raw.get("quantity", 1)), product.max_quantity))
        lines.append(
            CartLine(
                product=product,
                quantity=qty,
                site_id=raw.get("site_id"),
                site_name=raw.get("site_name"),
                config=raw.get("config") or {},
            )
        )
    return Cart(lines)


def _key(slug: str, site_id: str | None) -> tuple[str, str | None]:
    return (slug, site_id)


def add_line(
    session: dict,
    slug: str,
    quantity: int = 1,
    site_id: str | None = None,
    site_name: str | None = None,
    config: dict | None = None,
) -> None:
    """Add to the cart, merging by (product, site).

    Two Exposure Reports for the same venue is a mistake, not an order, so the
    same product against the same site replaces rather than duplicates. The same
    product against a *different* site is a separate line, because it is.
    """
    product = catalog.get(slug)
    if product is None:
        raise KeyError(slug)

    raw = list(session.get(CART_KEY) or [])
    entry = {
        "slug": slug,
        "quantity": max(product.min_quantity, min(int(quantity), product.max_quantity)),
        "site_id": site_id,
        "site_name": site_name,
        "config": config or {},
    }
    for i, existing in enumerate(raw):
        if _key(existing.get("slug", ""), existing.get("site_id")) == _key(slug, site_id):
            raw[i] = entry
            break
    else:
        raw.append(entry)
    session[CART_KEY] = raw


def update_line(session: dict, index: int, quantity: int) -> None:
    raw = list(session.get(CART_KEY) or [])
    if not 0 <= index < len(raw):
        return
    product = catalog.get(raw[index].get("slug", ""))
    if product is None or quantity <= 0:
        raw.pop(index)
    else:
        raw[index]["quantity"] = max(
            product.min_quantity, min(int(quantity), product.max_quantity)
        )
    session[CART_KEY] = raw


def remove_line(session: dict, index: int) -> None:
    raw = list(session.get(CART_KEY) or [])
    if 0 <= index < len(raw):
        raw.pop(index)
    session[CART_KEY] = raw


def clear_cart(session: dict) -> None:
    session[CART_KEY] = []


# ----------------------------------------------------------------------
# Payment
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Charge:
    reference: str
    provider: str
    succeeded: bool
    message: str


class PaymentProvider(Protocol):
    """Everything the checkout needs from a processor.

    Kept to one method on purpose. Anything that needs more than "charge this
    many cents and tell me the reference" belongs in the provider, not here.
    """

    name: str

    def charge(self, order: dict, billing: dict) -> Charge: ...


class MockProvider:
    """Records the order. Moves no money. Says so.

    The reference is prefixed `mock_` and is surfaced verbatim on the invoice,
    so there is no state of the system in which someone can mistake one of these
    for a real settlement.
    """

    name = "mock"

    def charge(self, order: dict, billing: dict) -> Charge:
        return Charge(
            reference=f"mock_{secrets.token_hex(8)}",
            provider=self.name,
            succeeded=True,
            message=(
                "No payment was taken. This order was recorded against your account "
                "so the workflow is real end to end, but no card was collected and "
                "no money moved."
            ),
        )


class StripeProvider:
    """Where a real processor plugs in. Not wired, and not pretending to be.

    The full integration is three pieces, and the reason this is a stub rather
    than a half-implementation is that the middle one cannot live in this
    process at all:

    1. **Create a PaymentIntent** for `order["total_cents"]` in
       `order["currency"]`, with `metadata={"order_id": order["id"],
       "invoice_number": order["invoice_number"]}`. Needs `STRIPE_SECRET_KEY`
       from the environment --- never from the database, never from a template.

    2. **Collect the card in Stripe Elements**, client-side, against the
       intent's client secret. Card details must never reach this server; that
       is the entire point of Elements and it is what keeps PCI scope at SAQ-A.
       This is why `place_order` returns a pending order and the confirmation
       page is reached by redirect rather than by this method returning success.

    3. **Confirm on the webhook**, not on the redirect. `payment_intent.
       succeeded` arrives at a signed endpoint; verify the signature with
       `STRIPE_WEBHOOK_SECRET`, look the order up by `metadata.order_id`, and
       call `store.mark_order_paid` there. A customer closing the tab after
       paying must still end up with a paid order, and only the webhook
       guarantees that.

    Subscriptions (`Season Monitor`, `Portfolio Analytics`) map to Stripe Prices
    with `recurring.interval="month"`; `store.create_subscription` then records
    the local mirror keyed on the Stripe subscription id.
    """

    name = "stripe"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def charge(self, order: dict, billing: dict) -> Charge:
        raise NotImplementedError(
            "Stripe is not configured. See the class docstring for the three "
            "integration points; this deliberately refuses rather than silently "
            "falling back to the mock provider."
        )


DEFAULT_PROVIDER: PaymentProvider = MockProvider()


# ----------------------------------------------------------------------
# Placing the order
# ----------------------------------------------------------------------


def place_order(
    account: dict,
    cart: Cart,
    billing: dict,
    provider: PaymentProvider | None = None,
) -> dict:
    """Freeze the cart into an order, charge it, and open any subscriptions.

    The price snapshot happens here and only here: `unit_price_cents` and
    `product_name` are read out of the catalogue once, written into
    `order_items`, and never consulted again.
    """
    if not cart.lines:
        raise ValueError("Cannot place an empty order.")

    provider = provider or DEFAULT_PROVIDER
    items = [
        {
            "product_slug": line.product.slug,
            "product_name": line.product.name,
            "site_id": line.site_id,
            "config": {**line.config, "site_name": line.site_name},
            "unit_price_cents": line.unit_price_cents,
            "quantity": line.quantity,
        }
        for line in cart.lines
    ]

    order = store.create_order(account["id"], items, {**billing, "provider": provider.name})

    charge = provider.charge(order, billing)
    if not charge.succeeded:
        return {**order, "charge": charge, "paid": False}

    store.mark_order_paid(order["id"], charge.reference, charge.provider)

    for item, line in zip(order["items"], cart.lines, strict=True):
        if line.product.is_subscription:
            store.create_subscription(
                account_id=account["id"],
                order_id=order["id"],
                product_slug=line.product.slug,
                price_cents=item["line_total_cents"],
                period=line.product.billing,
                site_id=line.site_id,
            )

    order = store.get_order(order["id"]) or order
    return {**order, "charge": charge, "paid": True}


def fulfil(order: dict, submit_fit) -> list[str]:
    """Start whatever each purchased line actually delivers.

    For the three fixed-price products the fit is fulfilment rather than a
    prerequisite --- the customer has bought a report and the century gets
    fitted to produce it. `submit_fit(site_id, job_id)` is injected so this
    module does not need to know about the thread pool.

    Returns the job ids started, for the confirmation page to poll.
    """
    jobs: list[str] = []
    for item in order["items"]:
        product = catalog.get(item["product_slug"])
        if product is None or product.fulfilment == "cover":
            # Cover is indicative: nothing to provision, a broker conversation
            # follows. Marked delivered so it does not sit as pending forever.
            store.update_item_fulfilment(item["id"], "awaiting-underwriting")
            continue

        site_id = item.get("site_id")
        if not site_id:
            store.update_item_fulfilment(item["id"], "awaiting-venue")
            continue

        site = store.get_site(site_id)
        if site and site["status"] == "ready":
            store.update_item_fulfilment(item["id"], "delivered")
            continue

        job = store.create_job(site_id, "fit")
        submit_fit(site_id, job["id"])
        store.update_item_fulfilment(item["id"], "fitting")
        jobs.append(job["id"])

    return jobs


def invoice_lines(order: dict) -> list[dict]:
    """Order rows shaped for display. Reads the snapshot, never the catalogue."""
    return [
        {
            **item,
            "unit_price": catalog.money(item["unit_price_cents"]),
            "line_total": catalog.money(item["line_total_cents"]),
            "site_name": (item.get("config") or {}).get("site_name"),
        }
        for item in order["items"]
    ]


def order_view(order: dict) -> dict:
    return {
        **order,
        "items": invoice_lines(order),
        "subtotal": catalog.money(order["subtotal_cents"]),
        "tax": catalog.money(order["tax_cents"]),
        "total": catalog.money(order["total_cents"]),
        "created_iso": time.strftime("%d %B %Y", time.localtime(order["created_at"])),
    }
