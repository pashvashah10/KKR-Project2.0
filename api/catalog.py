"""The four things a business can actually buy, defined as code.

Not a database table. A product catalogue at this stage changes when someone
edits a file and ships, not when someone edits a row in production, and keeping
it in code means the copy, the price and the fulfilment behaviour are reviewed
together in one diff.

The important rule is downstream of here: **a price is snapshotted into the
order line at purchase and never read back out of this file.** Changing
`price_cents` below changes what the next customer pays and nothing about what
a previous customer was charged. `store.create_order` copies both the price and
the display name into `order_items`; `api/tests/test_checkout.py` asserts that
moving a price afterwards leaves the historical order untouched.

### Why three of the four are fixed-price

Fitting a century of daily weather at a new coordinate takes 80-140 seconds and
every pricing endpoint is gated behind it. A storefront cannot quote an
arbitrary address instantly, and pretending otherwise means either a fake number
or a two-minute spinner before the visitor sees anything at all.

So the lineup absorbs the constraint rather than hiding it. Three products are
sold at a published list price and the fit happens *after* purchase, as
fulfilment --- which is how a report product works anyway. Only Parametric
Cover needs the engine before a number exists, and there the wait is shown as
the product working, with already-fitted example venues offered as the fast
path.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Product", "PRODUCTS", "BY_SLUG", "get", "price_for", "money"]


@dataclass(frozen=True, slots=True)
class Tier:
    """A volume band. `max_units` of `None` means "and above"."""

    label: str
    min_units: int
    max_units: int | None
    price_cents: int

    def covers(self, units: int) -> bool:
        return units >= self.min_units and (self.max_units is None or units <= self.max_units)


@dataclass(frozen=True, slots=True)
class Product:
    slug: str
    name: str
    tagline: str
    #: `fixed` --- one price. `per_unit` --- price times a quantity the customer
    #: picks. `tiered` --- banded by quantity. `quoted` --- the engine decides,
    #: and there is no list price to show.
    pricing_model: str
    price_cents: int
    unit: str
    billing: str  # once | month | year | contract
    #: `report` runs a fit then delivers. `monitor` opens a subscription.
    #: `analytics` provisions a portfolio workspace. `cover` binds nothing ---
    #: it produces an indicative quote and a broker conversation.
    fulfilment: str
    summary: str
    deliverables: tuple[str, ...]
    how_it_works: tuple[tuple[str, str], ...]
    included: tuple[str, ...]
    not_included: tuple[str, ...]
    faq: tuple[tuple[str, str], ...]
    turnaround: str
    tiers: tuple[Tier, ...] = ()
    requires_fit: bool = False
    min_quantity: int = 1
    max_quantity: int = 50
    accent: str = "s1"

    # ------------------------------------------------------------------
    @property
    def is_quoted(self) -> bool:
        return self.pricing_model == "quoted"

    @property
    def is_subscription(self) -> bool:
        return self.billing in ("month", "year")

    @property
    def price_label(self) -> str:
        if self.is_quoted:
            return "Priced by the engine"
        base = money(self.price_cents)
        if self.pricing_model == "tiered":
            return f"from {base}"
        return base

    @property
    def price_suffix(self) -> str:
        if self.is_quoted:
            return "per contract"
        parts = []
        if self.pricing_model in ("per_unit", "tiered"):
            parts.append(f"per {self.unit}")
        if self.billing == "month":
            parts.append("per month")
        elif self.billing == "year":
            parts.append("per year")
        elif self.billing == "once":
            parts.append("one-off")
        return " · ".join(parts)

    def unit_price_cents(self, quantity: int) -> int:
        """What one unit costs at this quantity.

        Tiered products price *every* unit at the band the total quantity falls
        into, rather than stacking bands. That is the simpler promise to make on
        a pricing page and the one customers assume when they read "from".
        """
        if self.pricing_model == "tiered":
            for tier in self.tiers:
                if tier.covers(quantity):
                    return tier.price_cents
            return self.tiers[-1].price_cents if self.tiers else self.price_cents
        return self.price_cents


def money(cents: int, currency: str = "$") -> str:
    """Cents to a display string. The only place integer cents become text."""
    whole, part = divmod(abs(int(cents)), 100)
    sign = "-" if cents < 0 else ""
    if part == 0:
        return f"{sign}{currency}{whole:,}"
    return f"{sign}{currency}{whole:,}.{part:02d}"


def price_for(product: Product, quantity: int) -> int:
    """Line total in cents."""
    return product.unit_price_cents(quantity) * max(1, int(quantity))


# ----------------------------------------------------------------------
# The lineup
# ----------------------------------------------------------------------


EXPOSURE_REPORT = Product(
    slug="exposure-report",
    name="Exposure Report",
    tagline="What a hundred years of weather says your venue is carrying.",
    pricing_model="per_unit",
    price_cents=2_400_00,
    unit="venue",
    billing="once",
    fulfilment="report",
    accent="s1",
    turnaround="Delivered within an hour of purchase.",
    summary=(
        "One venue, one coordinate, one century. We pull the daily record from the "
        "nearest long-serving GHCN station, fit a seasonal cycle and a warming "
        "response regressed on global forcing rather than on calendar time, fit the "
        "tails with extreme value theory, and project the result to 2125 under four "
        "emissions pathways. What comes back is the distribution your season sits "
        "inside — not a number, a distribution, with the uncertainty split into the "
        "parts you can do something about and the parts you cannot."
    ),
    deliverables=(
        "Century-long fitted record with provenance for every observation",
        "Regional amplification factor with block-bootstrap confidence bounds",
        "Trigger frequency for every peril that applies to your vertical, by decade to 2125",
        "Tail fits — GPD shape and scale, with the diagnostic plots behind them",
        "Walk-forward backtest: CRPS skill, PIT calibration, reliability curve",
        "Station quality assessment, including any detected inhomogeneity",
    ),
    how_it_works=(
        ("Match the station",
         "Your coordinates are matched against 38,000 GHCN-Daily stations carrying "
         "temperature and rainfall, scored on record length and currency discounted "
         "by distance. The closest station is usually not the right one."),
        ("Fit the century",
         "Local daily temperature is regressed on modelled global temperature, not on "
         "the year. That reproduces the 1940–75 aerosol plateau and the volcanic "
         "notches without being told they exist, and the coefficient is your "
         "amplification factor."),
        ("Project and validate",
         "Four SSP pathways through a two-box energy balance model, with the "
         "uncertainty decomposed three ways. Then the whole thing is scored against "
         "thirty years it never saw."),
    ),
    included=(
        "Full dashboard access for this venue, indefinitely",
        "The underlying fitted model, downloadable",
        "Every number traceable to a source and a method",
    ),
    not_included=(
        "Daily revenue analysis — that needs your revenue data, which is free to upload",
        "A contract. This is analysis, not cover.",
    ),
    faq=(
        ("How is this different from a climate risk score?",
         "A score is a number someone assigned. This is a fitted distribution you can "
         "price against, with the fit shown and the out-of-sample skill reported "
         "honestly — including where it is small."),
        ("What if there is no good station near me?",
         "You are told so on the configure page before you buy, and the report falls "
         "back to ERA5 reanalysis at your exact coordinates. Reanalysis has its own "
         "basis risk and we say which one you got."),
        ("How far back does the record go?",
         "As far as the station does. Most qualifying stations run 90–130 years; the "
         "match tells you the span before you commit."),
    ),
)

SEASON_MONITOR = Product(
    slug="season-monitor",
    name="Season Monitor",
    tagline="Your season, watched daily against the distribution it should be in.",
    pricing_model="per_unit",
    price_cents=390_00,
    unit="venue",
    billing="month",
    fulfilment="monitor",
    accent="s3",
    turnaround="Live from the next morning's observation.",
    summary=(
        "The Exposure Report tells you what a century says. This tells you where the "
        "season you are actually having sits inside it. Every morning the previous "
        "day's observation is scored against the fitted distribution, the "
        "season-to-date is compared with the same point in every prior year, and you "
        "are told when a run of days moves the season somewhere the model considers "
        "unusual — before the quarter closes, rather than after."
    ),
    deliverables=(
        "Daily percentile of yesterday against the fitted distribution",
        "Season-to-date versus all prior years at the same point",
        "Alerts when a rolling window crosses a peril threshold",
        "Trigger-probability tracking for any live contract on the venue",
        "Monthly written note on where the season landed and why",
    ),
    how_it_works=(
        ("Fit once",
         "The venue is fitted the same way an Exposure Report is fitted. That is the "
         "expensive part and it happens once."),
        ("Score daily",
         "Each new observation is scored against the fitted conditional distribution "
         "for that day of year, so a warm day in April and a warm day in August are "
         "judged against different yardsticks."),
        ("Alert on runs, not days",
         "One hot day is noise. The alert logic watches rolling windows, because the "
         "perils that cost money are the ones that persist."),
    ),
    included=(
        "Everything in the Exposure Report for each monitored venue",
        "Daily email digest and the live dashboard",
        "Cancel any time — billing stops at the end of the period",
    ),
    not_included=(
        "Weather forecasting. This scores what happened, not what will happen next week.",
        "Automatic contract settlement.",
    ),
    faq=(
        ("Is this a forecast?",
         "No, and deliberately so. Beyond about ten days a forecast has no skill; the "
         "distribution does. This tells you where you are in the distribution, which "
         "is the decision-relevant thing."),
        ("Do I need an Exposure Report first?",
         "No. The fit is included. If you already bought one for the same venue you "
         "are not charged for it twice."),
        ("What does cancelling do?",
         "Stops the next charge. The venue's fitted model and past reports stay in "
         "your account."),
    ),
)

PARAMETRIC_COVER = Product(
    slug="parametric-cover",
    name="Parametric Cover",
    tagline="A contract that pays on the gauge reading, not on a loss adjuster.",
    pricing_model="quoted",
    price_cents=0,
    unit="contract",
    billing="contract",
    fulfilment="cover",
    accent="s2",
    requires_fit=True,
    turnaround="Indicative quote in about two minutes at a new venue; instantly at an example venue.",
    summary=(
        "Pick the peril, the season, the daily payout and the limit. The engine "
        "simulates the fitted generator forward to the contract year, produces the "
        "trigger probability and the full payout distribution, and turns it into a "
        "two-sided price with every risk load named separately — capital, parameter "
        "uncertainty, basis, expense. You see the fair value and you see what sits on "
        "top of it, which is not how this is normally sold."
    ),
    deliverables=(
        "Fair value, bid, offer and mid, with each load itemised",
        "Trigger probability and the payout distribution behind it",
        "Suggested attachment point regressed from your own exposure",
        "Basis load shown explicitly against the settlement station's distance",
        "Ruin-adjusted hedge sizing against your reserves and burn",
    ),
    how_it_works=(
        ("Configure the trigger",
         "Peril, contract year, payout per triggering day, aggregate limit and "
         "attachment. If you have uploaded revenue, the attachment point is regressed "
         "from your own loss curve rather than guessed."),
        ("Price it",
         "Monte Carlo over the fitted generator under the chosen SSP pathway. The "
         "output is a distribution, not a point, and the loads are computed from its "
         "shape — CVaR, standard deviation, parameter spread."),
        ("Size it",
         "Every hedge loses money in expected dollars, because the premium contains "
         "the seller's load. Sizing therefore runs on expected log wealth subject to "
         "a ruin constraint, so two businesses with identical exposure and different "
         "cash positions correctly hedge different amounts."),
    ),
    included=(
        "Indicative pricing, itemised, with the model behind it inspectable",
        "Hedge sizing against your balance sheet",
        "Terms suggested from your exposure rather than from a template",
    ),
    not_included=(
        "Bound cover. Issuing a policy needs a licensed carrier or broker on the "
        "paper, and we are not one. The quote is indicative and subject to "
        "underwriting.",
        "Settlement administration.",
    ),
    faq=(
        ("Can I bind protection here today?",
         "No, and anyone telling you otherwise on a page like this is misleading you. "
         "What you get is a priced, itemised indicative quote and the analysis behind "
         "it, which is what a broker needs to place the risk. Placing it needs a "
         "licensed carrier or MGA, and we are neither."),
        ("Why is basis risk a separate line?",
         "Because it is real money and it is usually buried. The contract settles on "
         "a gauge that is some distance from your venue; that distance has a cost and "
         "you should see it rather than find it inside a margin."),
        ("What does 'indicative' actually mean?",
         "The model's price, before an underwriter has looked at it. Expect the "
         "final number to move — the point is that you know what it is made of."),
    ),
)

PORTFOLIO_ANALYTICS = Product(
    slug="portfolio-analytics",
    name="Portfolio Analytics",
    tagline="Many venues, one balance sheet, and the correlation between them.",
    pricing_model="tiered",
    price_cents=1_800_00,
    unit="venue",
    billing="month",
    fulfilment="analytics",
    accent="s4",
    min_quantity=3,
    max_quantity=200,
    turnaround="Workspace provisioned immediately; venues fit as they are added.",
    summary=(
        "A group with eleven sites does not have eleven independent weather problems. "
        "It has one, with a correlation structure — and the whole question is whether "
        "a bad July in one region is a bad July in the others. This fits every venue, "
        "then fits the dependence between them, so aggregate exposure is computed "
        "from a joint distribution instead of by adding up marginals and hoping."
    ),
    deliverables=(
        "Every venue fitted, with its own report and monitor",
        "Cross-site correlation of season outcomes, and of trigger events",
        "Aggregate loss distribution from the joint model, not the sum of marginals",
        "Diversification benefit quantified — what the portfolio saves over standalone",
        "Portfolio-level hedge sizing and a per-site attribution of it",
    ),
    how_it_works=(
        ("Fit the marginals",
         "Each venue gets the full treatment: station match, century fit, tails, "
         "projection."),
        ("Fit the dependence",
         "A Gaussian copula over season aggregates with a shared synoptic factor, so "
         "both the marginal distributions and the correlation survive into the "
         "simulation."),
        ("Aggregate honestly",
         "Adding up per-site expected losses gets the mean right and the tail badly "
         "wrong. The tail is where a portfolio hedge is decided, so the tail is what "
         "this computes."),
    ),
    included=(
        "Exposure Report and Season Monitor for every venue in the workspace",
        "Portfolio dashboard with per-site drill-down",
        "Quarterly review of the correlation structure",
    ),
    not_included=(
        "Cover. Same as Parametric Cover: indicative pricing only.",
        "Consolidation of your existing insurance programme.",
    ),
    faq=(
        ("Why a minimum of three venues?",
         "Below three there is no correlation structure worth fitting and you are "
         "better served buying individual monitors."),
        ("How is it billed?",
         "Per venue per month, banded. The band is set by your total venue count and "
         "applies to every venue, so crossing a threshold lowers the price on all of "
         "them. The discount at each threshold is modest by construction: with flat "
         "bands, anything steeper would make the eleventh venue cheaper in total than "
         "the tenth, and nobody should be able to lower their bill by buying more."),
        ("Can we add venues mid-period?",
         "Yes. New venues fit within a few minutes and the band re-evaluates at the "
         "next period."),
    ),
    # Flat bands bound how steep a threshold discount can be. At the 11-venue
    # boundary the most that can be given away is 1/11 (~9%), because anything
    # more makes eleven venues cheaper in total than ten --- a customer able to
    # cut their bill by buying more is a pricing bug, not a promotion. These
    # rates sit just inside that limit; `test_totals_never_decrease_with_volume`
    # enforces it over the whole range.
    tiers=(
        Tier("3–10 venues", 3, 10, 1_800_00),
        Tier("11–30 venues", 11, 30, 1_700_00),
        Tier("31+ venues", 31, None, 1_650_00),
    ),
)


PRODUCTS: tuple[Product, ...] = (
    EXPOSURE_REPORT,
    SEASON_MONITOR,
    PARAMETRIC_COVER,
    PORTFOLIO_ANALYTICS,
)

BY_SLUG: dict[str, Product] = {p.slug: p for p in PRODUCTS}


def get(slug: str) -> Product | None:
    return BY_SLUG.get(slug)
