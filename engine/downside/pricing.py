"""Turning a distribution into a price.

The framing here is the Quant Bible's market-making section (6.2), which lists
three determinants of a quote, and all three map cleanly onto weather risk:

**Theoretical value.** What the contract is worth: `E[payout]` under the
*forward-projected* distribution for the contract year. Not the historical
frequency. For a 2027 contract the two are nearly the same; for a 2055 contract
they are not, and quoting history there is simply mispricing.

**Market width.** The Bible's rule is that an uncertain quantity gets a wider
market --- a die roll can be quoted tight, the number of ping-pong balls in the
Empire State Building cannot. Here the uncertainty is measurable rather than
vibes-based: re-price under parameter draws and across scenarios, and let the
spread of resulting theos set the half-width. A near-dated rain-day contract at a
well-observed station quotes tight. A 2075 snow-drought contract quotes wide,
because it genuinely is.

**Inventory.** A book already long Southwest heat should not quote the next
Southwest heat contract symmetrically. The skew term shades the quote to attract
the offsetting side, exactly as the Bible describes giving up edge to reduce
exposure.

On top of those three, two loads that are specific to this asset class:

* **Capital / tail load.** Weather payouts are wildly skewed --- most seasons pay
  nothing, some pay the limit. Expected value alone does not compensate for the
  capital that has to sit behind the tail, so the load is driven by CVaR, not by
  standard deviation alone.
* **Basis risk load.** The build plan is blunt about this: a gauge 18 miles away
  is a materially different bet. Settlement happens at the station; the loss
  happens at the venue. That gap is priced.

Note on direction: every load *raises* the offer and *lowers* the bid. The seller
of protection is warehousing a skewed, poorly-diversifiable risk, and the premium
above fair value is what pays for that. This is the reverse of the operator's
view in `hedging.py`, where the same premium is a cost --- deliberately, since the
two screens describe the two sides of the same trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .config import Location, Peril, Scenario
from .site import SimulatedWeather, SiteModel

__all__ = [
    "Contract",
    "PayoutDistribution",
    "Quote",
    "evaluate_payout",
    "price_contract",
]


@dataclass(frozen=True, slots=True)
class Contract:
    """A parametric weather contract.

    `structure` decides how triggered days convert to money:

    * ``per_day``   --- pays `payout_per_day` for each triggered day, capped at
                        `max_days`. Sensible's per-day model.
    * ``binary``    --- pays `limit` in full if any day triggers.
                        WeatherPromise's whole-trip model.
    * ``aggregate`` --- pays for triggered days beyond an attachment point. The
                        operator-facing structure: it ignores the ordinary
                        friction a business already absorbs and covers the
                        genuinely bad season.
    """

    id: str
    location_id: str
    peril_id: str
    year: int
    doy_start: int
    doy_end: int
    structure: str = "per_day"
    payout_per_day: float = 5_000.0
    limit: float = 100_000.0
    max_days: int = 20
    attachment_days: int = 0
    scenario: str = "ssp245"

    @property
    def window(self) -> np.ndarray:
        if self.doy_end >= self.doy_start:
            return np.arange(self.doy_start, self.doy_end + 1, dtype=float)
        # Wrap across the new year, for a ski season running Nov-Apr.
        return np.concatenate(
            [
                np.arange(self.doy_start, 366, dtype=float),
                np.arange(1, self.doy_end + 1, dtype=float),
            ]
        )

    @property
    def n_days(self) -> int:
        return len(self.window)

    @property
    def max_payout(self) -> float:
        if self.structure == "binary":
            return self.limit
        if self.structure == "aggregate":
            return min(self.limit, self.payout_per_day * max(self.n_days - self.attachment_days, 0))
        return min(self.limit, self.payout_per_day * self.max_days)


@dataclass(slots=True)
class PayoutDistribution:
    payouts: np.ndarray = field(repr=False)
    trigger_days: np.ndarray = field(repr=False)

    @property
    def expected(self) -> float:
        return float(self.payouts.mean())

    @property
    def sd(self) -> float:
        return float(self.payouts.std(ddof=1))

    @property
    def prob_any(self) -> float:
        return float((self.payouts > 0).mean())

    @property
    def expected_days(self) -> float:
        return float(self.trigger_days.mean())

    def var(self, level: float = 0.99) -> float:
        return float(np.quantile(self.payouts, level))

    def cvar(self, level: float = 0.99) -> float:
        """Mean payout conditional on being in the worst `1 - level` tail."""
        threshold = self.var(level)
        tail = self.payouts[self.payouts >= threshold]
        return float(tail.mean()) if tail.size else threshold

    def quantiles(self, qs=(0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[str, float]:
        return {str(q): float(np.quantile(self.payouts, q)) for q in qs}

    def histogram(self, bins: int = 40) -> dict:
        counts, edges = np.histogram(self.payouts, bins=bins)
        return {
            "counts": counts.tolist(),
            "edges": [float(e) for e in edges],
        }


def evaluate_payout(contract: Contract, peril: Peril, sim: SimulatedWeather) -> PayoutDistribution:
    """Apply the contract structure to simulated weather."""
    series = sim.get(peril.variable)
    triggered = peril.triggered(series)

    if peril.statistic == "consecutive":
        # `triggered` is already per-path (did a long enough run occur).
        hit_any = np.asarray(triggered).ravel()
        days = hit_any.astype(float)
        payouts = np.where(hit_any, contract.limit, 0.0)
        return PayoutDistribution(payouts=payouts, trigger_days=days)

    days = triggered.sum(axis=1).astype(float)

    if contract.structure == "binary":
        payouts = np.where(days > 0, contract.limit, 0.0)
    elif contract.structure == "aggregate":
        excess = np.clip(days - contract.attachment_days, 0.0, None)
        payouts = np.minimum(excess * contract.payout_per_day, contract.limit)
    else:
        capped = np.minimum(days, contract.max_days)
        payouts = np.minimum(capped * contract.payout_per_day, contract.limit)

    return PayoutDistribution(payouts=payouts, trigger_days=days)


# ----------------------------------------------------------------------


@dataclass(slots=True)
class Quote:
    contract_id: str
    #: Fair value: expected payout under the projected distribution.
    theo: float
    #: Expected payout if the last 30 years were simply repeated. Shown beside
    #: `theo` because the gap between them is the entire argument for modelling
    #: the forward distribution instead of quoting history.
    theo_historical: float
    sd: float
    prob_trigger: float
    expected_days: float
    var99: float
    cvar99: float
    max_payout: float

    load_capital: float
    load_parameter: float
    load_basis: float
    load_expense: float
    risk_premium: float

    bid: float
    ask: float
    mid: float
    half_width: float
    inventory_skew: float

    #: Premium as a share of the maximum payout --- the rate-on-line an insurance
    #: or reinsurance counterparty would actually compare across deals.
    rate_on_line: float
    #: Ratio of charged premium to expected loss.
    loading_multiple: float
    scenario_theos: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # `slots=True` means there is no __dict__; iterate the declared fields.
        return {
            f.name: (
                float(getattr(self, f.name))
                if isinstance(getattr(self, f.name), (int, float, np.floating))
                and not isinstance(getattr(self, f.name), bool)
                else getattr(self, f.name)
            )
            for f in fields(self)
        }


# Load coefficients. Deliberately named and separated rather than folded into one
# fudge factor, so a quote can be explained line by line to a counterparty.
LAMBDA_CVAR = 0.16       # share of the CVaR-minus-mean capital band charged
LAMBDA_SD = 0.10         # standard-deviation load
LAMBDA_PARAM = 1.25      # multiple of model-uncertainty sd charged

#: Expense is charged on the *risk premium*, not on the limit.
#:
#: Charging a percentage of the limit looks harmless and is not: for a remote,
#: low-probability layer the limit is large while the expected loss is small, so
#: the expense term swamps everything. Boston's rain contract came out quoted at
#: 11x fair value, almost all of it expense, and at that price no operator should
#: hedge --- which is exactly what the hedging screen then reported.
#:
#: Real origination and settlement cost is mostly fixed per contract plus a
#: margin on the risk actually carried, which is what this charges.
EXPENSE_FIXED = 1_500.0  # origination, data, settlement per contract
EXPENSE_RATIO = 0.12     # margin on the risk premium carried
#: Basis load per km of gauge-to-venue distance, as a share of theo. A venue 20 km
#: from its settlement gauge carries roughly a 9% load on this scale.
BASIS_PER_KM = 0.0045
#: Metres of vertical separation per kilometre of horizontal equivalent. Vertical
#: separation is worth far more than horizontal: 100 m of elevation is ~0.65 degC
#: of lapse rate before any other effect, which no amount of walking sideways on
#: the flat will reproduce. Same constant the station matcher scores on, so the
#: gauge chosen and the gauge charged for are separated by the same metric.
BASIS_KM_PER_ELEVATION_M = 0.05
MIN_HALF_WIDTH_FRAC = 0.035


def effective_basis_km(location: Location) -> float:
    """Gauge-to-venue separation in horizontal-kilometre equivalents.

    Charging basis on horizontal distance alone underprices exactly the venues
    where basis is worst --- mountain sites, where the nearest long-record gauge
    is usually in the valley. Vail's gauge is 29 km away and 300 m up; Mammoth's
    nearest by distance is 1,150 m down. Those are not 29 km and 62 km of basis.
    """
    return abs(location.station_distance_km) + BASIS_KM_PER_ELEVATION_M * abs(
        location.station_elevation_delta_m
    )


def price_contract(
    contract: Contract,
    peril: Peril,
    model: SiteModel,
    location: Location,
    scenario: Scenario,
    all_scenarios: tuple[Scenario, ...],
    n_paths: int = 6000,
    inventory_notional: float = 0.0,
    inventory_capacity: float = 5_000_000.0,
    rng: np.random.Generator | None = None,
) -> tuple[Quote, PayoutDistribution]:
    """Produce a two-sided quote for one contract.

    `inventory_notional` is the book's existing signed exposure to this peril and
    region --- positive meaning already short protection (having sold a lot of
    it). It shades the quote toward the offsetting side.
    """
    rng = rng or np.random.default_rng(17)
    window = contract.window

    # --- theoretical value under the contract-year distribution ----------
    sim = model.simulate(contract.year, window, scenario, n_paths=n_paths, rng=rng)
    dist = evaluate_payout(contract, peril, sim)
    theo = dist.expected

    # --- the same contract priced off recent history ---------------------
    # Simulated at 2010 conditions, i.e. what a naive "look at the last few
    # decades" approach would produce.
    sim_hist = model.simulate(
        2010, window, scenario, n_paths=n_paths, rng=np.random.default_rng(rng.integers(1 << 30)),
        parameter_uncertainty=False,
    )
    theo_hist = evaluate_payout(contract, peril, sim_hist).expected

    # --- scenario spread: model uncertainty, the Bible's market width ----
    scenario_theos: dict[str, float] = {}
    for sc in all_scenarios:
        if sc.id == scenario.id:
            scenario_theos[sc.id] = theo
            continue
        s = model.simulate(
            contract.year, window, sc,
            n_paths=max(n_paths // 3, 800),
            rng=np.random.default_rng(rng.integers(1 << 30)),
        )
        scenario_theos[sc.id] = evaluate_payout(contract, peril, s).expected

    weights = np.array([sc.weight for sc in all_scenarios])
    weights = weights / weights.sum()
    theos = np.array([scenario_theos[sc.id] for sc in all_scenarios])
    centre = float(np.sum(weights * theos))
    sigma_model = float(np.sqrt(np.sum(weights * (theos - centre) ** 2)))

    # --- loads ------------------------------------------------------------
    load_capital = LAMBDA_CVAR * max(dist.cvar(0.99) - theo, 0.0) + LAMBDA_SD * dist.sd
    load_parameter = LAMBDA_PARAM * sigma_model
    load_basis = BASIS_PER_KM * effective_basis_km(location) * theo
    load_expense = EXPENSE_FIXED * _duration_factor(contract) + EXPENSE_RATIO * (
        theo + load_capital + load_parameter + load_basis
    )

    risk_premium = load_capital + load_parameter + load_basis + load_expense
    mid = theo + risk_premium

    # --- two-sided market --------------------------------------------------
    half_width = max(
        LAMBDA_PARAM * sigma_model + 0.5 * LAMBDA_SD * dist.sd,
        MIN_HALF_WIDTH_FRAC * contract.max_payout,
    )
    utilisation = float(np.clip(inventory_notional / max(inventory_capacity, 1.0), -1.5, 1.5))
    inventory_skew = utilisation * half_width * 0.6

    bid = mid - half_width - inventory_skew
    ask = mid + half_width - inventory_skew

    quote = Quote(
        contract_id=contract.id,
        theo=theo,
        theo_historical=theo_hist,
        sd=dist.sd,
        prob_trigger=dist.prob_any,
        expected_days=dist.expected_days,
        var99=dist.var(0.99),
        cvar99=dist.cvar(0.99),
        max_payout=contract.max_payout,
        load_capital=load_capital,
        load_parameter=load_parameter,
        load_basis=load_basis,
        load_expense=load_expense,
        risk_premium=risk_premium,
        bid=float(max(bid, 0.0)),
        ask=float(max(ask, 0.0)),
        mid=float(mid),
        half_width=float(half_width),
        inventory_skew=float(inventory_skew),
        rate_on_line=float(mid / max(contract.max_payout, 1e-9)),
        # Guard the ratio: with a near-zero fair value the loading multiple is
        # meaningless rather than large (it reached 2e13 on a contract that never
        # triggers). Anything past 50x is reported as 0 and read as "not a
        # meaningful ratio" by the caller.
        loading_multiple=float(mid / theo) if theo > max(mid, 1.0) * 0.02 else 0.0,
        scenario_theos={k: float(v) for k, v in scenario_theos.items()},
    )
    return quote, dist


def _duration_factor(contract: Contract) -> float:
    """Expense scales with window length, but sublinearly --- most of the cost is
    origination and settlement, which a longer window does not multiply."""
    return float(np.clip(np.sqrt(contract.n_days / 90.0), 0.35, 2.0))
