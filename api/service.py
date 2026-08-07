"""Domain layer: run the engine for a customer's own site, and cache the result.

Everything expensive happens here, once, in a background thread. The API layer
above is thin on purpose.

Two things this module exists to solve:

**Fitting is slow.** A century of daily observations, a mean and variance model,
a 200-replicate block bootstrap, tail fits, and a Monte Carlo grid come to
80-140 seconds per site. That cannot happen inside a request. So `fit_site` runs
on a worker thread, writes progress as it goes, and pickles the fitted model. Every
later call --- a quote, a hedge, a different target year --- loads that artefact and
returns in milliseconds.

**A customer's site is not one of ours.** The dashboard shipped with eight
locations chosen at build time. Here the coordinates, the season and the vertical
come from the customer, a `Location` is constructed from them, and the amplification
prior is derived from latitude rather than looked up in a table.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass

import numpy as np

from . import store

log = logging.getLogger(__name__)

# The engine lives alongside this package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from downside.backtest import assess_quality, detect_inhomogeneity, walk_forward  # noqa: E402
from downside.climatology import BASE_HI, BASE_LO  # noqa: E402
from downside.config import (  # noqa: E402
    PERILS,
    PERILS_BY_ID,
    SCENARIOS,
    SCENARIOS_BY_ID,
    ClimateNormals,
    Location,
    Peril,
)
from downside.hedging import analyse_hedge, fit_loss_curve, synthetic_revenue  # noqa: E402
from downside.pricing import Contract, evaluate_payout, price_contract  # noqa: E402
from downside.projection import project_variable, uncertainty_decomposition  # noqa: E402
from downside.site import SiteModel  # noqa: E402
from downside.sources import load_record  # noqa: E402
from downside.tails import fit_gpd  # noqa: E402

BASE_SCENARIO = SCENARIOS_BY_ID["ssp245"]
HISTORY_START, HISTORY_END = 1926, 2025
TARGET_YEARS = [2027, 2035, 2045, 2060, 2080, 2100, 2125]


def r3(x) -> float:
    return round(float(x), 3)


# ----------------------------------------------------------------------
# Turning a customer's inputs into something the engine can fit
# ----------------------------------------------------------------------


def amplification_prior_for(lat: float, lon: float, elevation_m: float) -> float:
    """Prior on local warming per degree of global warming.

    The shipped dashboard read this from a table of eight hand-tuned sites, which
    obviously does not generalise to a customer's coordinates. The physical
    regularities it stands in for are real and well established, so the prior is
    reconstructed from them:

    * warming increases with latitude (polar amplification), strongly so past 40N;
    * continental interiors warm faster than maritime margins, because the ocean
      buffers a coastline;
    * elevation adds a little through snow-albedo feedback in the mountains.

    It is only a prior. `forcing.shrink_amplification` weighs it against what the
    station's own record says, so a site with a long clean history barely feels it.
    """
    lat = abs(lat)
    base = 1.02 + 0.014 * max(lat - 30.0, 0.0)

    # Crude continentality: distance from the nearest US coast longitude band.
    # A real implementation would use a coastline distance raster; this captures
    # the first-order effect without one.
    coastal_west = abs(lon + 122.0)
    coastal_east = abs(lon + 76.0)
    inland = min(coastal_west, coastal_east)
    base *= 1.0 + 0.010 * min(inland, 22.0)

    base *= 1.0 + 0.00006 * max(elevation_m - 300.0, 0.0)
    return float(np.clip(base, 0.6, 1.9))


def build_location(site: dict) -> Location:
    """A `Location` for the engine, from a customer's site record.

    `normals` is only used by the surrogate generator, which is the last-resort
    source. When NOAA or ERA5 answers --- the normal case --- these values are
    never read.
    """
    lat, lon = float(site["lat"]), float(site["lon"])
    elev = float(site.get("elevation_m") or 0.0)
    return Location(
        id=site["id"],
        name=site["name"],
        region=f"{lat:.2f},{lon:.2f}",
        lat=lat,
        lon=lon,
        elevation_m=elev,
        station_id=site.get("station_id") or "",
        station_distance_km=float(site.get("station_km") or 0.0),
        vertical=site["vertical"],
        amplification_prior=amplification_prior_for(lat, lon, elev),
        observed_trend_c_per_century=1.3,
        season=(int(site["season_start_month"]), int(site["season_end_month"])),
        normals=_placeholder_normals(lat),
    )


def _placeholder_normals(lat: float) -> ClimateNormals:
    """Latitude-driven normals, used only if every real source is unreachable."""
    warm = 30.0 - 0.42 * abs(lat)
    cold = warm - 11.0
    amp = 6.0 + 0.22 * abs(lat)
    months = np.arange(12)
    phase = np.cos(2 * np.pi * (months - 6.5) / 12.0)
    return ClimateNormals(
        tmax_c=tuple(warm - amp * phase),
        tmin_c=tuple(cold - amp * phase),
        precip_mm=tuple(np.full(12, 80.0)),
        wet_days=tuple(np.full(12, 9.0)),
        wind_ms=tuple(np.full(12, 4.0)),
    )


def season_window(lo_month: int, hi_month: int) -> np.ndarray:
    starts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    lo = starts[lo_month - 1] + 1
    hi = starts[hi_month - 1] + lengths[hi_month - 1]
    if hi >= lo:
        return np.arange(lo, hi + 1, dtype=float)
    return np.concatenate([np.arange(lo, 366, dtype=float), np.arange(1, hi + 1, dtype=float)])


# ----------------------------------------------------------------------
# Model cache
# ----------------------------------------------------------------------


def model_path(site_id: str) -> Path:
    return store.MODEL_DIR / f"{site_id}.pkl"


def save_model(site_id: str, model: SiteModel) -> None:
    store.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path(site_id), "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(site_id: str) -> SiteModel | None:
    p = model_path(site_id)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        log.warning("could not load cached model for %s: %s", site_id, exc)
        return None


# ----------------------------------------------------------------------
# The fitting job
# ----------------------------------------------------------------------


@dataclass
class Progress:
    job_id: str

    def __call__(self, fraction: float, step: str) -> None:
        store.update_job(self.job_id, progress=round(fraction, 3), step=step, status="running")


def fit_site(site_id: str, job_id: str) -> None:
    """Fetch, fit, analyse and cache. Runs on a worker thread."""
    tick = Progress(job_id)
    started = time.time()
    try:
        site = store.get_site(site_id)
        if site is None:
            raise ValueError(f"site {site_id} not found")

        tick(0.05, "Locating the nearest long-record weather station")
        location = build_location(site)
        record = load_record(location, HISTORY_START, HISTORY_END)

        store.update_site(
            site_id,
            provenance=record.provenance,
            station_id=getattr(record, "location_id", "") or location.station_id,
        )

        tick(0.25, f"Fitting {record.n_years} years of daily observations")
        model = SiteModel.fit(location, record, BASE_SCENARIO, n_boot=160)
        save_model(site_id, model)

        tick(0.55, "Validating against thirty held-out years")
        window = season_window(*location.season)
        in_season = record.tmax_c[np.isin(record.doy, window.astype(int))]
        rel_threshold = float(np.percentile(in_season, 90))
        bt = walk_forward(
            record, model.response, "tmax_c", split_year=1995, trigger_threshold=rel_threshold
        )
        inhom = detect_inhomogeneity(record, model.tmax)
        quality = assess_quality(model.tmax.amplification, bt, inhom)
        quality["inhomogeneity"] = inhom

        tick(0.70, "Projecting to 2125 under four emissions pathways")
        store.put_analysis(site_id, "outlook", _outlook(model, window))
        store.put_analysis(site_id, "history", _history(record, model, window))
        store.put_analysis(
            site_id, "diagnostics", _diagnostics(model, bt, record, rel_threshold)
        )

        tick(0.88, "Simulating peril frequencies")
        store.put_analysis(site_id, "perils", _perils(model, record, window, location))
        store.put_analysis(site_id, "tail", _tail(record))

        store.update_site(
            site_id,
            status="ready",
            quality=__import__("json").dumps(quality),
            fitted_at=time.time(),
        )
        store.update_job(
            job_id, status="succeeded", progress=1.0, step="ready", finished_at=time.time()
        )
        log.info("fitted %s in %.1fs", site_id, time.time() - started)

    except Exception as exc:  # noqa: BLE001
        log.exception("fit failed for %s", site_id)
        store.update_site(site_id, status="failed")
        store.update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )


# ----------------------------------------------------------------------
# Analysis artefacts
# ----------------------------------------------------------------------


def _history(record, model, window) -> dict:
    years = np.unique(record.year)
    in_season = np.isin(record.doy, window.astype(int))
    g = model.response.baseline_shift(years.astype(float) + 0.5, BASE_LO, BASE_HI)
    frozen = np.full(len(window), 2025.0)
    return {
        "observed": [
            {
                "year": int(y),
                "tmax": r3(record.tmax_c[in_season & (record.year == y)].mean()),
                "precip": r3(record.precip_mm[record.year == y].sum()),
                "snow": r3(record.snow_mm[record.year == y].sum()),
            }
            for y in years
        ],
        "fitted": [
            {"year": int(y), "tmax": r3(model.tmax.predict_mean(window, np.full(len(window), gi), frozen).mean())}
            for y, gi in zip(years, g)
        ],
        "coverage": {k: r3(v) for k, v in record.coverage.items()},
        "substituted": [k for k, v in record.variable_source.items() if "not measured" in v],
    }


def _outlook(model, window) -> dict:
    fut = np.arange(2026, 2126, 2, dtype=float)
    return {
        "years": [int(y) for y in fut],
        "scenarios": {
            sc.id: [
                {"year": p.year, "mean": r3(p.mean), "lo": r3(p.band()[0]), "hi": r3(p.band()[1])}
                for p in project_variable(model.params, fut, window, sc)
            ]
            for sc in SCENARIOS
        },
        "uncertainty": [
            {k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in uncertainty_decomposition(project_variable(model.params, fut, window, BASE_SCENARIO))
        ],
    }


def _perils(model, record, window, location) -> dict:
    from downside.config import perils_for

    applicable = [p for p in perils_for(location) if record.usable(p.variable)]
    grid = {
        (sc.id, yr): model.simulate(
            yr, window, sc, n_paths=500, rng=np.random.default_rng(yr * 31 + (hash(sc.id) % 977))
        )
        for sc in SCENARIOS
        for yr in TARGET_YEARS
    }
    rows = []
    for p in applicable:
        spell = p.statistic == "consecutive"

        def measure(sim, p=p, spell=spell):
            hit = p.triggered(sim.get(p.variable))
            return r3(hit.mean() if spell else hit.sum(axis=1).mean())

        rows.append(
            {
                "id": p.id,
                "label": p.label,
                "variable": p.variable,
                "unit": p.unit,
                "threshold": r3(p.threshold),
                "description": p.description,
                "measure": "probability" if spell else "days",
                "statistic": p.statistic,
                "window_days": p.window_days,
                "series": {
                    sc.id: [measure(grid[(sc.id, yr)]) for yr in TARGET_YEARS] for sc in SCENARIOS
                },
            }
        )
    return {"target_years": TARGET_YEARS, "perils": rows}


def _tail(record) -> dict:
    wet = record.precip_mm[record.precip_mm > 0]
    gpd = fit_gpd(wet, threshold_quantile=0.95)
    return {
        "shape": r3(gpd.shape),
        "scale": r3(gpd.scale),
        "threshold": r3(gpd.threshold),
        "observed_max": r3(record.precip_mm.max()),
        "levels": [{"period": p, "level": r3(gpd.return_level(p))} for p in (2, 5, 10, 25, 50, 100, 250)],
    }


def _diagnostics(model, bt, record, rel_threshold) -> dict:
    return {
        "r2": r3(model.tmax.r_squared),
        "n_obs": int(model.tmax.mean_fit.n_obs),
        "amplification": r3(model.tmax.amplification),
        "amp_se": r3(model.tmax.amplification_se),
        "amp_se_classical": r3(model.tmax.amplification_se_classical),
        "se_inflation": r3(model.tmax.se_inflation),
        "ar1": r3(model.tmax.ar1),
        "backtest": {
            "train": list(bt.train_years),
            "test": list(bt.test_years),
            "rmse": r3(bt.rmse),
            "bias": r3(bt.bias),
            "crps_skill": r3(bt.crps_skill),
            "coverage_90": r3(bt.coverage_90),
            "coverage_50": r3(bt.coverage_50),
            "calibrated": bt.calibrated,
            "pit": bt.pit_histogram,
            "reliability": bt.reliability,
            "reliability_threshold": r3(rel_threshold),
        },
        "regression": [
            {k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in model.tmax.mean_fit.summary_rows()
        ],
    }


# ----------------------------------------------------------------------
# On-demand: exposure, quotes, hedging
# ----------------------------------------------------------------------


def revenue_series(site_id: str, record) -> tuple[np.ndarray, bool]:
    """The customer's own daily revenue, aligned to the weather record.

    Returns `(series, is_real)`. Where the customer has uploaded revenue it is
    matched by date and the rest of the record is left at zero and masked out of
    the fit. Where they have not, a placeholder is generated and flagged --- the
    loss curve is only as real as the revenue behind it, and the caller is
    expected to say so.
    """
    uploaded = dict(store.get_revenue(site_id))
    if len(uploaded) < 180:
        return synthetic_revenue(record, "Outdoor attraction"), False

    keys = np.datetime_as_string(record.dates, unit="D")
    series = np.array([uploaded.get(k, np.nan) for k in keys], dtype=float)
    return series, True


def exposure(site_id: str) -> dict:
    site = store.get_site(site_id)
    model = load_model(site_id)
    if site is None or model is None:
        raise LookupError("site not fitted")

    record = model.record
    revenue, is_real = revenue_series(site_id, record)
    mask = np.isfinite(revenue)
    if mask.sum() < 180:
        mask = np.ones(len(record), dtype=bool)
        revenue = np.nan_to_num(revenue)

    window = season_window(int(site["season_start_month"]), int(site["season_end_month"]))
    primary = _primary_peril(model, record, window, build_location(site))
    lc = fit_loss_curve(
        record,
        np.nan_to_num(revenue),
        primary.variable if primary else "precip_mm",
        variable_cost_ratio=float(site["variable_cost_ratio"]),
        mask=mask,
    )

    sim = model.simulate(TARGET_YEARS[0], window, BASE_SCENARIO, n_paths=2500,
                         rng=np.random.default_rng(7))
    seasonal_loss = lc.loss_at(sim.get(lc.variable)).sum(axis=1)

    return {
        "revenue_is_real": is_real,
        "revenue": store.revenue_summary(site_id) if is_real else None,
        "variable": lc.variable,
        "r2": r3(lc.r_squared),
        "breakpoint": r3(lc.breakpoint),
        "baseline_daily_margin": r3(lc.baseline_margin),
        "curve": [{"x": r3(r["x"]), "loss": r3(r["loss"])} for r in lc.curve_rows()],
        "segments": [
            {"from": r3(s["from"]), "to": r3(s["to"]), "slope": r3(s["slope"])}
            for s in lc.segment_slopes
        ],
        "expected_seasonal_loss": r3(seasonal_loss.mean()),
        "median_seasonal_loss": r3(np.median(seasonal_loss)),
        "p95_seasonal_loss": r3(np.percentile(seasonal_loss, 95)),
        "p99_seasonal_loss": r3(np.percentile(seasonal_loss, 99)),
        "primary_peril": primary.id if primary else None,
    }


#: A trigger that fires on more than this share of the season is describing the
#: local climate, not a peril. Charleston picked "warm snowline" --- base-area
#: temperature above 4 degC --- because in a subtropical spring it fires almost
#: every day and therefore scored highest on raw frequency.
MAX_TRIGGER_SHARE = 0.45


def _primary_peril(model, record, window, location) -> Peril | None:
    """The peril most worth underwriting at this site.

    Three filters, and every one of them exists because leaving it out produced
    a wrong answer:

    * **applicable to the vertical** --- a waterfront venue in South Carolina has
      no use for a snowline trigger, however often it fires;
    * **per-day statistic** --- aggregate cover counts independent triggering days,
      which a rolling-window peril does not produce;
    * **not ubiquitous** --- something that fires most days is the climate, and
      cover against it is a transfer with a premium attached.
    """
    from downside.config import perils_for

    sim = model.simulate(TARGET_YEARS[0], window, BASE_SCENARIO, n_paths=800,
                         rng=np.random.default_rng(3))
    season_days = len(window)
    best, best_days = None, 0.0
    for p in perils_for(location):
        if p.statistic != "daily" or not record.usable(p.variable):
            continue
        days = float(p.triggered(sim.get(p.variable)).sum(axis=1).mean())
        if days > season_days * MAX_TRIGGER_SHARE:
            continue
        if days > best_days:
            best, best_days = p, days
    return best if best_days >= 0.5 else None


def quote(site_id: str, peril_id: str, year: int, payout_per_day: float,
          limit: float, attachment_days: int, scenario_id: str = "ssp245") -> dict:
    site = store.get_site(site_id)
    model = load_model(site_id)
    if site is None or model is None:
        raise LookupError("site not fitted")
    peril = PERILS_BY_ID.get(peril_id)
    if peril is None:
        raise ValueError(f"unknown peril {peril_id}")

    window = season_window(int(site["season_start_month"]), int(site["season_end_month"]))
    scenario = SCENARIOS_BY_ID.get(scenario_id, BASE_SCENARIO)
    contract = Contract(
        id=f"{site_id}-{peril_id}-{year}",
        location_id=site_id,
        peril_id=peril_id,
        year=year,
        doy_start=int(window[0]),
        doy_end=int(window[-1]),
        structure="aggregate",
        payout_per_day=payout_per_day,
        limit=limit,
        attachment_days=attachment_days,
    )
    location = build_location(site)
    q, dist = price_contract(
        contract, peril, model, location, scenario, SCENARIOS,
        n_paths=2500, rng=np.random.default_rng(year),
    )
    body = {
        **q.to_dict(),
        "peril": peril.label,
        "peril_id": peril.id,
        "year": year,
        "structure": "aggregate",
        "attachment_days": attachment_days,
        "payout_per_day": payout_per_day,
        "limit": limit,
        "payout_quantiles": dist.quantiles(),
    }
    store.save_quote(
        site_id, peril_id, year,
        {"payout_per_day": payout_per_day, "limit": limit,
         "attachment_days": attachment_days, "scenario": scenario_id},
        body,
    )
    return body


def hedge(site_id: str, peril_id: str, year: int, payout_per_day: float, limit: float,
          attachment_days: int, reserves: float, monthly_burn: float) -> dict:
    site = store.get_site(site_id)
    model = load_model(site_id)
    if site is None or model is None:
        raise LookupError("site not fitted")
    peril = PERILS_BY_ID[peril_id]

    window = season_window(int(site["season_start_month"]), int(site["season_end_month"]))
    record = model.record
    revenue, is_real = revenue_series(site_id, record)
    mask = np.isfinite(revenue)
    if mask.sum() < 180:
        mask = np.ones(len(record), dtype=bool)
    lc = fit_loss_curve(
        record, np.nan_to_num(revenue), peril.variable,
        variable_cost_ratio=float(site["variable_cost_ratio"]), mask=mask,
    )

    sim = model.simulate(year, window, BASE_SCENARIO, n_paths=2500, rng=np.random.default_rng(99))
    loss = lc.loss_at(sim.get(peril.variable)).sum(axis=1)
    contract = Contract(
        id="hedge", location_id=site_id, peril_id=peril_id, year=year,
        doy_start=int(window[0]), doy_end=int(window[-1]), structure="aggregate",
        payout_per_day=payout_per_day, limit=limit, attachment_days=attachment_days,
    )
    payout = evaluate_payout(contract, peril, sim).payouts
    priced = quote(site_id, peril_id, year, payout_per_day, limit, attachment_days)

    # Ruin is measured on the shortfall against a normal season, not the gross
    # weather loss --- a business already budgets for ordinary weather.
    normal = float(np.median(loss))
    excess = loss - normal

    result = analyse_hedge(excess, payout, price=priced["ask"],
                           reserves=reserves, fixed_burn_monthly=monthly_burn)

    sensitivity = []
    for mult in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5):
        r_i = float(np.percentile(excess, 90) * mult)
        h_i = analyse_hedge(excess, payout, price=priced["ask"],
                            reserves=r_i, fixed_burn_monthly=r_i / 12.0)
        sensitivity.append({
            "reserves": r3(r_i), "recommended": r3(h_i.recommended_fraction),
            "kelly": r3(h_i.kelly_fraction), "ruin_unhedged": r3(h_i.unhedged_ruin),
            "ruin_hedged": r3(h_i.hedged_ruin),
        })

    return {
        "revenue_is_real": is_real,
        "normal_season_loss": r3(normal),
        "correlation": r3(result.correlation),
        "min_variance_ratio": r3(result.min_variance_ratio),
        "basis_risk": r3(result.basis_risk_share),
        "kelly": r3(result.kelly_fraction),
        "recommended": r3(result.recommended_fraction),
        "ruin_unhedged": r3(result.unhedged_ruin),
        "ruin_hedged": r3(result.hedged_ruin),
        "premium": r3(result.expected_cost),
        "reserves": r3(reserves),
        "monthly_burn": r3(monthly_burn),
        "frontier": [{k: r3(v) for k, v in row.items()} for row in result.frontier[::2]],
        "sensitivity": sensitivity,
        "quote": priced,
    }

def suggest_contract(site_id: str, year: int = TARGET_YEARS[0]) -> dict:
    """Propose contract terms from the site's own exposure.

    A customer should not have to guess an attachment point. Guessing low is the
    common mistake and it produces a contract that pays in most seasons --- at a
    45% rate on line that is a financing arrangement with a premium attached, not
    insurance. Attachment is therefore set from the *distribution* of triggering
    days: at roughly the 80th percentile, so the contract responds to a bad
    season and stays quiet in an ordinary one.
    """
    site = store.get_site(site_id)
    model = load_model(site_id)
    if site is None or model is None:
        raise LookupError("site not fitted")

    window = season_window(int(site["season_start_month"]), int(site["season_end_month"]))
    location = build_location(site)
    record = model.record
    primary = _primary_peril(model, record, window, location)
    if primary is None:
        return {
            "available": False,
            "reason": "No peril at this site triggers often enough to underwrite.",
        }

    sim = model.simulate(year, window, BASE_SCENARIO, n_paths=2000,
                         rng=np.random.default_rng(5))
    days = primary.triggered(sim.get(primary.variable)).sum(axis=1)

    exposure_now = exposure(site_id)
    per_day = round(max(exposure_now["baseline_daily_margin"], 500.0) * 0.55, -2)
    attach = int(np.percentile(days, 80)) + 1
    cover_days = max(int(np.percentile(days, 99)) - attach, 3)

    return {
        "available": True,
        "peril_id": primary.id,
        "peril": primary.label,
        "year": year,
        "attachment_days": attach,
        "payout_per_day": per_day,
        "limit": per_day * cover_days,
        "rationale": (
            f"In a typical {year} season this site sees {np.median(days):.0f} "
            f"{primary.label.lower()} days. Attaching at {attach} means the contract "
            f"stays quiet in an ordinary year and responds to roughly the worst one "
            f"season in five."
        ),
        "expected_days": r3(days.mean()),
        "median_days": r3(float(np.median(days))),
        "p95_days": r3(float(np.percentile(days, 95))),
    }
