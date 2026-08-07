"""Build the dashboard data bundle from real station records."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from downside.backtest import assess_quality, detect_inhomogeneity, walk_forward  # noqa: E402
from downside.climatology import BASE_HI, BASE_LO  # noqa: E402
from downside.config import (  # noqa: E402
    LOCATIONS,
    PERILS_BY_ID,
    SCENARIOS,
    SCENARIOS_BY_ID,
    perils_for,
)
from downside.hedging import analyse_hedge, fit_loss_curve, synthetic_revenue  # noqa: E402
from downside.pricing import Contract, evaluate_payout, price_contract  # noqa: E402
from downside.projection import project_variable, uncertainty_decomposition  # noqa: E402
from downside.site import SiteModel  # noqa: E402
from downside.sources import load_record  # noqa: E402
from downside.tails import fit_gpd  # noqa: E402

BASE = SCENARIOS_BY_ID["ssp245"]
TARGET_YEARS = [2027, 2040, 2055, 2075, 2100, 2125]
N_PATHS = 2500
#: Paths per cell of the peril grid (8 perils x 4 scenarios x 6 years). This
#: dominates runtime, and the quantity being estimated is a mean day count over
#: a season, which converges far faster than a tail statistic.
N_PATHS_GRID = 500


def season_window(lo_month: int, hi_month: int) -> np.ndarray:
    starts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    lo = starts[lo_month - 1] + 1
    hi = starts[hi_month - 1] + [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][hi_month - 1]
    if hi >= lo:
        return np.arange(lo, hi + 1, dtype=float)
    return np.concatenate([np.arange(lo, 366, dtype=float), np.arange(1, hi + 1, dtype=float)])


def r3(x) -> float:
    return round(float(x), 3)


def build_location(loc, verbose: bool = True) -> dict:
    t0 = time.time()
    record = load_record(loc, 1926, 2025)
    model = SiteModel.fit(loc, record, BASE, n_boot=160)
    window = season_window(*loc.season)
    rng = np.random.default_rng(11)

    out: dict = {
        "id": loc.id,
        "name": loc.name,
        "region": loc.region,
        "lat": loc.lat,
        "lon": loc.lon,
        "elevation_m": loc.elevation_m,
        "vertical": loc.vertical,
        "station": loc.station_id,
        "station_km": loc.station_distance_km,
        "season": list(loc.season),
        "provenance": record.provenance,
        "coverage": {k: round(v, 3) for k, v in record.coverage.items()},
        "patched": [k for k, v in record.variable_source.items() if "not measured" in v],
    }

    # --- observed history -------------------------------------------------
    years = np.unique(record.year)
    in_season = np.isin(record.doy, window.astype(int))
    out["history"] = [
        {
            "year": int(y),
            "tmax": r3(record.tmax_c[in_season & (record.year == y)].mean()),
            "precip": r3(record.precip_mm[record.year == y].sum()),
            "snow": r3(record.snow_mm[record.year == y].sum()),
        }
        for y in years
    ]

    g_hist = model.response.baseline_shift(years.astype(float) + 0.5, BASE_LO, BASE_HI)
    frozen = np.full(len(window), 2025.0)
    out["fitted"] = [
        {"year": int(y), "tmax": r3(model.tmax.predict_mean(window, np.full(len(window), g), frozen).mean())}
        for y, g in zip(years, g_hist)
    ]

    # --- diagnostics --------------------------------------------------------
    # Reliability needs a threshold this site actually crosses. A fixed 32 degC
    # is meaningless at a ski resort in its November-April season --- every
    # forecast probability is zero and the curve collapses to a single point at
    # the origin. The 90th percentile of in-season maxima is exceeded often
    # enough everywhere to say something.
    in_season_tmax = record.tmax_c[np.isin(record.doy, window.astype(int))]
    rel_threshold = float(np.percentile(in_season_tmax, 90))
    bt = walk_forward(
        record, model.response, "tmax_c", split_year=1995, trigger_threshold=rel_threshold
    )
    inhom = detect_inhomogeneity(record, model.tmax)
    out["quality"] = assess_quality(model.tmax.amplification, bt, inhom)
    out["quality"]["inhomogeneity"] = inhom
    out["diagnostics"] = {
        "r2": r3(model.tmax.r_squared),
        "n_obs": int(model.tmax.mean_fit.n_obs),
        "amplification": r3(model.tmax.amplification),
        "amp_se": r3(model.tmax.amplification_se),
        "amp_se_classical": r3(model.tmax.amplification_se_classical),
        "se_inflation": r3(model.tmax.se_inflation),
        "ar1": r3(model.tmax.ar1),
        "trend_f": r3(model.tmax.trend_f["f"]),
        "trend_p": model.tmax.trend_f["p"],
        "seasonal_f": r3(model.tmax.seasonal_trend_f["f"]),
        "occurrence_r2": r3(model.occurrence.pseudo_r2),
        "intensity_r2": r3(model.intensity.pseudo_r2),
        "backtest": {
            "train": list(bt.train_years),
            "test": list(bt.test_years),
            "rmse": r3(bt.rmse),
            "bias": r3(bt.bias),
            "crps_skill": r3(bt.crps_skill),
            "coverage_90": r3(bt.coverage_90),
            "coverage_50": r3(bt.coverage_50),
            "calibrated": bt.calibrated,
            "pit_max_deviation": r3(bt.pit_max_deviation),
            "pit": bt.pit_histogram,
            "reliability": bt.reliability,
            "reliability_threshold": r3(rel_threshold),
        },
        "regression": [
            {k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in model.tmax.mean_fit.summary_rows()
        ],
    }

    # --- projection ----------------------------------------------------------
    fut = np.arange(2026, 2126, 2, dtype=float)
    out["projection"] = {
        sc.id: [
            {"year": p.year, "mean": r3(p.mean), "lo": r3(p.band()[0]), "hi": r3(p.band()[1])}
            for p in project_variable(model.params, fut, window, sc)
        ]
        for sc in SCENARIOS
    }
    out["uncertainty"] = [
        {k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
        for row in uncertainty_decomposition(project_variable(model.params, fut, window, BASE))
    ]

    # --- perils: trigger frequency over time ----------------------------------
    perils = perils_for(loc)
    peril_rows: list[dict] = []
    ambient: dict = {}

    # Simulate each (scenario, year) cell once and evaluate every peril against
    # it, rather than re-simulating per peril. The perils read different
    # variables off the same weather, so the old loop was paying for eight
    # identical simulations to answer eight questions about one.
    grid: dict = {}
    for sc in SCENARIOS:
        for yr in TARGET_YEARS:
            grid[(sc.id, yr)] = model.simulate(
                yr, window, sc, n_paths=N_PATHS_GRID,
                rng=np.random.default_rng(yr * 31 + (hash(sc.id) % 977)),
            )

    for peril in perils:
        # A `consecutive` peril (a heat wave, a run of shutdown days) already
        # reduces to one value per path --- did a long enough run occur --- so its
        # series is the *probability* of a qualifying spell, not a day count.
        # Every other statistic is evaluated per day and summed over the season.
        spell = peril.statistic == "consecutive"

        def measure(sim, p=peril, spell=spell):
            hit = p.triggered(sim.get(p.variable))
            return r3(hit.mean() if spell else hit.sum(axis=1).mean())

        series = {
            sc.id: [measure(grid[(sc.id, yr)]) for yr in TARGET_YEARS]
            for sc in SCENARIOS
        }
        peril_rows.append(
            {
                "id": peril.id,
                "label": peril.label,
                "variable": peril.variable,
                "unit": peril.unit,
                "threshold": r3(peril.threshold),
                "description": peril.description,
                "window_days": peril.window_days,
                "series": series,
                "measure": "probability" if spell else "days",
                "usable": record.usable(peril.variable),
            }
        )

    out["perils"] = peril_rows
    out["target_years"] = TARGET_YEARS

    # --- ambient weather state: what the canvas renders -----------------------
    # Mean daily conditions in the operating window, per year and scenario. This
    # is what drives the animated backdrop, so the motion on screen is the same
    # number as the charts rather than a decorative loop.
    for sc in SCENARIOS:
        rows = []
        for yr in TARGET_YEARS:
            sim = grid[(sc.id, yr)]
            wet = sim.precip_mm >= 0.254
            rows.append(
                {
                    "year": yr,
                    "tmax": r3(sim.tmax_c.mean()),
                    "tmin": r3(sim.tmin_c.mean()),
                    "precip": r3(sim.precip_mm.mean()),
                    "wet_frac": r3(wet.mean()),
                    "heavy_frac": r3((sim.precip_mm >= 8.0).mean()),
                    "snow": r3(sim.snow_mm.mean()),
                    "snow_frac": r3((sim.snow_mm >= 5.0).mean()),
                    "wind": r3(sim.wind_ms.mean()),
                    "windy_frac": r3((sim.wind_ms >= 8.0).mean()),
                    "hot_frac": r3((sim.tmax_c >= 32.0).mean()),
                }
            )
        ambient[sc.id] = rows
    out["ambient"] = ambient

    # --- tails ----------------------------------------------------------------
    wet_amounts = record.precip_mm[record.precip_mm > 0]
    gpd = fit_gpd(wet_amounts, threshold_quantile=0.95)
    out["tail"] = {
        "shape": r3(gpd.shape),
        "scale": r3(gpd.scale),
        "threshold": r3(gpd.threshold),
        "n_exceedances": gpd.n_exceedances,
        "observed_max": r3(record.precip_mm.max()),
        "levels": [{"period": p, "level": r3(gpd.return_level(p))} for p in (2, 5, 10, 25, 50, 100, 250)],
    }

    # --- loss curve + pricing + hedge -----------------------------------------
    revenue = synthetic_revenue(record, loc.vertical)

    # A realistic contract: aggregate cover with an attachment, so it pays for a
    # genuinely bad season rather than for the ordinary friction the business
    # already absorbs.
    #
    # The peril is chosen by how often it actually fires, not by list order.
    # Taking the first usable peril gave Vail a high-wind contract that triggers
    # on 0.0% of seasons --- fair value zero, yet still quoted at $20k because the
    # expense load does not depend on fair value. A contract that cannot pay is
    # not a product, and quoting one is the most embarrassing possible output.
    # Only per-day perils are eligible for the aggregate structure, which counts
    # independent triggering days beyond an attachment. A rolling-window statistic
    # like snow drought does not produce independent days: every day inside a dry
    # spell registers, so Vail scored 174 "days", took an attachment of 174, and
    # then could never reach it inside a 181-day season. Selling a seasonal
    # accumulation peril properly needs a different contract shape.
    eligible = [
        p for p in perils if record.usable(p.variable) and p.statistic == "daily"
    ]
    scored = [
        (
            float(p.triggered(grid[(BASE.id, TARGET_YEARS[0])].get(p.variable)).sum(axis=1).mean()),
            p,
        )
        for p in eligible
    ]
    scored.sort(key=lambda t: -t[0])
    typical_days, primary = scored[0] if scored else (0.0, None)

    if primary is None or typical_days < 0.5:
        # Nothing at this site triggers often enough to underwrite.
        out["contract"] = None
        out["hedge"] = None
        out["loss_curve"] = None
        best = primary.label if primary else "none eligible"
        if verbose:
            print(f"  {loc.id:14s} no sellable peril (best {best} at "
                  f"{typical_days:.2f} days/season)", flush=True)
        return out

    # The loss curve is fitted on the *contract's own* settlement variable.
    #
    # Fitting it on rainfall regardless of the peril made the hedge meaningless:
    # a rain-driven loss against a heat-driven payout correlated at rho = -0.01,
    # so basis risk read 100% and no notional could reduce ruin. The operator's
    # loss and the contract's trigger have to be measured on the same quantity
    # before a hedge ratio means anything.
    lc = fit_loss_curve(record, revenue, primary.variable, variable_cost_ratio=0.30)
    x_cap = 55 if primary.variable == "precip_mm" else 1e9
    out["loss_curve"] = {
        "variable": primary.variable,
        "unit": primary.unit,
        "r2": r3(lc.r_squared),
        "breakpoint": r3(lc.breakpoint),
        "baseline": r3(lc.baseline_margin),
        "rows": [{"x": r3(r["x"]), "loss": r3(r["loss"])} for r in lc.curve_rows() if r["x"] <= x_cap],
        "segments": [
            {"from": r3(s["from"]), "to": r3(s["to"]), "slope": r3(s["slope"])} for s in lc.segment_slopes
        ],
    }

    attach = int(max(np.ceil(typical_days * 1.25), 1))
    per_day = round(max(lc.baseline_margin, 1000.0) * 0.55, -2) or 5000.0

    quotes = []
    for yr in TARGET_YEARS:
        contract = Contract(
            id=f"{loc.id}-{primary.id}-{yr}",
            location_id=loc.id,
            peril_id=primary.id,
            year=yr,
            doy_start=int(window[0]),
            doy_end=int(window[-1]),
            structure="aggregate",
            payout_per_day=per_day,
            limit=per_day * 12,
            attachment_days=attach,
        )
        q, dist = price_contract(
            contract, primary, model, loc, BASE, SCENARIOS,
            n_paths=N_PATHS, rng=np.random.default_rng(yr),
        )
        quotes.append(
            {
                "year": yr,
                "theo": r3(q.theo),
                "theo_hist": r3(q.theo_historical),
                "mid": r3(q.mid),
                "bid": r3(q.bid),
                "ask": r3(q.ask),
                "prob": r3(q.prob_trigger),
                "expected_days": r3(q.expected_days),
                "cvar99": r3(q.cvar99),
                "max_payout": r3(q.max_payout),
                "rate_on_line": r3(q.rate_on_line),
                "loading": r3(q.loading_multiple),
                "loads": {
                    "capital": r3(q.load_capital),
                    "parameter": r3(q.load_parameter),
                    "basis": r3(q.load_basis),
                    "expense": r3(q.load_expense),
                },
            }
        )
    out["contract"] = {
        "peril": primary.label,
        "peril_id": primary.id,
        "structure": "aggregate",
        "attachment_days": attach,
        "payout_per_day": per_day,
        "limit": per_day * 12,
        "quotes": quotes,
    }

    # --- hedge frontier --------------------------------------------------------
    sim_h = model.simulate(TARGET_YEARS[0], window, BASE, n_paths=N_PATHS, rng=np.random.default_rng(99))
    loss = lc.loss_at(sim_h.get(primary.variable)).sum(axis=1)
    contract0 = Contract(
        id="hedge", location_id=loc.id, peril_id=primary.id, year=TARGET_YEARS[0],
        doy_start=int(window[0]), doy_end=int(window[-1]), structure="aggregate",
        payout_per_day=per_day, limit=per_day * 12, attachment_days=attach,
    )
    payout = evaluate_payout(contract0, primary, sim_h).payouts

    # Hedge against the *unexpected* shortfall, not the whole weather loss.
    #
    # `loss_at` measures margin lost against a dry-day baseline, so summing it
    # over a season gives everything rain costs across the year. A business
    # already plans around a normal amount of rain; that figure is in its budget,
    # not a threat to its survival. Feeding the gross number in put ruin
    # probability at 94.9% before any hedge and recommended size at zero, because
    # no hedge could save a business that is ruined in the median season.
    #
    # What threatens solvency is a season materially worse than a normal one, so
    # ruin is measured on the excess over the median season.
    normal_season = float(np.median(loss))
    excess = loss - normal_season

    # Demo balance sheet. Real reserves and fixed burn are operator inputs and
    # are the two numbers that most change the recommended hedge.
    reserves = float(np.percentile(excess, 90) * 2.0)
    burn = reserves / 12.0
    ha = analyse_hedge(excess, payout, price=quotes[0]["ask"], reserves=reserves,
                       fixed_burn_monthly=burn)
    out["hedge"] = {
        "reserves": r3(reserves),
        "burn_monthly": r3(burn),
        "normal_season_loss": r3(normal_season),
        "loss_mean": r3(excess.mean()),
        "loss_p95": r3(np.percentile(excess, 95)),
        "correlation": r3(ha.correlation),
        "h_star": r3(ha.min_variance_ratio),
        "basis_risk": r3(ha.basis_risk_share),
        "kelly": r3(ha.kelly_fraction),
        "recommended": r3(ha.recommended_fraction),
        "ruin_unhedged": r3(ha.unhedged_ruin),
        "ruin_hedged": r3(ha.hedged_ruin),
        "frontier": [
            {k: r3(v) for k, v in row.items()} for row in ha.frontier[::2]
        ],
    }

    if verbose:
        flag = "" if out["quality"]["usable"] else f"  [{out['quality']['verdict'].upper()}]"
        print(f"  {loc.id:14s} {record.provenance:16s} {time.time() - t0:5.1f}s{flag}", flush=True)
    return out


def _safe_build(loc):
    try:
        return build_location(loc)
    except Exception as exc:  # noqa: BLE001
        print(f"  {loc.id:14s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenarios": [
            {"id": s.id, "label": s.label, "short": s.short, "description": s.description, "weight": s.weight}
            for s in SCENARIOS
        ],
        "locations": [],
    }

    # Sites are independent, so fan them across cores. Each worker does its own
    # network fetch, which is cached on disk after the first run.
    import multiprocessing as mp

    workers = min(4, len(LOCATIONS))
    with mp.Pool(workers) as pool:
        results = pool.map(_safe_build, list(LOCATIONS))

    order = {loc.id: i for i, loc in enumerate(LOCATIONS)}
    payload["locations"] = sorted(
        [r for r in results if r], key=lambda r: order.get(r["id"], 99)
    )

    dest = Path(__file__).resolve().parents[2] / "web" / "data" / "dashboard.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB, {len(payload['locations'])} locations)")


if __name__ == "__main__":
    main()
