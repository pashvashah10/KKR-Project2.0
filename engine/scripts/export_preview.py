"""Export a compact bundle of real engine output for the preview page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from downside.climatology import BASE_HI, BASE_LO  # noqa: E402
from downside.config import (  # noqa: E402
    LOCATIONS_BY_ID,
    PERILS_BY_ID,
    SCENARIOS,
    SCENARIOS_BY_ID,
)
from downside.hedging import fit_loss_curve, synthetic_revenue  # noqa: E402
from downside.projection import (  # noqa: E402
    project_variable,
    uncertainty_decomposition,
)
from downside.site import SiteModel  # noqa: E402
from downside.sources.synthetic import SyntheticSource  # noqa: E402
from downside.tails import fit_gpd  # noqa: E402

LOC_ID = "austin-tx"
SUMMER = np.arange(152, 244, dtype=float)


def main() -> None:
    loc = LOCATIONS_BY_ID[LOC_ID]
    record = SyntheticSource().fetch(loc, 1926, 2025)
    base = SCENARIOS_BY_ID["ssp245"]
    model = SiteModel.fit(loc, record, base, n_boot=200)

    out: dict = {
        "location": {
            "id": loc.id,
            "name": loc.name,
            "region": loc.region,
            "lat": loc.lat,
            "lon": loc.lon,
            "vertical": loc.vertical,
            "station": loc.station_id,
            "station_km": loc.station_distance_km,
        },
        "provenance": record.provenance,
    }

    # --- observed annual summer means + fitted climatology ----------------
    years = np.unique(record.year)
    sel_summer = np.isin(record.doy, SUMMER.astype(int))
    obs = []
    for y in years:
        m = sel_summer & (record.year == y)
        obs.append({"year": int(y), "tmax": round(float(record.tmax_c[m].mean()), 3)})
    out["observed_summer"] = obs

    g_hist = model.response.baseline_shift(years.astype(float) + 0.55, BASE_LO, BASE_HI)
    frozen = np.full(len(SUMMER), 2025.0)
    fitted = []
    for y, g in zip(years, g_hist):
        mu = model.tmax.predict_mean(SUMMER, np.full(len(SUMMER), g), frozen).mean()
        fitted.append({"year": int(y), "tmax": round(float(mu), 3), "global": round(float(g), 4)})
    out["fitted_summer"] = fitted

    # --- regression table --------------------------------------------------
    out["regression"] = {
        "rows": [
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
            for row in model.tmax.mean_fit.summary_rows()
        ],
        "r2": round(model.tmax.r_squared, 4),
        "n": int(model.tmax.mean_fit.n_obs),
        "amplification": round(model.tmax.amplification, 4),
        "amp_se_boot": round(model.tmax.amplification_se, 4),
        "amp_se_classical": round(model.tmax.amplification_se_classical, 4),
        "se_inflation": round(model.tmax.se_inflation, 2),
        "trend_f": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in model.tmax.trend_f.items()},
        "seasonal_f": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in model.tmax.seasonal_trend_f.items()
        },
        "ovb": {k: round(v, 4) if isinstance(v, float) else v for k, v in model.tmax.amplification_ovb.items()},
        "ar1": round(model.tmax.ar1, 4),
        "n_effective": round(model.tmax.n_effective, 1),
    }

    # --- 100-year projection under every scenario -------------------------
    fut = np.arange(2026, 2126, dtype=float)
    proj: dict = {}
    for sc in SCENARIOS:
        rows = project_variable(model.params, fut, SUMMER, sc)
        proj[sc.id] = [
            {
                "year": p.year,
                "mean": round(p.mean, 3),
                "lo": round(p.band()[0], 3),
                "hi": round(p.band()[1], 3),
            }
            for p in rows
        ]
    out["projection"] = proj
    out["scenarios"] = [
        {"id": s.id, "label": s.label, "short": s.short, "weight": s.weight, "description": s.description}
        for s in SCENARIOS
    ]

    base_rows = project_variable(model.params, fut, SUMMER, base)
    out["uncertainty"] = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        for row in uncertainty_decomposition(base_rows)
    ]

    # --- seasonal amplification curve --------------------------------------
    doy_grid = np.arange(1, 366, 5, dtype=float)
    amp_curve = model.tmax.seasonal_amplification(doy_grid)
    out["seasonal_amplification"] = [
        {"doy": int(d), "amp": round(float(a), 4)} for d, a in zip(doy_grid, amp_curve)
    ]

    # --- extreme value tail --------------------------------------------------
    wet = record.precip_mm[record.precip_mm > 0]
    gpd = fit_gpd(wet, threshold_quantile=0.95)
    rp = [2, 5, 10, 25, 50, 100, 250]
    out["return_levels"] = {
        "shape": round(gpd.shape, 4),
        "scale": round(gpd.scale, 4),
        "threshold": round(gpd.threshold, 3),
        "n_exceedances": gpd.n_exceedances,
        "rows": [{"period": p, "level": round(gpd.return_level(p), 2)} for p in rp],
        "observed_max": round(float(record.precip_mm.max()), 2),
    }

    # --- loss curve ----------------------------------------------------------
    revenue = synthetic_revenue(record, loc.vertical)
    lc = fit_loss_curve(record, revenue, "precip_mm", variable_cost_ratio=0.30)
    out["loss_curve"] = {
        "r2": round(lc.r_squared, 4),
        "breakpoint_mm": round(lc.breakpoint, 2),
        "baseline_margin": round(lc.baseline_margin, 0),
        "rows": [
            {"x": round(r["x"], 2), "loss": round(r["loss"], 0)}
            for r in lc.curve_rows()
            if r["x"] <= 60
        ],
        "segments": [
            {"from": round(s["from"], 1), "to": round(s["to"], 1), "slope": round(s["slope"], 0)}
            for s in lc.segment_slopes
        ],
        "synthetic_revenue": True,
    }

    # --- distribution of a peril's frequency, now vs 2075 -------------------
    peril = PERILS_BY_ID["extreme-heat"]
    freq = []
    for yr in (2026, 2050, 2075, 2100, 2125):
        row = {"year": yr}
        for sc in SCENARIOS:
            sim = model.simulate(yr, SUMMER, sc, n_paths=1200, rng=np.random.default_rng(yr))
            row[sc.id] = round(float(peril.triggered(sim.get(peril.variable)).sum(axis=1).mean()), 2)
        freq.append(row)
    out["heat_days"] = {
        "peril": peril.label,
        "threshold_c": round(peril.threshold, 2),
        "window": "Jun-Aug",
        "rows": freq,
    }

    dest = Path(__file__).resolve().parents[2] / "web" / "data" / "preview.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
