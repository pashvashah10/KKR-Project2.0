"""Seed the eight built-in venues as public example sites.

    python -m api.seed

### Why this exists

Fitting a century at a new coordinate is 80-140 seconds. Without example venues,
the first thing a visitor to `/configure` experiences is a two-minute wait before
any number appears, which is a terrible way to demonstrate a product whose whole
argument is that the numbers are real.

These eight are the locations the engine already ships with in
`downside.config.LOCATIONS`. Seeded here they become ordinary `sites` rows with
`is_example = 1`, fitted once, and readable by any visitor — so the configure
page can show a genuine engine-computed price in about two seconds while the
"add your own coordinates" path stays available beside it.

They are *examples*, not fixtures: they go through exactly the same station
match, the same fit, the same pricing as a customer venue. Nothing is
precomputed at build time and pasted in, which is what the earlier dashboard did
and what made it a showcase rather than a product.

Idempotent. Re-running matches on name and leaves anything already fitted alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from downside.config import LOCATIONS  # noqa: E402

from . import service, stations, store  # noqa: E402

log = logging.getLogger(__name__)

EXAMPLE_EMAIL = "examples@downside.local"
EXAMPLE_NAME = "Downside example venues"


def example_account() -> dict:
    account = store.account_by_email(EXAMPLE_EMAIL)
    if account is None:
        account = store.create_account(EXAMPLE_NAME, EXAMPLE_EMAIL)
    return account


def seed(fit: bool = True, only: str | None = None) -> list[dict]:
    """Create any missing example venues and fit them. Returns the site rows."""
    store.init()
    account = example_account()
    existing = {s["name"]: s for s in store.list_sites(account["id"])}

    out: list[dict] = []
    for loc in LOCATIONS:
        if only and only not in loc.id:
            continue

        site = existing.get(loc.name)
        if site is None:
            site = store.create_site(
                account["id"],
                name=loc.name,
                lat=loc.lat,
                lon=loc.lon,
                elevation_m=loc.elevation_m,
                vertical=loc.vertical,
                season_start_month=loc.season[0],
                season_end_month=loc.season[1],
                variable_cost_ratio=0.30,
                reserves=None,
                monthly_burn=None,
                is_example=1,
            )
            log.info("created example site %s (%s)", loc.name, site["id"])
        else:
            store.update_site(site["id"], is_example=1)

        # Re-match every run. The station index is a snapshot and a rebuild can
        # legitimately change which gauge wins; better to see that here than to
        # discover it inside a quote.
        match = stations.nearest_station(loc.lat, loc.lon, loc.elevation_m)
        if match is not None:
            store.update_site(site["id"], **service.station_columns(match))
            log.info(
                "  %s -> %s (%s) %.1f km, %d m delta%s",
                loc.name, match.station_id, match.name.strip(), match.distance_km,
                match.elevation_delta_m or 0,
                "  [MICROCLIMATE WARNING]" if match.elevation_warning else "",
            )
        else:
            log.warning("  %s -> no qualifying station; will use reanalysis", loc.name)

        current = store.get_site(site["id"]) or site
        if fit and current["status"] != "ready":
            job = store.create_job(site["id"], "fit")
            started = time.time()
            log.info("  fitting %s ...", loc.name)
            service.fit_site(site["id"], job["id"])
            current = store.get_site(site["id"]) or current
            log.info("  %s -> %s in %.0fs", loc.name, current["status"], time.time() - started)

        out.append(current)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api.seed",
        description="Seed and fit the eight built-in example venues.",
    )
    parser.add_argument("--no-fit", action="store_true",
                        help="Create the rows and match stations, but skip fitting.")
    parser.add_argument("--only", help="Substring of a location id, e.g. 'vail'.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sites = seed(fit=not args.no_fit, only=args.only)

    ready = sum(1 for s in sites if s["status"] == "ready")
    print(f"\n{ready}/{len(sites)} example venues ready.")
    for s in sites:
        flag = " [microclimate]" if s.get("station_elevation_warning") else ""
        print(
            f"  {s['status']:8s} {s['name']:26s} "
            f"{s.get('station_id') or 'ERA5':12s} "
            f"{(s.get('station_km') or 0):5.1f} km{flag}"
        )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
