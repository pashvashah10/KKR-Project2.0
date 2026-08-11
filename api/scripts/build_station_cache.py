"""Build the GHCN station bundle offline, so the server never needs NOAA to boot.

    python -m api.scripts.build_station_cache

Downloads `ghcnd-stations.txt` (11 MB) and `ghcnd-inventory.txt` (35 MB) from
NCEI, parses ~800k inventory rows down to the ~38k stations that carry
temperature and rainfall together, and writes `api/data/ghcn/stations_parsed.npz`
(a few hundred KB compressed).

Run this once on a machine with network access and commit the result. Every
deployment afterwards loads it in milliseconds with no egress at all, which
matters for two separate reasons: a sealed or air-gapped environment can still
match stations, and a NOAA outage cannot take the storefront down with it.

The bundle is a snapshot. Stations open and close, so it drifts --- rebuild it
when a venue matches to something that looks stale, or on a quarterly cron. It
is deliberately not auto-refreshed at runtime: a silent re-download that changes
which station a live contract settles on is exactly the kind of surprise this
codebase should not contain.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .. import stations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api.scripts.build_station_cache",
        description="Download and parse the GHCN-Daily station index into a numpy bundle.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=stations.CACHE_PATH,
        help=f"Where to write the bundle (default: {stations.CACHE_PATH}).",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep the downloaded .txt files instead of deleting them after parsing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if a bundle already exists at the output path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.output.exists() and not args.force:
        idx = stations._read_bundle(args.output)
        print(
            f"{args.output} already exists with {len(idx):,} stations. "
            f"Pass --force to rebuild."
        )
        return 0

    started = time.time()
    try:
        idx = stations.build_cache(dest=args.output, keep_raw=args.keep_raw)
    except stations.DataIngestionError as exc:
        print(f"\nBuild failed.\n{exc}\n", file=sys.stderr)
        print(
            "This script is the one place that is *supposed* to need network. "
            "Check that www.ncei.noaa.gov is reachable from here.",
            file=sys.stderr,
        )
        return 1

    size_kb = args.output.stat().st_size / 1024
    span = idx.last.astype(int) - idx.first.astype(int) + 1
    print(
        f"\nWrote {args.output} ({size_kb:,.0f} KB) in {time.time() - started:.1f}s\n"
        f"  stations with {'+'.join(stations.CORE_ELEMENTS)}: {len(idx):,}\n"
        f"  median record length: {int(sorted(span)[len(span) // 2])} years\n"
        f"  longest record:       {int(span.max())} years\n"
        f"\nCommit this file. The server will now boot with no NOAA dependency."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
