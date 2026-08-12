"""Package the eight example venues into a committed snapshot.

    python -m api.scripts.make_demo_snapshot

### Why this exists

A fresh deployment starts with an empty database, and the only way to fill it
was `api.seed` --- 5 to 20 minutes of fitting against NOAA, on a host that may
have a short startup timeout, limited memory and no outbound network. That made
the difference between "clone it" and "look at it" a fifteen-minute wait, which
is why the storefront had no public URL.

The snapshot removes the wait entirely. It carries a pruned database plus the
fitted model for each example venue, so a new instance restores in a second or
two and serves real, engine-computed numbers immediately, with no NOAA
dependency at boot.

### What it deliberately leaves out

Only the example account and its venues. Customer accounts, orders,
subscriptions, quotes and uploaded revenue are all excluded --- a demo fixture
committed to a public repository must not carry anyone's data, and the smaller
it is the faster a deploy comes up.

Rebuild it after re-seeding, or when the station bundle changes which gauge a
venue settles on.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

from .. import store

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "demo" / "snapshot.tar.gz"

#: Copied wholesale for the example account only.
SITE_TABLES = ("analyses",)


def build(dest: Path = SNAPSHOT) -> Path:
    src = store.DB_PATH
    if not src.exists():
        raise SystemExit(f"No database at {src}. Run `python -m api.seed` first.")

    with store.connect() as conn:
        account = conn.execute(
            "SELECT * FROM accounts WHERE email = ?", ("examples@downside.local",)
        ).fetchone()
        if account is None:
            raise SystemExit("No example account. Run `python -m api.seed` first.")
        sites = conn.execute(
            "SELECT * FROM sites WHERE account_id = ? AND is_example = 1 AND status = 'ready'",
            (account["id"],),
        ).fetchall()

    if not sites:
        raise SystemExit("No fitted example venues. Run `python -m api.seed` first.")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        out_db = tmpdir / "downside.db"
        models = tmpdir / "models"
        models.mkdir()

        # A fresh database with the current schema, rather than a copy with rows
        # deleted --- deletion leaves the customer data recoverable in free pages.
        fresh = sqlite3.connect(out_db)
        fresh.row_factory = sqlite3.Row
        fresh.executescript(store.SCHEMA)

        cols = [c[1] for c in fresh.execute("PRAGMA table_info(accounts)")]
        fresh.execute(
            f"INSERT INTO accounts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(account[c] for c in cols),
        )

        site_cols = [c[1] for c in fresh.execute("PRAGMA table_info(sites)")]
        kept = 0
        for site in sites:
            keys = site.keys()
            fresh.execute(
                f"INSERT INTO sites ({','.join(site_cols)}) "
                f"VALUES ({','.join('?' * len(site_cols))})",
                tuple(site[c] if c in keys else None for c in site_cols),
            )
            with store.connect() as conn:
                for table in SITE_TABLES:
                    for row in conn.execute(
                        f"SELECT * FROM {table} WHERE site_id = ?", (site["id"],)
                    ):
                        names = row.keys()
                        fresh.execute(
                            f"INSERT INTO {table} ({','.join(names)}) "
                            f"VALUES ({','.join('?' * len(names))})",
                            tuple(row[n] for n in names),
                        )

            pkl = store.MODEL_DIR / f"{site['id']}.pkl"
            if pkl.exists():
                shutil.copy2(pkl, models / pkl.name)
                kept += 1
            else:
                print(f"  ! no fitted model for {site['name']}, skipping its pickle")

        fresh.commit()
        fresh.execute("VACUUM")
        fresh.close()

        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(out_db, arcname="downside.db")
            for pkl in sorted(models.iterdir()):
                tar.add(pkl, arcname=f"models/{pkl.name}")

    mb = dest.stat().st_size / (1024 * 1024)
    print(f"\nwrote {dest} ({mb:.1f} MB)")
    print(f"  {len(sites)} example venues, {kept} fitted models")
    print("  contains no customer accounts, orders or revenue")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api.scripts.make_demo_snapshot",
        description="Package the fitted example venues for deployment.",
    )
    parser.add_argument("-o", "--output", type=Path, default=SNAPSHOT)
    args = parser.parse_args(argv)
    build(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
