"""Restore the committed demo snapshot when a deployment starts empty.

A hosted instance gets a blank disk. Without this it would come up with no
venues at all, and filling it means `api.seed` --- 5 to 20 minutes of fitting
against NOAA, which a platform's startup timeout will not tolerate and a
512 MB instance may not survive.

So a new instance restores `demo/snapshot.tar.gz` instead: a pruned database
plus the fitted model for each example venue. It takes a second or two, needs
no network, and the numbers are real --- the same artefacts a local fit
produces, packaged rather than recomputed.

**Only ever restores into an empty database.** It must never overwrite a real
one, so the check is for the absence of example venues rather than a flag
someone could get wrong, and any failure is logged and swallowed: a missing
snapshot should mean a bare storefront, not a service that will not boot.
"""

from __future__ import annotations

import gc
import logging
import sqlite3
import tarfile
from pathlib import Path

from . import store

log = logging.getLogger(__name__)

SNAPSHOT = Path(__file__).resolve().parents[1] / "demo" / "snapshot.tar.gz"

__all__ = ["restore_if_empty", "SNAPSHOT"]


def _count_examples() -> int:
    """Example venues on disk, using a connection this function actually closes.

    Deliberately not `store.list_example_sites()`. `store.connect()` returns a
    connection used as `with connect() as conn`, and that context manager
    commits a transaction --- it does **not** close the handle. The connection
    therefore outlives the call with its own page cache, and when it is finally
    collected it can flush stale pages over the database file.

    That is not theoretical: it silently undid the first working version of this
    restore. The empty database created by the check was written back on top of
    the snapshot, leaving a 128 KB file with no venues in it and a log line
    cheerfully reporting a successful restore of nothing.
    """
    if not store.DB_PATH.exists():
        return 0
    conn = sqlite3.connect(store.DB_PATH)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM sites WHERE is_example = 1 AND status = 'ready'"
            ).fetchone()[0]
        )
    except sqlite3.Error:
        return 0  # absent or unreadable counts as empty
    finally:
        conn.close()


def _clear_database_files() -> None:
    """Remove the database and its sidecars before writing a new one.

    A `-wal` left behind by a previous connection is replayed over whatever
    `downside.db` happens to contain, so restoring the file without clearing
    the write-ahead log restores the old contents instead.
    """
    # Drop any connection still held open by earlier store calls, so their
    # buffers cannot be flushed over the restored file.
    gc.collect()
    for path in (
        store.DB_PATH,
        store.DB_PATH.with_name(store.DB_PATH.name + "-wal"),
        store.DB_PATH.with_name(store.DB_PATH.name + "-shm"),
    ):
        path.unlink(missing_ok=True)


def _safe_members(tar: tarfile.TarFile):
    """Only the two shapes the snapshot is allowed to contain.

    A tarball is an archive format with absolute paths and `..` traversal, and
    this one is extracted with the service's own permissions. It is committed to
    our own repository rather than uploaded, so the risk is remote --- but the
    check costs three lines and removes the question entirely.
    """
    for member in tar.getmembers():
        name = member.name
        if not member.isfile():
            continue
        if name.startswith(("/", "..")) or ".." in Path(name).parts:
            log.warning("refusing suspicious path in snapshot: %s", name)
            continue
        if name == "downside.db" or (name.startswith("models/") and name.endswith(".pkl")):
            yield member


def restore_if_empty(snapshot: Path | None = None) -> bool:
    """Populate a blank instance from the snapshot. Returns whether it did.

    `snapshot` resolves at call time rather than as a default argument value.
    A default of `SNAPSHOT` would bind the module-level path once at import,
    so reassigning `demo.SNAPSHOT` afterwards would silently have no effect ---
    which is exactly what made the first version of the tests here pass against
    the real committed archive while appearing to use a fixture.
    """
    snapshot = snapshot or SNAPSHOT
    existing = _count_examples()
    if existing:
        log.info("%d example venues already present; leaving the database alone", existing)
        return False

    if not snapshot.exists():
        log.info(
            "no demo snapshot at %s; starting with an empty storefront "
            "(run `python -m api.seed` to fit the example venues)", snapshot,
        )
        return False

    try:
        store.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        store.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _clear_database_files()
        with tarfile.open(snapshot, "r:gz") as tar:
            for member in _safe_members(tar):
                target = (
                    store.DB_PATH if member.name == "downside.db"
                    else store.MODEL_DIR / Path(member.name).name
                )
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, open(target, "wb") as fh:
                    fh.write(source.read())

        # Bring a snapshot written against an older schema up to date.
        store.init()
        n = _count_examples()
        if n:
            log.info("restored the demo snapshot: %d example venues ready", n)
        else:
            log.warning("snapshot restored but no venues are readable from it")
        return n > 0

    except Exception:  # noqa: BLE001
        log.exception("could not restore the demo snapshot; continuing without it")
        return False
