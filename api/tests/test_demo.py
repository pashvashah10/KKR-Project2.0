"""Tests for restoring the committed demo snapshot.

This is what makes the application deployable: a hosted instance starts with a
blank disk, and without a restore it would need 5-20 minutes of NOAA fitting
before it had anything worth showing.

The regression test that matters is
`test_restore_survives_a_prior_connection`. The first working version of the
restore was silently undone by SQLite: `store.connect()` is used as
`with connect() as conn`, which commits a transaction but does **not** close
the handle, so the connection opened merely to *check* whether the database was
empty outlived the check and flushed its empty pages back over the freshly
restored file. The log cheerfully reported a successful restore of nothing.
"""

from __future__ import annotations

import sqlite3
import tarfile

import pytest

from api import demo


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from api import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(demo, "SNAPSHOT", tmp_path / "snapshot.tar.gz")
    store.init()
    return store


def make_snapshot(path, db, n_sites=2, n_models=2):
    """A miniature snapshot with the same shape as the committed one."""
    src = path.parent / "src.db"
    conn = sqlite3.connect(src)
    conn.executescript(db.SCHEMA)
    conn.execute(
        "INSERT INTO accounts (id,name,email,api_key,created_at) VALUES (?,?,?,?,?)",
        ("acct_x", "Examples", "examples@downside.local", "k", 0.0),
    )
    for i in range(n_sites):
        conn.execute(
            "INSERT INTO sites (id,account_id,name,lat,lon,vertical,season_start_month,"
            "season_end_month,variable_cost_ratio,status,is_example,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"site_{i}", "acct_x", f"Venue {i}", 42.0, -71.0, "Outdoor attraction",
             1, 12, 0.3, "ready", 1, 0.0),
        )
    conn.commit()
    conn.close()

    models = path.parent / "m"
    models.mkdir(exist_ok=True)
    for i in range(n_models):
        (models / f"site_{i}.pkl").write_bytes(b"not-a-real-pickle")

    with tarfile.open(path, "w:gz") as tar:
        tar.add(src, arcname="downside.db")
        for i in range(n_models):
            tar.add(models / f"site_{i}.pkl", arcname=f"models/site_{i}.pkl")
    return path


def test_restores_into_an_empty_database(db):
    make_snapshot(demo.SNAPSHOT, db)
    assert demo.restore_if_empty() is True
    assert len(db.list_example_sites()) == 2
    assert (db.MODEL_DIR / "site_0.pkl").exists()


def test_restore_survives_a_prior_connection(db):
    """The regression test for the bug that silently restored nothing.

    Touch the database through `store` first --- exactly what the old emptiness
    check did --- then restore. If the stale connection's pages are flushed over
    the restored file, this comes back empty.
    """
    db.list_example_sites()        # opens a connection that is never closed
    db.create_account("Someone", "x@y.test")

    make_snapshot(demo.SNAPSHOT, db)
    assert demo.restore_if_empty() is True
    assert len(db.list_example_sites()) == 2, "a stale connection clobbered the restore"


def test_a_stale_wal_does_not_resurrect_the_old_database(db):
    """A `-wal` left behind is replayed over whatever the db file contains."""
    db.create_account("Someone", "z@y.test")
    wal = db.DB_PATH.with_name(db.DB_PATH.name + "-wal")
    wal.write_bytes(b"stale write-ahead log")

    make_snapshot(demo.SNAPSHOT, db)
    assert demo.restore_if_empty() is True
    assert len(db.list_example_sites()) == 2
    assert not wal.exists() or wal.stat().st_size != len(b"stale write-ahead log")


def test_a_populated_database_is_never_overwritten(db):
    """The guard that stops a redeploy destroying real data."""
    make_snapshot(demo.SNAPSHOT, db)
    demo.restore_if_empty()

    account = db.create_account("Real Customer", "real@customer.test")
    db.create_site(
        account["id"], name="Real Venue", lat=1.0, lon=2.0, elevation_m=0.0,
        vertical="Outdoor attraction", season_start_month=1, season_end_month=12,
        variable_cost_ratio=0.3, reserves=None, monthly_burn=None, is_example=0,
    )

    assert demo.restore_if_empty() is False, "must not restore over a live database"
    assert db.account_by_email("real@customer.test") is not None
    assert len(db.list_sites(account["id"])) == 1


def test_a_missing_snapshot_is_not_fatal(db):
    """A bare storefront beats a service that will not boot."""
    assert not demo.SNAPSHOT.exists()
    assert demo.restore_if_empty() is False
    assert db.list_example_sites() == []


def test_a_corrupt_snapshot_is_not_fatal(db):
    demo.SNAPSHOT.write_bytes(b"this is not a gzip stream")
    assert demo.restore_if_empty() is False


def test_path_traversal_in_the_archive_is_refused(db, tmp_path):
    """A tarball can carry `..` and absolute paths; extraction is not blind."""
    evil = tmp_path / "evil.txt"
    evil.write_bytes(b"pwned")
    with tarfile.open(demo.SNAPSHOT, "w:gz") as tar:
        tar.add(evil, arcname="../../escaped.txt")
        tar.add(evil, arcname="/tmp/absolute.txt")

    demo.restore_if_empty()
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_only_expected_members_are_extracted(db, tmp_path):
    """Anything that is not the database or a model pickle is ignored."""
    make_snapshot(demo.SNAPSHOT, db)
    junk = tmp_path / "junk.sh"
    junk.write_bytes(b"#!/bin/sh\necho no")
    with tarfile.open(demo.SNAPSHOT, "a:") if False else tarfile.open(
        tmp_path / "s2.tar.gz", "w:gz"
    ) as tar:
        tar.add(junk, arcname="models/junk.sh")
    import shutil

    shutil.copy(tmp_path / "s2.tar.gz", demo.SNAPSHOT)
    demo.restore_if_empty()
    assert not (db.MODEL_DIR / "junk.sh").exists()


def test_the_committed_snapshot_is_real_and_populated():
    """Guards the artefact the deployment depends on."""
    from api import demo as real_demo

    if not real_demo.SNAPSHOT.exists():
        pytest.skip("snapshot not built; run `python -m api.scripts.make_demo_snapshot`")

    with tarfile.open(real_demo.SNAPSHOT, "r:gz") as tar:
        names = tar.getnames()

    assert "downside.db" in names
    pickles = [n for n in names if n.startswith("models/") and n.endswith(".pkl")]
    assert len(pickles) >= 8, f"expected the eight example venues, found {len(pickles)}"
    assert all(not n.startswith(("/", "..")) for n in names)


def test_the_committed_snapshot_carries_no_customer_data(tmp_path):
    """A demo fixture in a public repository must not contain anyone's data."""
    from api import demo as real_demo

    if not real_demo.SNAPSHOT.exists():
        pytest.skip("snapshot not built")

    with tarfile.open(real_demo.SNAPSHOT, "r:gz") as tar:
        member = tar.extractfile("downside.db")
        assert member is not None
        (tmp_path / "s.db").write_bytes(member.read())

    conn = sqlite3.connect(tmp_path / "s.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM revenue").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 0
        emails = [r[0] for r in conn.execute("SELECT email FROM accounts")]
        assert emails == ["examples@downside.local"], f"unexpected accounts: {emails}"
        non_example = conn.execute(
            "SELECT COUNT(*) FROM sites WHERE is_example != 1"
        ).fetchone()[0]
        assert non_example == 0
    finally:
        conn.close()
