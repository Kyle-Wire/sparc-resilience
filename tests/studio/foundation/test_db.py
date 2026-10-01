"""SQLite schema, migrations, the writer/reader split and ``--reindex``."""

from __future__ import annotations

import asyncio
import shutil
import threading

from sparc.studio import db as dbmod
from sparc.studio.db import Database, reindex, reindex_hook
from sparc.studio.workspace import Workspace, new_id, new_run_id, slugify

ALL_TABLES = {"projects", "config_versions", "runs", "jobs", "spans", "metrics", "artifacts", "warnings",
              "checkpoints", "unit_timings", "stage_timings", "resource_samples", "run_locks", "studies",
              "study_links", "scenarios", "results", "plans", "sweeps", "comparisons", "regions", "blobs", "exports",
              "findings", "settings"}


def test_migrate_creates_full_schema(tmp_path):
    db = Database(tmp_path / "studio.sqlite")
    assert db.migrate() == dbmod.SCHEMA_VERSION == 1
    assert set(db.tables()) == ALL_TABLES
    assert db.user_version() == 1
    assert db.fetchval("PRAGMA journal_mode") == "wal"
    assert db.migrate() == 1                                     # idempotent
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(jobs)")}
    assert {"last_cursor", "peak_rss_mb", "host_id", "blocked_json", "proc_create_time"} <= cols
    idx = {r["name"] for r in db.fetchall("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"jobs_status", "jobs_run", "metrics_job_name", "unit_timings_hu", "results_lookup"} <= idx
    db.close()


def test_single_writer_many_readers(tmp_path):
    db = Database(tmp_path / "studio.sqlite")
    db.migrate()
    errors = []

    def write(i):
        try:
            for k in range(50):
                db.execute("INSERT INTO settings (key, value_json) VALUES (?, ?)", (f"k{i}_{k}", dbmod.dumps(k)))
                db.fetchval("SELECT COUNT(*) FROM settings")
        except Exception as exc:                               # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert db.fetchval("SELECT COUNT(*) FROM settings") == 300

    async def awrites():
        await asyncio.gather(*(db.aexecute("INSERT INTO settings (key, value_json) VALUES (?, ?)", (f"a{i}", "1"))
                               for i in range(20)))
    asyncio.run(awrites())
    assert db.fetchval("SELECT COUNT(*) FROM settings WHERE key LIKE 'a%'") == 20
    with db.transaction() as conn:
        conn.execute("DELETE FROM settings")
        assert conn.execute("SELECT COUNT(*) AS n FROM settings").fetchone()["n"] == 0
    db.set_setting("x", {"a": float("nan")})
    assert db.get_setting("x") == {"a": None}
    db.close()


def test_ids_and_slugs():
    jid = new_id("j")
    assert jid.startswith("j_") and len(jid) == 10 and jid[2:].isalnum() and jid == jid.lower()
    assert new_id("export").startswith("ex_")
    rid = new_run_id("coarse60", now=1790000000)
    assert rid.startswith("20260921-") and "-coarse60-" in rid and len(rid.split("-")[-1]) == 4
    assert slugify("Providence, RI (2020)!") == "providence-ri-2020"
    assert slugify("Demo", taken={"demo", "demo-2"}) == "demo-3"


def test_reindex_rebuilds_jobs_and_runs_hooks(client, ctx, wait_job):
    calls = []

    @reindex_hook("test-feature", order=5)
    def _hook(db, workspace):
        calls.append(workspace.root)
        return {"rows": 0}

    try:
        jid = client.post("/api/jobs", json={"kind": "test.events", "params": {"seconds": 0.3}}).json()["id"]
        before = wait_job(client, jid)
        ws: Workspace = ctx.workspace
        n_spans = ctx.db.fetchval("SELECT COUNT(*) FROM spans WHERE job_id = ?", (jid,))
        n_metrics = ctx.db.fetchval("SELECT COUNT(*) FROM metrics WHERE job_id = ?", (jid,))
        # a separate copy of the workspace without its database
        copy = Workspace(ws.root.parent / "copy")
        shutil.copytree(ws.root / "jobs", copy.root / "jobs")
        db2 = Database(copy.db_path)
        summary = reindex(db2, copy)
        assert summary["jobs"] == 1 and summary["test-feature"] == {"rows": 0} and calls == [copy.root]
        row = db2.fetchone("SELECT * FROM jobs WHERE id = ?", (jid,))
        assert row["status"] == "succeeded" and row["kind"] == "test.events" and row["progress"] == 1.0
        assert dbmod.loads(row["result_json"]) == before["result"]
        assert db2.fetchval("SELECT COUNT(*) FROM spans WHERE job_id = ?", (jid,)) == n_spans
        assert db2.fetchval("SELECT COUNT(*) FROM metrics WHERE job_id = ?", (jid,)) == n_metrics
        assert db2.fetchval("SELECT count FROM warnings WHERE job_id = ? AND code = 'qa.coarse'", (jid,)) == 2
        assert db2.fetchval("SELECT COUNT(*) FROM artifacts WHERE job_id = ?", (jid,)) == 8
        db2.close()
    finally:
        dbmod._REINDEX_HOOKS[:] = [h for h in dbmod._REINDEX_HOOKS if h[1] != "test-feature"]


def test_reindex_flag_rebuilds_at_startup(make_app, wait_job):
    from fastapi.testclient import TestClient

    from tests.studio.conftest import AUTH

    app = make_app()
    with TestClient(app, headers=AUTH) as client:
        jid = client.post("/api/jobs", json={"kind": "test.sleep", "params": {"seconds": 0.1}}).json()["id"]
        wait_job(client, jid)
    ws = app.state.studio.workspace
    for suffix in ("", "-wal", "-shm"):
        p = ws.root / f"studio.sqlite{suffix}"
        if p.exists():
            p.unlink()
    with TestClient(make_app(reindex=True), headers=AUTH) as client:
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "succeeded" and job["kind"] == "test.sleep"
