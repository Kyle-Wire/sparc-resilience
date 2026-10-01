"""App factory, registries, OpenAPI, static SPA, lifecycle hooks, the CLI and the torch-free server process."""

from __future__ import annotations

import importlib.abc
import json
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from sparc.studio import app as appmod
from sparc.studio.jobs import kinds as kindsmod
from sparc.studio.routes import ROUTER_MODULES
from tests.studio.conftest import AUTH, wait_for

ROOT = Path(__file__).resolve().parents[3]
FEATURE_PACKAGES = ("sparc.studio.projects", "sparc.studio.runs", "sparc.studio.engine", "sparc.studio.studies",
                    "sparc.studio.exports", "sparc.studio.scenarios")
FEATURE_ROUTES = [f"sparc.studio.routes.{n}" for n in ROUTER_MODULES if n not in ("system", "jobs", "stream")]

# api.md §2–4: every foundation endpoint
FOUNDATION_ENDPOINTS = [
    ("get", "/auth"), ("get", "/api/health"), ("get", "/api/meta"), ("get", "/api/meta/event-schema"),
    ("get", "/api/settings"), ("put", "/api/settings"), ("get", "/api/system"), ("post", "/api/system/netcheck"),
    ("get", "/api/storage"), ("delete", "/api/storage/cache/{name}"), ("post", "/api/shutdown"),
    ("get", "/api/jobs"), ("post", "/api/jobs"), ("get", "/api/jobs/{jid}"), ("get", "/api/jobs/{jid}/tracker"),
    ("get", "/api/jobs/{jid}/spans"), ("get", "/api/jobs/{jid}/events"), ("get", "/api/jobs/{jid}/logs"),
    ("get", "/api/jobs/{jid}/logs/raw"), ("get", "/api/jobs/{jid}/metrics"), ("get", "/api/jobs/{jid}/warnings"),
    ("get", "/api/jobs/{jid}/resources"), ("post", "/api/jobs/{jid}/cancel"), ("post", "/api/jobs/{jid}/kill"),
    ("post", "/api/jobs/{jid}/retry"), ("patch", "/api/jobs/{jid}"), ("delete", "/api/jobs/{jid}"),
    ("get", "/api/queue"), ("post", "/api/queue/pause"), ("post", "/api/queue/resume"), ("get", "/api/timings"),
    ("get", "/api/stream"), ("get", "/api/jobs/{jid}/stream"),
]


class _Blocker(importlib.abc.MetaPathFinder):
    """Makes the feature packages look absent (ModuleNotFoundError naming exactly the missing module)."""

    def __init__(self, names):
        self.names = tuple(names)

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + ".") for n in self.names):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def no_features(monkeypatch):
    blocked = FEATURE_PACKAGES + tuple(FEATURE_ROUTES)
    for name in list(sys.modules):
        if any(name == n or name.startswith(n + ".") for n in blocked):
            monkeypatch.delitem(sys.modules, name)
    blocker = _Blocker(blocked)
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)


def test_app_starts_with_zero_feature_modules(no_features, make_app, wait_job):
    assert kindsmod.load_kind_modules(test_kinds=False) == []
    app = make_app()
    with TestClient(app, headers=AUTH) as client:
        assert client.get("/api/health").json()["ok"] is True
        meta = client.get("/api/meta").json()
        assert len(meta["stages"]) == 11
        paths = set(client.get("/openapi.json").json()["paths"])
        assert not any(p.startswith(("/api/projects", "/api/runs/", "/api/exports")) for p in paths)
        jid = client.post("/api/jobs", json={"kind": "test.sleep", "params": {"seconds": 0.1}}).json()["id"]
        assert wait_job(client, jid)["status"] == "succeeded"


def test_feature_modules_mount_when_present(monkeypatch, make_app):
    from sparc.studio import routes as routes_pkg

    mod = types.ModuleType("sparc.studio.routes.zz_feature")
    router = APIRouter()

    @router.get("/zz/ping")
    def ping():
        return {"pong": True}

    mod.router = router
    monkeypatch.setitem(sys.modules, "sparc.studio.routes.zz_feature", mod)
    monkeypatch.setattr(routes_pkg, "ROUTER_MODULES", ROUTER_MODULES + ["zz_feature", "zz_absent"])
    kind_mod = types.ModuleType("zz_feature_kinds")

    @kindsmod.job_kind("zz.noop", lane="medium", label="No-op")
    def noop(ctx, params):
        return {}

    monkeypatch.setitem(sys.modules, "zz_feature_kinds", kind_mod)
    monkeypatch.setattr(kindsmod, "KIND_MODULES", kindsmod.KIND_MODULES + ["zz_feature_kinds",
                                                                            "sparc.studio.zz_absent.kinds"])
    try:
        with TestClient(make_app(), headers=AUTH) as client:
            assert client.get("/api/zz/ping").json() == {"pong": True}
            kinds = {k["kind"] for k in client.get("/api/meta").json()["job_kinds"]}
            assert "zz.noop" in kinds
    finally:
        kindsmod.KINDS.pop("zz.noop", None)


def test_other_import_errors_are_raised(monkeypatch, make_app):
    from sparc.studio import routes as routes_pkg

    class Broken(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "sparc.studio.routes.zz_broken":
                raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")
            return None

    finder = Broken()
    sys.meta_path.insert(0, finder)
    monkeypatch.setattr(routes_pkg, "ROUTER_MODULES", ["system", "zz_broken"])
    try:
        with pytest.raises(ModuleNotFoundError):
            make_app()
    finally:
        sys.meta_path.remove(finder)


def test_openapi_describes_every_foundation_endpoint(client):
    doc = client.get("/openapi.json").json()
    for method, path in FOUNDATION_ENDPOINTS:
        assert path in doc["paths"], path
        assert method in doc["paths"][path], f"{method.upper()} {path}"
    schemas = doc["components"]["schemas"]
    for name in ("Job", "TrackerSnapshot", "Health", "Meta", "Settings", "SystemInfo", "StorageInfo",
                 "QueueState", "Timings", "EventsPage", "LogsPage"):
        assert name in schemas, name
    assert set(schemas["Job"]["properties"]) >= {"id", "kind", "lane", "executor", "status", "progress", "eta_lo",
                                                 "blocked", "current_path", "peak_rss_mb", "threads"}
    stream = doc["paths"]["/api/jobs/{jid}/stream"]["get"]["responses"]["200"]["content"]
    assert "text/event-stream" in stream
    assert "ErrorEnvelope" in schemas and "HTTPValidationError" not in schemas
    err = doc["paths"]["/api/jobs"]["post"]["responses"]
    assert err["422"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorEnvelope"
    assert "default" in err


def test_dump_openapi(tmp_path):
    out = tmp_path / "openapi.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-m", "sparc.studio", "--dump-openapi", str(out)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert doc["info"]["title"] == "SPARC Studio" and "/api/jobs/{jid}/tracker" in doc["paths"]


def test_static_spa(tmp_path, make_app):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>Studio</title>")
    (static / "assets" / "app-1a2b.js").write_text("console.log(1)")
    (static / "BUILD_INFO.json").write_text(json.dumps({"src_sha256": "abc", "vite": "8", "react": "19"}))
    with TestClient(make_app(static_dir=static), headers=AUTH) as client:
        for path in ("/", "/p/p_1/setup/data", "/r/x/lab"):
            r = client.get(path)
            assert r.status_code == 200 and "<title>Studio" in r.text and r.headers["cache-control"] == "no-store"
        r = client.get("/assets/app-1a2b.js")
        assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/assets/../index.html").status_code in (404, 200)       # never escapes static/
        r = client.get("/api/nothing-here")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        assert client.get("/BUILD_INFO.json").json()["src_sha256"] == "abc"
    with TestClient(make_app(static_dir=tmp_path / "empty"), headers=AUTH) as client:
        r = client.get("/")
        assert r.status_code == 200 and "Frontend not built" in r.text and "npm --prefix studio-web run build" in r.text


def test_system_storage_and_health(client, ctx):
    h = client.get("/api/health").json()
    assert h["workspace"] == str(ctx.workspace.root) and h["engine"] == {"state": "absent"}
    sysinfo = client.get("/api/system").json()
    assert sysinfo["cpu_count"] >= 1 and sysinfo["host_id"] and sysinfo["versions"]["python"]
    assert "torch" in sysinfo["versions"]
    (ctx.workspace.cache_dir / "ghcn_X.csv").write_text("a,b\n1,2\n")
    st = client.get("/api/storage").json()
    assert [c["name"] for c in st["cache"]] == ["ghcn_X.csv"] and st["free_bytes"] > 0
    assert client.delete("/api/storage/cache/ghcn_X.csv").json()["freed_bytes"] == 8
    assert client.delete("/api/storage/cache/ghcn_X.csv").status_code == 404
    assert client.delete("/api/storage/cache/%2e%2e").status_code in (404, 422)
    r = client.post("/api/system/netcheck", json={"hosts": ["127.0.0.1:9"]}).json()
    assert r["results"][0]["host"] == "127.0.0.1:9" and r["results"][0]["ok"] is False


def test_lifecycle_hooks(make_app):
    calls = []

    @appmod.startup_hook("zz-scan", phase="scan")
    def scan(sctx):
        calls.append(("scan", sctx.jobs is None))

    @appmod.startup_hook("zz-ready", phase="ready")
    async def ready(sctx):
        calls.append(("ready", sctx.jobs is not None))

    @appmod.startup_hook("zz-broken", phase="ready")
    def broken(sctx):
        raise RuntimeError("a failing hook never stops the server")

    @appmod.shutdown_hook("zz-stop")
    def stop(sctx):
        calls.append(("stop", True))

    try:
        with TestClient(make_app(), headers=AUTH) as client:
            assert client.get("/api/health").status_code == 200
        assert calls == [("scan", True), ("ready", True), ("stop", True)]
    finally:
        appmod._STARTUP[:] = [h for h in appmod._STARTUP if not h[2].startswith("zz-")]
        appmod._SHUTDOWN[:] = [h for h in appmod._SHUTDOWN if not h[1].startswith("zz-")]


def test_shutdown_with_stop_jobs_cancels_them(make_app, wait_job):
    app = make_app(shutdown_grace_s=10)
    with TestClient(app, headers=AUTH) as client:
        jid = client.post("/api/jobs", json={"kind": "test.sleep", "params": {"seconds": 60}}).json()["id"]
        wait_job(client, jid, ("running",))
        pid = app.state.studio.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (jid,))
        assert client.post("/api/shutdown", json={"stop_jobs": True}).status_code == 202
    from sparc.studio.db import Database

    db = Database(app.state.studio.workspace.db_path)
    assert db.fetchval("SELECT status FROM jobs WHERE id = ?", (jid,)) == "cancelled"
    db.close()
    wait_for(lambda: not _alive(pid), 10, what="the job process to exit")


def _alive(pid):
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_server_process_never_imports_torch(tmp_path):
    code = f"""
import sys, time
from fastapi.testclient import TestClient
from sparc.studio.app import create_app
from sparc.studio.settings import StudioSettings
app = create_app(StudioSettings(workspace={str(tmp_path / 'ws')!r}, token="t", public_hosts=["testserver"],
                                test_kinds=True, tail_interval_s=0.05))
H = {{"Authorization": "Bearer t", "Origin": "http://testserver"}}
with TestClient(app, headers=H) as c:
    for path in ("/api/health", "/api/meta", "/api/system", "/api/storage", "/api/settings", "/openapi.json",
                 "/api/meta/event-schema", "/api/timings"):
        assert c.get(path).status_code == 200, path
    jid = c.post("/api/jobs", json={{"kind": "test.events", "params": {{"seconds": 0.3}}}}).json()["id"]
    for _ in range(400):
        if c.get(f"/api/jobs/{{jid}}").json()["status"] == "succeeded":
            break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{{jid}}/tracker").status_code == 200
    import sparc.core.pipeline   # what /api/meta and the run kinds may import must stay torch-free too
assert "torch" not in sys.modules, sorted(m for m in sys.modules if m.startswith("torch"))[:5]
print("TORCH-FREE")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "TORCH-FREE" in proc.stdout


def _get(url, token=None, timeout=5.0):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def test_cli_start_reuse_and_shutdown(tmp_path):
    ws = tmp_path / "ws"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env.update(PYTHONPATH=str(ROOT), SPARC_STUDIO_TEST_KINDS="1")
    cmd = [sys.executable, "-m", "sparc.studio", "--workspace", str(ws), "--port", "0", "--no-browser",
           "--token", "t"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lock_path = ws / "studio.lock.json"
        wait_for(lambda: lock_path.exists(), 60, what="the lock file")
        lock = json.loads(lock_path.read_text())
        assert lock["pid"] == proc.pid and lock["port"] > 0 and oct(lock_path.stat().st_mode)[-3:] == "600"
        assert oct((ws / "token").stat().st_mode)[-3:] == "600" and (ws / "token").read_text().strip() == "t"
        base = f"http://127.0.0.1:{lock['port']}"
        health = wait_for(lambda: _try(lambda: _get(base + "/api/health")), 60, what="health")
        assert health["ok"] is True and health["workspace"] == str(ws.resolve())
        meta = _get(base + "/api/meta", token="t")
        assert len(meta["stages"]) == 11 and "seqLight" in meta["palettes"]
        assert {"test.sleep", "test.events"} <= {k["kind"] for k in meta["job_kinds"]}
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base + "/api/meta")
        assert exc.value.code == 401
        second = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert second.returncode == 0, second.stderr
        assert f"{base}/auth?t=t" in second.stdout and "already running" in second.stdout
        req = urllib.request.Request(base + "/api/shutdown", data=b"{}", method="POST",
                                     headers={"Authorization": "Bearer t", "Content-Type": "application/json"})
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=5).read()
        assert proc.wait(timeout=30) == 0
        assert not lock_path.exists()
        out = proc.stdout.read()
        assert f"{base}/auth?t=t" in out
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


def test_busy_port_falls_back_without_touching_the_occupant():
    from sparc.studio.cli import bind_socket

    occupant = bind_socket("127.0.0.1", 0)
    port = occupant.getsockname()[1]
    try:
        other = bind_socket("127.0.0.1", port)
        assert other.getsockname()[1] != port
        other.close()
    finally:
        occupant.close()


@pytest.mark.parametrize("argv", [["-m", "sparc", "studio", "--help"], ["-m", "sparc.core", "studio", "--help"],
                                  ["-m", "sparc.studio", "--help"]])
def test_help_through_core_delegation(argv):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "sparc studio" in proc.stdout and "--workspace" in proc.stdout and "--dump-openapi" in proc.stdout
