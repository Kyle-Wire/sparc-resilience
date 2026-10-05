"""Shared fixtures for every Studio backend test group (``tests/studio/<group>/``).

* ``studio_config`` / ``make_app`` / ``app`` - a Studio app on a temporary
  workspace with short internal intervals and the ``test.*`` kinds enabled;
* ``client`` - an authenticated ``TestClient`` (Bearer token + same-origin
  ``Origin``, so unsafe methods pass the CSRF check) whose lifespan is running;
* ``ctx`` - the app's :class:`~sparc.studio.app.StudioContext`;
* ``live_server`` - the app served by uvicorn on a real socket in a thread
  (``TestClient`` buffers whole responses, so SSE tests need this);
* ``wait_for`` / ``wait_job`` - polling helpers;
* fixture loaders: ``fixtures_dir``, ``load_jsonl``, ``selftest_events``,
  ``synth_run_dir`` (skips when the core fixture is absent).

Every test under ``tests/studio`` gets the ``studio`` marker.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOKEN = "studio-test-token"
ORIGIN = "http://testserver"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN}
FINAL = ("succeeded", "failed", "cancelled", "interrupted")


def pytest_configure(config):
    config.addinivalue_line("markers", "studio: SPARC Studio server and web app tests (tests/studio)")
    config.addinivalue_line("markers", "e2e: SPARC Studio end-to-end browser tests")
    config.addinivalue_line("filterwarnings", "ignore:Using `httpx` with `starlette.testclient`")


def pytest_collection_modifyitems(config, items):
    here = str(Path(__file__).resolve().parent)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.studio)


# ---------------------------------------------------------------------------
# apps and clients
# ---------------------------------------------------------------------------

@pytest.fixture
def studio_config(tmp_path) -> Callable[..., Any]:
    """``studio_config(**overrides)`` → a ``StudioSettings`` for a temporary workspace."""
    from sparc.studio.settings import StudioSettings

    def make(**overrides):
        kw = dict(workspace=tmp_path / "ws", token=TOKEN, public_hosts=["testserver"], test_kinds=True,
                  tail_interval_s=0.05, flush_interval_s=0.5, schedule_interval_s=0.2, exit_poll_s=0.05,
                  sample_interval_s=0.5, start_timeout_s=30.0, ping_interval_s=15.0)
        kw.update(overrides)
        return StudioSettings(**kw)

    return make


def kill_tree(pid: int) -> None:
    """Kill a job's process group (POSIX) or its process tree (Windows has no process groups to signal)."""
    import signal

    import psutil

    if hasattr(os, "killpg"):
        os.killpg(pid, signal.SIGKILL)
        return
    p = psutil.Process(pid)
    for c in p.children(recursive=True):
        try:
            c.kill()
        except psutil.Error:
            pass
    p.kill()


def kill_leftover_jobs(workspace) -> None:
    """Kill the process groups of jobs a test left running (jobs outlive their server by design)."""
    import psutil

    from sparc.studio.db import Database

    if not workspace.db_path.exists():
        return
    db = Database(workspace.db_path)
    try:
        rows = db.fetchall("SELECT pid, job_dir FROM jobs WHERE pid IS NOT NULL")
    except Exception:
        rows = []
    finally:
        db.close()
    for r in rows:
        try:
            p = psutil.Process(int(r["pid"]))
            if any(r["job_dir"] in part for part in p.cmdline()):
                kill_tree(int(r["pid"]))
        except (psutil.Error, ProcessLookupError, PermissionError, OSError):
            continue


@pytest.fixture
def make_app(studio_config):
    """``make_app(**overrides)`` → a fresh Studio FastAPI app (lifespan not started).

    At teardown, job processes the test left running are killed.
    """
    from sparc.studio.app import create_app

    made = []

    def make(**overrides):
        app = create_app(studio_config(**overrides))
        made.append(app)
        return app

    yield make
    for app in made:
        kill_leftover_jobs(app.state.studio.workspace)


@pytest.fixture
def app(make_app):
    return make_app()


@pytest.fixture
def client(app):
    """An authenticated TestClient with the app's lifespan running."""
    from fastapi.testclient import TestClient

    with TestClient(app, headers=AUTH) as c:
        yield c


@pytest.fixture
def ctx(client):
    return client.app.state.studio


def wait_for(pred: Callable[[], Any], timeout: float = 30.0, interval: float = 0.05, what: str = "condition"):
    """Poll ``pred`` until it returns a truthy value (returned) or fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        val = pred()
        if val:
            return val
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(interval)


@pytest.fixture
def wait_job():
    """``wait_job(client, jid, statuses=FINAL, timeout=60)`` → the job once its status is in ``statuses``."""
    def wait(client, jid: str, statuses=FINAL, timeout: float = 60.0) -> dict:
        statuses = (statuses,) if isinstance(statuses, str) else tuple(statuses)
        last = {}

        jobs = getattr(getattr(getattr(client, "app", None), "state", None), "studio", None)
        jobs = getattr(jobs, "jobs", None)

        def check():
            nonlocal last
            last = client.get(f"/api/jobs/{jid}").json()
            if last.get("status") not in statuses:
                return None
            # a final job's on_finish hook (export row, study children) has run too
            if last.get("status") in FINAL and jobs is not None and jid in getattr(jobs, "finishing", ()):
                return None
            return last

        try:
            return wait_for(check, timeout, 0.05, f"job {jid} to reach {statuses}")
        except AssertionError as exc:
            raise AssertionError(f"{exc}; last seen {last.get('status')}: {last}") from None

    return wait


class LiveServer:
    """A Studio app served by uvicorn on 127.0.0.1:<port> in a background thread."""

    def __init__(self, app, sock, server, thread):
        self.app, self.sock, self.server, self.thread = app, sock, server, thread
        self.port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.ctx = app.state.studio

    def client(self, **kw):
        import httpx

        headers = {"Authorization": f"Bearer {TOKEN}", "Origin": self.base}
        headers.update(kw.pop("headers", {}))
        return httpx.Client(base_url=self.base, headers=headers, timeout=kw.pop("timeout", 30.0), trust_env=False,
                            **kw)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=30)
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.fixture
def live_server(studio_config):
    """``live_server(**overrides)`` → a started :class:`LiveServer` (stopped at teardown)."""
    from sparc.studio.app import create_app
    from sparc.studio.cli import StudioServer, bind_socket, uvicorn_config

    servers: list[LiveServer] = []

    def start(**overrides) -> LiveServer:
        sock = bind_socket("127.0.0.1", 0)
        port = sock.getsockname()[1]
        app = create_app(studio_config(port=port, **overrides))
        server = StudioServer(uvicorn_config(app, log_level="warning"), app.state.studio)
        server.install_signal_handlers = lambda: None
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        wait_for(lambda: server.started, 30, 0.02, "uvicorn to start")
        live = LiveServer(app, sock, server, thread)
        servers.append(live)
        return live

    yield start
    for s in servers:
        s.stop()
        kill_leftover_jobs(s.ctx.workspace)


# ---------------------------------------------------------------------------
# fixture loaders
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def read_jsonl(path: str | os.PathLike) -> list[tuple[int, dict]]:
    """``(cursor, event)`` pairs of a JSONL file (cursor = byte offset of the line)."""
    out = []
    offset = 0
    with open(path, "rb") as f:
        for raw in f:
            if raw.endswith(b"\n") and raw.strip():
                out.append((offset, json.loads(raw)))
            offset += len(raw)
    return out


@pytest.fixture
def load_jsonl():
    return read_jsonl


@pytest.fixture
def selftest_events() -> list[tuple[int, dict]]:
    """The committed ``test.events`` fixture as ``(cursor, event)`` pairs."""
    return read_jsonl(FIXTURES / "selftest_events.jsonl")


@pytest.fixture
def synth_run_dir() -> Path:
    """``tests/studio/fixtures/synth_run`` (the recorded synthetic run of core-instrumentation); skips if absent."""
    d = FIXTURES / "synth_run"
    if not (d / "events.jsonl").is_file():
        pytest.skip("tests/studio/fixtures/synth_run is not present")
    return d


def sse_events(lines) -> list[dict]:
    """Parse SSE text lines into ``{id, event, data}`` dicts (comments skipped, ``data`` JSON-decoded)."""
    out, cur = [], {}
    for line in lines:
        if line == "":
            if cur:
                if "data" in cur:
                    try:
                        cur["data"] = json.loads(cur["data"])
                    except ValueError:
                        pass
                out.append(cur)
            cur = {}
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        cur[field] = value if field != "data" or "data" not in cur else cur["data"] + "\n" + value
    return out
