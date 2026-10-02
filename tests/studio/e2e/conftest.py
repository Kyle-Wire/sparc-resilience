"""End-to-end harness (SPEC §14.5): a real Studio server process driven by Playwright (Chromium).

* :func:`start_studio` runs ``python -m sparc.studio --workspace <tmp> --port 0 --no-browser
  --token e2e`` as a subprocess (this checkout first on ``PYTHONPATH``, so the server and its
  job workers import the code under test) and reads the URL from ``studio.lock.json``. At the
  end it asks the server to stop with its jobs and kills whatever process still names the
  workspace (job workers, the engine host).
* ``studio_page(server)`` opens an authenticated page. **Any console error or uncaught page
  error fails the test**; :meth:`StudioPage.shot` takes each step's screenshot in the light and
  the dark theme. Screenshots go to ``$SPARC_E2E_ARTIFACTS/<test>/`` when that is set (CI uploads
  it), else to the test's temporary folder.
* Chromium is ``$PLAYWRIGHT_CHROMIUM`` or the preinstalled one; the tests skip without it or
  without the Python ``playwright`` package. Browsers are never downloaded.

Every test here carries the ``e2e`` marker.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "studio" / "fixtures"
SYNTH_RUN = FIXTURES / "synth_run"
TOKEN = "e2e"
DEFAULT_CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
FINAL = ("succeeded", "failed", "cancelled", "interrupted")
VIEWPORT = {"width": 1440, "height": 900}


def chromium_path() -> str | None:
    path = os.environ.get("PLAYWRIGHT_CHROMIUM") or DEFAULT_CHROMIUM
    return path if Path(path).is_file() else None


def pytest_collection_modifyitems(config, items):
    here = str(Path(__file__).resolve().parent)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.e2e)


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

class ApiError(AssertionError):
    pass


class Studio:
    """A running ``sparc studio`` process and an authenticated API client for setting up tests."""

    def __init__(self, workspace: Path, proc: subprocess.Popen, base: str, log_path: Path):
        import httpx

        self.workspace, self.proc, self.base, self.log_path = workspace, proc, base, log_path
        self.http = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120.0,
                                 trust_env=False)

    # -- API helpers (setup and assertions; the journeys themselves go through the browser) --
    def call(self, method: str, path: str, **kw) -> Any:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise ApiError(f"{method} {path} → {r.status_code}: {r.text[:800]}")
        ct = r.headers.get("content-type", "")
        return r.json() if ct.startswith("application/json") else r.content

    def get(self, path: str, **kw) -> Any:
        return self.call("GET", path, **kw)

    def post(self, path: str, body: Any = None, **kw) -> Any:
        return self.call("POST", path, json=body if body is not None else {}, **kw)

    def create_demo(self, name: str, n: int, seed: int = 0) -> dict:
        """A synthetic-demo project made by the one shared generator with this ``n`` and ``seed``."""
        r = self.post("/api/projects", {"name": name, "template": "synthetic_demo", "options": {"n": n, "seed": seed}})
        return r["project"]

    def job(self, jid: str) -> dict:
        return self.get(f"/api/jobs/{jid}")

    def wait_job(self, jid: str, statuses=FINAL, timeout: float = 300.0) -> dict:
        statuses = (statuses,) if isinstance(statuses, str) else tuple(statuses)
        return wait_for(lambda: (j := self.job(jid))["status"] in statuses and j, timeout,
                        f"job {jid} to reach {statuses}", interval=0.25)

    def log_tail(self, n: int = 60) -> str:
        try:
            return "\n".join(self.log_path.read_text("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return ""

    def stop(self) -> None:
        try:
            self.http.post("/api/shutdown", json={"stop_jobs": True}, timeout=10)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.http.close()
        kill_workspace_processes(self.workspace)


def kill_workspace_processes(workspace: Path, grace_s: float = 5.0) -> None:
    """Stop every process whose command line names ``workspace`` (job workers, the engine host)."""
    import psutil

    ws = str(workspace)
    me = os.getpid()
    victims = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.pid != me and any(ws in part for part in (p.info["cmdline"] or [])):
                victims.append(p)
        except (psutil.Error, OSError):
            continue
    for p in victims:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(victims, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def wait_for(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        val = pred()
        if val:
            return val
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(interval)


def start_studio(workspace: Path, *, runner: str | None = None, env: dict | None = None,
                 timeout: float = 90.0) -> Studio:
    """Start the server on a free port for ``workspace`` and wait until it answers."""
    import httpx

    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace.parent / f"{workspace.name}.server.log"
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    full_env.pop("SPARC_STUDIO_RUNNER", None)
    full_env.pop("SPARC_STUDIO_HOME", None)
    if runner:
        full_env["SPARC_STUDIO_RUNNER"] = runner
    full_env.update(env or {})
    cmd = [sys.executable, "-m", "sparc.studio", "--workspace", str(workspace), "--port", "0", "--no-browser",
           "--token", TOKEN]
    log = open(log_path, "ab")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log.close()
    lock = workspace / "studio.lock.json"
    deadline = time.monotonic() + timeout
    base = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"sparc studio exited with {proc.returncode}:\n"
                                 + log_path.read_text("utf-8", "replace")[-3000:])
        try:
            info = json.loads(lock.read_text("utf-8"))
            if info.get("pid") == proc.pid and info.get("port"):
                base = f"http://127.0.0.1:{int(info['port'])}"
                r = httpx.get(base + "/api/health", timeout=2, trust_env=False)
                if r.status_code == 200 and r.json().get("ok"):
                    break
        except (OSError, ValueError, httpx.HTTPError):
            pass
        time.sleep(0.2)
    else:
        proc.kill()
        raise AssertionError("sparc studio did not answer /api/health:\n" + log_path.read_text("utf-8", "replace")[-3000:])
    studio = Studio(workspace, proc, base, log_path)
    # the registries start in the background: wait until the runs routes answer
    wait_for(lambda: studio.http.get("/api/runs").status_code == 200, 60, "the runs registry")
    return studio


@pytest.fixture
def studio_server(tmp_path):
    """``studio_server(runner=None, env=None)`` → a started :class:`Studio` (stopped at teardown)."""
    started: list[Studio] = []

    def start(**kw) -> Studio:
        s = start_studio(tmp_path / f"ws{len(started)}", **kw)
        started.append(s)
        return s

    yield start
    for s in started:
        s.stop()


# ---------------------------------------------------------------------------
# the browser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def browser():
    exe = chromium_path()
    if exe is None:
        pytest.skip("Chromium not found (set PLAYWRIGHT_CHROMIUM)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("the playwright Python package is not installed")
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=exe, headless=True, args=["--disable-gpu", "--no-sandbox"])
        yield b
        b.close()


class StudioPage:
    """A page on the Studio SPA that records console and page errors and takes themed screenshots."""

    def __init__(self, page, base: str, shots_dir: Path, counter=None):
        self.page, self.base, self.shots_dir = page, base, shots_dir
        self.errors: list[str] = []
        self.allowed: list[re.Pattern] = []
        self.counter = counter or itertools.count(1)   # shared by the pages of one test
        page.on("console", self._on_console)
        page.on("pageerror", lambda exc: self.errors.append(f"pageerror: {exc}"))

    def _on_console(self, msg) -> None:
        if msg.type == "error":
            loc = msg.location or {}
            self.errors.append(f"console.error: {msg.text} ({loc.get('url', '')}:{loc.get('lineNumber', '')})")

    def allow_error(self, pattern: str) -> None:
        """Let console errors matching ``pattern`` through (only for errors a step provokes on purpose)."""
        self.allowed.append(re.compile(pattern))

    def unexpected_errors(self) -> list[str]:
        return [e for e in self.errors if not any(p.search(e) for p in self.allowed)]

    def goto(self, path: str, *, wait: str = "domcontentloaded") -> None:
        self.page.goto(self.base + path, wait_until=wait)

    def shot(self, name: str) -> None:
        """Screenshots of the current view in the light and the dark theme."""
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{next(self.counter):02d}_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        for scheme in ("light", "dark"):
            self.page.emulate_media(color_scheme=scheme)
            self.page.wait_for_timeout(120)
            self.page.screenshot(path=str(self.shots_dir / f"{stem}_{scheme}.png"))
        self.page.emulate_media(color_scheme="light")

    def assert_no_errors(self) -> None:
        bad = self.unexpected_errors()
        assert not bad, "browser console errors:\n" + "\n".join(bad[:20])


@pytest.fixture
def shots_dir(request, tmp_path) -> Path:
    root = os.environ.get("SPARC_E2E_ARTIFACTS")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    return Path(root) / name if root else tmp_path / "shots"


@pytest.fixture
def studio_page(request, browser, shots_dir):
    """``studio_page(server, **context_options)`` → an authenticated :class:`StudioPage`.

    At teardown every page is checked for console errors (any one fails the test).
    """
    contexts, pages = [], []
    counter = itertools.count(1)

    def open_page(server: Studio, **ctx_kw) -> StudioPage:
        kw = {"viewport": VIEWPORT, "color_scheme": "light", "accept_downloads": True}
        kw.update(ctx_kw)
        ctx = browser.new_context(**kw)
        ctx.set_default_timeout(30_000)
        contexts.append(ctx)
        page = ctx.new_page()
        sp = StudioPage(page, server.base, shots_dir, counter)
        pages.append(sp)
        sp.goto(f"/auth?t={TOKEN}&next=/")
        page.wait_for_selector("#root *", state="attached")
        return sp

    yield open_page
    try:
        rep = getattr(request.node, "rep_call", None)
        if rep is not None and rep.failed:      # the state each page was in when the test failed
            for i, sp in enumerate(pages):
                try:
                    shots_dir.mkdir(parents=True, exist_ok=True)
                    sp.page.screenshot(path=str(shots_dir / f"zz_failure_page{i}.png"), full_page=True)
                except Exception:
                    pass
        for sp in pages:
            sp.assert_no_errors()
    finally:
        for ctx in contexts:
            ctx.close()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    setattr(item, f"rep_{rep.when}", rep)
    return rep


# ---------------------------------------------------------------------------
# page helpers shared by the journeys
# ---------------------------------------------------------------------------

_RAIL_JS = """() => [...document.querySelectorAll('[aria-label=Stages] .rail-chip')]
  .map(e => [e.dataset.stage, (e.querySelector('.status > span:not(.meta)') || {}).textContent || ''])"""


def rail_states(page) -> list[tuple[str, str]]:
    """The Mission Control stage rail as ``[(stage id, state text)]`` in display order."""
    return [tuple(x) for x in page.evaluate(_RAIL_JS)]


def drag_on(page, locator, start: tuple[float, float], end: tuple[float, float], steps: int = 8) -> None:
    """Press at ``start`` and release at ``end`` (fractions of the element's box)."""
    locator.scroll_into_view_if_needed()
    box = wait_for(locator.bounding_box, 10, "the element to have a box")
    x0, y0 = box["x"] + box["width"] * start[0], box["y"] + box["height"] * start[1]
    x1, y1 = box["x"] + box["width"] * end[0], box["y"] + box["height"] * end[1]
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up()


def dd_of(scope, term: str):
    """The ``<dd>`` that follows the ``<dt>`` reading exactly ``term`` inside ``scope``."""
    return scope.locator("dt", has_text=re.compile(rf"^\s*{re.escape(term)}\s*$")).first.locator("xpath=following-sibling::dd[1]")


def wait_view_ready(page, what: str, timeout: float = 60) -> None:
    """Wait until a page under ``main`` has rendered its view (a heading below the page title, or an
    empty state) and finished loading (no spinner). An error state is a ``role=alert`` the caller checks."""
    wait_for(lambda: page.locator("main :is(h1, h2, h3)").count() >= 2 or page.locator("main .empty").count(),
             timeout, f"the {what} view")
    wait_for(lambda: not page.locator("main .spinner").count(), timeout, f"the {what} view to load")
