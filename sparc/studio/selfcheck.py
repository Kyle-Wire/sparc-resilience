"""``python -m sparc.studio.selfcheck``: check that SPARC Studio works on this machine.

Runs the core of the start-to-end journey against a real server in a throw-away
workspace, through the same HTTP API the web app uses:

1. start ``sparc studio`` and wait for ``/api/health``;
2. create the synthetic demo project;
3. launch a fast run, cancel it mid-fit and check it ends ``cancelled``;
4. resume it from its checkpoint and wait for it to finish;
5. open the Scenario Lab engine on the run;
6. run an exact scenario (+10 canopy everywhere) and read its result;
7. build a decision pack through Exports;
8. shut the server down and check no worker or engine process is left behind.

Each step prints PASS or FAIL with its time; the exit code is 0 only when every
step passed. On GitHub Actions a failure is also written as an ``::error``
annotation with the server log tail, so CI failures are readable from the
check run alone. The workspace is deleted unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

FINAL = ("succeeded", "failed", "cancelled", "interrupted")


class CheckFailed(AssertionError):
    pass


class Server:
    """A ``sparc studio`` child process and an authenticated client."""

    def __init__(self, workspace: Path, timeout: float):
        import httpx

        self.workspace = workspace
        self.token = secrets.token_urlsafe(16)
        self.log_path = workspace.parent / "server.log"
        workspace.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop("SPARC_STUDIO_HOME", None)
        cmd = [sys.executable, "-m", "sparc.studio", "--workspace", str(workspace), "--port", "0", "--no-browser",
               "--token", self.token]
        kw: dict = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kw["start_new_session"] = True
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, **kw)
        lock = workspace / "studio.lock.json"
        deadline = time.monotonic() + timeout
        self.base = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise CheckFailed(f"sparc studio exited with code {self.proc.returncode}")
            try:
                info = json.loads(lock.read_text("utf-8"))
                if info.get("pid") == self.proc.pid and info.get("port"):
                    base = f"http://127.0.0.1:{int(info['port'])}"
                    if httpx.get(base + "/api/health", timeout=2, trust_env=False).status_code == 200:
                        self.base = base
                        break
            except (OSError, ValueError, httpx.HTTPError):
                pass
            time.sleep(0.25)
        if self.base is None:
            raise CheckFailed(f"sparc studio did not answer /api/health within {timeout:.0f} s")
        self.http = httpx.Client(base_url=self.base, headers={"Authorization": f"Bearer {self.token}"},
                                 timeout=120.0, trust_env=False)
        self.wait(lambda: self.http.get("/api/runs").status_code == 200, 120, "the runs registry")

    # -- API ----------------------------------------------------------------
    def call(self, method: str, path: str, **kw):
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise CheckFailed(f"{method} {path} -> {r.status_code}: {r.text[:600]}")
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.content

    def get(self, path: str, **kw):
        return self.call("GET", path, **kw)

    def post(self, path: str, body=None, **kw):
        return self.call("POST", path, json=body if body is not None else {}, **kw)

    @staticmethod
    def wait(pred, timeout: float, what: str, interval: float = 0.5):
        deadline = time.monotonic() + timeout
        while True:
            val = pred()
            if val:
                return val
            if time.monotonic() > deadline:
                raise CheckFailed(f"timed out after {timeout:.0f} s waiting for {what}")
            time.sleep(interval)

    def wait_job(self, jid: str, timeout: float) -> dict:
        job = self.wait(lambda: (j := self.get(f"/api/jobs/{jid}"))["status"] in FINAL and j, timeout,
                        f"job {jid} to finish")
        return job

    def log_tail(self, n: int = 80) -> str:
        try:
            return "\n".join(self.log_path.read_text("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return ""

    def job_error(self, jid: str) -> str:
        try:
            job = self.get(f"/api/jobs/{jid}")
            tail = self.get(f"/api/jobs/{jid}/logs", params={"limit": 60})
            lines = tail.get("lines") if isinstance(tail, dict) else None
            return json.dumps(job.get("error") or job.get("result"), default=str)[:1500] + (
                "\n" + "\n".join(str(x) for x in (lines or [])[-40:]) if lines else "")
        except Exception as exc:                                  # noqa: BLE001 - diagnostics only
            return f"(could not read job {jid}: {exc})"

    # -- shutdown -----------------------------------------------------------
    def descendants(self) -> list:
        import psutil

        try:
            return psutil.Process(self.proc.pid).children(recursive=True)
        except psutil.Error:
            return []

    def shutdown(self, timeout: float = 60.0) -> list[str]:
        """Stop the server; return the command lines of processes it left behind."""
        import psutil

        kids = self.descendants()
        try:
            self.http.post("/api/shutdown", json={"stop_jobs": True}, timeout=15)
        except Exception:                                         # noqa: BLE001 - the server may close first
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=15)
            raise CheckFailed(f"the server did not exit within {timeout:.0f} s of POST /api/shutdown")
        finally:
            self.http.close()
        _, alive = psutil.wait_procs(kids, timeout=20)
        left = []
        for p in alive:
            try:
                left.append(" ".join(p.cmdline())[:200])
                p.kill()
            except psutil.Error:
                pass
        return left


def _annotate(title: str, text: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    body = text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title=selfcheck: {title}::{body[:60000]}", flush=True)


def run(args) -> int:
    root = Path(args.dir).resolve() if args.dir else Path(tempfile.mkdtemp(prefix="sparc-selfcheck-"))
    ws = root / "workspace"
    print(f"SPARC Studio self-check ({sys.platform}, Python {sys.version.split()[0]})")
    print(f"workspace: {ws}")
    state: dict = {}
    server: Server | None = None
    results: list[tuple[str, bool, float, str]] = []

    def step(name: str, fn) -> bool:
        t0 = time.monotonic()
        try:
            note = fn() or ""
            results.append((name, True, time.monotonic() - t0, note))
            print(f"  PASS  {name:<38} {time.monotonic() - t0:6.1f} s  {note}", flush=True)
            return True
        except Exception as exc:                                  # noqa: BLE001 - every failure is reported
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            results.append((name, False, time.monotonic() - t0, detail))
            print(f"  FAIL  {name:<38} {time.monotonic() - t0:6.1f} s  {detail}", flush=True)
            extra = ""
            if state.get("job"):
                extra += "\n--- job ---\n" + (server.job_error(state["job"]) if server else "")
            extra += "\n--- server log ---\n" + (server.log_tail() if server else "")
            if args.verbose or os.environ.get("GITHUB_ACTIONS") == "true":
                print(traceback.format_exc() + extra, flush=True)
            _annotate(name, f"{detail}\n{traceback.format_exc()}{extra}")
            return False

    def start():
        nonlocal server
        server = Server(ws, timeout=args.start_timeout)
        return server.base

    def demo():
        p = server.post("/api/projects", {"name": "Self-check city", "template": "synthetic_demo",
                                          "options": {"n": 40, "seed": 0}})["project"]
        state["pid"] = p["id"]
        return p["id"]

    def cancel():
        r = server.post(f"/api/projects/{state['pid']}/runs", {"mode": "fast", "label": "self-check"})
        state["rid"], state["job"] = r["run"]["id"], r["job"]["id"]

        def fitting():
            stages = {s["id"]: s["state"] for s in server.get(f"/api/runs/{state['rid']}")["stages"]}
            job = server.get(f"/api/jobs/{state['job']}")
            if job["status"] in FINAL:
                raise CheckFailed(f"the run ended {job['status']} before it could be cancelled")
            return stages.get("S2_S3") == "running"

        server.wait(fitting, args.run_timeout, "the run to start fitting (S2-S3)")
        t0 = time.monotonic()
        server.post(f"/api/jobs/{state['job']}/cancel")
        job = server.wait_job(state["job"], 120)
        if job["status"] != "cancelled":
            raise CheckFailed(f"cancel ended the job {job['status']} (expected cancelled)")
        return f"cancelled in {time.monotonic() - t0:.1f} s"

    def resume():
        job = server.post(f"/api/runs/{state['rid']}/resume")
        state["job"] = job["id"]
        job = server.wait_job(state["job"], args.run_timeout)
        if job["status"] != "succeeded":
            raise CheckFailed(f"the resumed run ended {job['status']}")
        run = server.get(f"/api/runs/{state['rid']}")["run"]
        if run["status"] != "complete":
            raise CheckFailed(f"run status {run['status']} after the resume (expected complete)")
        r2 = ((job.get("result") or {}).get("metrics") or {}).get("r2")
        return f"held-out R2 {r2:.2f}" if isinstance(r2, (int, float)) else ""

    def engine():
        state["job"] = None
        r = server.post(f"/api/runs/{state['rid']}/engine/open")
        if isinstance(r, dict) and r.get("id", "").startswith("j_"):
            state["job"] = r["id"]
        st = server.wait(lambda: (e := server.get(f"/api/runs/{state['rid']}/engine"))["state"]
                         in ("ready", "error", "incompatible") and e, args.run_timeout, "the engine to load")
        if st["state"] != "ready":
            raise CheckFailed(f"engine state {st['state']}: {st.get('error')}")
        return f"rss {st.get('rss_mb') or 0:.0f} MB"

    def exact():
        sc = server.post(f"/api/projects/{state['pid']}/scenarios", {"doc": {
            "name": "Self-check +10 canopy", "edits": [{"lever": "canopy", "mode": "add", "amount": 10}]}})
        r = server.post(f"/api/scenarios/{sc['id']}/run", {"run_id": state["rid"]})
        if r.get("job"):
            state["job"] = r["job"]["id"]
            job = server.wait_job(state["job"], args.run_timeout)
            if job["status"] != "succeeded":
                raise CheckFailed(f"the exact scenario job ended {job['status']}")
        results_ = [x for x in server.get(f"/api/scenarios/{sc['id']}")["results"]
                    if x["run_id"] == state["rid"] and x["kind"] == "exact"]
        if not results_:
            raise CheckFailed("no exact result was recorded for the scenario")
        state["result"] = results_[0]["id"]
        res = server.get(f"/api/results/{state['result']}")
        city = ((res.get("city") or res.get("summary") or {}))
        est = city.get("estimate") if isinstance(city, dict) else None
        return f"city mean {est:+.3f}" if isinstance(est, (int, float)) else state["result"]

    def pack():
        r = server.post("/api/exports", {"kind": "decision_pack", "project_id": state["pid"],
                                         "params": {"result_id": state["result"]}})
        state["job"] = r["job"]["id"]
        eid = r["export"]["id"]
        ex = server.wait(lambda: (e := server.get(f"/api/exports/{eid}"))["status"] in ("ready", "failed") and e,
                         args.run_timeout, "the decision pack")
        if ex["status"] != "ready":
            raise CheckFailed("the decision pack export failed")
        data = server.get(f"/api/exports/{eid}/download")
        if not isinstance(data, (bytes, bytearray)) or data[:2] != b"PK":
            raise CheckFailed("the decision pack download is not a zip file")
        return f"{len(data) / 1024:.0f} kB"

    def stop():
        left = server.shutdown()
        if left:
            raise CheckFailed("processes left running after shutdown:\n" + "\n".join(left))
        return ""

    ok = step("start the server", start)
    if ok:
        for name, fn in (("create the demo project", demo), ("launch, then cancel mid-fit", cancel),
                         ("resume from the checkpoint", resume), ("open the Scenario Lab engine", engine),
                         ("run an exact scenario", exact), ("export a decision pack", pack)):
            if not step(name, fn):
                ok = False
                break
        ok = step("shut down cleanly", stop) and ok
    n_pass = sum(1 for r in results if r[1])
    print(f"{'OK' if ok else 'FAILED'}: {n_pass}/{len(results)} steps passed")
    if not ok:
        print(f"server log: {server.log_path if server else ws.parent / 'server.log'}")
    if not args.keep and ok:
        shutil.rmtree(root, ignore_errors=True)
    elif args.keep or not ok:
        print(f"kept: {root}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sparc.studio.selfcheck",
                                description="Check that SPARC Studio works on this machine (about 2-5 minutes).")
    p.add_argument("--dir", help="folder for the throw-away workspace (default: a new temporary folder)")
    p.add_argument("--keep", action="store_true", help="keep the workspace and server log afterwards")
    p.add_argument("--start-timeout", type=float, default=180.0, help="seconds to wait for the server to start")
    p.add_argument("--run-timeout", type=float, default=1200.0, help="seconds to allow each run, engine or export")
    p.add_argument("-v", "--verbose", action="store_true", help="print tracebacks and the server log on failure")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
