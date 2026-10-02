"""Engine states as the API reports them (api.md §7.2) and the server-side engine actions.

``sctx.services["engine"]`` is an :class:`EngineService`: the :class:`~.client.EngineClient` of the
workspace plus

* :meth:`~EngineService.state` - the host state of ``/api/health`` (``absent``, ``starting``, ``ready``,
  ``busy``, ``recycling``, ``error``);
* :meth:`~EngineService.host_info` - ``GET /api/engine``;
* :meth:`~EngineService.run_status` - ``GET /api/runs/{rid}/engine``: ``no_checkpoint`` → ``incompatible``
  (``studio/engine/status.json`` written by the host, valid while the checkpoint is unchanged) → ``queued`` /
  ``loading`` (an ``engine.open`` job; its tracker gives progress and the step) → ``busy`` / ``ready``
  (loaded in the host) → ``loading`` (an exact request the host is serving loads the cold run first) →
  ``error`` (the last open failed) → ``cold``;
* :meth:`~EngineService.preflight` - checkpoint, pickle trust and memory before a request that loads a run:
  ``409 no_checkpoint``, ``409 untrusted_pickle`` (the action re-imports the run with trust, naming the
  risk), ``409 engine_memory`` (estimated RSS ≈ 3.5 × checkpoint bytes + 0.3 GB, plus 1 GB, must fit in
  available memory; ``detail.holders`` lists loaded runs and live jobs, ``action`` evicts the least recently
  used run or stops the biggest job).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from sparc.studio.engine.client import EngineClient, EngineError, HostGone
from sparc.studio.errors import ApiError
from sparc.studio.workspace import read_json

log = logging.getLogger("sparc.studio.engine")

__all__ = ["EngineService", "estimate_rss_gb", "get_service", "TRUST_RISK", "OPEN_KIND"]

OPEN_KIND = "engine.open"
TRUST_RISK = ("checkpoint files execute code when loaded; trust only folders you produced")
_STATUS_TTL_S = 0.5


def estimate_rss_gb(checkpoint_bytes: float) -> float:
    """SPEC §7.6: ≈ 3.5 × checkpoint bytes + 0.3 GB."""
    return 3.5 * float(checkpoint_bytes) / 1e9 + 0.3


def get_service(sctx) -> "EngineService":
    svc = sctx.services.get("engine")
    if svc is None:
        svc = EngineService(sctx)
        sctx.services["engine"] = svc
    return svc


class EngineService:
    def __init__(self, sctx):
        self.sctx = sctx
        settings = sctx.settings()
        self.client = EngineClient(sctx.workspace, idle_min=settings.engine_idle_min,
                                   threads=settings.engine_threads)
        self._lock = threading.Lock()
        self._status: tuple[float, dict | None] | None = None

    # ------------------------------------------------------------------ host

    def limits(self) -> dict:
        s = self.sctx.settings()
        return {"max_runs": int(s.engine_max_runs), "budget_gb": float(s.engine_mem_budget_gb),
                "idle_min": float(s.engine_idle_min)}

    def host_status(self, *, fresh: bool = False) -> dict | None:
        now = time.monotonic()
        with self._lock:
            hit = self._status
        if not fresh and hit is not None and now - hit[0] < _STATUS_TTL_S:
            return hit[1]
        st = self.client.status(timeout=5.0)
        with self._lock:
            self._status = (now, st)
        return st

    def invalidate(self) -> None:
        with self._lock:
            self._status = None

    def state(self) -> str:
        if self.client.starting:
            return "starting"
        if not self.client.alive():
            return "error" if self.client.last_error else "absent"
        st = self.host_status()
        if st is None:
            return "error"
        if st.get("recycle"):
            return "recycling"
        return "busy" if st.get("busy") else "ready"

    def host_info(self) -> dict:
        """``GET /api/engine``."""
        lim = self.limits()
        st = self.host_status() if self.client.alive() else None
        queue = [r["id"] for r in self.sctx.db.fetchall(
            "SELECT id FROM jobs WHERE lane = 'engine' AND status IN ('queued','blocked') "
            "ORDER BY priority DESC, created_utc ASC")]
        busy = None
        if st and st.get("busy"):
            busy = st["busy"].get("job_id")
        return {"state": self.state(), "pid": (st or {}).get("pid"), "rss_mb": (st or {}).get("rss_mb"),
                "budget_gb": lim["budget_gb"], "max_runs": lim["max_runs"],
                "runs": [{"run_id": r["run_id"], "est_rss_mb": float(r.get("est_rss_mb") or 0.0),
                          "loaded_utc": r.get("loaded_utc") or "", "last_used_utc": r.get("last_used_utc") or ""}
                         for r in (st or {}).get("runs") or []],
                "busy_job_id": busy, "queue": queue}

    def reconnect(self) -> bool:
        """Server start: keep a live host (pid + create_time match), else remove its stale files."""
        info = self.client.info()
        if info is None:
            self.client.cleanup_stale()
            return False
        log.info("reconnected to engine host %s", info.get("pid"))
        self.invalidate()
        return True

    def restart(self) -> bool:
        """Kill the host (evicting every engine); it starts again on the next request."""
        killed = self.client.kill()
        self.invalidate()
        self.publish(None, "absent")
        return killed

    def shutdown(self) -> None:
        self.client.stop()
        self.invalidate()

    def evict(self, run_id: str) -> bool:
        if not self.client.alive():
            return False
        try:
            res = self.client.request("close", run_id=run_id, start=False, timeout=120.0,
                                      payload={"limits": self.limits()})
        except (HostGone, EngineError) as exc:
            log.warning("evicting %s failed: %s", run_id, exc)
            return False
        self.invalidate()
        self.publish(run_id, "cold")
        return bool(res.get("closed"))

    def publish(self, run_id: str | None, state: str, progress: float | None = None, rss_mb: float | None = None):
        try:
            self.sctx.hub.publish("engine.status", {"run_id": run_id, "state": state, "progress": progress,
                                                    "rss_mb": rss_mb})
        except Exception:
            pass

    # ------------------------------------------------------------------ runs

    def _row(self, rid: str) -> dict:
        row = self.sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (rid,))
        if row is None:
            raise ApiError("not_found", f"no run {rid!r}")
        return row

    @staticmethod
    def studio_dir(row: dict) -> Path:
        return Path(row["studio_dir"]) if row.get("studio_dir") else Path(row["run_dir"]) / "studio"

    def checkpoint_bytes(self, row: dict) -> int | None:
        p = Path(row["run_dir"]) / "checkpoint.pkl"
        try:
            return int(p.stat().st_size)
        except OSError:
            return None

    def trusted(self, run_id: str) -> bool:
        reg = self.sctx.services.get("registry")
        if reg is None:
            return False
        try:
            return bool(reg.trusted(run_id))
        except Exception:
            return False

    def incompatible(self, row: dict) -> dict | None:
        """The host's ``incompatible`` record of this checkpoint, if any."""
        from sparc.core.session import checkpoint_key

        st = read_json(self.studio_dir(row) / "engine" / "status.json")
        if not isinstance(st, dict) or st.get("state") != "incompatible":
            return None
        if st.get("ckpt_key") != checkpoint_key(row["run_dir"]):
            return None
        return st

    def open_job(self, rid: str, statuses=("queued", "blocked", "starting", "running", "cancelling")) -> dict | None:
        marks = ",".join("?" for _ in statuses)
        return self.sctx.db.fetchone(f"SELECT * FROM jobs WHERE kind = ? AND run_id = ? AND status IN ({marks}) "
                                     f"ORDER BY created_utc DESC", (OPEN_KIND, rid, *statuses))

    def loaded(self, rid: str) -> dict | None:
        st = self.host_status() if self.client.alive() else None
        for r in (st or {}).get("runs") or []:
            if r.get("run_id") == rid:
                return r
        return None

    def trust_action(self, row: dict) -> dict:
        rec = read_json(self.studio_dir(row) / "import.json") or {}
        return {"kind": "open", "label": f"Trust this run's checkpoint ({TRUST_RISK})", "method": "POST",
                "path": "/api/runs/import",
                "body": {"dir": str(row["run_dir"]), "project_id": row.get("project_id"),
                         "config_path": rec.get("config_path"), "trust_pickles": True}}

    def run_status(self, rid: str) -> dict:
        """``GET /api/runs/{rid}/engine`` (api.md §7.2)."""
        row = self._row(rid)
        nbytes = self.checkpoint_bytes(row)
        est = round(estimate_rss_gb(nbytes or 0) * 1024.0, 1)
        out: dict[str, Any] = {"state": "cold", "progress": None, "step": None, "rss_mb": None, "est_rss_mb": est,
                               "code_match": None, "loaded_utc": None, "last_used_utc": None, "error": None,
                               "job_id": None, "action": None}
        side = read_json(Path(row["run_dir"]) / "checkpoint.json") or {}
        if side.get("code_sha256"):
            from sparc.studio.engine.store import code_sha

            out["code_match"] = side["code_sha256"] == code_sha()
        if nbytes is None:
            out["state"] = "no_checkpoint"
            return out
        inc = self.incompatible(row)
        if inc is not None:
            err = inc.get("error") or {}
            out.update(state="incompatible", error={"type": err.get("type") or "Incompatible",
                                                    "message": (err.get("message") or "") + " — refit S2/S3 from "
                                                    "the launch snapshot, or use preview only"},
                       action={"kind": "run_job", "label": "Refit S2/S3 (rerun from the launch snapshot)",
                               "method": "POST", "path": f"/api/runs/{rid}/rerun",
                               "body": {"use_current_config": False}})
            return out
        job = self.open_job(rid)
        if job is not None:
            out["job_id"] = job["id"]
            if job["status"] in ("queued", "blocked"):
                out["state"] = "queued"
                return out
            out["state"] = "loading"
            out["progress"] = job.get("progress")
            out["step"] = self._step(job)
            return out
        loaded = self.loaded(rid)
        if loaded is not None:
            st = self.host_status() or {}
            busy = (st.get("busy") or {})
            out.update(state="busy" if busy.get("run_id") == rid else "ready",
                       rss_mb=float(loaded.get("est_rss_mb") or 0.0) or None,
                       loaded_utc=loaded.get("loaded_utc"), last_used_utc=loaded.get("last_used_utc"),
                       job_id=busy.get("job_id") if busy.get("run_id") == rid else None)
            if loaded.get("code_match") is not None:
                out["code_match"] = loaded["code_match"]
            return out
        busy = ((self.host_status() or {}).get("busy") or {}) if self.client.alive() else {}
        if busy.get("run_id") == rid and busy.get("job_id"):
            # an exact request on a cold run: the host loads the run first
            out.update(state="loading", job_id=busy["job_id"])
            jrow = self.sctx.db.fetchone("SELECT * FROM jobs WHERE id = ?", (busy["job_id"],))
            if jrow is not None:
                out.update(progress=jrow.get("progress"), step=self._step(jrow))
            return out
        if not self.trusted(rid):
            out.update(state="error", error={"type": "UntrustedPickle",
                                             "message": "this run was imported without trusting its checkpoint: "
                                                        + TRUST_RISK},
                       action=self.trust_action(row))
            return out
        last = self.sctx.db.fetchone("SELECT * FROM jobs WHERE kind = ? AND run_id = ? ORDER BY created_utc DESC",
                                     (OPEN_KIND, rid))
        if last is not None and last["status"] == "failed":
            from sparc.studio import db as dbmod

            err = dbmod.loads(last.get("error_json")) or {}
            out.update(state="error", error={"type": str(err.get("type") or "Error"),
                                             "message": str(err.get("message") or "")}, job_id=last["id"],
                       action={"kind": "open_engine", "label": "Open engine", "method": "POST",
                               "path": f"/api/runs/{rid}/engine/open", "body": {}})
            return out
        out["action"] = {"kind": "open_engine", "label": "Open engine", "method": "POST",
                         "path": f"/api/runs/{rid}/engine/open", "body": {}}
        return out

    def _step(self, job: dict) -> str | None:
        """The current step of a loading job: the last tick's label ("312/525 MB"), else its current task."""
        tailer = getattr(self.sctx.jobs, "tailers", {}).get(job["id"]) if self.sctx.jobs is not None else None
        st = getattr(tailer, "state", None) if tailer is not None else None
        tick = (st or {}).get("tick") if isinstance(st, dict) else None
        path = (st or {}).get("current_path") if isinstance(st, dict) else None
        from sparc.studio import db as dbmod

        path = path or dbmod.loads(job.get("current_path"))
        name = None
        if path:
            last = str(path[-1])
            name = last.split(":", 1)[-1].split("[", 1)[0]
        labels = {"load_run": "Loading data", "unpickle": "Loading checkpoint", "mediators": "Fitting mediators",
                  "engine_init": "Baseline pass"}
        step = labels.get(name or "", name)
        if tick and tick.get("label") and name == "unpickle":
            step = f"Loading checkpoint {tick['label']}"
        return step

    # ------------------------------------------------------------------ preflight

    def preflight(self, rid: str, *, memory: bool = True) -> None:
        """Checks before a request that opens a run's engine (``engine.open``, and any exact request on a run that
        is not loaded, which the host loads first); raises ``ApiError``.  The memory check is skipped while the
        run is loaded or an ``engine.open`` of it is already queued or loading (that one passed it)."""
        row = self._row(rid)
        nbytes = self.checkpoint_bytes(row)
        if nbytes is None:
            raise ApiError("no_checkpoint", "this run has no checkpoint.pkl (it did not reach S3); exact "
                                            "scenarios need one", detail={"run_id": rid})
        if not self.trusted(rid):
            raise ApiError("untrusted_pickle", "this run was imported without trusting its checkpoint: "
                           + TRUST_RISK, detail={"run_id": rid, "run_dir": str(row["run_dir"])},
                           action=self.trust_action(row))
        if memory and self.loaded(rid) is None and self.open_job(rid) is None:
            self.memory_check(rid, nbytes)

    def memory_check(self, rid: str, nbytes: int) -> None:
        from sparc.studio.jobs.resources import memory_available_gb

        need = estimate_rss_gb(nbytes) + 1.0
        avail = memory_available_gb()
        if need <= avail:
            return
        holders = []
        st = self.host_status() if self.client.alive() else None
        for r in (st or {}).get("runs") or []:
            holders.append({"run_id": r["run_id"], "rss_gb": round(float(r.get("est_rss_mb") or 0) / 1024.0, 2)})
        jobs = self.sctx.jobs
        samples = getattr(jobs, "samples", {}) if jobs is not None else {}
        for jid, smp in samples.items():
            if smp and smp.get("rss_mb"):
                holders.append({"job_id": jid, "rss_gb": round(float(smp["rss_mb"]) / 1024.0, 2)})
        action = None
        loaded = [h for h in holders if h.get("run_id")]
        if loaded:
            victim = loaded[0]["run_id"]
            action = {"kind": "open", "label": f"Evict run {victim} from the engine", "method": "POST",
                      "path": f"/api/runs/{victim}/engine/evict", "body": {}}
        else:
            big = max((h for h in holders if h.get("job_id")), key=lambda h: h["rss_gb"], default=None)
            if big is not None:
                action = {"kind": "open", "label": f"Stop job {big['job_id']}", "method": "POST",
                          "path": f"/api/jobs/{big['job_id']}/cancel", "body": {}}
        raise ApiError("engine_memory", f"opening this run needs about {need:.1f} GB of memory; "
                                        f"{avail:.1f} GB is available",
                       detail={"needed_gb": round(need, 2), "available_gb": round(avail, 2), "holders": holders},
                       action=action)
