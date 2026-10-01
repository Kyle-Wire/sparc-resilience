"""Resource monitor and preflight checks (SPEC §5.13, §10.4).

:class:`ResourceSampler` samples every live job's process tree every 2 s
with psutil (RSS sum, CPU %, process count, threads).  Each sample goes to
the job's SSE subscribers as a transient ``resource`` event (no id); one
sample every 10 s is stored in ``resource_samples``; ``jobs.peak_rss_mb`` is
maintained; ``storage.low`` is broadcast while free workspace disk is below
2 GB (again every 5 minutes while it stays low).

The preflight helpers compare estimates with what the machine has:
memory = ``psutil.virtual_memory().available`` minus the RSS of other live
jobs (refuse when estimate + 1 GB exceeds it), disk = free bytes of the
workspace (refuse when the estimate + 1 GB exceeds it).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from typing import Any, Callable

log = logging.getLogger("sparc.studio.jobs")

__all__ = ["sample_tree", "ResourceSampler", "memory_available_gb", "preflight_memory", "preflight_disk",
           "STORAGE_LOW_BYTES", "STORE_EVERY_S"]

STORAGE_LOW_BYTES = 2 * 1024 ** 3
STORE_EVERY_S = 10.0
STORAGE_REPEAT_S = 300.0
_GB = 1024 ** 3


def _psutil():
    import psutil

    return psutil


class _TreeSampler:
    """Keeps ``psutil.Process`` objects between samples so ``cpu_percent`` measures the interval."""

    def __init__(self):
        self.trees: dict[int, dict[int, Any]] = {}      # root pid → {pid: Process}

    def sample(self, pid: int) -> dict | None:
        psutil = _psutil()
        known = self.trees.get(pid, {})
        try:
            root = known.get(pid) or psutil.Process(pid)
            members = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            self.trees.pop(pid, None)
            return None
        rss = cpu = 0.0
        threads = n = 0
        seen: dict[int, Any] = {}
        for p in members:
            proc = known.get(p.pid) or p
            try:
                with proc.oneshot():
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        continue
                    rss += proc.memory_info().rss
                    cpu += proc.cpu_percent(None)
                    threads += proc.num_threads()
                n += 1
                seen[proc.pid] = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if n == 0:
            self.trees.pop(pid, None)
            return None
        self.trees[pid] = seen
        return {"rss_mb": round(rss / 2 ** 20, 1), "cpu_pct": round(cpu, 1), "n_procs": n, "threads": threads}

    def forget(self, keep: set[int]) -> None:
        for pid in list(self.trees):
            if pid not in keep:
                self.trees.pop(pid, None)


_SHARED = _TreeSampler()


def sample_tree(pid: int) -> dict | None:
    """``{rss_mb, cpu_pct, n_procs, threads}`` of ``pid`` and its descendants, or None when it is gone.

    The engine host item uses this for ``engine.status``.  CPU % is measured
    since the previous call for the same processes (0 on the first call).
    """
    return _SHARED.sample(pid)


def memory_available_gb(exclude_rss_mb: float = 0.0) -> float:
    vm = _psutil().virtual_memory()
    return max(0.0, vm.available / _GB - exclude_rss_mb / 1024)


def preflight_memory(estimate_gb: float, other_rss_mb: float = 0.0) -> dict | None:
    """None when ``estimate_gb + 1`` fits in available memory minus other live jobs' RSS, else a reason dict."""
    avail = memory_available_gb(other_rss_mb)
    need = float(estimate_gb) + 1.0
    if need > avail:
        return {"reason": f"needs about {need:.1f} GB of memory; {avail:.1f} GB available",
                "needed_gb": round(need, 2), "available_gb": round(avail, 2)}
    return None


def preflight_disk(path, need_bytes: float) -> dict | None:
    """None when ``need_bytes + 1 GB`` fits on the disk holding ``path``, else a reason dict."""
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return None
    need = float(need_bytes) + _GB
    if need > free:
        return {"reason": f"needs about {need / _GB:.1f} GB of disk; {free / _GB:.1f} GB free",
                "needed_gb": round(need / _GB, 2), "free_gb": round(free / _GB, 2)}
    return None


class ResourceSampler:
    """Periodic sampler of live jobs (see the module docstring).

    ``live_jobs()`` returns ``{job_id: pid}`` for the jobs to sample;
    ``on_sample(job_id, sample)`` receives each sample (the manager keeps
    the latest for out-of-memory labelling and peak RSS).
    """

    def __init__(self, *, db, hub, workspace_root, live_jobs: Callable[[], dict[str, int]],
                 on_sample: Callable[[str, dict], None] | None = None, interval: float = 2.0,
                 store_every: float = STORE_EVERY_S):
        self.db = db
        self.hub = hub
        self.root = workspace_root
        self.live_jobs = live_jobs
        self.on_sample = on_sample
        self.interval = interval
        self.store_every = store_every
        self.sampler = _TreeSampler()
        self.last_store: dict[str, float] = {}
        self.peak: dict[str, float] = {}
        self.latest: dict[str, dict] = {}
        self._low_sent = 0.0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> "ResourceSampler":
        self._task = asyncio.create_task(self._run(), name="resource-sampler")
        return self

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("resource sampling failed")
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
            except asyncio.TimeoutError:
                pass

    async def tick(self) -> None:
        jobs = dict(self.live_jobs())
        samples = await asyncio.to_thread(self._sample_all, jobs)
        now = time.time()
        store: list[tuple] = []
        peaks: list[tuple] = []
        for jid, sample in samples.items():
            if sample is None:
                continue
            sample = {"ts": round(now, 3), **sample}
            self.latest[jid] = sample
            if self.on_sample is not None:
                self.on_sample(jid, sample)
            self.hub.publish_job(jid, ("resource", sample))
            if sample["rss_mb"] > self.peak.get(jid, 0.0):
                self.peak[jid] = sample["rss_mb"]
                peaks.append((sample["rss_mb"], jid))
            if now - self.last_store.get(jid, 0.0) >= self.store_every:
                self.last_store[jid] = now
                store.append((jid, sample["ts"], sample["rss_mb"], sample["cpu_pct"], sample["n_procs"],
                              sample["threads"]))
        for jid in list(self.latest):
            if jid not in jobs:
                self.latest.pop(jid, None)
                self.last_store.pop(jid, None)
        if store or peaks:
            def _write(conn):
                if store:
                    conn.executemany("INSERT INTO resource_samples (job_id, ts, rss_mb, cpu_pct, n_procs, threads) "
                                     "VALUES (?,?,?,?,?,?)", store)
                for rss, jid in peaks:
                    conn.execute("UPDATE jobs SET peak_rss_mb = ? WHERE id = ? AND (peak_rss_mb IS NULL OR "
                                 "peak_rss_mb < ?)", (rss, jid, rss))
            await self.db.atransaction(_write)
        await self._check_storage(now)

    def _sample_all(self, jobs: dict[str, int]) -> dict[str, dict | None]:
        self.sampler.forget({pid for pid in jobs.values() if pid})
        return {jid: self.sampler.sample(pid) for jid, pid in jobs.items() if pid}

    async def _check_storage(self, now: float) -> None:
        try:
            free = (await asyncio.to_thread(shutil.disk_usage, self.root)).free
        except OSError:
            return
        if free < STORAGE_LOW_BYTES:
            if now - self._low_sent >= STORAGE_REPEAT_S:
                self._low_sent = now
                self.hub.publish("storage.low", {"free_bytes": int(free), "threshold_bytes": STORAGE_LOW_BYTES})
        else:
            self._low_sent = 0.0
