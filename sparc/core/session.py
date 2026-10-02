"""Open a finished run for interactive scenarios (SPEC §7.6, §11 item 17).

:func:`open_run` turns a run directory with a ``checkpoint.pkl`` into a
:class:`RunSession`: the data and folds rebuilt from the run's config, the
fitted ensemble and influence ranges from the checkpoint, the mediator chain,
a :class:`~sparc.core.scenarios.ScenarioEngine` and, on first use, a
:class:`~sparc.core.response.ResponseEngine`.  The Studio engine host keeps a
few of these in memory and answers scenario requests from them.

Loading reports through :mod:`sparc.core.progress`:

* ``task load_run`` - the data and folds (``baselines.load_run``);
* ``task unpickle`` - the checkpoint, read through a byte-counting reader
  that ticks ``312/525 MB`` (unit ``unpickle``; a ``k == n`` tick completes
  the planned unit) and checks for a cancel request between reads;
* ``task mediators`` (unit ``mediator_fit``) under ``limit_threads``;
* ``task engine_init`` - the ScenarioEngine; its baseline pass ticks
  ``engine_pass`` unless a cached ``base_fold`` (K × n) is given.

Every GWRF base model gets ``n_jobs = threads`` (a runtime setting: the
fitted trees are unchanged), so an engine request never oversubscribes the
machine.  A checkpoint whose classes no longer match the code (an
``AttributeError`` / ``ImportError`` while unpickling or in the first
evaluation) raises :class:`IncompatibleCheckpoint`.

``emulator.emulator_for_run`` keeps its own loader; ``tests/studio/engine``
asserts both paths give identical scenario deltas.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sparc.core import progress
from sparc.core.config import CoreConfig, core_config_from_dict, load_core_config

log = logging.getLogger(__name__)

__all__ = ["RunSession", "IncompatibleCheckpoint", "config_for_run", "open_run", "checkpoint_key", "code_digest",
           "DEFAULT_DROP"]

DEFAULT_DROP = ("cv_distance", "causal_arrays")
_MB = 1024 * 1024
_INCOMPATIBLE = (AttributeError, ImportError)            # ModuleNotFoundError is an ImportError


class IncompatibleCheckpoint(RuntimeError):
    """The checkpoint was written by core code whose classes no longer match (refit S2/S3, or use preview only)."""


# ---------------------------------------------------------------------------
# config resolution (SPEC §4.3)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _usable(cfg: CoreConfig, run_dir: Path) -> bool:
    """A config whose data file exists (or a frame-input run, rebuilt from ``input_frame.parquet``)."""
    if (run_dir / "input_frame.parquet").exists():
        return True
    try:
        return Path(cfg.data_path).is_file()
    except (ValueError, TypeError):
        return False


def _apply_modes(cfg: CoreConfig, run_dir: Path, args: dict | None, manifest: dict | None) -> CoreConfig:
    """The run's mode overrides (fast / coarse / CV curve) from ``launch.json`` args, else the manifest."""
    from sparc.core.pipeline import apply_mode_overrides

    a = dict(args or {})
    if not a and manifest:
        co = (manifest.get("qa") or {}).get("coarse") or {}
        a = {"fast": bool(manifest.get("fast_mode")), "coarse": co.get("cell_m")}
    return apply_mode_overrides(cfg, fast=bool(a.get("fast")), coarse=a.get("coarse"), cv_curve=a.get("cv_curve"),
                                frame_given=(run_dir / "input_frame.parquet").exists())


def config_for_run(run_dir, fallback: CoreConfig | str | os.PathLike | None = None, *,
                   studio_dir: str | os.PathLike | None = None) -> CoreConfig:
    """The config a run was made with, resolved in the order of SPEC §4.3.

    1. ``<studio_dir>/launch.json`` (``config_raw`` + ``config_dir``, with the launch mode overrides);
       ``studio_dir`` defaults to ``<run_dir>/studio``;
    2. ``manifest.config`` + ``manifest.provenance.config_dir``;
    3. ``fallback``: a :class:`CoreConfig` or a config path given at import.

    A candidate whose data file is missing yields to the next one; the first candidate is returned when none
    has its data (the caller's ``load_run`` then reports the missing file).
    """
    run_dir = Path(run_dir)
    sdir = Path(studio_dir) if studio_dir is not None else run_dir / "studio"
    manifest = _read_json(run_dir / "manifest.json")
    cands: list[CoreConfig] = []
    launch = _read_json(sdir / "launch.json")
    if launch and isinstance(launch.get("config_raw"), dict):
        try:
            cfg = core_config_from_dict(copy.deepcopy(launch["config_raw"]),
                                        base_dir=launch.get("config_dir") or run_dir)
            cands.append(_apply_modes(cfg, run_dir, launch.get("args"), manifest))
        except (OSError, ValueError) as exc:
            log.warning("%s: launch.json config unusable: %s", run_dir, exc)
    if manifest and isinstance(manifest.get("config"), dict):
        base = (manifest.get("provenance") or {}).get("config_dir") or run_dir
        try:
            cfg = core_config_from_dict(copy.deepcopy(manifest["config"]), base_dir=base)
            cands.append(_apply_modes(cfg, run_dir, None, manifest))
        except (OSError, ValueError) as exc:
            log.warning("%s: manifest config unusable: %s", run_dir, exc)
    if fallback is not None:
        cfg = fallback if isinstance(fallback, CoreConfig) else load_core_config(fallback)
        cands.append(_apply_modes(copy.deepcopy(cfg), run_dir, None, manifest))
    if not cands:
        raise FileNotFoundError(f"{run_dir}: no launch.json, manifest config or fallback config")
    for cfg in cands:
        if _usable(cfg, run_dir):
            return cfg
    return cands[0]


# ---------------------------------------------------------------------------
# checkpoint identity
# ---------------------------------------------------------------------------

def checkpoint_key(run_dir) -> str | None:
    """``"<mtime_ns>:<size>"`` of ``checkpoint.pkl`` (the cache key of exact results and ``base_fold.npy``)."""
    try:
        st = os.stat(Path(run_dir) / "checkpoint.pkl")
    except OSError:
        return None
    return f"{st.st_mtime_ns}:{st.st_size}"


def code_digest() -> str:
    """SHA-256 of the fingerprinted core sources (the checkpoint sidecar's ``code_sha256``)."""
    from sparc.core.pipeline import _code_digest

    return _code_digest()


# ---------------------------------------------------------------------------
# the byte-counting reader
# ---------------------------------------------------------------------------

class _CountingReader(io.RawIOBase):
    """A read-only file wrapper for ``pickle.load`` that counts bytes, ticks progress and checks for cancel."""

    def __init__(self, fh, total: int, callback: Callable[[int, int], Any] | None = None):
        super().__init__()
        self.fh = fh
        self.total = max(int(total), 1)
        self.n_mb = max(1, int(math.ceil(self.total / _MB)))
        self.done = 0
        self.callback = callback
        self._last_mb = -1

    def readable(self) -> bool:
        return True

    def _count(self, n: int) -> None:
        if n <= 0:
            return
        self.done += n
        mb = min(self.n_mb, int(self.done // _MB))
        if mb != self._last_mb and mb >= 1:
            self._last_mb = mb
            if mb < self.n_mb:              # the final k == n tick is sent by finish()
                progress.tick(mb, self.n_mb, unit="unpickle", label=f"{mb}/{self.n_mb} MB")
            if self.callback is not None:
                self.callback(self.done, self.total)
            progress.check_cancel()

    def read(self, n: int = -1) -> bytes:
        b = self.fh.read(n)
        self._count(len(b))
        return b

    def readinto(self, b) -> int:
        n = self.fh.readinto(b)
        self._count(n or 0)
        return n

    def readline(self, size: int = -1) -> bytes:
        b = self.fh.readline(size)
        self._count(len(b))
        return b

    def finish(self) -> None:
        progress.tick(self.n_mb, self.n_mb, unit="unpickle", label=f"{self.n_mb}/{self.n_mb} MB")
        if self.callback is not None:
            self.callback(self.total, self.total)


def _unpickle(path: Path, progress_cb=None) -> dict:
    total = path.stat().st_size
    with open(path, "rb") as fh:
        reader = _CountingReader(fh, total, progress_cb)
        try:
            state = pickle.load(reader)
        except _INCOMPATIBLE as exc:
            raise IncompatibleCheckpoint(f"{type(exc).__name__}: {exc}") from exc
        reader.finish()
    if not isinstance(state, dict) or "ensemble" not in state:
        raise ValueError(f"{path} is not a SPARC core checkpoint (no fitted ensemble; S3 not reached?)")
    return state


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

@dataclass
class RunSession:
    """Everything a scenario request on one run needs (see the module docstring)."""

    run_dir: Path
    cfg: CoreConfig
    data: Any
    folds: Any
    manifest: dict
    predictions: Any
    ensemble: Any
    influence: Any
    mediators: Any
    engine: Any
    responses: dict = field(default_factory=dict)
    layers: Any = None
    causal: dict | None = None
    code_match: bool | None = None
    load_seconds: dict = field(default_factory=dict)
    checkpoint_bytes: int = 0
    checkpoint_key: str | None = None
    code_sha: str | None = None
    scenarios: list = field(default_factory=list)
    threads: int = 2
    _resp: Any = None

    @property
    def resp(self):
        """The ResponseEngine, built on first use (marginals, emulator builds)."""
        if self._resp is None:
            from sparc.core.response import ResponseEngine

            scales = (self.cfg.raw.get("influence") or {}).get("scales") or (0.5, 1.0, 2.0)
            self._resp = ResponseEngine(self.engine, influence_scales=scales)
        return self._resp

    @property
    def ranges_m(self) -> dict:
        return dict(self.influence.ranges_m)

    @property
    def base_fold(self) -> np.ndarray:
        return self.engine.base_fold

    def gwrf_models(self) -> list:
        """Every GWRF base model of every fold stack."""
        return [m for st in self.ensemble.stacks for m in (st.base or []) if type(m).__name__ == "GWRFModel"]

    def set_threads(self, n: int) -> None:
        """Override ``GWRFModel.n_jobs`` on every fold (the forests' prediction threads)."""
        self.threads = max(1, int(n))
        for m in self.gwrf_models():
            m.n_jobs = self.threads


def _strip_arrays(d):
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items()
                if not str(k).startswith(("tau_hat", "per_point")) and not isinstance(v, np.ndarray)}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def open_run(run_dir, cfg: CoreConfig | None = None, *, threads: int = 2, base_fold: np.ndarray | None = None,
             drop=DEFAULT_DROP, progress_cb: Callable[[int, int], Any] | None = None,
             studio_dir: str | os.PathLike | None = None) -> RunSession:
    """Load ``run_dir`` (with its ``checkpoint.pkl``) into a :class:`RunSession` (see the module docstring).

    ``cfg`` defaults to :func:`config_for_run`.  ``base_fold`` is a cached (K × n) baseline pass of the
    ScenarioEngine (``engine.base_fold`` of an earlier open of the same checkpoint and code); a wrong shape
    is ignored with a warning.  ``drop`` lists what is not kept in memory: ``cv_distance`` and the causal
    per-cell arrays (``causal_arrays``).  ``progress_cb(bytes_read, bytes_total)`` follows the unpickle.
    """
    from sparc.core.baselines import load_run
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import ScenarioEngine

    run_dir = Path(run_dir)
    pkl = run_dir / "checkpoint.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"{run_dir} has no checkpoint.pkl (the run did not reach S3)")
    threads = max(1, int(threads))
    cfg = cfg if cfg is not None else config_for_run(run_dir, studio_dir=studio_dir)
    seconds: dict[str, float] = {}
    drop = set(drop or ())
    with progress.run_dir_scope(run_dir), progress.limit_threads(threads):
        t = time.perf_counter()
        with progress.task("load_run"):
            data, folds, manifest, pred = load_run(run_dir, cfg)
        seconds["load_run"] = round(time.perf_counter() - t, 3)

        progress.check_cancel()
        t = time.perf_counter()
        key = checkpoint_key(run_dir)
        with progress.task("unpickle", key="checkpoint.pkl"):
            state = _unpickle(pkl, progress_cb)
        seconds["unpickle"] = round(time.perf_counter() - t, 3)
        ens, inf = state["ensemble"], state["influence"]
        responses = dict(state.get("responses") or {})
        causal = state.get("causal")
        scenarios = list(state.get("scenarios") or [])
        if "cv_distance" in drop:
            state.pop("cv_distance", None)
        del state

        session = RunSession(run_dir=run_dir, cfg=cfg, data=data, folds=folds, manifest=manifest, predictions=pred,
                             ensemble=ens, influence=inf, mediators=None, engine=None, responses=responses,
                             checkpoint_bytes=int(pkl.stat().st_size), checkpoint_key=key, scenarios=scenarios,
                             threads=threads)
        session.set_threads(threads)

        progress.check_cancel()
        t = time.perf_counter()
        if cfg.mediators:
            with progress.task("mediators", unit="mediator_fit"):
                session.mediators = MediatorChain(cfg.mediators).fit(data.frame)
        seconds["mediators"] = round(time.perf_counter() - t, 3)

        progress.check_cancel()
        t = time.perf_counter()
        if base_fold is not None:
            bf = np.asarray(base_fold, dtype=float)
            if bf.shape != (len(ens.stacks), data.n):
                log.warning("%s: cached base_fold has shape %s, expected %s; recomputing", run_dir, bf.shape,
                            (len(ens.stacks), data.n))
                base_fold = None
        with progress.task("engine_init"):
            try:
                session.engine = ScenarioEngine(data, cfg, ens, dict(inf.ranges_m), session.mediators,
                                                base_fold=base_fold)
            except _INCOMPATIBLE as exc:
                raise IncompatibleCheckpoint(f"{type(exc).__name__}: {exc}") from exc
        seconds["engine_init"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        try:
            from sparc.core.opendata import load_layers

            session.layers = load_layers(cfg, data) if (cfg.raw.get("planner") or {}).get("layers") else None
        except Exception as exc:              # optional input: a missing or stale table never blocks scenarios
            log.warning("%s: planner layers unavailable: %s", run_dir, exc)
            session.layers = None
        if causal is None:
            causal = _read_json(run_dir / "causal.json")
        session.causal = _strip_arrays(causal) if (causal is not None and "causal_arrays" in drop) else causal
        seconds["layers"] = round(time.perf_counter() - t, 3)

        side = _read_json(run_dir / "checkpoint.json") or {}
        session.code_sha = code_digest()
        session.code_match = (side.get("code_sha256") == session.code_sha) if side.get("code_sha256") else None
        session.load_seconds = seconds
    return session
