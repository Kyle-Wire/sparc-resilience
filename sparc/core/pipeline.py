"""Run the core pipeline end to end (S0 → S7) and write a run directory.

Outputs (``run_dir``, by default ``cfg.output.dir``/<run name>/):

* ``manifest.json`` — config, versions, timings, QA and every summary metric
* ``predictions.parquet`` — per point: target, OOF base predictions, stacked
  prediction, conformal interval
* ``influence.json`` — S1 ranges, anisotropy, target ACF
* ``physics.json`` — per-fold fitted physics parameters
* ``response_<var>.parquet`` + ``response_curves.json`` — S4 maps and curves
* ``scenarios.json`` (+ ``scenario_deltas.parquet``, ``scenario_detail.npz``) — S5
* ``cv_distance.json`` — optional skill-vs-distance CV curve
* ``climate.json`` — CMIP6 projections × adaptation (threshold exposure, offsets)
* ``causal.json`` (+ ``causal_cells.parquet``) — S6 estimates, sensitivity and model-vs-causal audit
* ``optimize.json`` (+ ``allocation.parquet``) — S7
* ``report.md``, ``methods.md``, ``model_card.md``, ``environment.txt``
* ``input_frame.parquet`` — the input table of a run made from an in-memory frame
* ``checkpoint.pkl`` (+ ``checkpoint.json``) — fitted state after each expensive stage (``resume=True``)
* ``run_state.json`` — status, current stage and done set, rewritten at every stage boundary

Every file is written atomically (:mod:`sparc.core.runio`) and reported to
the progress sink (:mod:`sparc.core.progress`): ``run.start``, ``run.plan``,
``run.dir``, one ``stage`` span per stage (or ``stage.skip`` with its
reason), ``artifact``, ``checkpoint``, QA warnings and ``run.end``.

Planning (no torch): :data:`STAGE_NODES`, :func:`plan_stages`,
:func:`fingerprint_sections` and :func:`checkpoint_status` share the gating
predicates that :func:`run_core` uses, so a plan cannot drift from what runs.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import socket
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.core import progress, runio
from sparc.core.config import CoreConfig, load_core_config
from sparc.core.cv import make_spatial_folds
from sparc.core.data import CoreData, load_core_data, prepare_frame
from sparc.core.ensemble import fit_ensemble
from sparc.core.features import build_context
from sparc.core.mediators import MediatorChain

log = logging.getLogger(__name__)

ALL_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")

# Stage nodes of a run, in execution order.  ``checkpoint_key`` is the done-set entry that marks the node as
# restored from the checkpoint; ``manifest_timing_key`` the manifest ``timings_s`` entry that times it (the
# climate stage is timed inside S5).
STAGE_NODES = (
    {"id": "S0", "label": "Data and QA", "desc": "Load the table, clean, coarsen or window it, build the grid",
     "checkpoint_key": None, "manifest_timing_key": "S0"},
    {"id": "S1", "label": "Area of influence", "desc": "Correlograms, influence ranges and anisotropy",
     "checkpoint_key": "S3", "manifest_timing_key": "S1"},
    {"id": "S2_S3", "label": "Base models and stacker", "desc": "Cross-fitted base models and the stacker",
     "checkpoint_key": "S3", "manifest_timing_key": "S2_S3"},
    {"id": "baselines", "label": "Reference baselines", "desc": "Standard models scored on the same folds",
     "checkpoint_key": "baselines", "manifest_timing_key": "baselines"},
    {"id": "cv_curve", "label": "Skill vs distance", "desc": "Re-fits on smaller blocks and random points",
     "checkpoint_key": "cv_curve", "manifest_timing_key": "cv_curve"},
    {"id": "S4", "label": "Response surfaces", "desc": "Dose sweeps, saturation, own and footprint effects",
     "checkpoint_key": "S4", "manifest_timing_key": "S4"},
    {"id": "S5", "label": "Scenarios", "desc": "Configured scenarios through the fitted model",
     "checkpoint_key": "S5", "manifest_timing_key": "S5"},
    {"id": "climate", "label": "Climate and adaptation", "desc": "CMIP6 warming with adaptation scenarios",
     "checkpoint_key": "climate", "manifest_timing_key": None},
    {"id": "S6", "label": "Causal validation", "desc": "DML, spillover, CATE, dose-response, audit",
     "checkpoint_key": "S6", "manifest_timing_key": "S6"},
    {"id": "S7", "label": "Budget optimisation", "desc": "Greedy allocation and its closed-loop check",
     "checkpoint_key": None, "manifest_timing_key": "S7"},
    {"id": "finish", "label": "Manifest and report", "desc": "Manifest, report, methods and model card",
     "checkpoint_key": None, "manifest_timing_key": None},
)
STAGE_IDS = tuple(n["id"] for n in STAGE_NODES)
CHECKPOINT_KEY = {n["id"]: n["checkpoint_key"] for n in STAGE_NODES if n["checkpoint_key"]}
_LABELS = {n["id"]: n["label"] for n in STAGE_NODES}

CHECKPOINT = "checkpoint.pkl"
CHECKPOINT_JSON = "checkpoint.json"
RUN_STATE = "run_state.json"
INPUT_FRAME = "input_frame.parquet"
MANIFEST_SCHEMA = 2
# Manifest sections written after the run (post-run actions, studies); a re-run keeps them.
POST_RUN_KEYS = ("planner", "uncertainty", "simcheck", "multiverse", "placebo", "emulator", "post_run")
# Instrumentation modules: editing them must not invalidate saved fits.
_FINGERPRINT_SKIP = frozenset({"progress.py", "runio.py"})
FINGERPRINT_SECTIONS = ("data", "core", "s4", "s5", "climate", "s6", "s7", "code")
_CORE_KEYS = ("data", "qa", "encodings", "predictors", "influence", "cv", "models", "physics", "stacker")
_SECTION_KEYS = {"s4": ("actionable", "response", "mediators", "coupling"), "s5": ("scenarios", "joint_scenarios"),
                 "climate": ("climate",), "s6": ("causal",), "s7": ("optimize", "planner")}
_ENGINE_STAGES = frozenset({"S4", "S5", "S6", "S7"})


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: Path, obj) -> None:
    runio.write_json_atomic(path, _jsonable(obj))


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CoreResult:
    cfg: CoreConfig
    data: CoreData
    run_dir: Path | None
    influence: object = None
    ensemble: object = None
    responses: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)
    causal: dict = field(default_factory=dict)
    optimize: dict = field(default_factory=dict)
    cv_distance: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    climate: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mode overrides and gating predicates (shared by run_core and plan_stages)
# ---------------------------------------------------------------------------


def apply_mode_overrides(cfg: CoreConfig, fast: bool = False, coarse: float | None = None,
                         cv_curve: bool | None = None, frame_given: bool = False) -> CoreConfig:
    """Apply the run modes to ``cfg`` in place (idempotent) and return it.

    ``fast`` shrinks the problem (8k-point window unless a frame is given or
    ``data.subsample`` is set, ≤ 3 folds, ≤ 200 epochs, two λ candidates,
    ≤ 9 permutations, GP-free baselines); ``coarse`` sets ``data.coarse_m``;
    ``cv_curve`` overrides ``cv.distance_curve.enabled``."""
    if coarse:
        cfg.raw["data"]["coarse_m"] = float(coarse)
    if cv_curve is not None:
        cfg.raw["cv"].setdefault("distance_curve", {})["enabled"] = bool(cv_curve)
    if fast:
        if not frame_given and not cfg.data.get("subsample"):
            cfg.raw["data"]["subsample"] = 8000
        cfg.raw["cv"]["n_folds"] = min(3, int(cfg.raw["cv"]["n_folds"]))
        cfg.raw["stacker"]["epochs"] = min(200, int(cfg.raw["stacker"]["epochs"]))
        cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1]
        cfg.raw["influence"]["n_perm"] = min(9, int(cfg.raw["influence"].get("n_perm", 19)))
        if cfg.raw["cv"].get("baselines", True) is True:
            cfg.raw["cv"]["baselines"] = ["hgb_xy", "hgb_focal", "idw"]     # no GP fits in smoke runs
    return cfg


def _first_requested(stages, after: str | None = None) -> str | None:
    order = list(ALL_STAGES)
    start = order.index(after) + 1 if after in order else 0
    return next((s for s in order[start:] if s in stages), None)


def _wants_s2(stages) -> bool:
    """S2_S3 (with the baselines and CV curve that hang off it) runs when any later stage is requested."""
    return bool({"S2", "S3", "S4", "S5", "S6", "S7"} & set(stages))


def _wants_s4(stages) -> bool:
    """S4 runs when requested and whenever S6 (model effects) or S7 (benefit maps) needs its responses."""
    return bool({"S4", "S6", "S7"} & set(stages))


def _s6_runs(cfg: CoreConfig, stages, responses_available) -> tuple[bool, str | None, list[str]]:
    """(runs, skip reason, treatments that get model effects — those with an S4 response)."""
    if "S6" not in stages:
        return False, "not_requested", []
    c = cfg.raw.get("causal") or {}
    if not c.get("enabled", True):
        return False, "disabled_by_config:causal.enabled", []
    if not c.get("treatments"):
        return False, "no_treatments", []
    avail = set(responses_available or ())
    return True, None, [t for t in c["treatments"] if t in avail]


def _s7_runs(cfg: CoreConfig, stages, responses) -> tuple[bool, str | None]:
    if "S7" not in stages:
        return False, "not_requested"
    o = cfg.raw.get("optimize") or {}
    if not o.get("enabled", True):
        return False, "disabled_by_config:optimize.enabled"
    if not o.get("budget"):
        return False, "no_budget"
    if o.get("variable") not in set(responses or ()):
        return False, "no_responses"
    return True, None


def _climate_runs(cfg: CoreConfig, stages) -> tuple[bool, str | None]:
    if not (cfg.raw.get("climate") or {}).get("enabled"):
        return False, "disabled_by_config:climate.enabled"
    if "S5" not in stages:
        return False, "requires_S5"
    return True, None


def _baselines_list(cfg: CoreConfig) -> tuple[str, ...]:
    """The reference baselines this config fits (empty when ``cv.baselines`` is off)."""
    bl = cfg.raw["cv"].get("baselines", True)
    if not bl:
        return ()
    if bl is True:
        from sparc.core.baselines import BASELINES

        return tuple(BASELINES)
    return tuple(str(b) for b in bl)


def _cv_curve_sizes(cfg: CoreConfig, block_m: float | None, dx: float | None, extent: float | None) -> list[float]:
    """The ``cv.distance_curve.block_m`` sizes that get a partition (0 = random points); [] when disabled."""
    from sparc.core.diagnostics import curve_partitions

    dcfg = cfg.raw["cv"].get("distance_curve") or {}
    if not dcfg.get("enabled"):
        return []
    sizes = [float(b) for b in dcfg.get("block_m", (0, 500, 1000))]
    kept, _ = curve_partitions(sizes, dx, extent, block_m)
    return [0.0 if lab.startswith("random") else b for b, _buf, lab in kept]


def _cv_block(cfg: CoreConfig, dx: float, extent: float,
              influence_block_m: float | None) -> tuple[float, float, float | None]:
    """(block, buffer, raised_from): the main CV block — ``cv.block_m`` or the S1 range capped at a third of
    the extent — raised to 3 grid cells when smaller (``raised_from`` is then the smaller value, else None)."""
    cv = cfg.raw["cv"]
    block = cv.get("block_m", "auto")
    if block in (None, "auto"):
        block = influence_block_m if influence_block_m is not None else 10 * dx
        # keep ≥ 3 blocks per fold on small study areas (coordinates, not the target)
        block = float(min(block, extent / 3.0))
    raised_from = None
    if float(block) < 3.0 * dx:
        # A block smaller than a few cells is random-point CV in disguise:
        # neighbours of every test point sit in training and scores leak.
        raised_from, block = float(block), 3.0 * dx
    buf = cv.get("buffer_m", "auto")
    if buf in (None, "auto"):
        buf = float(block) / 3.0
    return float(block), float(buf), raised_from


def _block_and_buffer(cfg: CoreConfig, influence, data: CoreData) -> tuple[float, float]:
    extent = min(np.ptp(data.x), np.ptp(data.y_coord))
    block, buf, raised_from = _cv_block(cfg, data.grid.dx, extent,
                                        influence.block_size_m if influence is not None else None)
    if raised_from is not None:
        log.warning("cv.block_m=%.1f m is below 3 grid cells (%.1f m); raising it", raised_from, block)
        progress.warn("cv.block_raised", f"the CV block ({raised_from:.0f} m) is below 3 grid cells; raised to "
                      f"{block:.0f} m", block_m=block, requested_m=raised_from)
    return block, buf


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _base_model_names(cfg: CoreConfig) -> list[str]:
    """Base models :func:`sparc.core.base_models.build_base_models` builds, in its order."""
    m, ph = cfg.raw.get("models") or {}, cfg.raw.get("physics") or {}
    names = [n for n in ("ols", "mgwr", "gwrf", "gam") if m.get(n, True)]
    if m.get("physics", True) and ph.get("enabled", True) and (ph.get("roles") or {}):
        names.append("physics")
    return names


def _fits_advection(cfg: CoreConfig) -> bool:
    ph = cfg.raw.get("physics") or {}
    fa = ph.get("fit_advection", "auto")
    adv_on = (ph.get("wind") is not None) if fa in ("auto", None) else bool(fa)
    return "physics" in _base_model_names(cfg) and adv_on and bool(ph.get("select_advection", True))


def _s2s3_units(cfg: CoreConfig, K: int) -> dict[str, int]:
    """Planned units of one cross-fitted ensemble (S2_S3 or one CV-curve partition), checkpoint excluded."""
    names = _base_model_names(cfg)
    u = {f"base_fit:{n}": K for n in names}
    if _fits_advection(cfg):
        u["adv_refit"] = K
    s = cfg.raw["stacker"]
    lambdas = list(s.get("tune_lambda") or [s.get("lambda_pde", 1.0)]) if "physics" in names else [0.0]
    if s.get("allow_residual_off", True):
        u["stacker_fit:mean"] = K
        u["stacker_fit:nnls"] = K
    u["stacker_fit:residual"] = K * len(lambdas)
    return u


def _lever_passes(cfg: CoreConfig, var: str) -> int:
    """Engine passes of one S4 sweep: non-zero doses + the marginal passes of ``ResponseEngine.marginals``."""
    spec = cfg.actionable.get(var) or {}
    doses = [float(d) for d in spec.get("doses", [0, 5, 10, 20, 30])]
    enc = cfg.raw.get("encodings") or {}
    no_focal = set(enc.get("categorical") or []) | set(enc.get("circular_degrees") or [])
    moved = [var] + [med for med, s in (cfg.mediators or {}).items() if var in (s.get("parents") or [])]
    n_scales = len(cfg.raw["influence"].get("scales", (0.5, 1.0, 2.0)))
    physics = 1 if "physics" in _base_model_names(cfg) else 0
    marginal = 2 + n_scales * sum(1 for c in moved if c not in no_focal) + physics
    return sum(1 for d in doses if d != 0.0) + marginal


def _add(units: dict, key: str, n: int) -> None:
    if n:
        units[key] = units.get(key, 0) + int(n)


def _plan(cfg: CoreConfig, stages, done=frozenset(), *, n_points: int | None = None, extent: float | None = None,
          dx: float | None = None, block_m: float | None = None, write: bool = True,
          n_climate_models: int | None = None) -> list[dict]:
    """Plan nodes for an already mode-overridden ``cfg`` (no data access)."""
    req, done = set(stages), set(done)
    K = int(cfg.raw["cv"]["n_folds"])
    ck = 1 if write else 0
    nodes: dict[str, dict] = {}

    def put(nid, state, reason=None, units=None):
        nodes[nid] = {"id": nid, "label": _LABELS[nid], "state": state, "reason": reason,
                      "units": {k: int(v) for k, v in (units or {}).items() if v},
                      "checkpoint_key": CHECKPOINT_KEY.get(nid), "est_s": None, "est_lo": None, "est_hi": None}

    def required(sid, after):
        return None if sid in req else (f"required_by:{_first_requested(req, after)}"
                                        if _first_requested(req, after) else None)

    put("S0", "will_run", required("S0", "S0"), {"s0_load": 1})
    if "S3" in done:
        put("S1", "cached", "checkpoint")
    else:
        put("S1", "will_run", required("S1", "S1"), {"s1_influence": 1})

    wants2 = _wants_s2(req)
    if not wants2:
        put("S2_S3", "skipped", "not_requested")
    elif "S3" in done:
        put("S2_S3", "cached", "checkpoint")
    else:
        u = _s2s3_units(cfg, K)
        _add(u, "checkpoint_save", ck)
        put("S2_S3", "will_run", None if {"S2", "S3"} & req else f"required_by:{_first_requested(req, 'S3')}", u)

    bl = _baselines_list(cfg)
    bl_units = {f"baseline_fit:{m}": K for m in bl}
    if not wants2:
        put("baselines", "skipped", "not_requested")
    elif not bl:
        put("baselines", "skipped", "disabled_by_config:cv.baselines")
    elif "baselines" in done:
        put("baselines", "cached", "checkpoint")
    else:
        put("baselines", "will_run", None, {**bl_units, "checkpoint_save": ck})

    if not wants2:
        put("cv_curve", "skipped", "not_requested")
    elif not (cfg.raw["cv"].get("distance_curve") or {}).get("enabled"):
        put("cv_curve", "skipped", "disabled_by_config:cv.distance_curve.enabled")
    elif "cv_curve" in done:
        put("cv_curve", "cached", "checkpoint")
    else:
        P = len(_cv_curve_sizes(cfg, block_m, dx, extent))
        if P == 0:
            put("cv_curve", "skipped", "no_partitions")
        else:
            u = {k: P * v for k, v in _s2s3_units(cfg, K).items()}
            for k, v in bl_units.items():
                _add(u, k, P * v)
            _add(u, "checkpoint_save", ck)
            put("cv_curve", "will_run", None, u)

    engine_any = bool(_ENGINE_STAGES & req)
    responses: set = set()
    if not _wants_s4(req):
        put("S4", "skipped", "not_requested")
    elif "S4" in done:
        put("S4", "cached", "checkpoint")
        responses = set(cfg.actionable)
    else:
        put("S4", "will_run", None if "S4" in req else f"required_by:{'S6' if 'S6' in req else 'S7'}",
            {"engine_pass": sum(_lever_passes(cfg, v) for v in cfg.actionable), "checkpoint_save": ck})
        responses = set(cfg.actionable)

    from sparc.core.scenarios import specs_from_config

    n_specs = len(specs_from_config(cfg))
    if "S5" not in req:
        put("S5", "skipped", "not_requested")
    elif "S5" in done:
        put("S5", "cached", "checkpoint")
    else:
        put("S5", "will_run", None, {"engine_pass": n_specs, "checkpoint_save": ck})

    runs_c, reason_c = _climate_runs(cfg, req)
    if not runs_c:
        put("climate", "skipped", reason_c)
    elif "climate" in done:
        put("climate", "cached", "checkpoint")
    else:
        cc = cfg.raw.get("climate") or {}
        m = 0 if cc.get("source", "table") == "table" else int(n_climate_models if n_climate_models is not None else 24)
        put("climate", "will_run", None, {"climate_model": m, "checkpoint_save": ck})

    runs6, reason6, eff_tr = _s6_runs(cfg, req, responses)
    if not runs6:
        put("S6", "skipped", reason6)
    elif "S6" in done:
        put("S6", "cached", "checkpoint")
    else:
        u: dict[str, int] = {}
        for t in (cfg.raw["causal"].get("treatments") or []):
            if t not in cfg.predictors:
                continue                                     # skipped by run_causal_validation (not in the data)
            for step in ("dml", "spillover", "cate", "dr", "sens"):
                _add(u, f"causal_step:{step}", 1)
            if t in eff_tr:
                _add(u, "engine_pass", 8)                    # adoption slope + 7-point own-only PD curve
                _add(u, "causal_step:audit", 1)
        _add(u, "checkpoint_save", ck)
        put("S6", "will_run", None, u)

    runs7, reason7 = _s7_runs(cfg, req, responses)
    if not runs7:
        put("S7", "skipped", reason7)
    else:
        put("S7", "will_run", None, {"engine_pass": 1, "pareto": 1})
    put("finish", "will_run")

    # The engine's baseline pass (task engine_init) runs in the first stage that needs the engine.
    if engine_any:
        needs = {"S4": bool(cfg.actionable), "S5": n_specs > 0, "S6": bool(eff_tr), "S7": True}
        for sid in ("S4", "S5", "S6", "S7"):
            if nodes[sid]["state"] == "will_run" and needs[sid]:
                _add(nodes[sid]["units"], "engine_pass", 1)
                break
    return [nodes[i] for i in STAGE_IDS]


def plan_event(nodes: list[dict], n_points: int | None = None, max_bytes: int = 3600) -> dict:
    """``run.plan`` fields, compact enough for one progress line: unset estimates are left out and, if
    still too long, the labels too (they are in :data:`STAGE_NODES`) — never the units."""
    slim = [{k: v for k, v in n.items() if not (k.startswith("est_") and v is None)} for n in nodes]
    ev = {"nodes": slim, "total_units": plan_totals(nodes), "n_points": n_points}
    if len(json.dumps(ev, separators=(",", ":"), ensure_ascii=False).encode()) > max_bytes:
        ev["nodes"] = [{k: v for k, v in n.items() if k != "label"} for n in slim]
    return ev


def plan_totals(nodes: list[dict]) -> dict[str, int]:
    """Planned units over the nodes that will run."""
    tot: dict[str, int] = {}
    for n in nodes:
        if n["state"] == "will_run":
            for k, v in n["units"].items():
                _add(tot, k, v)
    return tot


def plan_stages(cfg: CoreConfig, stages=ALL_STAGES, fast: bool = False, coarse: float | None = None,
                cv_curve: bool | None = None, done=frozenset(), n_points: int | None = None,
                extent: float | None = None, dx: float | None = None, *, block_m: float | None = None,
                write: bool = True) -> list[dict]:
    """Plan nodes (SPEC §5.4) of a ``run_core`` call, without running it (and without torch).

    ``done`` is a checkpoint done set (those nodes become ``cached``).  When
    ``n_points``, ``extent`` or ``dx`` is unknown, S0 runs on demand (a data
    load, ~0.3 s for Providence) to count the CV-curve partitions.  The main
    CV block is known only after S1 when ``cv.block_m`` is ``auto``; pass
    ``block_m`` once known (``run_core`` re-plans after S1).  ``write=False``
    plans no ``checkpoint_save`` units (nothing is written)."""
    if not isinstance(cfg, CoreConfig):
        cfg = load_core_config(cfg)
    c = apply_mode_overrides(copy.deepcopy(cfg), fast=fast, coarse=coarse, cv_curve=cv_curve)
    if n_points is None or extent is None or dx is None:
        try:
            data = load_core_data(c)
        except (OSError, ValueError, KeyError) as exc:      # plan without geometry (a counts-only estimate)
            log.info("plan_stages: S0 unavailable (%s); planning without geometry", exc)
        else:
            n_points = data.n if n_points is None else n_points
            dx = float(data.grid.dx) if dx is None else dx
            extent = float(min(np.ptp(data.x), np.ptp(data.y_coord))) if extent is None else extent
    if block_m is None and dx is not None and extent is not None and \
            c.raw["cv"].get("block_m", "auto") not in (None, "auto"):
        block_m = _cv_block(c, dx, extent, None)[0]
    return _plan(c, stages, done, n_points=n_points, extent=extent, dx=dx, block_m=block_m, write=write)


# ---------------------------------------------------------------------------
# Fingerprints and checkpoints
# ---------------------------------------------------------------------------


def _code_files() -> list[Path]:
    """Core sources a checkpoint depends on (the instrumentation modules are excluded)."""
    return [p for p in sorted(Path(__file__).parent.glob("*.py")) if p.name not in _FINGERPRINT_SKIP]


def _code_digest() -> str:
    h = hashlib.sha256()
    for src in _code_files():
        h.update(src.name.encode())
        h.update(src.read_bytes())
    return h.hexdigest()


def _hash(obj) -> str:
    from sparc.core.provenance import _str_keys

    return hashlib.sha256(json.dumps(_str_keys(obj), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _data_identity(cfg: CoreConfig, frame: pd.DataFrame | None) -> str:
    """Identity of the input: the frame's content, or the data (and join) files' path, size and mtime."""
    h = hashlib.sha256()
    if frame is not None:
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    else:
        for path in [cfg.data_path] + [cfg.resolve_path(j["path"]) for j in cfg.data.get("join") or []]:
            try:
                st = Path(path).stat()
                h.update(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                h.update(f"{path}:missing".encode())
    return h.hexdigest()[:16]


def _fingerprint(cfg: CoreConfig, fast: bool, frame: pd.DataFrame | None, data_id: str | None = None) -> str:
    """Identity of a run for resuming: effective config, input data and the
    core source code (a code change invalidates saved fits; a docs commit
    does not, and neither do edits of ``progress.py`` / ``runio.py``)."""
    from sparc.core.provenance import _str_keys

    h = hashlib.sha256()
    h.update(json.dumps(_str_keys(cfg.raw), sort_keys=True, default=str).encode())
    h.update(str(bool(fast)).encode())
    h.update((data_id or _data_identity(cfg, frame)).encode())
    h.update(_code_digest().encode())
    return h.hexdigest()[:16]


def fingerprint_sections(cfg: CoreConfig, fast: bool, frame: pd.DataFrame | None = None, *,
                         coarse: float | None = None, cv_curve: bool | None = None) -> dict[str, str]:
    """Per-section hashes of what a checkpoint depends on (SPEC §11 item 6).

    ``data`` the input identity; ``core`` the data/qa/encodings/predictors/
    influence/cv (minus ``distance_curve.enabled``) and models/physics/stacker
    config plus the fast flag; ``s4`` actionable/response/mediators/coupling;
    ``s5`` scenarios; ``climate``; ``s6`` causal; ``s7`` optimize/planner;
    ``code`` the core sources.  Mode overrides are applied first (they are
    idempotent), so a raw project config and the effective config of the run
    give the same sections."""
    from sparc.core.provenance import _str_keys

    c = apply_mode_overrides(copy.deepcopy(cfg), fast=fast, coarse=coarse, cv_curve=cv_curve,
                             frame_given=frame is not None)
    raw = _str_keys(c.raw)
    core = {k: copy.deepcopy(raw.get(k)) for k in _CORE_KEYS}
    cv = dict(core.get("cv") or {})
    dc = dict(cv.get("distance_curve") or {})
    dc.pop("enabled", None)
    cv["distance_curve"] = dc
    core["cv"] = cv
    core["fast"] = bool(fast)
    out = {"data": _data_identity(c, frame), "core": _hash(core)}
    for name, keys in _SECTION_KEYS.items():
        out[name] = _hash({k: raw.get(k) for k in keys})
    out["code"] = _code_digest()[:16]
    return out


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _run_args(run_dir: Path) -> dict:
    """Mode arguments of a run: Studio's ``launch.json`` args, else the manifest's fast flag."""
    launch = _read_json(run_dir / "studio" / "launch.json") or {}
    args = dict(launch.get("args") or {})
    if "fast" not in args:
        m = _read_json(run_dir / "manifest.json") or {}
        if "fast_mode" in m:
            args["fast"] = bool(m["fast_mode"])
    return args


def checkpoint_status(run_dir, cfg: CoreConfig | None = None, fast: bool | None = None) -> dict:
    """What a resume of ``run_dir`` would reuse, from ``checkpoint.json`` — the pickle is never opened.

    Returns ``{present, done, bytes, saved_utc, fingerprint, matches: {data, code, config}, changed_sections,
    fingerprint_match}``.  ``matches.code`` compares with the current core sources; with ``cfg`` (the
    config a resume would use; mode arguments come from ``fast`` or the run's ``launch.json``/manifest) the
    data and config sections are compared too, and ``fingerprint_match`` says whether ``run_core(resume=True)``
    would accept the checkpoint.  Unknown values are None (e.g. a checkpoint written before ``checkpoint.json``
    existed)."""
    run_dir = Path(run_dir)
    pkl = run_dir / CHECKPOINT
    side = _read_json(run_dir / CHECKPOINT_JSON) if pkl.exists() else None
    present = pkl.exists()
    out = {"present": present, "done": None, "bytes": None, "saved_utc": None, "fingerprint": None,
           "matches": {"data": None, "code": None, "config": None}, "changed_sections": None,
           "fingerprint_match": None}
    if not present:
        return out
    st = pkl.stat()
    out["bytes"] = int(st.st_size)
    out["saved_utc"] = _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not side:
        return out
    out.update(done=list(side.get("done") or []), saved_utc=side.get("saved_utc") or out["saved_utc"],
               fingerprint=side.get("fingerprint"))
    out["matches"]["code"] = side.get("code_sha256") == _code_digest()
    if cfg is not None:
        if not isinstance(cfg, CoreConfig):
            cfg = load_core_config(cfg)
        args = _run_args(run_dir)
        fast = bool(args.get("fast", False)) if fast is None else bool(fast)
        coarse, cv_curve = args.get("coarse"), args.get("cv_curve")
        frame = pd.read_parquet(run_dir / INPUT_FRAME) if (run_dir / INPUT_FRAME).exists() else None
        cur = fingerprint_sections(cfg, fast, frame, coarse=coarse, cv_curve=cv_curve)
        old = side.get("sections") or {}
        changed = [k for k in FINGERPRINT_SECTIONS if old.get(k) != cur.get(k)]
        out["changed_sections"] = changed
        out["matches"]["data"] = "data" not in changed
        out["matches"]["config"] = not any(k in changed for k in ("core", "s4", "s5", "climate", "s6", "s7"))
        eff = apply_mode_overrides(copy.deepcopy(cfg), fast=fast, coarse=coarse, cv_curve=cv_curve,
                                   frame_given=frame is not None)
        out["fingerprint_match"] = _fingerprint(eff, fast, frame, cur["data"]) == side.get("fingerprint")
    return out


def _save_checkpoint(run_dir: Path | None, state: dict, sections: dict | None = None) -> None:
    """Pickle ``state`` atomically, then write the ``checkpoint.json`` sidecar and emit ``checkpoint saved``."""
    if run_dir is None:
        return
    import pickle

    t = time.perf_counter()
    with runio.atomic_open(run_dir / CHECKPOINT, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    nbytes = int((run_dir / CHECKPOINT).stat().st_size)
    done = sorted(state.get("done", ()))
    runio.write_json_atomic(run_dir / CHECKPOINT_JSON, {
        "schema": 1, "fingerprint": state.get("fingerprint"), "sections": dict(sections or {}), "done": done,
        "bytes": nbytes, "saved_utc": _utc(), "code_sha256": _code_digest()})
    log.info("checkpoint saved (%s)", ", ".join(done))
    progress.checkpoint("saved", done=done, bytes=nbytes, elapsed_s=round(time.perf_counter() - t, 3),
                        fingerprint=state.get("fingerprint"))


def _checkpoint_mismatch(run_dir: Path, done, fingerprint: str, changed: list[str] | None) -> None:
    log.warning("checkpoint in %s is from a different config/data/code version — ignoring it", run_dir)
    progress.warn("checkpoint.mismatch", "the checkpoint is from a different config, data or code version; "
                  "every stage is refitted", changed_sections=changed)
    progress.checkpoint("mismatch", done=done or [], fingerprint=fingerprint, changed_sections=changed)


def _load_checkpoint(run_dir: Path | None, fingerprint: str, sections: dict | None = None) -> dict:
    if run_dir is None or not (run_dir / CHECKPOINT).exists():
        return {}
    side = _read_json(run_dir / CHECKPOINT_JSON)
    if side and side.get("fingerprint") and side["fingerprint"] != fingerprint:
        # the sidecar settles it: no need to unpickle a checkpoint that cannot be used
        old = side.get("sections") or {}
        changed = [k for k in FINGERPRINT_SECTIONS if sections and old.get(k) != sections.get(k)] or None
        _checkpoint_mismatch(run_dir, side.get("done"), fingerprint, changed)
        return {}
    import pickle

    t = time.perf_counter()
    with progress.task("unpickle", key=CHECKPOINT):
        with open(run_dir / CHECKPOINT, "rb") as fh:
            state = pickle.load(fh)
    if state.get("fingerprint") != fingerprint:
        _checkpoint_mismatch(run_dir, sorted(state.get("done", ())), fingerprint, None)
        return {}
    done = sorted(state.get("done", ()))
    log.info("resuming from checkpoint (done: %s)", ", ".join(done))
    progress.checkpoint("loaded", done=done, bytes=int((run_dir / CHECKPOINT).stat().st_size),
                        elapsed_s=round(time.perf_counter() - t, 3), fingerprint=fingerprint)
    return state


def _peek_done(run_dir: Path | None, fingerprint: str) -> set:
    """The done set a resume will load, from the sidecar alone (for the first plan, before the unpickle)."""
    if run_dir is None or not (run_dir / CHECKPOINT).exists():
        return set()
    side = _read_json(run_dir / CHECKPOINT_JSON) or {}
    return set(side.get("done") or ()) if side.get("fingerprint") == fingerprint else set()


# ---------------------------------------------------------------------------
# run_core
# ---------------------------------------------------------------------------


def _job_id() -> str | None:
    return getattr(getattr(progress, "_S", None), "job", None) or os.environ.get(progress.ENV_JOB) or None


class _RunState:
    """``run_state.json``: rewritten atomically at the start, at every stage boundary and at the end."""

    def __init__(self, run_dir: Path | None, fingerprint: str):
        self.path = None if run_dir is None else run_dir / RUN_STATE
        self.doc = {"schema": 1, "status": "running", "pid": os.getpid(), "job": _job_id(),
                    "host": socket.gethostname(), "started_utc": _utc(), "updated_utc": _utc(), "stage": None,
                    "done": [], "fingerprint": fingerprint, "events_path": progress.sink_path(), "error": None,
                    "meta": {}}

    def write(self, **fields) -> None:
        meta = fields.pop("meta", None)
        self.doc.update(fields)
        if meta:
            self.doc["meta"].update(meta)
        if self.path is None:
            return
        self.doc["updated_utc"] = _utc()
        try:
            runio.write_json_atomic(self.path, self.doc)
        except OSError as exc:                       # the run itself must not fail on its status file
            log.warning("could not write %s: %s", self.path, exc)


def run_core(cfg: CoreConfig | str | Path, stages=ALL_STAGES, fast: bool = False, frame: pd.DataFrame | None = None,
             write: bool = True, resume: bool = False, cv_curve: bool | None = None,
             coarse: float | None = None, run_dir: Path | str | None = None,
             run_meta: dict | None = None) -> CoreResult:
    """Run the requested stages.  ``fast`` shrinks the problem (8k-point
    window, 3 folds, fewer epochs) for smoke runs and CI.  With ``write``,
    fitted state is checkpointed after S3, the baselines, the CV distance
    curve, S4, S5, climate and S6; ``resume`` reuses a checkpoint whose
    fingerprint matches.  ``cv_curve`` overrides
    ``cv.distance_curve.enabled``; ``coarse`` (metres) overrides
    ``data.coarse_m`` — the full extent averaged onto coarser cells, the
    resolution of the validation studies.

    ``run_dir`` replaces the default ``output.dir/<name>[_fast][_coarse<M>]``;
    ``run_meta`` (Studio's run identity and lineage) is recorded in the
    ``run.start`` event and the manifest.  Progress, warnings and artifacts
    go to the :mod:`sparc.core.progress` sink when one is configured, and a
    cancel request stops the run at the next stage boundary (or a finer
    safe point) with :class:`~sparc.core.progress.Cancelled`."""
    if not isinstance(cfg, CoreConfig):
        cfg = load_core_config(cfg)
    apply_mode_overrides(cfg, fast=fast, coarse=coarse, cv_curve=cv_curve, frame_given=frame is not None)
    run = _Run(cfg, stages, fast, frame, write, resume, run_dir, run_meta)
    run.cv_curve_arg = cv_curve
    return run.execute()


class _Run:
    """One ``run_core`` invocation: stage spans, plan, checkpoints and run state."""

    def __init__(self, cfg: CoreConfig, stages, fast: bool, frame, write: bool, resume: bool, run_dir, run_meta):
        self.cfg, self.fast, self.frame, self.resume = cfg, bool(fast), frame, bool(resume)
        self.stages = set(stages)
        self.requested = [s for s in ALL_STAGES if s in self.stages]
        self.run_meta = dict(run_meta or {})
        if write:
            if run_dir is None:
                cm = cfg.data.get("coarse_m")
                run_dir = cfg.output_dir / (cfg.name + ("_fast" if fast else "") +
                                            (f"_coarse{float(cm):g}" if cm else ""))
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = None
        self.run_dir = run_dir
        self.data_id = _data_identity(cfg, frame) if run_dir is not None else ""
        self.fp = _fingerprint(cfg, fast, frame, self.data_id) if run_dir is not None else ""
        self.sections = fingerprint_sections(cfg, fast, frame) if run_dir is not None else {}
        self.timings: dict[str, float] = {}           # manifest timings_s
        self.stage_s: dict[str, float] = {}           # every stage that ran, climate and finish included
        self.stages_run: list[str] = []
        self.state: dict = {}
        self.done: set = set()
        self.planned_done: set = set()
        self.geom: dict = {}
        self.block_m: float | None = None
        self.n_climate_models: int | None = None
        self.overrides: dict[str, dict] = {}
        self.last_plan = None
        self.engine = None
        self.resp = None
        self.rs = _RunState(run_dir, self.fp)
        self.result: CoreResult | None = None
        self.cv_curve_arg: bool | None = None

    # ------------------------------------------------------------- helpers
    def plan(self) -> None:
        """Emit ``run.plan`` when the plan or the point count changed (cached nodes, geometry, CV partitions
        once the block is known, the CMIP6 model count, an S7 with nothing to allocate)."""
        nodes = _plan(self.cfg, self.stages, self.planned_done, write=self.run_dir is not None,
                      block_m=self.block_m, n_climate_models=self.n_climate_models, **self.geom)
        for nid, units in self.overrides.items():
            for n in nodes:
                if n["id"] == nid:
                    n["units"] = {k: int(v) for k, v in units.items() if v}
        key = (nodes, self.geom.get("n_points"))
        if key != self.last_plan:
            self.last_plan = key
            progress.emit("run.plan", **plan_event(nodes, self.geom.get("n_points")))

    @contextmanager
    def stage(self, sid: str):
        """A stage span: cancel check first, run_state at both boundaries, wall time recorded."""
        progress.check_cancel()
        self.rs.write(stage=sid, done=sorted(self.done))
        t = time.time()
        with progress.stage(sid, label=_LABELS[sid]) as sp:
            yield sp
        self.stage_s[sid] = time.time() - t
        self.stages_run.append(sid)
        self.rs.write(stage=sid, done=sorted(self.done))

    def skip(self, sid: str, reason: str) -> None:
        progress.skip(sid, reason)

    def save(self, key: str) -> None:
        self.done.add(key)
        self.state["done"] = self.done
        _save_checkpoint(self.run_dir, self.state, self.sections)

    def artifact(self, path: Path, role: str, stage: str | None = None) -> None:
        progress.artifact(path, role=role, stage=stage)

    def write_json(self, name: str, obj, role: str, stage: str | None = None) -> None:
        if self.run_dir:
            _write_json(self.run_dir / name, obj)
            self.artifact(self.run_dir / name, role, stage)

    def write_parquet(self, name: str, df: pd.DataFrame, role: str, stage: str | None = None) -> None:
        if self.run_dir:
            runio.write_parquet_atomic(df, self.run_dir / name)
            self.artifact(self.run_dir / name, role, stage)

    def ensure_engine(self):
        """ScenarioEngine (baseline pass) + ResponseEngine, built once, in the first stage that needs them."""
        if self.engine is None:
            from sparc.core.response import ResponseEngine
            from sparc.core.scenarios import ScenarioEngine

            r, cfg = self.result, self.cfg
            with progress.task("engine_init"):
                mediators = MediatorChain(cfg.mediators).fit(r.data.frame) if cfg.mediators else None
                self.engine = ScenarioEngine(r.data, cfg, r.ensemble, dict(r.influence.ranges_m), mediators)
                self.resp = ResponseEngine(self.engine,
                                           influence_scales=cfg.raw["influence"].get("scales", (0.5, 1.0, 2.0)))
        return self.engine, self.resp

    # --------------------------------------------------------------- drive
    def execute(self) -> CoreResult:
        cfg, run_dir = self.cfg, self.run_dir
        from sparc.core.provenance import config_hash

        status, error = "failed", None
        try:
            with (progress.run_dir_scope(run_dir) if run_dir is not None else nullcontext()), \
                    progress.span("run", cfg.name, stages=self.requested, fast=self.fast,
                                  coarse=cfg.data.get("coarse_m"), resume=self.resume, cv_curve=self.cv_curve_arg,
                                  config_sha256=config_hash(cfg.raw), code_sha256=_code_digest(),
                                  run_meta=self.run_meta) as rsp:
                try:
                    if self.resume:
                        self.planned_done = _peek_done(run_dir, self.fp)
                    self.plan()
                    if run_dir is not None:
                        progress.emit("run.dir", run_dir=str(run_dir.resolve()), fingerprint=self.fp)
                    self.rs.write(status="running")
                    result = self.body()
                finally:
                    rsp.set(timings_s={k: round(v, 2) for k, v in self.timings.items()}, done=sorted(self.done))
            status = "succeeded"
            return result
        except (progress.Cancelled, KeyboardInterrupt):
            status = "cancelled"
            raise
        except BaseException as exc:
            import traceback

            error = {"type": type(exc).__name__, "message": str(exc)[:2000],
                     "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]}
            raise
        finally:
            self.rs.write(status=status, error=error, done=sorted(self.done))

    def body(self) -> CoreResult:
        cfg, run_dir, frame, stages = self.cfg, self.run_dir, self.frame, self.stages

        # ------------------------------------------------------------- S0
        with self.stage("S0") as sp:
            t = time.time()
            data = prepare_frame(frame, cfg) if frame is not None else load_core_data(cfg)
            self.timings["S0"] = time.time() - t
            result = self.result = CoreResult(cfg=cfg, data=data, run_dir=run_dir)
            from sparc.core.provenance import provenance, sha256_file, sha256_frame

            result.provenance = provenance(cfg, sha256_frame(frame) if frame is not None else
                                           sha256_file(cfg.data_path), "frame" if frame is not None else "file")
            if frame is None and cfg.data.get("join"):
                result.provenance["join_sha256"] = {j["path"]: sha256_file(cfg.resolve_path(j["path"]))
                                                    for j in cfg.data["join"]}
            if frame is not None and run_dir is not None:
                # placebo children: the table they were fitted on differs from data.path; keep it readable
                self.write_parquet(INPUT_FRAME, frame, "input_frame")
                result.provenance["input_frame"] = INPUT_FRAME
            qa = data.qa
            sp.summary = {"n_points": int(data.n), "n_input": qa.get("n_input"),
                          "n_dropped": qa.get("n_dropped_nonfinite"), "clipped": qa.get("clipped") or {},
                          "grid_shape": list(data.grid.shape), "cell_m": float(data.grid.dx),
                          "fill_fraction": qa.get("grid_fill_fraction"), "background": qa.get("background"),
                          "noise_floor": qa.get("target_rounding_noise_sd")}
        log.info("S0: %d points, grid %s, cell %.2f m", data.n, data.grid.shape, data.grid.dx)
        for f in qa.get("flags") or []:
            progress.warn(f"qa.{f['code']}", f.get("message", ""), severity=f.get("severity"))
        co = qa.get("coarse") or {}
        self.rs.write(meta={"n_points": int(data.n), "cell_m": float(data.grid.dx), "grid_shape": list(data.grid.shape),
                            "coarse_m": co.get("cell_m"), "subsample_window_n": qa.get("subsample_window_n")})
        self.geom = {"n_points": int(data.n), "dx": float(data.grid.dx),
                     "extent": float(min(np.ptp(data.x), np.ptp(data.y_coord)))}
        self.state = _load_checkpoint(run_dir, self.fp, self.sections) if (self.resume and run_dir) else {}
        self.done = set(self.state.get("done", ()))
        self.state = {**self.state, "fingerprint": self.fp, "done": self.done}
        self.planned_done = set(self.done)
        self.plan()

        # ------------------------------------------------------------- S1
        if "S3" in self.done:
            influence = self.state["influence"]
            self.skip("S1", "checkpoint")
            self.timings["S1"] = 0.0
        else:
            from sparc.core.influence import compute_influence

            with self.stage("S1") as sp:
                t = time.time()
                influence = compute_influence(data, cfg.raw["influence"], seed=int(cfg.raw["cv"].get("seed", 0)))
                self.timings["S1"] = time.time() - t
                for c, r in influence.ranges_m.items():
                    progress.metric("influence.range_m", r, unit="m", predictor=c)
                    ratio = (influence.anisotropy.get(c) or {}).get("ratio")
                    progress.metric("influence.anisotropy_ratio", ratio, predictor=c)
                sp.summary = {"block_size_m": influence.block_size_m, "L_prior_m": influence.L_prior_m,
                              "target_resid_range_m": influence.target_resid_range_m}
        result.influence = influence
        ranges = dict(influence.ranges_m)
        self.write_json("influence.json", influence.to_dict(), "influence", "S1")
        self.block_m = _cv_block(cfg, data.grid.dx, self.geom["extent"], influence.block_size_m)[0]
        self.plan()

        if not _wants_s2(stages):
            for sid in ("S2_S3", "baselines", "cv_curve", "S4", "S5"):
                self.skip(sid, "not_requested")
            self.skip("climate", _climate_runs(cfg, stages)[1])
            for sid in ("S6", "S7"):
                self.skip(sid, "not_requested")
            return self.finish()

        # ---------------------------------------------------------- S2+S3
        t = time.time()
        if "S3" in self.done:
            ctx = build_context(data.frame, data, ranges, cfg)
            folds, ens = self.state["folds"], self.state["ensemble"]
            self.skip("S2_S3", "checkpoint")
        else:
            with self.stage("S2_S3") as sp:
                ctx = build_context(data.frame, data, ranges, cfg)
                block, buf = _block_and_buffer(cfg, influence, data)
                folds = make_spatial_folds(data.coords, n_folds=int(cfg.raw["cv"]["n_folds"]), block_m=block,
                                           buffer_m=buf, seed=int(cfg.raw["cv"].get("seed", 42)))
                ens = fit_ensemble(ctx, folds, cfg, ranges_m=ranges, intercept_range_m=influence.target_resid_range_m,
                                   L_init=influence.L_prior_m, seed=int(cfg.raw["cv"].get("seed", 0)))
                self.state.update(influence=influence, folds=folds, ensemble=ens)
                self.save("S3")
                st = ens.metrics["stacker"]
                sp.summary = {"stacker_rmse": st["rmse"], "stacker_r2": st["r2"], "stacker_choice": ens.stacker_choice,
                              "block_m": folds.block_m, "n_folds": folds.n_folds}
        self.folds = folds
        result.ensemble = ens
        if data.zones is not None and ens.halfwidth_adaptive is not None:
            from sparc.core.ensemble import interval_diagnostics

            ens.metrics["stacker"]["interval_diagnostics"] = interval_diagnostics(
                data.y, ens.oof_pred, ens.halfwidth, ens.halfwidth_adaptive, folds.fold_id, ens.dist_train,
                groups={"zone": data.zones})
        self.timings["S2_S3"] = time.time() - t
        self.rs.write(meta={"cv": {"n_folds": int(folds.n_folds), "block_m": float(folds.block_m),
                                   "buffer_m": float(folds.buffer_m), "seed": int(cfg.raw["cv"].get("seed", 42))}})
        if run_dir:
            out = pd.DataFrame({"id": data.ids, "x_m": data.x, "y_m": data.y_coord, "fold": folds.fold_id,
                                "target": data.target_raw, "dT": data.y, "dT_pred": ens.oof_pred,
                                "pred": ens.oof_pred + data.background,
                                "pi_lo": ens.oof_pred + data.background - ens.halfwidth,
                                "pi_hi": ens.oof_pred + data.background + ens.halfwidth})
            if ens.halfwidth_adaptive is not None:
                out["pi_lo_adaptive"] = ens.oof_pred + data.background - ens.halfwidth_adaptive
                out["pi_hi_adaptive"] = ens.oof_pred + data.background + ens.halfwidth_adaptive
                out["dist_train_m"] = ens.dist_train
            if data.zones is not None:
                out["zone"] = data.zones
            for c in ens.oof_base.columns:
                out[f"oof_{c}"] = ens.oof_base[c].to_numpy(float) + data.background
            self.write_parquet("predictions.parquet", out, "predictions", "S2_S3")
            if ens.has_physics:
                self.write_json("physics.json", [st.physics.params for st in ens.stacks], "physics", "S2_S3")

        # ------------------------------------------------- reference baselines
        models = _baselines_list(cfg)
        if models:
            t = time.time()
            if "baselines" in self.done:
                result.baselines = self.state["baselines"]
                self.skip("baselines", "checkpoint")
            else:
                from sparc.core.baselines import compare_baselines

                with self.stage("baselines") as sp:
                    result.baselines = compare_baselines(data.X.to_numpy(float), data.coords, data.y, ens.oof_pred,
                                                         folds, models=models, seed=int(cfg.raw["cv"].get("seed", 0)),
                                                         XF=ctx.XF.to_numpy(float))
                    self.state["baselines"] = result.baselines
                    self.save("baselines")
                    sp.summary = {"verdict": result.baselines["verdict"],
                                  "best_baseline": result.baselines["best_baseline"]}
            log.info("baselines: %s", result.baselines["verdict"])
            self.timings["baselines"] = time.time() - t
            self.write_json("baselines.json", result.baselines, "baselines", "baselines")
        else:
            self.skip("baselines", "disabled_by_config:cv.baselines")

        # ------------------------------------------------- CV distance diagnostic
        dcfg = cfg.raw["cv"].get("distance_curve") or {}
        if not dcfg.get("enabled"):
            self.skip("cv_curve", "disabled_by_config:cv.distance_curve.enabled")
        else:
            t = time.time()
            if "cv_curve" in self.done:
                result.cv_distance = self.state["cv_distance"]
                self.skip("cv_curve", "checkpoint")
            elif not _cv_curve_sizes(cfg, folds.block_m, data.grid.dx, self.geom["extent"]):
                from sparc.core.diagnostics import curve_partitions

                for sk in curve_partitions(dcfg.get("block_m", (0, 500, 1000)), data.grid.dx, self.geom["extent"],
                                           folds.block_m)[1]:
                    progress.warn("cv.partition_skipped", f"{sk['block_m']:g} m blocks are outside "
                                  f"[{sk['lo']:.0f}, {sk['hi']:.0f}] m for this study area", **sk)
                self.skip("cv_curve", "no_partitions")
            else:
                from sparc.core.diagnostics import cv_distance_curve

                with self.stage("cv_curve") as sp:
                    result.cv_distance = cv_distance_curve(ctx, data, cfg, influence, ens, folds,
                                                           block_sizes=dcfg.get("block_m", (0, 500, 1000)),
                                                           seed=int(cfg.raw["cv"].get("seed", 42)),
                                                           baselines=cfg.raw["cv"].get("baselines", True),
                                                           main_baselines=result.baselines or None)
                    self.state["cv_distance"] = result.cv_distance
                    self.save("cv_curve")
                    sp.summary = {"n_rows": len(result.cv_distance["rows"])}
            self.timings["cv_curve"] = time.time() - t
            if result.cv_distance:
                self.write_json("cv_distance.json", result.cv_distance, "cv_distance", "cv_curve")

        if not (_ENGINE_STAGES & stages):
            for sid in ("S4", "S5"):
                self.skip(sid, "not_requested")
            self.skip("climate", _climate_runs(cfg, stages)[1])
            for sid in ("S6", "S7"):
                self.skip(sid, "not_requested")
            return self.finish()
        self.s4_s7(data, ens, ranges)
        return self.finish()

    # ------------------------------------------------------------ S4 – S7
    def s4_s7(self, data: CoreData, ens, ranges: dict) -> None:
        cfg, run_dir, stages, result = self.cfg, self.run_dir, self.stages, self.result

        t = time.time()
        if not _wants_s4(stages):
            self.skip("S4", "not_requested")
        elif "S4" in self.done:
            result.responses = self.state["responses"]
            self.skip("S4", "checkpoint")
        else:
            with self.stage("S4") as sp:
                levers = list(cfg.actionable)
                for v, var in enumerate(levers, start=1):
                    progress.check_cancel()
                    _eng, resp = self.ensure_engine()
                    with progress.task("variable", k=v, n=len(levers), key=var):
                        result.responses[var] = resp.sweep(var)
                self.state["responses"] = result.responses
                self.save("S4")
                sp.summary = {v: {"mean_own_effect": r.summary.get("mean_own_effect"),
                                  "median_d90": r.summary.get("median_d90")} for v, r in result.responses.items()}
        if result.responses:
            for var, vr in result.responses.items():
                self.write_parquet(f"response_{var}.parquet",
                                   vr.maps.assign(id=data.ids, x_m=data.x, y_m=data.y_coord), "response_map", "S4")
            self.write_json("response_curves.json", {v: {"summary": r.summary, "curve": r.curve.to_dict(orient="list")}
                                                     for v, r in result.responses.items()}, "response_curves", "S4")
        self.timings["S4"] = time.time() - t

        t = time.time()
        deltas: dict = {}
        if "S5" not in stages:
            self.skip("S5", "not_requested")
        else:
            from sparc.core.scenarios import specs_from_config

            if "S5" in self.done:
                result.scenarios, deltas = self.state["scenarios"], self.state["scenario_deltas"]
                detail = self.state.get("scenario_detail")
                self.skip("S5", "checkpoint")
            else:
                with self.stage("S5") as sp:
                    specs = specs_from_config(cfg)
                    detail = {}
                    for s, spec in enumerate(specs, start=1):
                        progress.check_cancel()
                        engine, _resp = self.ensure_engine()
                        with progress.task("scenario", k=s, n=len(specs), key=spec.name) as tsp:
                            res = engine.run(spec)
                            summ = res.summary()
                            tsp.metrics.update(mean_delta=summ["mean_delta"], se=summ["mean_delta_se"],
                                               frac_extrapolated=summ["frac_extrapolated"])
                        result.scenarios.append(summ)
                        deltas[spec.name] = res.delta
                        detail[spec.name] = {"folds": np.asarray(res.delta_folds, np.float32),
                                             "sd": np.asarray(res.delta_sd, np.float32),
                                             "ex": np.asarray(res.extrapolation, np.float32)}
                        progress.metric("scenario.mean_delta", summ["mean_delta"], scenario=spec.name)
                        progress.metric("scenario.se", summ["mean_delta_se"], scenario=spec.name)
                        progress.metric("scenario.frac_extrapolated", summ["frac_extrapolated"], scenario=spec.name)
                    self.state.update(scenarios=result.scenarios, scenario_deltas=deltas, scenario_detail=detail)
                    self.save("S5")
                    sp.summary = {"n_scenarios": len(specs)}
            if run_dir and deltas:
                self.write_json("scenarios.json", result.scenarios, "scenarios", "S5")
                self.write_parquet("scenario_deltas.parquet", pd.DataFrame(deltas).assign(id=data.ids),
                                   "scenario_deltas", "S5")
                if (cfg.raw.get("output") or {}).get("scenario_detail", True) and detail \
                        and all(n in detail for n in deltas):
                    self.write_scenario_detail(data, list(deltas), detail)

        runs_c, reason_c = _climate_runs(cfg, stages)
        if not runs_c:
            self.skip("climate", reason_c)
        else:
            if "climate" in self.done:
                result.climate = self.state["climate"]
                self.skip("climate", "checkpoint")
            else:
                def on_models(m: int) -> None:
                    self.n_climate_models = int(m)
                    self.plan()

                with self.stage("climate") as sp:
                    result.climate = climate_stage(cfg, data, result.scenarios, deltas, on_models=on_models)
                    self.state["climate"] = result.climate
                    self.save("climate")
                    sp.summary = {"n_models": result.climate.get("n_models"), "source": result.climate.get("source")}
            self.write_json("climate.json", result.climate, "climate", "climate")
        self.timings["S5"] = time.time() - t

        # ------------------------------------------------------------- S6
        t = time.time()
        runs6, reason6, eff_tr = _s6_runs(cfg, stages, set(result.responses))
        if not runs6:
            self.skip("S6", reason6)
        else:
            if "S6" in self.done:
                result.causal = self.state["causal"]
                self.skip("S6", "checkpoint")
            else:
                from sparc.core.causal import run_causal_validation

                with self.stage("S6") as sp:
                    effects = {}
                    for i, tr in enumerate(eff_tr, start=1):
                        progress.check_cancel()
                        _engine, resp = self.ensure_engine()
                        with progress.task("model_effects", k=i, n=len(eff_tr), key=tr):
                            x = data.frame[tr].to_numpy(float)
                            pos = x[x > 0] if (x > 0).mean() < 0.95 else x
                            t_grid = np.quantile(pos, np.linspace(0.1, 0.9, 7))
                            effects[tr] = resp.model_effects(tr, result.responses[tr], t_grid=t_grid)
                    result.causal = run_causal_validation(data, cfg.raw["causal"], self.folds, ranges,
                                                          model_effects=effects,
                                                          seed=int(cfg.raw["cv"].get("seed", 0)))
                    for tr, me in effects.items():
                        if tr in (result.causal.get("treatments") or {}):
                            result.causal["treatments"][tr]["model_effects"] = me
                    self.state["causal"] = result.causal
                    self.save("S6")
                    sp.summary = {"treatments": list((result.causal.get("treatments") or {})),
                                  "n_flags": len(result.causal.get("flags") or [])}
            if result.scenarios:
                causal_crosscheck(result.scenarios, result.causal)
            self.write_json("causal.json", _strip_arrays(result.causal), "causal", "S6")
            self.write_causal_cells(data)
            if result.scenarios:
                self.write_json("scenarios.json", result.scenarios, "scenarios", "S6")
        self.timings["S6"] = time.time() - t

        # ------------------------------------------------------------- S7
        t = time.time()
        runs7, reason7 = _s7_runs(cfg, stages, set(result.responses))
        if not runs7:
            self.skip("S7", reason7)
        else:
            from sparc.core.optimize import optimise_allocation

            with self.stage("S7") as sp:
                ocfg = cfg.raw["optimize"]
                var = ocfg["variable"]
                engine_new = self.engine is None
                engine, _resp = self.ensure_engine()
                eq = equity_scores(cfg, data)
                cap, weight, constraint, objective = optimizer_layers(cfg, data, var, ranges)
                opt = optimise_allocation(engine, result.responses[var], float(ocfg["budget"]),
                                          cost_per_unit=cfg.actionable[var].get("cost_per_unit",
                                                                                ocfg.get("cost_per_unit", 1.0)),
                                          equity_scores=eq, equity_focus=float(ocfg.get("equity_focus", 0.0)),
                                          cap=cap, benefit_weight=weight, min_dose=float(ocfg.get("min_dose", 0.0)))
                opt.update(constraint=constraint, objective=objective)
                result.optimize = opt
                if "status" in opt:                  # nothing to allocate: no closed loop, no Pareto sweep
                    self.overrides["S7"] = {"engine_pass": 1 if engine_new else 0}
                    self.plan()
                sp.summary = {k: opt.get(k) for k in ("planned_total_cooling", "realized_total_cooling",
                                                      "n_cells_treated", "status") if k in opt}
                if run_dir:
                    self.write_json("optimize.json", {k: v for k, v in opt.items()
                                                      if k not in ("dose", "closed_loop_delta", "planned_benefit")},
                                    "optimize", "S7")
                    self.write_parquet("allocation.parquet", pd.DataFrame({
                        "id": data.ids, "x_m": data.x, "y_m": data.y_coord, "dose": opt.get("dose"),
                        "closed_loop_delta": opt.get("closed_loop_delta")}), "allocation", "S7")
        self.timings["S7"] = time.time() - t

    def write_scenario_detail(self, data: CoreData, names: list[str], detail: dict) -> None:
        """``scenario_detail.npz``: ids, names (JSON as uint8) and per scenario i ``f{i}`` (K×n fold Δ),
        ``sd{i}`` (jackknife SD) and ``ex{i}`` (extrapolation score), float32."""
        ids = np.asarray(data.ids)
        if ids.dtype == object:
            ids = ids.astype(str)
        arrays = {"ids": ids, "names": np.frombuffer(json.dumps(names).encode("utf-8"), dtype=np.uint8)}
        for i, n in enumerate(names):
            d = detail[n]
            arrays[f"f{i}"] = np.asarray(d["folds"], np.float32)
            arrays[f"sd{i}"] = np.asarray(d["sd"], np.float32)
            arrays[f"ex{i}"] = np.asarray(d["ex"], np.float32)
        runio.write_npz_atomic(self.run_dir / "scenario_detail.npz", compressed=True, **arrays)
        self.artifact(self.run_dir / "scenario_detail.npz", "scenario_detail", "S5")

    def write_causal_cells(self, data: CoreData) -> None:
        """``causal_cells.parquet``: per cell and treatment the R-learner CATE and the model's adoption and
        own-cell slopes (the arrays ``causal.json`` leaves out), row-aligned with ``predictions.parquet``."""
        if not self.run_dir:
            return
        cols: dict[str, np.ndarray] = {}
        for t, r in ((self.result.causal or {}).get("treatments") or {}).items():
            if not isinstance(r, dict):
                continue
            tau = (r.get("cate") or {}).get("tau_hat")
            if tau is not None:
                cols[f"cate:{t}"] = np.asarray(tau, np.float32)
            me = r.get("model_effects") or {}
            if me.get("per_point_slope") is not None:
                cols[f"mslope:{t}"] = np.asarray(me["per_point_slope"], np.float32)
            if me.get("per_point_own_slope") is not None:
                cols[f"mslope_own:{t}"] = np.asarray(me["per_point_own_slope"], np.float32)
        if cols:
            self.write_parquet("causal_cells.parquet", pd.DataFrame({"id": data.ids, **cols}), "causal_cells", "S6")

    # ------------------------------------------------------------- finish
    def finish(self) -> CoreResult:
        r = self.result
        with self.stage("finish"):
            _finish(r, self.timings, self.fast, folds=getattr(self, "folds", None), extras=self.manifest_extras())
            for name, role in (("manifest.json", "manifest"), ("environment.txt", "environment"),
                               ("methods.md", "methods"), ("model_card.md", "model_card"), ("report.md", "report")):
                if r.run_dir is not None:
                    self.artifact(r.run_dir / name, role, "finish")
        return r

    def manifest_extras(self) -> dict:
        ens = self.result.ensemble
        detail: dict = {"stages": {k: round(v, 3) for k, v in self.stage_s.items()}}
        if ens is not None:
            et = dict(getattr(ens, "timings", None) or {})
            detail["ensemble"] = {k: v for k, v in et.items() if k != "fold_model_s"}
            if et.get("fold_model_s"):
                detail["fold_model_s"] = et["fold_model_s"]
            if ens.has_physics:
                detail["physics_fit_s"] = [(getattr(st.physics.model, "fit_info", None) or {}).get("seconds")
                                           for st in ens.stacks]
        return {"schema_version": MANIFEST_SCHEMA, "stages_run": list(self.stages_run), "timings_detail": detail,
                "run_meta": self.run_meta}


def optimizer_layers(cfg: CoreConfig, data: CoreData, var: str, ranges: dict):
    """Plantable-space cap and people weighting for S7 from the planner layers:
    ``(cap, benefit_weight, constraint label, objective label)``."""
    from sparc.core import operators as ops
    from sparc.core.opendata import load_layers

    ocfg = cfg.raw["optimize"]
    layers = load_layers(cfg, data) if (cfg.raw.get("planner") or {}).get("layers") else None
    cap, weight = None, None
    constraint, objective = "unconstrained (no plantable-space layer)", "total cooling"
    canopy = cfg.physics_role("canopy")
    if layers is not None and var == canopy and ocfg.get("plantable", True):
        from sparc.core.planner import plantable_headroom

        share = float((cfg.raw.get("planner") or {}).get("paved_plantable_share", 0.2))
        cap = plantable_headroom(data.frame[var].to_numpy(float), layers, share)
        constraint = f"plantable space (WorldCover open land + {share:.0%} of built-up area)"
    if layers is not None and ocfg.get("objective", "cooling") == "people":
        g = data.grid
        r = g.rasterize(np.nan_to_num(layers["people"].to_numpy(float)))
        sigma = max(float(ranges.get(var, 300.0)) / 2.0 / g.dx, 1.0)
        dens = g.sample(ops.masked_gaussian(r, np.isfinite(r), sigma))
        weight = np.nan_to_num(dens) / max(float(np.nanmean(dens)), 1e-9)
        objective = "resident-weighted cooling (HRSL)"
    return cap, weight, constraint, objective


_optimizer_layers = optimizer_layers


def equity_scores(cfg: CoreConfig, data: CoreData) -> np.ndarray | None:
    """The ``optimize.equity_column`` score per cell, aligned with ``data.ids`` (carried through S0 as an
    auxiliary column); cells without a value get the mean score.  None when not configured or absent."""
    col = (cfg.raw.get("optimize") or {}).get("equity_column")
    v = (getattr(data, "aux", None) or {}).get(col) if col else None
    if v is None:
        return None
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return None
    return np.where(ok, v, float(np.mean(v[ok])))


def _site_latlon(cfg: CoreConfig, data: CoreData) -> tuple[float, float]:
    from pyproj import Transformer

    crs = cfg.data.get("crs")
    if not crs:
        raise ValueError("climate: set climate.site: [lat, lon] (the data have no CRS)")
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    s = cfg.coord_scale
    lon, lat = tr.transform(float(np.mean(data.x)) / s, float(np.mean(data.y_coord)) / s)
    return float(lat), float(lon)


def default_adaptation(scenarios: list[dict], max_extrapolated: float = 0.2) -> list[str]:
    """Packages (several variables) plus, per variable, the largest dose that
    stays within observed conditions."""
    names, best = [], {}
    for sc in scenarios:
        real = sc.get("mean_realized") or {}
        if sc.get("frac_extrapolated", 1.0) > max_extrapolated:
            continue
        if len(real) > 1:
            names.append(sc["name"])
        elif len(real) == 1:
            (v, d), = real.items()
            if v not in best or abs(d) > abs(best[v][1]):
                best[v] = (sc["name"], d)
    return names + [n for n, _ in best.values()]


def climate_stage(cfg: CoreConfig, data: CoreData, scenarios: list[dict], deltas: dict, on_models=None) -> dict:
    """CMIP6 change factors × observed field × adaptation scenarios.  ``on_models(M)`` is told the number of
    CMIP6 models once the catalogue is read (``climate.source: cmip6``)."""
    from sparc.core.climate import cmip6_change_factors, summarize_projections

    cc = cfg.raw["climate"]
    lat, lon = (float(cc["site"][0]), float(cc["site"][1])) if cc.get("site") else _site_latlon(cfg, data)
    if cc.get("source", "table") == "table":
        path = cfg.resolve_path(cc.get("table"))
        if path is None or not path.exists():
            raise FileNotFoundError(f"climate.table not found: {path} (write one with `sparc core climate`)")
        factors = pd.read_csv(path)
    else:
        cache = cfg.output_dir / str(cc.get("cache", "cache"))
        factors = cmip6_change_factors(lat, lon, cache, experiments=tuple(cc["experiments"]),
                                       periods={k: tuple(v) for k, v in cc["periods"].items()},
                                       months=tuple(cc["months"]), variable=cc["variable"], on_models=on_models)
        runio.write_text_atomic(cache / f"cmip6_{cc['variable']}_{lat:.3f}_{lon:.3f}.csv",
                                factors.to_csv(index=False))
    factors = factors[factors["experiment"].isin(cc["experiments"]) & factors["period"].isin(list(cc["periods"]))]
    units = (data.target_units or "").lower()
    fahrenheit = units in ("degf", "f", "°f", "fahrenheit")
    thresholds = cc.get("thresholds") or ([90.0, 95.0] if fahrenheit else [32.0, 35.0])
    names = cc.get("adaptation") or default_adaptation(scenarios)
    adapt = {n: deltas[n] for n in names if n in deltas}
    out = summarize_projections(np.asarray(data.target_raw, float), factors, adapt, thresholds,
                                to_units=1.8 if fahrenheit else 1.0)
    out.update({"site": {"lat": lat, "lon": lon}, "source": cc.get("source", "table"),
                "variable": cc["variable"], "months": list(cc["months"]),
                "baseline": str(factors["baseline"].iloc[0]) if "baseline" in factors and len(factors) else "1995-2014",
                "n_models": int(factors["model"].nunique()), "units": data.target_units})
    log.info("climate: %d models; median %s warming %s", out["n_models"],
             "2041-2060 SSP2-4.5", next((f"{p['warming']['median']:+.2f}" for p in out["projections"]
                                         if p["experiment"] == "ssp245" and p["period"] == "2041-2060"), "n/a"))
    return out


def causal_crosscheck(scenarios: list[dict], causal: dict) -> list[dict]:
    """Attach a linear causal estimate to every scenario whose edited
    variables all have a spillover estimate: Σ_j (θ_own+θ_nbr)_j × mean
    realised change_j, with a 95% band from the SEs (independence assumed).
    It is a local-slope extrapolation — a cross-check on magnitude and sign,
    not a replacement for the model's scenario."""
    tr = (causal or {}).get("treatments") or {}
    for sc in scenarios:
        real = sc.get("mean_realized") or {}
        if not real or any(not isinstance((tr.get(v) or {}).get("spillover"), dict) for v in real):
            continue
        est, var = 0.0, 0.0
        for v, d in real.items():
            sp = tr[v]["spillover"]
            th, se = sp.get("theta_sum"), sp.get("se_sum")
            if th is None or se is None or not np.isfinite(th) or not np.isfinite(se):
                break
            est += float(th) * float(d)
            var += (float(se) * float(d)) ** 2
        else:
            half = 1.96 * float(np.sqrt(var))
            sc["causal_linear"] = {"delta": est, "lo": est - half, "hi": est + half,
                                   "model_within": bool(est - half <= sc["mean_delta"] <= est + half)}
    return scenarios


def _strip_arrays(d):
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items() if not str(k).startswith("tau_hat")
                and not str(k).startswith("per_point")}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def _finish(result: CoreResult, timings: dict, fast: bool, folds=None, extras: dict | None = None) -> CoreResult:
    """Build the manifest and write it with the report, methods, model card and environment lock.

    A manifest already in the run directory keeps its post-run sections
    (:data:`POST_RUN_KEYS`) and a post-run ``baselines`` section this run
    did not recompute; the merge happens under the run lock."""
    from sparc.core.report import build_manifest, render_report

    result.manifest = build_manifest(result, timings, fast, folds)
    result.manifest.update(extras or {})
    if result.run_dir:
        from sparc.core.provenance import environment_lock
        from sparc.core.writeup import methods_markdown, model_card_markdown

        path = result.run_dir / "manifest.json"
        with runio.run_lock(result.run_dir):
            old = _read_json(path) or {}
            for k in POST_RUN_KEYS:
                if k in old and k not in result.manifest:
                    result.manifest[k] = old[k]
            post_sections = {e.get("section") for e in (old.get("post_run") or []) if isinstance(e, dict)}
            if not result.baselines and "baselines" in old and "baselines" in post_sections:
                result.manifest["baselines"] = old["baselines"]
            _write_json(path, result.manifest)
        runio.write_text_atomic(result.run_dir / "environment.txt", environment_lock())
        runio.write_text_atomic(result.run_dir / "methods.md", methods_markdown(result.manifest))
        runio.write_text_atomic(result.run_dir / "model_card.md", model_card_markdown(result.manifest))
        runio.write_text_atomic(result.run_dir / "report.md", render_report(result))
    return result
