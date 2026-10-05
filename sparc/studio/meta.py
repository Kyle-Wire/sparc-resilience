"""Catalogs served by ``GET /api/meta`` (api.md §2): stages, job kinds, outputs, palettes, unit costs, modes.

* ``stages`` come from ``sparc.core.pipeline.STAGE_NODES`` (imported lazily)
  with :data:`STATIC_STAGES` as the fallback and the source of labels and
  descriptions the core table lacks.
* ``output_catalog`` comes from ``sparc.core.catalog.OUTPUTS`` (lazily), or
  ``[]`` while that module does not exist; ``run_tab_outputs`` groups it by
  ``OutputSpec.view`` over the fixed run-tab vocabulary.
* ``palettes`` are the four OKLab ramps of the results-page template plus the
  categorical colours, ported verbatim; :func:`lut` is the Python twin of the
  template's ``lut()`` (SPEC §6.3 golden values).
* ``unit_costs`` is the seed table of SPEC §5.4 (:mod:`.jobs.eta`).
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

from sparc.studio import __version__
from sparc.studio.jobs.eta import unit_costs
from sparc.studio.schemas.common import SELECTION_KINDS

log = logging.getLogger("sparc.studio")

__all__ = ["STATIC_STAGES", "RAMPS", "CAT", "RUN_TAB_IDS", "EDIT_MODES", "MODES", "WARNING_CODES", "stages",
           "output_catalog", "palettes", "lut", "lut_hex", "build_meta"]

#: (id, label, desc, checkpoint_key, manifest_timing_key) - SPEC §5.3
STATIC_STAGES: list[dict] = [
    {"id": "S0", "label": "Data & QA", "desc": "Load the points, apply QA flags and build the 30 m grid.",
     "checkpoint_key": None, "manifest_timing_key": "S0"},
    {"id": "S1", "label": "Influence", "desc": "Estimate each predictor's area of influence and the CV block size.",
     "checkpoint_key": "S3", "manifest_timing_key": "S1"},
    {"id": "S2_S3", "label": "Models & stacker",
     "desc": "Fit the base models per spatial fold and the physics-informed stacker.",
     "checkpoint_key": "S3", "manifest_timing_key": "S2_S3"},
    {"id": "baselines", "label": "Baselines", "desc": "Fit reference baselines on the same folds.",
     "checkpoint_key": "baselines", "manifest_timing_key": "baselines"},
    {"id": "cv_curve", "label": "CV distance curve", "desc": "Repeat the fit at other block sizes.",
     "checkpoint_key": "cv_curve", "manifest_timing_key": "cv_curve"},
    {"id": "S4", "label": "Response", "desc": "Dose-response curves and marginal effects per lever.",
     "checkpoint_key": "S4", "manifest_timing_key": "S4"},
    {"id": "S5", "label": "Scenarios", "desc": "Configured scenarios with per-fold deltas.",
     "checkpoint_key": "S5", "manifest_timing_key": "S5"},
    {"id": "climate", "label": "Climate", "desc": "Warming offsets from CMIP6 or a climate table.",
     "checkpoint_key": "climate", "manifest_timing_key": None},
    {"id": "S6", "label": "Causal audit", "desc": "Causal cross-checks of the model's effects.",
     "checkpoint_key": "S6", "manifest_timing_key": "S6"},
    {"id": "S7", "label": "Budget", "desc": "Budget allocation and the Pareto frontier.",
     "checkpoint_key": None, "manifest_timing_key": "S7"},
    {"id": "finish", "label": "Documents", "desc": "Manifest, report, methods and model card.",
     "checkpoint_key": None, "manifest_timing_key": None},
]
_STAGE_IDS = [s["id"] for s in STATIC_STAGES]

RAMPS: dict[str, list[str]] = {
    "seqLight": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "seqDark": ["#1d2f45", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    "divLight": ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#f19a8f", "#d6403f", "#8a1f22"],
    "divDark": ["#9ec5f4", "#3987e5", "#1f4f8a", "#383835", "#8f3434", "#e66767", "#f6b3ab"],
}
#: categorical series colours (template --s1, --s2, --s3) and the grey for 0 / no data (light theme)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1"]

RUN_TAB_IDS = ("overview", "data", "accuracy", "distance", "influence", "response", "causal", "scenarios", "climate",
               "heat", "budget", "planner", "lab", "validation", "uncertainty", "provenance", "track", "map", "docs", "files")
EDIT_MODES = ("add", "set", "scale", "floor", "ceiling", "fill_headroom", "to_percentile", "per_cell")
MODES = [
    {"id": "fast", "label": "Fast", "desc": "Quick check: fewer models and tuning steps; minutes, not hours."},
    {"id": "coarse", "label": "Coarse", "desc": "Full pipeline on coarser cells (default 60 m): a faithful preview.",
     "default_coarse_m": 60},
    {"id": "full", "label": "Full", "desc": "Every model at full 30 m resolution: the run decisions rest on."},
]
#: registered warning codes (SPEC §5.3) → (label, run tab that explains them)
WARNING_CODES: list[dict] = [
    {"code": "qa.classed_target", "label": "Target looks classed (few distinct values)", "view": "data"},
    {"code": "qa.albedo_scale", "label": "Albedo scale looks off", "view": "data"},
    {"code": "qa.canopy_scale", "label": "Canopy scale looks off", "view": "data"},
    {"code": "qa.impervious_scale", "label": "Impervious scale looks off", "view": "data"},
    {"code": "qa.cover_overlap", "label": "Cover fractions overlap", "view": "data"},
    {"code": "qa.dose_scale", "label": "Dose scale differs from the data", "view": "data"},
    {"code": "qa.coarse", "label": "Coarse cells", "view": "data"},
    {"code": "qa.window", "label": "Data subsampled to a window", "view": "data"},
    {"code": "cv.block_raised", "label": "CV block size raised", "view": "accuracy"},
    {"code": "cv.partition_skipped", "label": "CV partition skipped", "view": "distance"},
    {"code": "checkpoint.mismatch", "label": "Checkpoint does not match", "view": "files"},
    {"code": "influence.few_cells", "label": "Too few cells for an influence range", "view": "influence"},
    {"code": "influence.constant_predictor", "label": "Constant predictor", "view": "influence"},
    {"code": "physics.missing_roles", "label": "Physics roles missing", "view": "accuracy"},
    {"code": "physics.antiphysical_a", "label": "Anti-physical physics coefficient", "view": "accuracy"},
    {"code": "causal.hole_scale", "label": "Causal hole scale", "view": "causal"},
    {"code": "causal.blp_constant", "label": "Constant best linear predictor", "view": "causal"},
    {"code": "causal.dag_unavailable", "label": "DAG audit unavailable", "view": "causal"},
    {"code": "causal.no_confounders", "label": "No confounders", "view": "causal"},
    {"code": "causal.controls_missing", "label": "Causal controls missing", "view": "causal"},
    {"code": "causal.treatment_missing", "label": "Causal treatment missing", "view": "causal"},
    {"code": "causal.audit_flag", "label": "Causal audit flag", "view": "causal"},
    {"code": "baselines.stack_not_better", "label": "Stack not better than a baseline", "view": "distance"},
    {"code": "interval.coverage_below_target", "label": "Interval coverage below target", "view": "accuracy"},
    {"code": "reproduce.input_changed", "label": "Input changed since the run", "view": "validation"},
    {"code": "planner.hot_days_skipped", "label": "Hot days skipped", "view": "planner"},
    {"code": "planner.gpkg_skipped", "label": "GeoPackage skipped", "view": "planner"},
    {"code": "layers.missing", "label": "Planner layers missing", "view": "planner"},
    {"code": "layers.empty_points", "label": "Planner layers have empty points", "view": "planner"},
    {"code": "forcing.lsm_unavailable", "label": "Land-surface forcing unavailable", "view": "provenance"},
    {"code": "forcing.station_unavailable", "label": "Forcing station unavailable", "view": "provenance"},
    {"code": "climate.model_skipped", "label": "Climate model skipped", "view": "climate"},
    {"code": "network.retry", "label": "Network request retried", "view": "track"},
]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def _normalise_stage(node: Any, static: dict[str, dict], ckpt_keys: dict) -> dict | None:
    if isinstance(node, str):
        sid, extra = node, {}
    elif isinstance(node, dict):
        sid, extra = node.get("id"), node
    elif isinstance(node, (tuple, list)) and node:
        sid = node[0]
        extra = {"label": node[1]} if len(node) > 1 and isinstance(node[1], str) else {}
    else:
        sid, extra = getattr(node, "id", None), {k: getattr(node, k) for k in ("label", "desc", "checkpoint_key",
                                                                              "manifest_timing_key")
                                                 if hasattr(node, k)}
    if sid not in static:
        return None
    out = dict(static[sid])
    for key in ("label", "desc", "checkpoint_key", "manifest_timing_key"):
        if extra.get(key) is not None:
            out[key] = extra[key]
    if sid in ckpt_keys:
        out["checkpoint_key"] = ckpt_keys[sid]
    return out


def stages() -> list[dict]:
    """Stage table: ``sparc.core.pipeline.STAGE_NODES`` when importable, else :data:`STATIC_STAGES`."""
    static = {s["id"]: s for s in STATIC_STAGES}
    try:
        pipeline = importlib.import_module("sparc.core.pipeline")
    except Exception as exc:              # core import trouble must never break /api/meta
        log.debug("sparc.core.pipeline not importable (%s); static stage list", exc)
        return [dict(s) for s in STATIC_STAGES]
    nodes = getattr(pipeline, "STAGE_NODES", None)
    if not nodes:
        return [dict(s) for s in STATIC_STAGES]
    ckpt = getattr(pipeline, "CHECKPOINT_KEY", None)
    ckpt = dict(ckpt) if isinstance(ckpt, dict) else {}
    seq = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    out = [s for s in (_normalise_stage(n, static, ckpt) for n in seq) if s is not None]
    have = {s["id"] for s in out}
    if not out:
        return [dict(s) for s in STATIC_STAGES]
    for sid in _STAGE_IDS:                 # a core table that omits a stage still gets the full rail
        if sid not in have:
            out.append(dict(static[sid]))
    order = {sid: i for i, sid in enumerate(_STAGE_IDS)}
    return sorted(out, key=lambda s: order[s["id"]])


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _spec_dict(spec: Any) -> dict:
    get = spec.get if isinstance(spec, dict) else (lambda k, d=None: getattr(spec, k, d))
    return {"id": str(get("id")), "label": str(get("label", get("id"))), "group": str(get("group", "")),
            "files": [str(f) for f in (get("files") or ())], "produced_by": str(get("produced_by", "")),
            "view": str(get("view", "")), "formats": [str(f) for f in (get("formats") or ())],
            "manifest_key": get("manifest_key")}


def output_catalog() -> list[dict]:
    """``sparc.core.catalog.OUTPUTS`` as dicts, or ``[]`` while the catalog does not exist."""
    try:
        catalog = importlib.import_module("sparc.core.catalog")
    except ModuleNotFoundError as exc:
        if exc.name not in ("sparc.core.catalog",):
            log.warning("output catalog unavailable: %s", exc)
        return []
    except Exception as exc:
        log.warning("output catalog unavailable: %s", exc)
        return []
    out = []
    for spec in getattr(catalog, "OUTPUTS", None) or []:
        try:
            out.append(_spec_dict(spec))
        except Exception:
            continue
    return out


def run_tab_outputs(catalog: list[dict]) -> dict[str, list[str]]:
    tabs: dict[str, list[str]] = {t: [] for t in RUN_TAB_IDS}
    for spec in catalog:
        if spec["view"] in tabs:
            tabs[spec["view"]].append(spec["id"])
    return tabs


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

def palettes() -> dict[str, list[str]]:
    return {**{k: list(v) for k, v in RAMPS.items()}, "cat": list(CAT)}


def _hex2rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _delin(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def _to_lab(h: str) -> tuple[float, float, float]:
    r, g, b = (_lin(c) for c in _hex2rgb(h))
    l_ = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` (half up)."""
    return int(math.floor(x + 0.5))


def _from_lab(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
           -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    return tuple(_js_round(255 * min(1.0, max(0.0, _delin(c)))) for c in rgb)  # type: ignore[return-value]


def lut(stops: list[str], n: int = 256) -> list[tuple[int, int, int]]:
    """``n`` RGB entries interpolated linearly in OKLab between equally spaced ``stops`` (SPEC §6.3)."""
    labs = [_to_lab(s) for s in stops]
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(labs) - 1)
        k = min(len(labs) - 2, int(math.floor(t)))
        f = t - k
        out.append(_from_lab(tuple(v + (labs[k + 1][j] - v) * f for j, v in enumerate(labs[k]))))
    return out


def lut_hex(stops: list[str], n: int = 256) -> list[str]:
    return ["#%02x%02x%02x" % c for c in lut(stops, n)]


# ---------------------------------------------------------------------------
# the whole document
# ---------------------------------------------------------------------------

def build_meta(*, include_test_kinds: bool = True) -> dict:
    from sparc.studio.jobs.kinds import kinds_listing

    catalog = output_catalog()
    return {
        "version": __version__,
        "schema_version": 1,
        "event_schema_version": 1,
        "stages": stages(),
        "job_kinds": kinds_listing(include_test=include_test_kinds),
        "output_catalog": catalog,
        "palettes": palettes(),
        "unit_costs": unit_costs(),
        "modes": [dict(m) for m in MODES],
        "run_tab_outputs": run_tab_outputs(catalog),
        "edit_modes": list(EDIT_MODES),
        "selection_kinds": list(SELECTION_KINDS),
        "warning_codes": [dict(w) for w in WARNING_CODES],
    }
