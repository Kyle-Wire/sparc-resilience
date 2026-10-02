"""``validate_deep``: every config check of SPEC §9.4, as api.md ``Issue`` rows.

``validate_deep(raw, project_dir)`` checks the user's raw ``core:`` block
(merged with DEFAULTS, as core would run it) against the schema and the data:

* types (``CoreConfigModel``) and unknown keys (info);
* files exist (data, joins, forcing, climate table, planner layers);
  header columns exist and are numeric (a sample of 50k rows);
* target/id/x/y set; coord_unit valid; a CRS where the climate site
  fallback, open data, GeoTIFF exports or the people objective need one;
* predictors non-empty; levers are predictors; doses within bounds and
  including 0; direction valid; mediators, coupling and roles name
  predictors; scenario and package variables are levers;
* causal treatments, confounders and exclusions are predictors;
* ``climate.table`` exists when ``source: table``; ``optimize.variable`` is
  a lever; ``objective: people`` needs ``planner.layers``; the equity column
  exists; the forcing JSON parses.

Issue codes are stable (the UI keys help text on them).  The column sample
is cached per file (path, size, mtime).
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from sparc.studio.projects.config_schema import ROLE_NAMES, CoreConfigModel
from sparc.studio.projects.config_service import build_core_config, merged_raw

__all__ = ["validate_deep", "validate_report", "fast_overrides", "coarse_preview", "table_columns", "has_errors"]

SAMPLE_ROWS = 50000
_COLS_CACHE: "OrderedDict[tuple, dict[str, bool]]" = OrderedDict()
_COLS_LOCK = threading.Lock()
_COLS_MAX = 32


def has_errors(issues: list[dict]) -> bool:
    return any(i.get("level") == "error" for i in issues)


def table_columns(path: str | os.PathLike, rows: int = SAMPLE_ROWS) -> dict[str, bool]:
    """``{column: is_numeric}`` from the first ``rows`` rows of a CSV/parquet table (cached by size + mtime)."""
    from sparc.studio.projects.files import read_table_head

    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns, rows)
    with _COLS_LOCK:
        hit = _COLS_CACHE.get(key)
        if hit is not None:
            _COLS_CACHE.move_to_end(key)
            return hit
    df = read_table_head(p, rows=rows)
    out = {str(c): bool(pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
           for c in df.columns}
    with _COLS_LOCK:
        _COLS_CACHE[key] = out
        while len(_COLS_CACHE) > _COLS_MAX:
            _COLS_CACHE.popitem(last=False)
    return out


class _Issues:
    def __init__(self):
        self.rows: list[dict] = []
        self.typed: set[str] = set()

    def add(self, level: str, path: str, code: str, message: str, fix: dict | None = None) -> None:
        if level == "error" and path in self.typed and code != "type":
            return                                   # the type error at this path already says it
        row = {"level": level, "path": path, "code": code, "message": message}
        if fix is not None:
            row["fix"] = fix
        self.rows.append(row)


def _loc(loc) -> str:
    return ".".join(str(p) for p in loc)


def _schema_issues(eff: dict, out: _Issues) -> None:
    try:
        CoreConfigModel.model_validate(eff)
    except ValidationError as exc:
        seen = set()
        for e in exc.errors():
            path = _loc(e.get("loc", ()))
            # pydantic reports each union member; keep the first message per path
            parts = path.split(".")
            while parts and (parts[-1] in ("bool", "float", "int", "str", "list[float]") or
                             parts[-1].startswith(("literal[", "function-", "DagAudit", "list[", "dict["))):
                parts.pop()
            path = ".".join(parts)
            if path in seen:
                continue
            seen.add(path)
            out.typed.add(path)
            out.add("error", path, "type", f"{path}: {e.get('msg', 'invalid value')}")
    _unknown_keys(eff, CoreConfigModel, "", out)


def _models_in(annotation) -> tuple[list[type], str]:
    """``(models, shape)`` of a field annotation: shape is ``model`` (a sub-model, possibly optional),
    ``map`` (dict of models), ``list`` (list of models) or ``other``."""
    import typing

    origin = typing.get_origin(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation], "model"
    if origin is dict:
        args = typing.get_args(annotation)
        inner = args[1] if len(args) == 2 else None
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [inner], "map"
        return [], "other"
    if origin is list:
        args = typing.get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return [args[0]], "list"
        return [], "other"
    if origin is not None:                              # a union: the sub-model member, if any
        for a in typing.get_args(annotation):
            if isinstance(a, type) and issubclass(a, BaseModel):
                return [a], "model"
    return [], "other"


def _unknown_keys(d: Any, model: type[BaseModel], prefix: str, out: _Issues) -> None:
    """Info for every key the schema does not know (checked on the raw shape, valid or not)."""
    if not isinstance(d, dict):
        return
    fields = model.model_fields
    for k, v in d.items():
        k = str(k)
        if k not in fields:
            out.add("info", f"{prefix}{k}", "unknown_key",
                    f"'{prefix}{k}' is not a key Studio knows; it is kept and passed to core as is")
            continue
        models, shape = _models_in(fields[k].annotation)
        if not models:
            continue
        if shape == "model":
            _unknown_keys(v, models[0], f"{prefix}{k}.", out)
        elif shape == "map" and isinstance(v, dict):
            for name, item in v.items():
                _unknown_keys(item, models[0], f"{prefix}{k}.{name}.", out)
        elif shape == "list" and isinstance(v, list):
            for i, item in enumerate(v):
                _unknown_keys(item, models[0], f"{prefix}{k}.{i}.", out)


def _resolve(project_dir: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (project_dir / q)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def validate_deep(raw: dict, project_dir: str | os.PathLike, *, rows: int = SAMPLE_ROWS) -> list[dict]:
    """Every check of SPEC §9.4 on the raw ``core:`` block; returns ``Issue`` rows (errors first)."""
    from sparc.core.config import UNIT_TO_METRES
    from sparc.core.provenance import _str_keys

    pdir = Path(project_dir)
    out = _Issues()
    raw = _str_keys(raw or {})
    eff = merged_raw(raw)
    _schema_issues(eff, out)

    def sec(name) -> dict:
        v = eff.get(name)
        return v if isinstance(v, dict) else {}

    d = sec("data")
    plist = eff.get("predictors") if isinstance(eff.get("predictors"), list) else []   # else: the type error says it
    preds = [str(p) for p in plist if isinstance(p, (str, int, float))]
    pset = set(preds)
    act = {str(k): (v if isinstance(v, dict) else {}) for k, v in (eff.get("actionable") or {}).items()} \
        if isinstance(eff.get("actionable"), dict) else {}
    phys, causal, clim, opt = sec("physics"), sec("causal"), sec("climate"), sec("optimize")
    planner = eff.get("planner") if isinstance(eff.get("planner"), dict) else {}
    crs = d.get("crs") or d.get("reproject_to")

    # -- required keys and units ------------------------------------------------------------
    if not d.get("target"):
        out.add("error", "data.target", "required", "data.target (the temperature column) is required")
    for key in ("x", "y"):
        if not d.get(key):
            out.add("error", f"data.{key}", "required", f"data.{key} (coordinate column) is required")
    if not d.get("id"):
        out.add("warn", "data.id", "id_missing", "No id column: rows are numbered, so ids change when rows are "
                "dropped or the file is re-exported; joins and layers need a stable id")
    unit = str(d.get("coord_unit", "m")).lower()
    if unit not in UNIT_TO_METRES:
        out.add("error", "data.coord_unit", "bad_coord_unit",
                f"coord_unit {unit!r} is not one of {', '.join(sorted(UNIT_TO_METRES))}")
    if not preds and "predictors" not in out.typed:
        out.add("error", "predictors", "no_predictors", "List at least one predictor column")

    # -- files and columns ---------------------------------------------------------------------
    cols: dict[str, bool] | None = None
    if not d.get("path"):
        out.add("error", "data.path", "required", "data.path (the point table) is required")
    else:
        from sparc.studio.projects.files import is_csv_name

        p = _resolve(pdir, str(d["path"]))
        if not p.is_file():
            out.add("error", "data.path", "missing_file", f"data file not found: {d['path']}")
        elif not is_csv_name(p.name):
            out.add("error", "data.path", "data_not_csv",
                    f"{p.name}: core reads the point table as CSV; convert it, or join a parquet table to a CSV "
                    "through data.join")
        else:
            try:
                cols = dict(table_columns(p, rows))
            except Exception as exc:                 # noqa: BLE001 - reported, not raised
                out.add("error", "data.path", "unreadable", f"cannot read {d['path']}: {getattr(exc, 'message', exc)}")
    for i, j in enumerate(d.get("join") or []):
        if not isinstance(j, dict) or not j.get("path"):
            out.add("error", f"data.join.{i}.path", "required", "each join needs a path")
            continue
        jp = _resolve(pdir, str(j["path"]))
        if not jp.is_file():
            out.add("error", f"data.join.{i}.path", "missing_file", f"join table not found: {j['path']}")
            continue
        try:
            jcols = table_columns(jp, rows)
        except Exception as exc:                     # noqa: BLE001
            out.add("error", f"data.join.{i}.path", "unreadable", f"cannot read {j['path']}: {exc}")
            continue
        key = j.get("key") or j.get("on") or j.get("True") or d.get("id")
        right = j.get("right_key") or j.get("right_on") or ("id" if "id" in jcols else key)
        if cols is not None and key and key not in cols:
            out.add("error", f"data.join.{i}.key", "missing_column", f"join key {key!r} is not a data column")
        if right and right not in jcols:
            out.add("error", f"data.join.{i}.right_key", "missing_column",
                    f"join key {right!r} is not a column of {Path(str(j['path'])).name}")
        if cols is not None:
            for c, isnum in jcols.items():
                if c != right and c not in cols:
                    cols[c] = isnum
    categorical = set((sec("encodings").get("categorical") or []))

    def need_column(path: str, col: Any, *, numeric: bool = True, level: str = "error") -> None:
        if cols is None or col in (None, ""):
            return
        if str(col) not in cols:
            out.add(level, path, "missing_column", f"column {col!r} is not in the data")
        elif numeric and not cols[str(col)] and str(col) not in categorical:
            out.add(level, path, "not_numeric", f"column {col!r} is not numeric")

    need_column("data.target", d.get("target"))
    need_column("data.x", d.get("x"))
    need_column("data.y", d.get("y"))
    need_column("data.id", d.get("id"), numeric=False)
    need_column("data.zone", d.get("zone"), numeric=False, level="warn")
    bg = d.get("background")
    if isinstance(bg, str) and bg != "median":
        need_column("data.background", bg)
    for i, c in enumerate(preds):
        need_column(f"predictors.{i}", c)
    need_column("optimize.equity_column", opt.get("equity_column"))
    for kind in ("categorical", "circular_degrees"):
        for i, c in enumerate(sec("encodings").get(kind) or []):
            if c not in pset:
                out.add("warn", f"encodings.{kind}.{i}", "encoding_not_predictor", f"{c!r} is not a predictor")

    # -- levers ----------------------------------------------------------------------------------
    for var, spec in act.items():
        base = f"actionable.{var}"
        if var not in pset:
            out.add("error", base, "lever_not_predictor", f"lever {var!r} is not a predictor")
        lo, hi = _num(spec.get("min")), _num(spec.get("max"))
        if lo is not None and hi is not None and lo > hi:
            out.add("error", f"{base}.min", "bounds_inverted", f"{var}: min {lo:g} is above max {hi:g}")
        span = (hi - lo) if lo is not None and hi is not None and hi >= lo else None
        doses = spec.get("doses")
        if isinstance(doses, list):
            vals = [_num(x) for x in doses]
            for k, x in enumerate(vals):
                if x is None:
                    continue
                if x < 0:
                    out.add("error", f"{base}.doses.{k}", "dose_out_of_bounds",
                            f"{var}: dose {x:g} is negative (doses are magnitudes; direction sets the sign)")
                elif span is not None and x > span + 1e-12:
                    out.add("error", f"{base}.doses.{k}", "dose_out_of_bounds",
                            f"{var}: dose {x:g} exceeds the lever's range {lo:g}–{hi:g}")
            if vals and not any(x == 0 for x in vals if x is not None):
                fixed = [0] + [x for x in doses]
                out.add("warn", f"{base}.doses", "dose_zero_missing", f"{var}: the dose ladder should include 0",
                        fix={"path": f"{base}.doses", "value": fixed})
        direction = str(spec.get("direction", "increase")).lower()
        if direction not in ("increase", "decrease"):
            out.add("error", f"{base}.direction", "bad_direction", f"{var}: direction must be increase or decrease")
    for i, c in enumerate(eff.get("coupling") or []):
        for k, col in enumerate((c or {}).get("sum") or [] if isinstance(c, dict) else []):
            if col not in pset:
                out.add("error", f"coupling.{i}.sum.{k}", "coupling_not_predictor", f"{col!r} is not a predictor")
    meds = eff.get("mediators") if isinstance(eff.get("mediators"), dict) else {}
    for med, spec in meds.items():
        if med not in pset:
            out.add("error", f"mediators.{med}", "mediator_not_predictor", f"mediator {med!r} is not a predictor")
        spec = spec if isinstance(spec, dict) else {}
        for key in ("parents", "context"):
            for k, c in enumerate(spec.get(key) or []):
                if c not in pset:
                    out.add("error", f"mediators.{med}.{key}.{k}", "mediator_not_predictor",
                            f"mediator {key[:-1] if key == 'parents' else key} {c!r} is not a predictor")

    # -- physics roles and forcing -----------------------------------------------------------------
    roles = phys.get("roles") if isinstance(phys.get("roles"), dict) else {}
    for role, col in roles.items():
        if col in (None, ""):
            continue
        if role not in ROLE_NAMES:
            out.add("warn", f"physics.roles.{role}", "unknown_role",
                    f"{role!r} is not a physics role ({', '.join(ROLE_NAMES)})")
        if col not in pset:
            out.add("error", f"physics.roles.{role}", "role_not_predictor",
                    f"role {role} is mapped to {col!r}, which is not a predictor")
    mapped = {r for r, c in roles.items() if c}
    if preds and not {"canopy", "impervious"} <= mapped:
        miss = sorted({"canopy", "impervious"} - mapped)
        out.add("warn", "physics.roles", "roles_missing",
                f"No {' or '.join(miss)} role: placebo, simcheck, planner and emulator need these")
    if phys.get("enabled", True) and preds and not mapped:
        out.add("info", "physics.roles", "physics_no_roles", "No roles mapped: the physics base model is skipped")
    forcing = phys.get("forcing")
    if forcing:
        fp = _resolve(pdir, str(forcing))
        if not fp.is_file():
            out.add("error", "physics.forcing", "missing_file", f"forcing file not found: {forcing}")
        else:
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
                if not isinstance(blob, dict) or not isinstance(blob.get("physics"), dict):
                    raise ValueError("no 'physics' block")
            except (OSError, ValueError) as exc:
                out.add("error", "physics.forcing", "forcing_invalid", f"forcing file {forcing} is not valid: {exc}")

    # -- scenarios -----------------------------------------------------------------------------------
    for i, s in enumerate(eff.get("scenarios") or []):
        if not isinstance(s, dict):
            continue
        var = s.get("variable")
        if var not in act:
            out.add("error", f"scenarios.{i}.variable", "scenario_not_actionable",
                    f"scenario {s.get('name')!r}: {var!r} is not a lever (add it to actionable)")
            continue
        lo, hi = _num(act[var].get("min")), _num(act[var].get("max"))
        for k, inc in enumerate(s.get("increments") or []):
            x = _num(inc)
            if x is not None and lo is not None and hi is not None and abs(x) > hi - lo:
                out.add("warn", f"scenarios.{i}.increments.{k}", "increment_out_of_bounds",
                        f"scenario {s.get('name')!r}: +{x:g} exceeds the lever's range {lo:g}–{hi:g}")
    for i, j in enumerate(eff.get("joint_scenarios") or []):
        if not isinstance(j, dict):
            continue
        for k, iv in enumerate(j.get("interventions") or []):
            var = (iv or {}).get("variable") if isinstance(iv, dict) else None
            if var not in act:
                out.add("error", f"joint_scenarios.{i}.interventions.{k}.variable", "scenario_not_actionable",
                        f"package {j.get('name')!r}: {var!r} is not a lever")

    # -- causal --------------------------------------------------------------------------------------
    if causal.get("enabled", True):
        for i, t in enumerate(causal.get("treatments") or []):
            if t not in pset:
                out.add("error", f"causal.treatments.{i}", "causal_not_predictor",
                        f"causal treatment {t!r} is not a predictor")
        for key in ("confounders", "exclude_controls"):
            m = causal.get(key) if isinstance(causal.get(key), dict) else {}
            for t, lst in m.items():
                if t not in pset:
                    out.add("error", f"causal.{key}.{t}", "causal_not_predictor", f"{t!r} is not a predictor")
                for k, c in enumerate(lst or [] if isinstance(lst, list) else []):
                    if c not in pset:
                        out.add("error", f"causal.{key}.{t}.{k}", "causal_not_predictor",
                                f"{key[:-1].replace('_', ' ')} {c!r} is not a predictor")
        contrast = causal.get("contrast") if isinstance(causal.get("contrast"), dict) else {}
        for t in contrast:
            if t not in pset:
                out.add("warn", f"causal.contrast.{t}", "causal_not_predictor", f"{t!r} is not a predictor")

    # -- climate ---------------------------------------------------------------------------------------
    if clim.get("enabled"):
        if str(clim.get("source", "table")) == "table":
            if not clim.get("table"):
                out.add("error", "climate.table", "climate_table_missing",
                        "climate.source is 'table' but climate.table is not set (fetch CMIP6 change factors)")
            else:
                tp = _resolve(pdir, str(clim["table"]))
                if not tp.is_file():
                    out.add("error", "climate.table", "missing_file", f"climate table not found: {clim['table']}")
                else:
                    try:
                        have = set(table_columns(tp, 100))
                        need = {"model", "experiment", "period", "delta_K"}
                        if not need <= have:
                            out.add("error", "climate.table", "climate_table_columns",
                                    f"climate table lacks {', '.join(sorted(need - have))}")
                    except Exception as exc:         # noqa: BLE001
                        out.add("error", "climate.table", "unreadable", f"cannot read the climate table: {exc}")
        if not clim.get("site") and not crs:
            out.add("error", "data.crs", "needs_crs",
                    "Climate projections need climate.site or data.crs (the site defaults to the data centroid)")

    # -- optimize and planner ------------------------------------------------------------------------------
    layers = planner.get("layers") if isinstance(planner, dict) else None
    people = str(opt.get("objective", "cooling")) == "people"
    if layers:
        lp = _resolve(pdir, str(layers))
        if not lp.is_file():
            out.add("error" if people else "warn", "planner.layers", "missing_file",
                    f"people/land-cover layers not found: {layers}")
    if opt.get("enabled", True):
        var = opt.get("variable")
        if var and var not in act:
            out.add("error", "optimize.variable", "optimize_not_actionable",
                    f"optimize.variable {var!r} is not a lever")
        if people and not layers:
            out.add("error", "planner.layers", "needs_layers",
                    "objective 'people' weights cooling by residents: set planner.layers (fetch people & land cover)")
        if people and not crs:
            out.add("error", "data.crs", "needs_crs", "objective 'people' needs data.crs to place residents on the grid")
        if opt.get("plantable", True) and var and roles.get("canopy") == var and not layers:
            out.add("info", "optimize.plantable", "plantable_off",
                    "The plantable-space cap needs planner.layers; without it canopy doses are not capped")
        if var and opt.get("budget") in (None, 0, 0.0):
            out.add("info", "optimize.budget", "no_budget", "No budget: the optimisation stage (S7) is skipped")
    if not crs:
        out.add("info", "data.crs", "no_crs", "Without data.crs there are no open-data inputs, no lon/lat on maps "
                "and no GeoTIFF exports")
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out.rows, key=lambda r: order[r["level"]])


# ---------------------------------------------------------------------------
# mode previews (validate endpoint)
# ---------------------------------------------------------------------------

def _leaves(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_leaves(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def fast_overrides(raw: dict, project_dir: str | os.PathLike) -> dict:
    """``{path: {from, to}}`` of what ``--fast`` changes in this config (core ``apply_mode_overrides``)."""
    from sparc.core.pipeline import apply_mode_overrides

    cfg = build_core_config(raw, project_dir, apply_forcing=False)
    before = _leaves(copy.deepcopy(cfg.raw))
    try:
        after = _leaves(apply_mode_overrides(cfg, fast=True).raw)
    except (TypeError, ValueError):
        return {}
    return {k: {"from": before.get(k), "to": v} for k, v in after.items() if before.get(k) != v}


def coarse_preview(raw: dict, project_dir: str | os.PathLike, cell_m: float | None = None) -> dict | None:
    """Cells a coarse run would have: ``{cell_m, n_input, n_cells, fine_cell_m, ok}`` (None without data)."""
    from sparc.core.grid import estimate_lattice_spacing
    from sparc.studio.projects.files import read_table_head

    d = merged_raw(raw).get("data") or {}
    if not d.get("path") or d.get("reproject_to"):
        return None
    p = _resolve(Path(project_dir), str(d["path"]))
    x, y = d.get("x"), d.get("y")
    try:
        if not p.is_file() or not table_columns(p) or x not in table_columns(p) or y not in table_columns(p):
            return None
        df = read_table_head(p, rows=None, columns=[x, y])
        from sparc.core.config import UNIT_TO_METRES

        s = UNIT_TO_METRES.get(str(d.get("coord_unit", "m")).lower(), 1.0)
        xs = pd.to_numeric(df[x], errors="coerce").to_numpy(float) * s
        ys = pd.to_numeric(df[y], errors="coerce").to_numpy(float) * s
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 3:
            return None
        fine = float(d.get("cell_m") or estimate_lattice_spacing(xs, ys))
    except Exception:                                 # noqa: BLE001 - a preview, never an error
        return None
    cm = float(cell_m or d.get("coarse_m") or 60.0)
    ix = np.floor((xs - xs.min() + 0.5 * fine) / cm).astype(np.int64)
    iy = np.floor((ys - ys.min() + 0.5 * fine) / cm).astype(np.int64)
    n_cells = int(np.unique(iy * (int(ix.max()) + 1) + ix).size)
    return {"cell_m": cm, "n_input": int(len(xs)), "n_cells": n_cells, "fine_cell_m": round(fine, 3),
            "ok": bool(cm >= 1.5 * fine)}


def validate_report(raw: dict, project_dir: str | os.PathLike) -> dict:
    """``POST /config/validate``: ``{ok, issues, fast_overrides, coarse_preview}``."""
    issues = validate_deep(raw, project_dir)
    return {"ok": not has_errors(issues), "issues": issues, "fast_overrides": fast_overrides(raw, project_dir),
            "coarse_preview": coarse_preview(raw, project_dir)}
