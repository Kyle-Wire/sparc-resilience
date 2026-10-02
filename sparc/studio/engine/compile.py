"""The scenario compiler (SPEC §7.3): a ``ScenarioDoc`` becomes one core ``Intervention`` per edit.

The same compiler serves the preview, the cost estimate and the exact run, so all three see the same
per-cell changes.  Edits apply in list order, as in core: a mode that depends on the current value
(``floor``, ``ceiling``, ``to_percentile``, ``fill_headroom``) is computed from the value left by the edits
before it.

=================  ===========================================================================================
Mode               Intervention
=================  ===========================================================================================
``add a``          ``Intervention(var, "add", a, where=mask)``
``set v``          ``Intervention(var, "set", v, where=mask)``
``scale f``        ``Intervention(var, "scale", f, where=mask)``
``floor v``        ``per_point = max(v − x, 0)`` on the mask
``ceiling v``      ``per_point = min(v − x, 0)``
``fill_headroom``  ``per_point = f × plantable_headroom(canopy, layers, paved_share)`` (canopy role only)
``to_percentile``  ``max(q_p − x, 0)`` for increase levers, ``min(q_p − x, 0)`` for decrease levers
``per_cell ref``   ``plan:<plid>`` (dose signed by direction), ``blob:<id>`` (Int32 idx + Float32 values),
                   ``csv:<project-relative path>`` (``id,change`` or ``id,lever,change``; joined by id)
=================  ===========================================================================================

**Predicted clamping** is computed by core itself: the interventions go through
``ScenarioEngine.apply`` on a light stand-in that has the run's data and config but no mediators, so the
bounds rule (never push a cell further outside its bounds than its baseline) and the coupling rules are
exactly the engine's.

**Guardrails.**  Blocked without ``options.expert``: an edit of a non-actionable predictor, and an edit of a
mediator whose parents are also edited (core overwrites it, ``mediators.py``).  Always blocked: an unknown
lever, a missing amount, ``fill_headroom`` on a lever that is not the canopy role.  Warned: an amount beyond
1 sd of ``qa.dose_scale`` (likely extrapolation), values beyond the 0.5–99.5th percentile, an empty
selection, ``fill_headroom`` without the planner layers, a direction opposite to the configured one, and
per-cell edits resampled from another run's grid.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError

log = logging.getLogger("sparc.studio.engine")

__all__ = ["Compiled", "CompiledEdit", "compile_scenario", "content_hash", "canonical_doc", "doc_dict",
           "clamp_values", "parse_brush", "lever_cost", "PER_CELL_PREFIXES"]

PER_CELL_PREFIXES = ("plan:", "blob:", "csv:")
_TOL = 1e-9


# ---------------------------------------------------------------------------
# documents and hashes
# ---------------------------------------------------------------------------

def doc_dict(doc: Any) -> dict:
    """A ScenarioDoc (model or dict) as a plain JSON dict with defaults filled in."""
    from sparc.studio.scenarios.schemas import ScenarioDoc

    if hasattr(doc, "model_dump"):
        return doc.model_dump(mode="json")
    return ScenarioDoc.model_validate(doc).model_dump(mode="json")


def _strip_none(obj):
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def canonical_doc(doc: Any) -> dict:
    """The part of a ScenarioDoc its ``content_hash`` covers: ``{edits, regions, costs, options}``.

    Names, tags, notes and edit labels are cosmetic and excluded; a missing ``where`` is ``{kind: all}``.
    """
    d = doc_dict(doc)
    edits = []
    for e in d.get("edits") or []:
        e = {k: v for k, v in e.items() if k != "label"}
        if e.get("where") is None:
            e["where"] = {"kind": "all"}
        edits.append(_strip_none(e))
    opts = {"clip_to_support": True, "mediators": True, "expert": False, **(d.get("options") or {})}
    return {"edits": edits, "regions": _strip_none(d.get("regions") or {}),
            "costs": {k: {"per_unit": float(v["per_unit"])} for k, v in (d.get("costs") or {}).items()},
            "options": {k: bool(opts[k]) for k in ("clip_to_support", "mediators", "expert")}}


def content_hash(doc: Any) -> str:
    """sha256 of the canonical JSON of ``{edits, regions, costs, options}`` (SPEC §7.2)."""
    text = json.dumps(canonical_doc(doc), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class CompiledEdit:
    index: int
    lever: str
    mode: str                         # the doc's mode
    core_mode: str                    # add | set | scale
    amount: float
    where: np.ndarray | None          # bool[n]; None = every cell
    per_point: np.ndarray | None      # float64[n] for the per-point modes
    requested: np.ndarray             # float64[n]: the change this edit asks for (before clamping)
    label: str | None = None
    ref: str | None = None

    def intervention(self) -> dict:
        """The host payload of this edit (primitive values and numpy arrays only, api.md §13)."""
        return {"variable": self.lever, "mode": self.core_mode, "amount": float(self.amount),
                "where": None if self.where is None else np.asarray(self.where, dtype=bool),
                "per_point": None if self.per_point is None else np.asarray(self.per_point, dtype=np.float64)}


@dataclass
class Compiled:
    run_id: str
    n: int
    doc: dict
    content_hash: str
    options: dict
    edits: list[CompiledEdit] = field(default_factory=list)
    requested: dict[str, np.ndarray] = field(default_factory=dict)
    realized: dict[str, np.ndarray] = field(default_factory=dict)
    lever_cells: dict[str, np.ndarray] = field(default_factory=dict)
    edited: np.ndarray | None = None
    levers: dict[str, dict] = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    people: float | None = None
    portable: bool = True
    est_exact_s: float = 0.0
    costs: dict[str, float] = field(default_factory=dict)

    @property
    def blocking(self) -> list[dict]:
        return [w for w in self.warnings if w.get("blocking")]

    @property
    def union_cells(self) -> int:
        return int(self.edited.sum()) if self.edited is not None else 0

    def interventions(self) -> list[dict]:
        return [e.intervention() for e in self.edits]

    def specs(self, name: str = "scenario"):
        """The core ``ScenarioSpec`` of this compilation."""
        from sparc.core.scenarios import Intervention, ScenarioSpec

        return ScenarioSpec(name=name, interventions=[
            Intervention(e.lever, e.core_mode, float(e.amount), where=e.where, per_point=e.per_point)
            for e in self.edits])

    def dx(self) -> dict[str, np.ndarray]:
        """Per-lever predicted realised change (what the emulator previews)."""
        return {k: v for k, v in self.realized.items()}

    def response(self, emulator: dict | None = None) -> dict:
        """``POST /compile`` (api.md §7.3)."""
        return {"content_hash": self.content_hash, "portable": bool(self.portable), "levers": self.levers,
                "union_cells": self.union_cells, "people": self.people,
                "warnings": [{"code": w["code"], "message": w["message"], "edit_index": w.get("edit_index"),
                              "blocking": bool(w.get("blocking"))} for w in self.warnings],
                "est_exact_s": round(float(self.est_exact_s), 2),
                "emulator": emulator or {"usable": False, "hatched": False, "reasons": []}}

    def lever_summary(self) -> dict:
        """The ``compiled`` block of ``spec.json``."""
        return {"levers": self.levers, "edits": [
            {"index": e.index, "lever": e.lever, "mode": e.mode, "core_mode": e.core_mode, "amount": e.amount,
             "label": e.label, "ref": e.ref, "n_cells": int(np.count_nonzero(e.requested))} for e in self.edits],
            "union_cells": self.union_cells, "requested_sha256": self.requested_sha256()}

    def requested_sha256(self) -> str:
        h = hashlib.sha256()
        for e in self.edits:
            h.update(e.lever.encode())
            h.update(np.ascontiguousarray(e.requested, dtype="<f8").tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def clamp_values(x: np.ndarray, base: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``ScenarioEngine.apply``'s bounds rule: never push a value further outside ``[lo, hi]`` than its
    baseline, and never "correct" a baseline that already sits outside."""
    x = np.where(x > hi, np.maximum(hi, np.minimum(x, base)), x)
    x = np.where(x < lo, np.minimum(lo, np.maximum(x, base)), x)
    return x


class _ApplyShim:
    """What ``ScenarioEngine.apply`` reads from ``self``: the data, the config and the bounds rule."""

    def __init__(self, data, cfg):
        from sparc.core.scenarios import ScenarioEngine

        self.data = data
        self.cfg = cfg
        self.mediators = None
        self._apply = ScenarioEngine.apply
        self._bounds_fn = ScenarioEngine._bounds

    def _bounds(self, var: str):
        return self._bounds_fn(self, var)

    def apply(self, spec):
        return self._apply(self, spec)


def lever_bounds(cfg, var: str) -> tuple[float, float]:
    """``ScenarioEngine._bounds``: the lever's ``min``/``max``, else ``qa.clip``, else unbounded."""
    from sparc.core.scenarios import ScenarioEngine

    holder = type("CfgHolder", (), {})()
    holder.cfg = cfg
    return ScenarioEngine._bounds(holder, var)


def lever_cost(cfg_raw: dict, doc: dict, var: str) -> float:
    """Cost per unit of ``var``: the scenario's cost model, else the lever's ``cost_per_unit``, else 1."""
    c = ((doc.get("costs") or {}).get(var) or {}).get("per_unit")
    if c is not None:
        return float(c)
    spec = (cfg_raw.get("actionable") or {}).get(var) or {}
    if spec.get("cost_per_unit") is not None:
        return float(spec["cost_per_unit"])
    return float(((cfg_raw.get("optimize") or {}).get("cost_per_unit")) or 1.0)


def parse_brush(brush: dict | None, n: int) -> dict[str, np.ndarray]:
    """``{lever: SparseEdit}`` → ``{lever: float64[n]}`` per-cell increments (api.md §0.4)."""
    out: dict[str, np.ndarray] = {}
    for lever, se in (brush or {}).items():
        d = se.model_dump() if hasattr(se, "model_dump") else dict(se)
        try:
            idx = np.frombuffer(base64.b64decode(d["idx"], validate=True), dtype="<i4")
            val = np.frombuffer(base64.b64decode(d["val"], validate=True), dtype="<f4")
        except (KeyError, ValueError, TypeError):
            raise _bad(f"brush.{lever}", "a brush edit needs base64 Int32 idx and Float32 val", "brush")
        if idx.size != val.size:
            raise _bad(f"brush.{lever}", f"brush idx has {idx.size} entries and val {val.size}", "brush")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise _bad(f"brush.{lever}", f"brush indices must lie in [0, {n})", "brush")
        arr = out.setdefault(lever, np.zeros(n))
        np.add.at(arr, idx.astype(np.int64), val.astype(np.float64))
    return out


def _bad(path: str, msg: str, code: str = "compile") -> ApiError:
    return ApiError("validation", msg, detail={"errors": [{"path": path, "message": msg, "code": code}]})


def _warn(out: list, code: str, message: str, index: int | None, blocking: bool = False) -> None:
    out.append({"code": code, "message": message, "edit_index": index, "blocking": bool(blocking)})


def _fmt(v: float) -> str:
    return f"{v:.3g}".replace("-", "−")


# ---------------------------------------------------------------------------
# per-cell references
# ---------------------------------------------------------------------------

def _rows_by_id(ids_this, ids_other) -> tuple[np.ndarray, np.ndarray]:
    """``(rows_this, rows_other)`` of the ids both runs share."""
    a = np.asarray(ids_this).astype(str)
    b = np.asarray(ids_other).astype(str)
    pos = {v: i for i, v in enumerate(a)}
    rows_o = np.array([j for j, v in enumerate(b) if v in pos], dtype=np.int64)
    rows_t = np.array([pos[b[j]] for j in rows_o], dtype=np.int64)
    return rows_t, rows_o


def _resample(values_other: np.ndarray, grid_other, grid_this) -> np.ndarray:
    """Per-cell values of another run's grid, moved to this run's grid by nearest cell centre (run frame)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([grid_other.x, grid_other.y]))
    d, j = tree.query(np.column_stack([grid_this.x, grid_this.y]))
    out = np.asarray(values_other, dtype=np.float64)[j]
    out[d > 1.5 * max(float(grid_other.dx), float(grid_this.dx))] = 0.0      # outside the other run's extent
    return out


def _other_values(ctx, other_run: str | None, values: np.ndarray, reader, warnings: list, index: int,
                  what: str) -> np.ndarray:
    """Values that belong to another run, on this run's rows: by id when the ids align, else resampled."""
    if not other_run or other_run == ctx.run_id:
        return values
    if reader is None:
        raise _bad(f"edits.{index}.per_cell_ref", f"{what} belongs to run {other_run}; it cannot be read here")
    octx = reader.get(other_run)
    og = octx.grid
    if og is None:
        raise _bad(f"edits.{index}.per_cell_ref", f"the grid of run {other_run} is not known")
    if og.n == ctx.grid.n and np.array_equal(np.asarray(og.ids).astype(str), np.asarray(ctx.grid.ids).astype(str)):
        return values
    _warn(warnings, "resampled", f"{what} comes from run {other_run}, whose grid differs: its values were "
          "resampled to the nearest cell centre", index)
    return _resample(values, og, ctx.grid)


def _per_cell(ctx, edit: dict, index: int, lever: str, direction: str, *, db, reader, project_dir,
              warnings: list) -> tuple[np.ndarray, bool]:
    """``(per_point float64[n], portable)`` of a ``per_cell`` edit."""
    n = ctx.grid.n
    ref = str(edit.get("per_cell_ref") or "")
    if not ref.startswith(PER_CELL_PREFIXES):
        raise _bad(f"edits.{index}.per_cell_ref", "per_cell_ref must be plan:<id>, blob:<id> or csv:<path>")
    kind, _, key = ref.partition(":")
    if kind == "plan":
        row = db.fetchone("SELECT run_id, dir FROM plans WHERE id = ?", (key,)) if db is not None else None
        pdir = Path(row["dir"]) if row and row.get("dir") else ctx.studio_dir / "plans" / key
        if not (pdir / "dose.npy").exists():
            raise _bad(f"edits.{index}.per_cell_ref", f"no plan {key!r}", "unknown_plan")
        params = _read_json(pdir / "params.json") or {}
        pl = ((params.get("params") or {}).get("lever")) or lever
        if pl != lever:
            _warn(warnings, "plan_lever", f"plan {key} allocates {pl}, not {lever}", index)
        dose = np.asarray(np.load(pdir / "dose.npy", allow_pickle=False), dtype=np.float64)
        dose = _other_values(ctx, row.get("run_id") if row else None, dose, reader, warnings, index, f"plan {key}")
        if dose.size != n:
            raise _bad(f"edits.{index}.per_cell_ref", f"plan {key} has {dose.size} cells; this run has {n}")
        sign = -1.0 if direction == "decrease" else 1.0
        return sign * dose, False
    if kind == "blob":
        row = db.fetchone("SELECT run_id, kind, path FROM blobs WHERE id = ?", (key,)) if db is not None else None
        if row is None or row.get("kind") != "edit":
            raise _bad(f"edits.{index}.per_cell_ref", f"no edit blob {key!r}", "unknown_blob")
        try:
            raw = Path(row["path"]).read_bytes()
        except OSError:
            raise _bad(f"edits.{index}.per_cell_ref", f"blob {key!r} is gone", "unknown_blob")
        m = len(raw) // 8
        idx = np.frombuffer(raw[:4 * m], dtype="<i4").astype(np.int64)
        val = np.frombuffer(raw[4 * m:8 * m], dtype="<f4").astype(np.float64)
        other = row.get("run_id")
        if other and other != ctx.run_id:
            if reader is None:
                raise _bad(f"edits.{index}.per_cell_ref", f"blob {key} belongs to run {other}")
            og = reader.get(other).grid
            full = np.zeros(og.n)
            ok = (idx >= 0) & (idx < og.n)
            np.add.at(full, idx[ok], val[ok])
            return _other_values(ctx, other, full, reader, warnings, index, f"blob {key}"), False
        out = np.zeros(n)
        ok = (idx >= 0) & (idx < n)
        if not ok.all():
            _warn(warnings, "blob_range", f"blob {key}: {int((~ok).sum())} indices outside the grid were ignored",
                  index)
        np.add.at(out, idx[ok], val[ok])
        return out, False
    # csv:<project-relative path>
    if project_dir is None:
        raise _bad(f"edits.{index}.per_cell_ref", "csv references need the run's project")
    from sparc.studio.security import safe_path

    try:
        path = safe_path(key, project_dir)
    except ApiError:
        raise _bad(f"edits.{index}.per_cell_ref", f"{key!r} is not a file of the project")
    if not path.is_file():
        raise _bad(f"edits.{index}.per_cell_ref", f"no file {key!r} in the project", "missing_file")
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols or not ({"change", "value"} & set(cols)):
        raise _bad(f"edits.{index}.per_cell_ref", "the CSV needs columns id and change (or value)")
    if "lever" in cols:
        df = df[df[cols["lever"]].astype(str) == lever]
    ids = np.asarray(ctx.grid.ids).astype(str)
    pos = {v: i for i, v in enumerate(ids)}
    out = np.zeros(n)
    unknown = 0
    is_value = "change" not in cols
    x = ctx.data.frame[lever].to_numpy(float) if is_value else None
    for rid, v in zip(df[cols["id"]].astype(str), df[cols["change" if not is_value else "value"]].astype(float)):
        r = pos.get(rid)
        if r is None:
            r = pos.get(str(int(float(rid)))) if _intlike(rid) else None
        if r is None:
            unknown += 1
            continue
        out[r] = (v - x[r]) if is_value else v
    if unknown:
        _warn(warnings, "unknown_ids", f"{key}: {unknown} ids are not cells of this run and were ignored", index)
    return out, True


def _intlike(s: str) -> bool:
    try:
        return float(s).is_integer()
    except ValueError:
        return False


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------

def compile_scenario(ctx, doc: Any, *, db=None, reader=None, brush: dict | None = None,
                     project_dir: str | Path | None = None, cost_model=None, threads: int = 2,
                     engine_loaded: bool | None = None) -> Compiled:
    """Compile ``doc`` on the run of ``ctx`` (a :class:`~sparc.studio.runs.reader.RunContext`).

    ``db`` reads saved regions, blobs and plans; ``reader`` (a ``RunReader``) resolves references to other
    runs; ``brush`` adds per-cell increments per lever (``POST /preview``); ``cost_model`` estimates the exact
    run (``engine_loaded=False`` adds the engine load).  Raises ``422 validation`` for a malformed edit or
    selection and ``404 output_missing`` when the run's data cannot be rebuilt.
    """
    from sparc.core.scenarios import Intervention, ScenarioSpec
    from sparc.studio.runs import layers as L
    from sparc.studio.runs.selection import RunSource, is_portable, resolve

    d = doc_dict(doc)
    data = ctx.data
    if data is None:
        raise ApiError("output_missing", f"the run's data cannot be rebuilt: {ctx.data_error or 'unknown'}",
                       detail={"output": "data", "produced_by": "stage:S0", "expected_path": None})
    cfg = ctx.cfg
    raw = cfg.raw
    src = RunSource(ctx, db)
    n = src.n
    if data.n != n:
        raise ApiError("output_missing", "the run's data and grid disagree; re-index the run",
                       detail={"output": "grid", "produced_by": "stage:S0", "expected_path": None})
    opts = {"clip_to_support": True, "mediators": True, "expert": False, **(d.get("options") or {})}
    expert = bool(opts.get("expert"))
    out = Compiled(run_id=ctx.run_id, n=n, doc=d, content_hash=content_hash(d), options=opts)
    warnings = out.warnings
    actionable = raw.get("actionable") or {}
    predictors = list(cfg.predictors)
    mediators = raw.get("mediators") or {}
    canopy_role = ((raw.get("physics") or {}).get("roles") or {}).get("canopy")
    dose_scale = (((ctx.manifest or {}).get("qa") or {}).get("dose_scale")) or {}
    frame = data.frame
    base = {v: frame[v].to_numpy(dtype=np.float64) for v in predictors}
    cur = {v: base[v].copy() for v in predictors}
    edits = list(d.get("edits") or [])
    brush_arr = parse_brush(brush, n)
    for lever, arr in brush_arr.items():
        edits.append({"lever": lever, "mode": "per_cell", "amount": 1.0, "per_cell_ref": "__brush__",
                      "where": None, "label": "brush"})
    edited_levers = {e.get("lever") for e in edits}

    for i, e in enumerate(edits):
        lever = str(e.get("lever"))
        mode = str(e.get("mode"))
        if lever not in predictors:
            raise _bad(f"edits.{i}.lever", f"{lever!r} is not a predictor of this run", "unknown_lever")
        if lever not in actionable:
            _warn(warnings, "not_actionable", f"{lever} is not an actionable lever of this run"
                  + ("" if expert else ": expert mode is needed to edit it"), i, blocking=not expert)
        if lever in mediators:
            parents = [p for p in (mediators[lever] or {}).get("parents") or [] if p in edited_levers]
            if parents:
                _warn(warnings, "mediator_with_parents",
                      f"{lever} follows {', '.join(parents)}, which this scenario also edits: core recomputes "
                      f"{lever} from its parents and overwrites this edit" + ("" if expert else
                                                                             " (expert mode to keep it)"),
                      i, blocking=not expert)
        direction = str((actionable.get(lever) or {}).get("direction", "increase")).lower()
        where_spec = e.get("where") or {"kind": "all"}
        if e.get("per_cell_ref") == "__brush__":
            mask = None
        else:
            mask = resolve(src, where_spec)
            if not is_portable(where_spec, db):
                out.portable = False
            if mask is not None and not mask.any():
                _warn(warnings, "empty_selection", f"edit {i + 1} ({lever}) selects no cells", i)
        sel = np.ones(n, dtype=bool) if mask is None else mask
        x = cur[lever]
        lo, hi = lever_bounds(cfg, lever)
        amount = e.get("amount")
        core_mode, core_amount, per_point = "add", 0.0, None

        def need_amount():
            if amount is None or not math.isfinite(float(amount)):
                raise _bad(f"edits.{i}.amount", f"mode {mode} needs an amount", "missing")
            return float(amount)

        if mode == "add":
            core_mode, core_amount = "add", need_amount()
            target = np.where(sel, x + core_amount, x)
        elif mode == "set":
            core_mode, core_amount = "set", need_amount()
            target = np.where(sel, core_amount, x)
        elif mode == "scale":
            core_mode, core_amount = "scale", need_amount()
            target = np.where(sel, x * core_amount, x)
        elif mode in ("floor", "ceiling"):
            v = need_amount()
            pp = np.maximum(v - x, 0.0) if mode == "floor" else np.minimum(v - x, 0.0)
            per_point = np.where(sel, pp, 0.0)
        elif mode == "to_percentile":
            p = e.get("percentile") if e.get("percentile") is not None else amount
            if p is None or not 0 <= float(p) <= 100:
                raise _bad(f"edits.{i}.percentile", "to_percentile needs a percentile in [0, 100]")
            q = float(np.nanpercentile(base[lever], float(p)))
            pp = np.minimum(q - x, 0.0) if direction == "decrease" else np.maximum(q - x, 0.0)
            per_point = np.where(sel, pp, 0.0)
        elif mode == "fill_headroom":
            f = need_amount()
            if lever != canopy_role:
                raise _bad(f"edits.{i}.mode", f"fill_headroom applies to the canopy lever ({canopy_role or 'no canopy role'})"
                           f", not {lever}", "canopy_only")
            share = e.get("paved_share")
            if share is None:
                share = float((raw.get("planner") or {}).get("paved_plantable_share", 0.2))
            lay = L.people_layers(ctx)
            if lay is None:
                _warn(warnings, "no_layers", "fill_headroom without the planner layers: the headroom is the room "
                      f"left below the lever's maximum ({_fmt(hi)}) instead of plantable space", i)
                head = np.clip(hi - x, 0.0, None)
            else:
                from sparc.core.planner import plantable_headroom

                head = plantable_headroom(x, lay, float(share))
            per_point = np.where(sel, f * np.nan_to_num(head), 0.0)
        elif mode == "per_cell":
            if e.get("per_cell_ref") == "__brush__":
                pp, portable = brush_arr[lever], False
            else:
                pp, portable = _per_cell(ctx, e, i, lever, direction, db=db, reader=reader, project_dir=project_dir,
                                         warnings=warnings)
            out.portable = out.portable and portable
            scale = float(amount) if amount is not None else 1.0
            per_point = np.where(sel, np.nan_to_num(pp) * scale, 0.0)
        else:
            raise _bad(f"edits.{i}.mode", f"unknown edit mode {mode!r}")
        if per_point is not None:
            target = x + per_point
        requested = np.where(sel, target - x, 0.0)
        new = clamp_values(target, base[lever], lo, hi)
        cur[lever] = new
        out.edits.append(CompiledEdit(index=i, lever=lever, mode=mode, core_mode=core_mode, amount=core_amount,
                                      where=None if mask is None or mask.all() else mask,
                                      per_point=per_point, requested=requested, label=e.get("label"),
                                      ref=e.get("per_cell_ref") if mode == "per_cell" else None))
        # warnings on the request itself
        moved = requested != 0
        if moved.any():
            mean_req = float(np.mean(requested[moved]))
            sd = (dose_scale.get(lever) or {}).get("sd")
            sd = float(sd) if sd else float(np.nanstd(base[lever]))
            if sd > 0 and abs(mean_req) > sd:
                _warn(warnings, "beyond_sd", f"edit {i + 1} changes {lever} by {_fmt(mean_req)} on average, more "
                      f"than 1 sd of the layer ({_fmt(sd)}): likely extrapolation", i)
            p_lo, p_hi = np.nanpercentile(base[lever], [0.5, 99.5])
            beyond = (target[moved] > p_hi + _TOL) | (target[moved] < p_lo - _TOL)
            if beyond.any():
                _warn(warnings, "beyond_p995", f"edit {i + 1} takes {beyond.mean():.0%} of its cells beyond the "
                      f"observed range of {lever} (0.5–99.5th percentile {_fmt(p_lo)}–{_fmt(p_hi)})", i)
            if lever in actionable and ((direction == "increase" and mean_req < 0) or
                                        (direction == "decrease" and mean_req > 0)):
                _warn(warnings, "opposite_direction", f"edit {i + 1} {'lowers' if mean_req < 0 else 'raises'} "
                      f"{lever}, against its configured direction ({direction})", i)

    # requested per lever, predicted realised change through core's own apply()
    for ce in out.edits:
        out.requested[ce.lever] = out.requested.get(ce.lever, np.zeros(n)) + ce.requested
        cells = out.lever_cells.get(ce.lever, np.zeros(n, dtype=bool))
        out.lever_cells[ce.lever] = cells | (ce.requested != 0)
    if out.edits:
        spec = ScenarioSpec(name="compile", interventions=[
            Intervention(ce.lever, ce.core_mode, float(ce.amount), where=ce.where, per_point=ce.per_point)
            for ce in out.edits])
        _new, realized = _ApplyShim(data, cfg).apply(spec)
        out.realized = {k: np.asarray(v, dtype=np.float64) for k, v in realized.items()}
    edited = np.zeros(n, dtype=bool)
    for v in out.realized.values():
        edited |= np.abs(v) > 0
    out.edited = edited

    people = None
    try:
        if "people" in L.layer_defs(ctx):
            people = np.asarray(L.layer_array(ctx, "people"), dtype=np.float64)
    except ApiError:
        people = None
    out.people = float(np.nansum(people[edited])) if people is not None else None
    for lever, req in out.requested.items():
        cells = out.lever_cells[lever]
        real = out.realized.get(lever, np.zeros(n))
        k = int(cells.sum())
        per_unit = lever_cost(raw, d, lever)
        out.costs[lever] = per_unit
        clipped = np.abs(real - req) > _TOL * np.maximum(1.0, np.abs(req))
        out.levers[lever] = {
            "n_cells": k,
            "mean_requested": float(req[cells].mean()) if k else 0.0,
            "total_requested": float(req[cells].sum()) if k else 0.0,
            "predicted_mean_realised": float(real[cells].mean()) if k else 0.0,
            "clipped_share": float(clipped[cells].mean()) if k else 0.0,
            "est_cost": float(np.abs(real).sum() * per_unit),
        }
    for name, spec in (d.get("regions") or {}).items():
        out.regions[str(name)] = resolve(src, spec)
        if not is_portable(spec, db):
            out.portable = False
    out.est_exact_s = estimate_exact_s(n, cost_model=cost_model, threads=threads, engine_loaded=engine_loaded)
    return out


def estimate_exact_s(n: int, *, cost_model=None, threads: int = 2, engine_loaded: bool | None = None,
                     n_passes: int = 1, checkpoint_bytes: int | None = None) -> float:
    """Seconds of an exact run of ``n_passes`` engine passes (plus the engine load when it is cold)."""
    from sparc.studio.jobs.eta import CostModel

    cm = cost_model or CostModel()
    s = n_passes * cm.seconds("engine_pass", n, threads)
    if engine_loaded is False:
        s += load_seconds(n, cost_model=cm, threads=threads, checkpoint_bytes=checkpoint_bytes)
    return float(s)


def load_seconds(n: int, *, cost_model=None, threads: int = 2, checkpoint_bytes: int | None = None) -> float:
    """Seconds to open the engine: unpickle (1 s per 20 MB) + mediator fit + the baseline pass."""
    from sparc.studio.jobs.eta import CostModel

    cm = cost_model or CostModel()
    unpickle = cm.seconds("unpickle", n, threads)
    if checkpoint_bytes:
        unpickle = float(checkpoint_bytes) / 20e6
    return float(unpickle + cm.seconds("mediator_fit", n, threads) + cm.seconds("engine_pass", n, threads))
