"""The emulator preview (SPEC §7.5, api.md §7.3): ``ΔT = Σ_levers sparc.core.emulator.emulate(em_v, grid, dx_v)``.

``dx_v`` is the compiler's predicted realised change of lever ``v`` (bounds and coupling applied, as the
emulator's own validation edits are), so the preview previews exactly what the exact run will be asked to
do.  Everything runs in the API process under ``threadpool_limits(1)``:

* ``emulator.npz`` / ``emulator.json`` are loaded once per run (cached by file stat) into the arrays
  ``emulate`` takes (``own``, per channel ``coef`` / ``weight`` / ``sigma_cells``, the physics ``dq`` and
  kernel), on the run's core ``Grid`` (``RunContext.data.grid``, the grid the emulator was built on);
* requests are **single-flight, latest-wins per run**: a request whose ``request_seq`` is older than one
  already in flight or waiting, or that waited behind a computation while a newer one arrived, returns
  ``409 superseded``.  Once a run's previews are idle any ``request_seq`` is accepted again, so a reloaded
  page (whose sequence restarts at 1) is not locked out by the numbers of an earlier one;
* the response is a packed body (``delta`` float32[n], ``edited`` bitset bytes) with ``X-SPARC-Offsets``
  and ``X-SPARC-Summary``.

**Trust per lever** from the emulator's validation: ``good`` when ``patch_pass_rate ≥ 0.9`` and the uniform
relative error ≤ 0.35, else ``rough``; ``none`` without an emulator.  **Hatching**: more than 3,000 edited
cells, a ``rough`` lever editing more than 1,000 cells, or a city-wide edit of a lever whose uniform
relative error exceeds 1.0.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError

log = logging.getLogger("sparc.studio.engine")

__all__ = ["RunEmulator", "load_emulator", "lever_trust", "emulator_status", "emulator_info", "levers",
           "preview", "PreviewFlights", "FLIGHTS", "HATCH_CELLS", "ROUGH_CELLS", "build_emulator_action",
           "compute_delta"]

HATCH_CELLS = 3000
ROUGH_CELLS = 1000
CITY_WIDE_SHARE = 0.5
GOOD_PASS_RATE = 0.9
GOOD_REL_ERR = 0.35
UNIFORM_UNRELIABLE = 1.0


def build_emulator_action(run_id: str) -> dict:
    return {"kind": "build_emulator", "label": "Build emulator (≈1 min fast, ≈4–5 min per lever full)",
            "method": "POST", "path": f"/api/runs/{run_id}/actions/emulator", "body": {}}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

@dataclass
class RunEmulator:
    meta: dict
    levers: dict[str, dict]                    # var → the dict ``emulate`` takes
    grid: Any                                  # sparc.core.grid.Grid
    ids: np.ndarray | None = None
    trust: dict[str, str] = field(default_factory=dict)


def lever_trust(validation: dict | None) -> str:
    v = validation or {}
    pr = v.get("patch_pass_rate")
    rel = (v.get("uniform") or {}).get("rel_err")
    if pr is None:
        return "rough"
    return "good" if float(pr) >= GOOD_PASS_RATE and rel is not None and float(rel) <= GOOD_REL_ERR else "rough"


_CACHE: dict[str, tuple[tuple, RunEmulator]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 8


def _stat(p: Path):
    try:
        st = p.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def load_emulator(ctx) -> RunEmulator | None:
    """The run's emulator on its grid, or None when ``emulator.npz``/``.json`` are missing."""
    npz, js = ctx.run_dir / "emulator.npz", ctx.run_dir / "emulator.json"
    key = (_stat(npz), _stat(js), ctx.key)
    if key[0] is None or key[1] is None:
        return None
    with _CACHE_LOCK:
        hit = _CACHE.get(ctx.run_id)
        if hit is not None and hit[0] == key:
            return hit[1]
    meta = json.loads(js.read_text("utf-8"))
    data = ctx.data
    grid = data.grid if data is not None else (ctx.grid.core_grid() if ctx.grid is not None else None)
    if grid is None:
        return None
    levers: dict[str, dict] = {}
    with np.load(npz, allow_pickle=False) as z:
        ids = z["ids"] if "ids" in z.files else None
        kernel = z["physics_kernel"] if "physics_kernel" in z.files else None
        for var, lv in (meta.get("levers") or {}).items():
            if f"{var}__own" not in z.files:
                continue
            em = {"own": np.asarray(z[f"{var}__own"], dtype=np.float64),
                  "channels": [{"sigma_cells": float(ch["sigma_cells"]),
                                "coef": np.asarray(z[ch["coef"]], dtype=np.float64),
                                "weight": np.asarray(z[ch["weight"]], dtype=np.float64)}
                               for ch in lv.get("channels") or []],
                  "physics": ({"dq": np.asarray(z[f"{var}__dq"], dtype=np.float64), "kernel": kernel}
                              if lv.get("physics") and f"{var}__dq" in z.files and kernel is not None else None)}
            levers[var] = em
    em = RunEmulator(meta=meta, levers=levers, grid=grid, ids=ids,
                     trust={v: lever_trust((meta["levers"][v] or {}).get("validation")) for v in levers})
    with _CACHE_LOCK:
        _CACHE[ctx.run_id] = (key, em)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
    return em


def _aligned(ctx, em: RunEmulator) -> bool:
    if em.ids is None or ctx.grid is None:
        return True
    return em.ids.size == ctx.grid.n and np.array_equal(np.asarray(em.ids).astype(str),
                                                        np.asarray(ctx.grid.ids).astype(str))


# ---------------------------------------------------------------------------
# status, levers
# ---------------------------------------------------------------------------

def _uniform_rel(meta: dict, var: str) -> float | None:
    v = ((meta.get("levers") or {}).get(var) or {}).get("validation") or {}
    r = (v.get("uniform") or {}).get("rel_err")
    return float(r) if r is not None else None


def emulator_status(ctx, compiled, em: RunEmulator | None = None) -> dict:
    """``{usable, hatched, reasons}`` of previewing ``compiled`` (the compile response's ``emulator``)."""
    if em is None:
        try:
            em = load_emulator(ctx)
        except Exception as exc:              # a broken emulator file: no preview, but compile still answers
            log.warning("%s: emulator unreadable: %s", ctx.run_id, exc)
            em = None
    levers = [v for v, cells in compiled.lever_cells.items() if cells.any()]
    if em is None:
        return {"usable": False, "hatched": False, "reasons": ["no emulator for this run (build it, or run exact)"]}
    missing = [v for v in levers if v not in em.levers]
    if missing:
        return {"usable": False, "hatched": False,
                "reasons": [f"the emulator has no {', '.join(missing)} lever"]}
    if not _aligned(ctx, em):
        return {"usable": False, "hatched": False, "reasons": ["the emulator was built on other cells; rebuild it"]}
    hatched, reasons = hatch(em, compiled)
    return {"usable": True, "hatched": hatched, "reasons": reasons}


def hatch(em: RunEmulator, compiled) -> tuple[bool, list[str]]:
    """The hatching rules of SPEC §7.5."""
    reasons: list[str] = []
    n_edit = compiled.union_cells
    if n_edit > HATCH_CELLS:
        reasons.append(f"Preview unreliable at this scale ({n_edit:,} edited cells > {HATCH_CELLS:,}) — run exact")
    for var, cells in compiled.lever_cells.items():
        k = int(cells.sum())
        if not k:
            continue
        if em.trust.get(var) == "rough" and k > ROUGH_CELLS:
            reasons.append(f"Preview unreliable: {var} has a rough emulator and edits {k:,} cells "
                           f"(> {ROUGH_CELLS:,}) — run exact")
        rel = _uniform_rel(em.meta, var)
        if k >= CITY_WIDE_SHARE * compiled.n and rel is not None and rel > UNIFORM_UNRELIABLE:
            reasons.append(f"Preview unreliable at this scale (uniform {var} rel. error {rel:.0%}) — run exact")
    return bool(reasons), reasons


def emulator_info(ctx) -> dict:
    """``GET /api/runs/{rid}/emulator``."""
    em = load_emulator(ctx)
    if em is None:
        return {"present": False, "kernel_cells": None, "levers": {}, "action": build_emulator_action(ctx.run_id)}
    levers = {}
    for var, lv in (em.meta.get("levers") or {}).items():
        levers[var] = {"design_dose": lv.get("design_dose"), "bounds": [float(b) for b in lv.get("bounds") or []],
                       "direction": str(lv.get("direction") or "increase"),
                       "trust": em.trust.get(var, "none"), "validation": lv.get("validation") or {}}
    return {"present": True, "kernel_cells": em.meta.get("kernel_cells"), "levers": levers,
            "action": None if _aligned(ctx, em) else build_emulator_action(ctx.run_id)}


def levers(ctx) -> list[dict]:
    """``GET /api/runs/{rid}/levers`` (api.md §7.3)."""
    from sparc.core.catalog import unit_label
    from sparc.studio.runs import layers as L

    raw = ctx.cfg_raw
    act = raw.get("actionable") or {}
    roles = (raw.get("physics") or {}).get("roles") or {}
    role_of = {v: k for k, v in roles.items()}
    meds = raw.get("mediators") or {}
    dose_scale = (((ctx.manifest or {}).get("qa") or {}).get("dose_scale")) or {}
    try:
        em = load_emulator(ctx)
    except Exception:
        em = None
    has_layers = L.people_layers(ctx) is not None if ctx.data is not None else False
    info = L.lever_info(ctx)
    out = []
    for var, spec in act.items():
        spec = spec or {}
        lv = (em.meta.get("levers") or {}).get(var) if em is not None else None
        val = (lv or {}).get("validation") or {}
        design = (lv or {}).get("design_dose")
        if design is None and ctx.data is not None and var in ctx.data.frame:
            from sparc.core.emulator import design_dose

            design = design_dose(ctx.cfg, var, ctx.data.frame[var].to_numpy(float))
        sd = (dose_scale.get(var) or {}).get("sd")
        out.append({
            "var": var, "label": (info.get(var) or {}).get("label") or var.replace("_", " "),
            "unit": unit_label(spec.get("unit")) or "units",
            "min": float(spec["min"]) if spec.get("min") is not None else None,
            "max": float(spec["max"]) if spec.get("max") is not None else None,
            "direction": "decrease" if str(spec.get("direction", "increase")).lower() == "decrease" else "increase",
            "doses": sorted({float(d) for d in spec.get("doses") or [] if float(d) != 0}),
            "cost_per_unit": float(spec.get("cost_per_unit") if spec.get("cost_per_unit") is not None else
                                   (raw.get("optimize") or {}).get("cost_per_unit") or 1.0),
            "design_dose": float(design) if design is not None else None,
            "sd": float(sd) if sd is not None else None,
            "headroom_available": bool(role_of.get(var) == "canopy" and has_layers),
            "role": role_of.get(var),
            "mediator_children": [m for m, s in meds.items() if var in ((s or {}).get("parents") or [])],
            "emulator": {"available": lv is not None, "trust": em.trust.get(var, "none") if lv is not None else "none",
                         "patch_pass_rate": val.get("patch_pass_rate"),
                         "uniform_rel_err": (val.get("uniform") or {}).get("rel_err")},
        })
    return out


# ---------------------------------------------------------------------------
# single flight
# ---------------------------------------------------------------------------

class _Flight:
    def __init__(self):
        self.meta = threading.Lock()
        self.run = threading.Lock()
        self.latest: int | None = None          # the newest request_seq of the current burst
        self.pending = 0                        # requests in flight or waiting


class PreviewFlights:
    """Per-run single-flight, latest-wins gate (``409 superseded`` for stale requests, see the module docstring)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, _Flight] = {}

    def _flight(self, run_id: str) -> _Flight:
        with self._lock:
            f = self._runs.get(run_id)
            if f is None:
                f = self._runs[run_id] = _Flight()
            return f

    def run(self, run_id: str, seq: int, fn):
        f = self._flight(run_id)
        with f.meta:
            if f.pending and f.latest is not None and seq < f.latest:
                raise _superseded(seq, f.latest)
            f.latest = seq
            f.pending += 1
        try:
            with f.run:
                with f.meta:
                    if f.latest > seq:
                        raise _superseded(seq, f.latest)
                return fn()
        finally:
            with f.meta:
                f.pending -= 1
                if f.pending == 0:
                    f.latest = None


def _superseded(seq: int, latest: int) -> ApiError:
    return ApiError("superseded", f"preview {seq} was superseded by {latest}",
                    detail={"request_seq": seq, "latest_seq": latest})


FLIGHTS = PreviewFlights()


# ---------------------------------------------------------------------------
# the preview
# ---------------------------------------------------------------------------

def compute_delta(em: RunEmulator, dx: dict[str, np.ndarray], n: int) -> np.ndarray:
    """``Σ_v emulate(em_v, grid, dx_v)`` (core numpy/scipy) over the levers that move."""
    from sparc.core import emulator as emulator_mod

    delta = np.zeros(n)
    for var, d in dx.items():
        if not np.any(d):
            continue
        delta = delta + emulator_mod.emulate(em.levers[var], em.grid, d)
    return delta


def preview(ctx, body, *, db=None, reader=None, project_dir=None, flights: PreviewFlights | None = None) -> tuple[bytes, dict]:
    """``POST /api/runs/{rid}/preview`` → ``(packed body, headers)``.

    Errors: ``404 no_emulator`` (action ``build_emulator``), ``409 superseded``, ``422 validation``.
    """
    from threadpoolctl import threadpool_limits

    from sparc.studio.engine.compile import compile_scenario
    from sparc.studio.runs.grid import pack_arrays
    from sparc.studio.runs.selection import mask_bytes

    b = body.model_dump(mode="json") if hasattr(body, "model_dump") else dict(body)
    seq = int(b.get("request_seq") or 0)

    def work():
        with threadpool_limits(1):
            em = load_emulator(ctx)
            if em is None:
                raise ApiError("no_emulator", "this run has no emulator: build it for instant previews "
                               "(exact runs still work)", action=build_emulator_action(ctx.run_id))
            if not _aligned(ctx, em):
                raise ApiError("no_emulator", "the emulator was built on other cells than this run's grid: rebuild it",
                               action=build_emulator_action(ctx.run_id))
            doc = {"name": "preview", "edits": b.get("edits") or [], "options": b.get("options") or {}}
            comp = compile_scenario(ctx, doc, db=db, reader=reader, brush=b.get("brush"), project_dir=project_dir)
            missing = [v for v, cells in comp.lever_cells.items() if cells.any() and v not in em.levers]
            if missing:
                raise ApiError("no_emulator", f"the emulator has no {', '.join(missing)} lever",
                               action=build_emulator_action(ctx.run_id))
            delta = compute_delta(em, comp.dx(), comp.n)
            hatched, reasons = hatch(em, comp)
            return comp, delta, hatched, reasons, em

    comp, delta, hatched, reasons, em = (flights or FLIGHTS).run(ctx.run_id, seq, work)
    edited = comp.edited if comp.edited is not None else np.zeros(delta.size, dtype=bool)
    # the summary header is JSON without NaN (api.md §0.1); a cell without an emulator value counts as 0
    fin = np.where(np.isfinite(delta), delta, 0.0)
    total = float(np.sum(fin))
    outside = float(np.sum(fin[~edited]))
    trusts = [em.trust.get(v, "none") for v, cells in comp.lever_cells.items() if cells.any()]
    trust = "none" if "none" in trusts else "rough" if "rough" in trusts else "good"
    summary = {"mean": float(np.mean(fin)) if fin.size else 0.0,
               "edited_mean": float(np.mean(fin[edited])) if edited.any() else 0.0,
               "n_edited": int(edited.sum()),
               "outside_share": (outside / total) if abs(total) > 1e-12 else 0.0,
               "trust": trust, "hatched": bool(hatched), "reasons": reasons, "request_seq": seq}
    payload, offsets = pack_arrays([("delta", delta.astype(np.float32)),
                                    ("edited", np.frombuffer(mask_bytes(edited), dtype=np.uint8))])
    headers = {"X-SPARC-Offsets": json.dumps(offsets, separators=(",", ":")),
               "X-SPARC-Summary": json.dumps(summary, separators=(",", ":"), allow_nan=False),
               "Cache-Control": "no-store"}
    return payload, headers
