"""Comparisons with paired standard errors (SPEC §7.8, §7.12, api.md §7.6).

Items (``ItemRef``) are exact results, configured scenarios, plans and the baseline:

* ``result`` - ``results/<id>/cells.parquet`` delta and ``folds.npy``;
* ``configured`` - the ``sc:<slug>`` layer (``scenario_deltas.parquet``) with folds from
  ``scenario_detail.npz``, else from a ``configured`` result made by "Re-run exactly";
* ``plan`` - the plan's closed-loop result when verified, else the planned benefit (no folds);
* ``baseline`` - Δ = 0 everywhere (its folds are zero: A − baseline has A's own SE).

Pairs get ``SE(A−B) = std_k(mean_i(Δ_A,ki − Δ_B,ki))·√(K−1)`` when both items have folds of the same fold
models (``paired: true``); otherwise ``√(SE_A² + SE_B²)`` (``paired: false``), and items without folds are
listed in ``needs_exact``.  Difference maps are written as ``comparisons/<cid>/diff_<a>__<b>.npy`` (layer key
``cmp:<cid>:<a>__<b>``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["resolve_item", "compare", "comparison_out", "list_comparisons", "delete", "ref_dict"]


def ref_dict(ref: Any) -> dict:
    d = ref.model_dump(mode="json", exclude_none=True) if hasattr(ref, "model_dump") else dict(ref)
    kind = d.get("kind")
    if kind in ("result", "plan") and not d.get("id"):
        raise ApiError("validation", f"a {kind} item needs an id",
                       detail={"errors": [{"path": "items", "message": "id required", "code": "missing"}]})
    if kind == "configured" and not d.get("slug"):
        raise ApiError("validation", "a configured item needs a slug",
                       detail={"errors": [{"path": "items", "message": "slug required", "code": "missing"}]})
    return {k: v for k, v in d.items() if k in ("kind", "id", "slug")}


def _folds_ok(f, n: int) -> np.ndarray | None:
    if f is None:
        return None
    f = np.asarray(f, dtype=np.float64)
    return f if f.ndim == 2 and f.shape[1] == n and f.shape[0] >= 2 else None


def resolve_item(ctx, db, ref: Any) -> dict:
    """``{ref, label, delta[n], folds[K,n] | None, se, cost, kind}`` of an item on the run of ``ctx``."""
    from sparc.studio.engine import stats as S
    from sparc.studio.engine import store
    from sparc.studio.runs import layers as L

    d = ref_dict(ref)
    n = ctx.grid.n
    kind = d["kind"]
    if kind == "baseline":
        return {"ref": d, "label": "Baseline (today)", "delta": np.zeros(n), "folds": None, "se": 0.0, "cost": 0.0,
                "zero": True, "edited": np.zeros(n, dtype=bool)}
    if kind == "result":
        row = store.result_row(db, d["id"])
        if row["run_id"] != ctx.run_id:
            raise ApiError("validation", f"result {d['id']} belongs to run {row['run_id']}",
                           detail={"errors": [{"path": "items", "message": "another run", "code": "run"}]})
        cells = store.read_cells(Path(row["dir"]))
        delta = cells["delta"].to_numpy(np.float64)
        folds = _folds_ok(store.read_folds(Path(row["dir"])), n)
        summ = read_json(Path(row["dir"]) / "summary.json") or {}
        label = ((summ.get("scenario") or {}).get("name")) or (dbmod.loads(row.get("summary_json"), {}) or {}).get(
            "name") or d["id"]
        edited = np.zeros(n, dtype=bool)
        for c in cells.columns:
            if c.startswith("realized_"):
                edited |= np.abs(cells[c].to_numpy(np.float64)) > 0
        return {"ref": d, "label": str(label), "delta": delta, "folds": folds, "se": S.masked_se(folds),
                "cost": ((summ.get("cost") or {}).get("total")), "zero": False, "edited": edited}
    if kind == "configured":
        slug = d["slug"]
        match = next((s for s in ctx.configured_scenarios() if s["slug"] == slug), None)
        if match is None:
            raise ApiError("validation", f"no configured scenario {slug!r} on this run",
                           detail={"errors": [{"path": "items", "message": "unknown slug", "code": "unknown"}]})
        delta = np.asarray(L.layer_array(ctx, f"sc:{slug}"), dtype=np.float64)
        det = ctx.scenario_detail()
        folds = _folds_ok(det["folds"].get(match["name"]) if det else None, n)
        if folds is None:
            rr = db.fetchall("SELECT * FROM results WHERE run_id = ? AND kind = 'configured' AND stale = 0 "
                             "ORDER BY created_utc DESC", (ctx.run_id,))
            for r in rr:
                spec = read_json(Path(r["dir"]) / "spec.json") or {}
                if spec.get("configured_slug") == slug:
                    folds = _folds_ok(store.read_folds(Path(r["dir"])), n)
                    if folds is not None:
                        delta = store.read_cells(Path(r["dir"]))["delta"].to_numpy(np.float64)
                    break
        summaries = {str(s.get("name")): s for s in (ctx.manifest.get("scenarios") or []) if isinstance(s, dict)}
        s = summaries.get(match["name"]) or {}
        se = S.masked_se(folds) if folds is not None else s.get("mean_delta_se")
        per_unit = {v: float(((ctx.cfg_raw.get("actionable") or {}).get(v) or {}).get("cost_per_unit") or 1.0)
                    for v in (s.get("mean_realized") or {})}
        cost = sum(abs(float(m)) * n * per_unit[v] for v, m in (s.get("mean_realized") or {}).items()) \
            if s.get("mean_realized") else None
        return {"ref": d, "label": match["name"], "delta": delta, "folds": folds, "se": se, "cost": cost,
                "zero": False, "edited": np.ones(n, dtype=bool)}
    if kind == "plan":
        prow = db.fetchone("SELECT * FROM plans WHERE id = ?", (d["id"],))
        if prow is None or prow["run_id"] != ctx.run_id:
            raise ApiError("validation", f"no plan {d['id']!r} on this run",
                           detail={"errors": [{"path": "items", "message": "unknown plan", "code": "unknown"}]})
        pdir = Path(prow["dir"])
        planned = read_json(pdir / "planned.json") or {}
        realised = read_json(pdir / "realised.json") or {}
        label = f"Plan: {prow.get('name') or prow['id']}"
        treated = np.asarray(np.load(pdir / "dose.npy", allow_pickle=False), dtype=np.float64) > 0
        if realised.get("result_id"):
            try:
                rrow = store.result_row(db, realised["result_id"])
                delta = store.read_cells(Path(rrow["dir"]))["delta"].to_numpy(np.float64)
                folds = _folds_ok(store.read_folds(Path(rrow["dir"])), n)
                return {"ref": d, "label": label, "delta": delta, "folds": folds, "se": S.masked_se(folds),
                        "cost": planned.get("total_cost"), "zero": False, "edited": treated}
            except ApiError:
                pass
        benefit = np.asarray(np.load(pdir / "planned_benefit.npy", allow_pickle=False), dtype=np.float64)
        return {"ref": d, "label": label + " (planned)", "delta": -benefit, "folds": None, "se": None,
                "cost": planned.get("total_cost"), "zero": False, "edited": treated}
    raise ApiError("validation", f"unknown item kind {kind!r}",
                   detail={"errors": [{"path": "items", "message": "unknown kind", "code": "kind"}]})


def _regions(ctx, db, names: list[str] | None) -> dict[str, np.ndarray]:
    """Requested regions: saved region ids (``rg_…``) or ``zone:<code>``."""
    from sparc.studio.runs.selection import RunSource, resolve

    out: dict[str, np.ndarray] = {}
    src = RunSource(ctx, db)
    for nm in names or []:
        if nm.startswith("zone:"):
            out[nm] = resolve(src, {"kind": "zones", "values": [nm[5:]]})
        else:
            row = db.fetchone("SELECT name, spec_json FROM regions WHERE id = ?", (nm,))
            if row is None:
                raise ApiError("validation", f"no region {nm!r}",
                               detail={"errors": [{"path": "regions", "message": "unknown region", "code": "unknown"}]})
            out[row.get("name") or nm] = resolve(src, {"kind": "region", "id": nm})
    return out


def compare(db, ctx, items: list, *, regions: list[str] | None = None, thresholds: list[float] | None = None) -> dict:
    """``POST /api/runs/{rid}/compare`` → ``Comparison`` (written to ``comparisons/<cid>/``)."""
    from sparc.core import runio
    from sparc.studio.engine import stats as S
    from sparc.studio.runs import layers as L
    from sparc.studio.runs.common import clean, likely

    if not 2 <= len(items) <= 4:
        raise ApiError("validation", "compare takes 2 to 4 items",
                       detail={"errors": [{"path": "items", "message": "2–4 items", "code": "count"}]})
    unit = ctx.units.get("target", "°F")
    res = [resolve_item(ctx, db, it) for it in items]
    n = ctx.grid.n
    regs = _regions(ctx, db, regions)
    cid = new_id("comparison")
    cdir = Path(ctx.studio_dir) / "comparisons" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    out_items = []
    for it in res:
        edited = it["edited"]
        city = likely(float(np.mean(it["delta"])), it["se"], unit, what="the city") or likely(0.0, None, unit)
        ed = S.masked_likely(it["delta"], it["folds"], edited, unit, what="the edited area") if edited.any() \
            and not edited.all() else None
        out_items.append({"ref": it["ref"], "label": it["label"], "city": city, "edited": ed,
                          "cost": float(it["cost"]) if it["cost"] is not None else None,
                          "has_folds": it["folds"] is not None or it["zero"]})
    pairs = []
    for a in range(len(res)):
        for b in range(a + 1, len(res)):
            A, B = res[a], res[b]
            fa = A["folds"] if A["folds"] is not None else (np.zeros_like(B["folds"]) if A["zero"] and B["folds"]
                                                             is not None else None)
            fb = B["folds"] if B["folds"] is not None else (np.zeros_like(A["folds"]) if B["zero"] and A["folds"]
                                                             is not None else None)
            city = S.pair_likely(A["delta"], fa, A["se"], B["delta"], fb, B["se"], np.ones(n, dtype=bool), unit,
                                 what=f"{A['label']} vs {B['label']}")
            rr = {}
            for nm, mk in regs.items():
                lk = S.pair_likely(A["delta"], fa, A["se"], B["delta"], fb, B["se"], mk, unit)
                if lk is not None:
                    rr[nm] = lk
            diff = (A["delta"] - B["delta"]).astype(np.float32)
            with runio.atomic_open(cdir / f"diff_{a}__{b}.npy", "wb") as fh:
                np.save(fh, diff, allow_pickle=False)
            pairs.append({"a": a, "b": b, "city": city, "regions": rr, "layer_key": f"cmp:{cid}:{a}__{b}"})
    equity: dict[str, dict] = {}
    exposure: list[dict] = []
    lay = L.people_layers(ctx) if ctx.data is not None else None
    if lay is not None and "people" in lay:
        from sparc.core.planner import benefit_by_group, exposure_table

        obs = np.asarray(ctx.data.target_raw, dtype=np.float64)
        people = np.nan_to_num(lay["people"].to_numpy(float))
        thr = thresholds or (ctx.cfg_raw.get("climate") or {}).get("thresholds") or (
            [90.0, 95.0] if unit == "°F" else [32.0, 35.0])
        for it in res:
            if it["zero"]:
                continue
            try:
                g = benefit_by_group(-it["delta"], lay)
                equity[it["label"]] = {k: float(v["concentration_index"]) for k, v in g.items()}
            except Exception as exc:          # too few residents for quintiles
                log.info("equity of %s skipped: %s", it["label"], exc)
            for r in exposure_table(obs, people, [float(t) for t in thr], {}, it["delta"]):
                exposure.append({"item": it["label"], **r})
    cpc = {}
    for it in res:
        cost = it["cost"]
        cpc[it["label"]] = (-float(np.sum(it["delta"])) / float(cost)) if cost else None
    needs = [it["ref"] for it in res if it["folds"] is None and not it["zero"]]
    out = clean({"id": cid, "items": out_items, "pairs": pairs, "equity": equity, "exposure": exposure,
                 "cooling_per_cost": cpc, "needs_exact": needs})
    now = utc_now()
    write_json_atomic(cdir / "items.json", {"schema": 1, "id": cid, "run_id": ctx.run_id,
                                            "items": [it["ref"] for it in res], "regions": regions or [],
                                            "thresholds": thresholds, "created_utc": now})
    write_json_atomic(cdir / "summary.json", out)
    db.insert("comparisons", {"id": cid, "run_id": ctx.run_id, "items_json": dbmod.dumps([it["ref"] for it in res]),
                              "dir": str(cdir), "summary_json": dbmod.dumps(out), "created_utc": now})
    return out


def comparison_out(db, cid: str) -> dict:
    row = db.fetchone("SELECT * FROM comparisons WHERE id = ?", (cid,))
    if row is None:
        raise ApiError("not_found", f"no comparison {cid!r}")
    s = read_json(Path(row["dir"]) / "summary.json") if row.get("dir") else None
    return s if isinstance(s, dict) else (dbmod.loads(row.get("summary_json"), {}) or {})


def list_comparisons(db, run_id: str) -> list[dict]:
    return [{"id": r["id"], "items": dbmod.loads(r.get("items_json"), []) or [], "created_utc": r.get("created_utc")
             or ""} for r in db.fetchall("SELECT * FROM comparisons WHERE run_id = ? ORDER BY created_utc DESC",
                                         (run_id,))]


def delete(db, cid: str) -> None:
    from sparc.studio.engine import store

    row = db.fetchone("SELECT * FROM comparisons WHERE id = ?", (cid,))
    if row is None:
        raise ApiError("not_found", f"no comparison {cid!r}")
    if row.get("dir"):
        store.rmtree_under(Path(row["dir"]), "comparisons")
    db.execute("DELETE FROM comparisons WHERE id = ?", (cid,))
