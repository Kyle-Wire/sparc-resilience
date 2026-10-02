"""The scenario compiler (SPEC §7.3): every edit mode, predicted clamping and the guardrails."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from sparc.studio.engine.compile import clamp_values, compile_scenario, content_hash, lever_bounds

ALL = {"kind": "all"}


def _compile(ctx, run_ctx, edits, **opts):
    doc = {"name": "t", "edits": edits, "options": opts}
    return compile_scenario(run_ctx, doc, db=ctx.db, reader=ctx.services["reader"],
                            project_dir=ctx.db.fetchone("SELECT dir FROM projects")["dir"])


def _x(run_ctx, var):
    return run_ctx.data.frame[var].to_numpy(float)


def _clamped(run_ctx, var, target):
    lo, hi = lever_bounds(run_ctx.cfg, var)
    x = _x(run_ctx, var)
    return clamp_values(target, x, lo, hi) - x


def _mask_top(run_ctx, frac):
    t = run_ctx.predictions["target"].to_numpy(float)
    k = int(round(frac * t.size))
    out = np.zeros(t.size, dtype=bool)
    out[np.argsort(-t, kind="stable")[:k]] = True
    return out


def test_add_set_scale_on_a_selection(ctx, run_ctx):
    sel = {"kind": "top", "column": "pred:target", "frac": 0.25, "direction": "highest"}
    mask = _mask_top(run_ctx, 0.25)
    x = _x(run_ctx, "canopy")
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "add", "amount": 7.5, "where": sel}])
    e = comp.edits[0]
    assert e.core_mode == "add" and e.amount == 7.5 and e.per_point is None
    np.testing.assert_array_equal(e.where, mask)
    np.testing.assert_allclose(e.requested, np.where(mask, 7.5, 0.0))
    comp = _compile(ctx, run_ctx, [{"lever": "albedo", "mode": "set", "amount": 0.35, "where": sel}])
    a = _x(run_ctx, "albedo")
    np.testing.assert_allclose(comp.edits[0].requested, np.where(mask, 0.35 - a, 0.0))
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "scale", "amount": 1.5}])
    assert comp.edits[0].where is None
    np.testing.assert_allclose(comp.edits[0].requested, 0.5 * x)


def test_floor_ceiling_percentile_per_point(ctx, run_ctx):
    x = _x(run_ctx, "canopy")
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "floor", "amount": 30, "where": ALL}])
    np.testing.assert_allclose(comp.edits[0].per_point, np.maximum(30 - x, 0))
    imp = _x(run_ctx, "impervious")
    comp = _compile(ctx, run_ctx, [{"lever": "impervious", "mode": "ceiling", "amount": 60, "where": ALL}])
    np.testing.assert_allclose(comp.edits[0].per_point, np.minimum(60 - imp, 0))
    # increase lever: raise to the 75th percentile; decrease lever: lower to the 25th
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "to_percentile", "percentile": 75}])
    np.testing.assert_allclose(comp.edits[0].per_point, np.maximum(np.nanpercentile(x, 75) - x, 0))
    comp = _compile(ctx, run_ctx, [{"lever": "impervious", "mode": "to_percentile", "amount": 25}])
    np.testing.assert_allclose(comp.edits[0].per_point, np.minimum(np.nanpercentile(imp, 25) - imp, 0))


def test_fill_headroom_uses_plantable_space(ctx, run_ctx):
    from sparc.core.planner import plantable_headroom
    from sparc.studio.runs import layers as L

    x = _x(run_ctx, "canopy")
    lay = L.people_layers(run_ctx)
    assert lay is not None, "the demo project ships planner layers"
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "fill_headroom", "amount": 0.5, "paved_share": 0.3}])
    np.testing.assert_allclose(comp.edits[0].per_point, 0.5 * plantable_headroom(x, lay, 0.3))
    with pytest.raises(Exception) as exc:
        _compile(ctx, run_ctx, [{"lever": "albedo", "mode": "fill_headroom", "amount": 0.5}])
    assert getattr(exc.value, "code", None) == "validation"


def test_per_cell_plan_blob_and_csv(client, ctx, run_ctx, synth_run):
    from pathlib import Path

    from sparc.studio.scenarios import plans

    rid, _ = synth_run
    n = run_ctx.grid.n
    row = plans.create(ctx.db, run_ctx, {"lever": "canopy", "budget": 500.0}, "p")
    dose = np.load(Path(row["dir"]) / "dose.npy").astype(float)
    assert (dose > 0).any()
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "per_cell", "per_cell_ref": f"plan:{row['id']}"}])
    np.testing.assert_allclose(comp.edits[0].per_point, dose)
    assert not comp.portable
    # brushed edit blob: Int32 idx + Float32 values
    idx = np.array([0, 5, 9], dtype="<i4")
    val = np.array([3.0, -2.0, 4.5], dtype="<f4")
    r = client.put(f"/api/runs/{rid}/blobs?kind=edit", content=idx.tobytes() + val.tobytes(),
                   headers={"X-SPARC-Count": "3", "Content-Type": "application/octet-stream"})
    assert r.status_code == 201, r.text
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "per_cell", "per_cell_ref": f"blob:{r.json()['blob_id']}"}])
    want = np.zeros(n)
    want[idx] = val
    np.testing.assert_allclose(comp.edits[0].per_point, want, rtol=1e-6)
    # project CSV joined by id
    pdir = Path(ctx.db.fetchone("SELECT dir FROM projects")["dir"])
    ids = np.asarray(run_ctx.grid.ids)
    (pdir / "data" / "design.csv").write_text("id,change\n" + f"{ids[3]},2.5\n{ids[7]},-1\nnot-a-cell,9\n")
    comp = _compile(ctx, run_ctx, [{"lever": "canopy", "mode": "per_cell", "per_cell_ref": "csv:data/design.csv"}])
    want = np.zeros(n)
    want[3], want[7] = 2.5, -1.0
    np.testing.assert_allclose(comp.edits[0].per_point, want)
    assert any(w["code"] == "unknown_ids" for w in comp.warnings)


def test_predicted_clamping_is_the_engine_bounds_rule(ctx, run_ctx):
    """Edits beyond the bounds: core's own apply() (through the compiler) equals the bounds rule applied by hand,
    including edits that apply in order on the same lever."""
    from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

    edits = [{"lever": "canopy", "mode": "add", "amount": 70},
             {"lever": "canopy", "mode": "floor", "amount": 95},
             {"lever": "albedo", "mode": "set", "amount": 0.001},
             {"lever": "impervious", "mode": "add", "amount": -60}]
    comp = _compile(ctx, run_ctx, edits)
    can = _x(run_ctx, "canopy")
    step1 = can + _clamped(run_ctx, "canopy", can + 70)
    lo, hi = lever_bounds(run_ctx.cfg, "canopy")
    step2 = clamp_values(step1 + np.maximum(95 - step1, 0), can, lo, hi)
    np.testing.assert_allclose(comp.realized["canopy"], step2 - can, atol=1e-12)
    a = _x(run_ctx, "albedo")
    np.testing.assert_allclose(comp.realized["albedo"], _clamped(run_ctx, "albedo", np.full(a.size, 0.001)), atol=1e-12)
    imp = _x(run_ctx, "impervious")
    np.testing.assert_allclose(comp.realized["impervious"], _clamped(run_ctx, "impervious", imp - 60), atol=1e-12)
    assert comp.levers["canopy"]["clipped_share"] > 0 and comp.levers["albedo"]["clipped_share"] > 0
    # and core's ScenarioEngine.apply on the same data and config (no mediators) gives the same realised values
    eng = object.__new__(ScenarioEngine)
    eng.data, eng.cfg, eng.mediators = run_ctx.data, run_ctx.cfg, None
    spec = ScenarioSpec(name="x", interventions=[Intervention(e.lever, e.core_mode, e.amount, where=e.where,
                                                              per_point=e.per_point) for e in comp.edits])
    _new, realized = eng.apply(spec)
    for v in realized:
        np.testing.assert_array_equal(realized[v], comp.realized[v])


def test_mediator_with_parent_is_blocked_without_expert(ctx, run_ctx):
    edits = [{"lever": "canopy", "mode": "add", "amount": 10}, {"lever": "ndvi", "mode": "add", "amount": 0.1}]
    comp = _compile(ctx, run_ctx, edits)
    codes = {w["code"]: w for w in comp.blocking}
    assert "mediator_with_parents" in codes and codes["mediator_with_parents"]["edit_index"] == 1
    comp = _compile(ctx, run_ctx, edits, expert=True)
    assert not comp.blocking
    assert {w["code"] for w in comp.warnings} >= {"mediator_with_parents", "not_actionable"}
    # the mediator alone (no parent edited) is only non-actionable
    comp = _compile(ctx, run_ctx, edits[1:])
    assert [w["code"] for w in comp.blocking] == ["not_actionable"]


def test_non_actionable_lever_needs_expert(ctx, run_ctx):
    comp = _compile(ctx, run_ctx, [{"lever": "elevation", "mode": "add", "amount": 1}])
    assert [w["code"] for w in comp.blocking] == ["not_actionable"]
    assert not _compile(ctx, run_ctx, [{"lever": "elevation", "mode": "add", "amount": 1}], expert=True).blocking
    with pytest.raises(Exception) as exc:
        _compile(ctx, run_ctx, [{"lever": "no_such_column", "mode": "add", "amount": 1}])
    assert getattr(exc.value, "code", None) == "validation"


def test_warnings(ctx, run_ctx):
    comp = _compile(ctx, run_ctx, [
        {"lever": "canopy", "mode": "add", "amount": 60},                                     # > 1 sd
        {"lever": "albedo", "mode": "add", "amount": 0.01, "where": {"kind": "cells", "ids": []}},   # empty
        {"lever": "canopy", "mode": "add", "amount": -1},                                      # wrong direction
    ])
    codes = {(w["code"], w["edit_index"]) for w in comp.warnings}
    assert ("beyond_sd", 0) in codes and ("beyond_p995", 0) in codes
    assert ("empty_selection", 1) in codes
    assert ("opposite_direction", 2) in codes
    assert not comp.blocking


def test_content_hash_ignores_names(run_ctx):
    a = {"name": "A", "tags": ["x"], "notes": "n", "edits": [{"lever": "canopy", "mode": "add", "amount": 5,
                                                               "label": "street trees"}]}
    b = copy.deepcopy(a)
    b.update(name="B", tags=[], notes="")
    b["edits"][0]["label"] = "other"
    b["edits"][0]["where"] = {"kind": "all"}
    assert content_hash(a) == content_hash(b)
    b["options"] = {"clip_to_support": False}
    assert content_hash(a) != content_hash(b)


def test_compile_endpoint(client, synth_run):
    rid, _ = synth_run
    doc = {"name": "c", "edits": [{"lever": "canopy", "mode": "add", "amount": 10,
                                   "where": {"kind": "top", "column": "pred:target", "frac": 0.1,
                                             "direction": "highest"}}],
           "costs": {"canopy": {"per_unit": 2.0}}}
    r = client.post(f"/api/runs/{rid}/compile", json={"scenario": doc})
    assert r.status_code == 200, r.text
    out = r.json()
    lv = out["levers"]["canopy"]
    assert lv["n_cells"] == out["union_cells"] > 0
    assert lv["mean_requested"] == pytest.approx(10.0)
    assert lv["est_cost"] == pytest.approx(2.0 * lv["predicted_mean_realised"] * lv["n_cells"])
    assert out["people"] is not None and out["portable"] is True
    assert out["emulator"]["usable"] is False and out["est_exact_s"] > 0
