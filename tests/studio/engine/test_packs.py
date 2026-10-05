"""Decision, plan and compare packs (SPEC §7.13): the ``export.*_pack`` job kinds run in workers."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from tests.studio.engine.conftest import make_result


async def _submit(ctx, kind: str, params: dict, project_id: str, eid: str) -> dict:
    return await ctx.jobs.submit(kind, {"export_id": eid, **params}, project_id=project_id)


def _run_pack(client, ctx, wait_job, kind, params, project_id, *, with_row=True):
    import anyio

    from sparc.studio.workspace import new_id, utc_now

    eid = new_id("export")
    if with_row:                                        # what POST /api/exports ([S]) does before enqueuing
        ctx.db.insert("exports", {"id": eid, "project_id": project_id, "run_id": None, "kind": kind.split(".")[1],
                                  "ref": None, "options_json": "{}", "job_id": None, "path": None, "bytes": None,
                                  "status": "running", "created_utc": utc_now()})
    job = client.portal.call(_submit, ctx, kind, params, project_id, eid) if hasattr(client, "portal") else \
        anyio.run(_submit, ctx, kind, params, project_id, eid)
    done = wait_job(client, job["id"], timeout=180)
    assert done["status"] == "succeeded", done
    return eid, done["result"]


def _zip(path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def test_decision_pack_for_an_exact_result(client, ctx, demo, run_ctx, synth_run, wait_job):
    import rasterio

    row = make_result(ctx, run_ctx)
    eid, res = _run_pack(client, ctx, wait_job, "export.decision_pack", {"result_id": row["id"]}, demo["id"])
    assert res["export_id"] == eid and res["draft"] is False and res["bytes"] > 0
    assert f"/exports/{eid}/" in res["path"] and res["path"].endswith(".zip")
    z = _zip(res["path"])
    names = set(z.namelist())
    assert {"brief.html", "scenario.json", "summary.json", "cells.csv", "delta.tif", "realized_canopy.tif",
            "hex_250m.csv", "hex_500m.csv", "README.txt"} <= names
    brief = z.read("brief.html").decode()
    assert brief.lower().startswith("<!doctype html>") and "DRAFT" not in brief
    assert "data:image/png;base64," in brief and "Provenance" in brief and "likely range" in brief
    readme = z.read("README.txt").decode()
    assert "NEGATIVE = COOLER" in readme and "°F" in readme
    import pandas as pd

    cells = pd.read_csv(io.BytesIO(z.read("cells.csv")))
    assert list(cells.columns[:7]) == ["id", "lon", "lat", "zone", "delta", "delta_sd", "extrapolation"]
    assert "realised_canopy" in cells.columns and len(cells) == run_ctx.grid.n
    assert np.isfinite(cells["lon"]).all()
    with z.open("delta.tif") as f, rasterio.MemoryFile(f.read()) as mem, mem.open() as ds:
        assert ds.crs.to_epsg() == 32619
        g = run_ctx.grid
        assert ds.width == g.nx and ds.height == g.ny
        assert ds.transform.a == pytest.approx(g.dx) and ds.transform.e == pytest.approx(-g.dx)
        band = ds.read(1)
        # the north-west cell of the raster is row 0: check one data cell maps back to its run cell
        i = int(np.argmax(np.isfinite(cells["delta"].to_numpy())))
        assert band[g.ny - 1 - g.iy[i], g.ix[i]] == pytest.approx(cells["delta"].iloc[i], rel=1e-5)
    hexes = pd.read_csv(io.BytesIO(z.read("hex_250m.csv")))
    assert {"cooling", "people", "n_cells", "lon", "lat"} <= set(hexes.columns)
    exp = ctx.db.fetchone("SELECT * FROM exports WHERE id = ?", (eid,))
    assert exp["status"] == "ready" and exp["path"] == res["path"] and exp["bytes"] == res["bytes"]


def test_preview_only_pack_is_draft(client, ctx, demo, run_ctx, synth_run, wait_job, fake_emulator):
    rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    sc = client.post(f"/api/projects/{demo['id']}/scenarios", json={"doc": {
        "name": "Shade", "anchor_run_id": rid,
        "edits": [{"lever": "canopy", "mode": "add", "amount": 10,
                   "where": {"kind": "top", "column": "pred:target", "frac": 0.2, "direction": "highest"}}]}}).json()
    eid, res = _run_pack(client, ctx, wait_job, "export.decision_pack", {"result_id": sc["id"]}, demo["id"])
    assert res["draft"] is True
    z = _zip(res["path"])
    brief = z.read("brief.html").decode()
    assert "DRAFT" in brief and "preview only" in brief
    import json

    summ = json.loads(z.read("summary.json"))
    assert summ["draft"] is True and summ["city"]["se"] is None and summ["city"]["lo"] is None
    # the preview computes no extrapolation scores: the brief says so instead of a definite "0%"
    assert summ["extrapolated_edited"] is None and summ["summary"]["frac_extrapolated_edited"] is None
    assert "Edited cells outside observed conditions</td><td>not computed (preview)</td>" in brief
    assert "was not computed for this preview" in brief
    assert "DRAFT" in z.read("README.txt").decode()
    opts = json.loads(ctx.db.fetchone("SELECT options_json FROM exports WHERE id = ?", (eid,))["options_json"])
    assert opts["draft"] is True


def test_plan_and_compare_packs(client, ctx, demo, run_ctx, synth_run, wait_job):
    rid, _ = synth_run
    plan = client.post(f"/api/runs/{rid}/plans", json={"params": {"lever": "canopy", "budget": 1500}, "name": "Trees",
                                                        "verify": False}).json()["plan"]
    _eid, res = _run_pack(client, ctx, wait_job, "export.plan_pack", {"plan_id": plan["id"]}, demo["id"],
                          with_row=False)
    z = _zip(res["path"])
    names = set(z.namelist())
    assert {"brief.html", "cells.csv", "delta.tif", "field_list.csv", "logger_sites.csv", "before_after_pairs.csv",
            "pareto.csv", "README.txt"} <= names
    assert res["draft"] is True                       # not verified: planned numbers only
    import pandas as pd

    fl = pd.read_csv(io.BytesIO(z.read("field_list.csv")))
    assert {"rank", "id", "lon", "lat", "dose"} <= set(fl.columns) and len(fl) == plan["planned"]["n_cells_treated"]
    row = make_result(ctx, run_ctx)
    cmp_ = client.post(f"/api/runs/{rid}/compare", json={"items": [{"kind": "result", "id": row["id"]},
                                                                   {"kind": "configured", "slug": "cooling-package"}]}).json()
    _eid, res = _run_pack(client, ctx, wait_job, "export.compare_pack", {"comparison_id": cmp_["id"]}, demo["id"],
                          with_row=False)
    z = _zip(res["path"])
    assert {"brief.html", "items.csv", "pairs.csv", "diff_0__1.tif", "summary.json", "README.txt"} <= set(z.namelist())
    pairs = pd.read_csv(io.BytesIO(z.read("pairs.csv")))
    assert bool(pairs["paired"].iloc[0]) is True


def test_narrative_range_reads_from_the_smaller_to_the_larger_cooling():
    from sparc.studio.scenarios.packs import narrative

    res = {"city": {"estimate": -0.656, "lo": -0.81, "hi": -0.50},
           "regions": [{"name": "edited", "mean": {"estimate": -1.66, "lo": -2.07, "hi": -1.25}}]}
    assert "by 1.66 °F (likely range 1.25–2.07 °F)" in narrative("Corridor", res, "°F", 10, 1.0, False)


def test_realised_change_maps_show_the_size_of_the_change():
    """Unchanged cells take the lightest colour and a decrease reads like an increase of the same size."""
    import io

    from PIL import Image

    from sparc.studio.scenarios.packs import map_png

    class G:                                   # a 2 × 2 run grid
        nx = ny = 2

        @staticmethod
        def raster(v):
            return np.asarray(v, dtype=np.float64).reshape(2, 2)

    def px(values):
        im = np.asarray(Image.open(io.BytesIO(map_png(G, np.asarray(values), magnitude=True))).convert("RGBA"))
        return {tuple(im[0, 0]), tuple(im[0, -1]), tuple(im[-1, 0]), tuple(im[-1, -1])}, im

    cut, im = px([0.0, 0.0, -10.0, -10.0])
    grow, _ = px([0.0, 0.0, 10.0, 10.0])
    assert cut == grow and len(cut) == 2
    light, dark = sorted(cut, key=lambda c: -sum(c[:3]))
    assert sum(light[:3]) > sum(dark[:3])


def test_pack_provenance_names_the_code_commit_of_runs_with_a_null_git_commit():
    """Studio-launched runs written before the fix have ``manifest.git_commit: null`` but
    ``provenance.git.commit`` set: the decision brief's provenance table names that commit, as the run Overview
    and the report export do."""
    from types import SimpleNamespace

    from sparc.studio.scenarios.packs import _provenance

    full = "07a4e7d59daea143a7f9f09a3751a5f9268b6e10"
    ctx = SimpleNamespace(run_id="r1", run_dir="/tmp/r1", checkpoint_json={},
                          manifest_raw={"git_commit": None, "provenance": {"git": {"commit": full}}})
    assert _provenance(ctx, {})["git commit"] == full
    ctx.manifest_raw = {"git_commit": "07a4e7d"}                          # older manifests: the short hash
    assert _provenance(ctx, {})["git commit"] == "07a4e7d"
