"""Exports, the report builder and the findings notebook (SPEC §6.7, §6.9, §6.10, api.md §10–11).

Export jobs run in real worker processes on the committed synthetic fixture run (demo city, EPSG:32619):
the bundle ZIP and its README, GeoTIFFs readable by rasterio in the run's CRS, the packaged results page,
the report and its live preview (caveats written from the numbers), and findings with their image and
export.  ``POST /api/exports`` creates the row before the job, also for the engine's pack kinds.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests.studio.studies.conftest import RUN_ID

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82")
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'


def _export(client, wait_job, pid: str, kind: str, params: dict, status: str = "succeeded") -> dict:
    r = client.post("/api/exports", json={"kind": kind, "project_id": pid, "params": params})
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["export"]["status"] == "running" and out["export"]["job_id"] == out["job"]["id"]
    assert out["job"]["params"]["export_id"] == out["export"]["id"] and out["job"]["kind"] == f"export.{kind}"
    j = wait_job(client, out["job"]["id"], timeout=180)
    assert j["status"] == status, j
    return client.get(f"/api/exports/{out['export']['id']}").json()


def _zip(client, ex: dict) -> zipfile.ZipFile:
    r = client.get(f"/api/exports/{ex['id']}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert len(r.content) == ex["bytes"]
    return zipfile.ZipFile(io.BytesIO(r.content))


# ---------------------------------------------------------------------------
# bundle and GIS
# ---------------------------------------------------------------------------

def test_bundle_lists_the_requested_files_readme_and_no_checkpoint(client, ctx, demo, fixture_run, wait_job):
    rid, rd = fixture_run
    (rd / "checkpoint.pkl").write_bytes(b"\x80\x04N.")                 # a (tiny) checkpoint to leave out
    ex = _export(client, wait_job, demo["id"], "bundle",
                 {"run_id": rid, "outputs": ["predictions", "scenarios", "manifest", "checkpoint"]})
    assert ex["status"] == "ready" and ex["run_id"] == rid and ex["kind"] == "bundle"
    assert Path(ex["path"]).parent == Path(demo["dir"]) / "exports" / ex["id"]
    names = sorted(_zip(client, ex).namelist())
    assert names == sorted(f"{rid}/{n}" for n in ("predictions.parquet", "scenarios.json", "manifest.json",
                                                  "README.txt", "contents.json"))
    z = _zip(client, ex)
    readme = z.read(f"{rid}/README.txt").decode()
    assert "negative = cooler" in readme and "°F" in readme and "predictions.parquet" in readme
    assert "dT_pred" in readme                                        # the column dictionary, with units
    contents = json.loads(z.read(f"{rid}/contents.json"))
    assert {c["path"] for c in contents["files"]} == {"predictions.parquet", "scenarios.json", "manifest.json"}
    # every output by default, still without the checkpoint unless asked for
    ex2 = _export(client, wait_job, demo["id"], "bundle", {"run_id": rid})
    all_names = _zip(client, ex2).namelist()
    assert f"{rid}/checkpoint.pkl" not in all_names and f"{rid}/response_canopy.parquet" in all_names
    assert not any("/studio/" in n for n in all_names)
    ex3 = _export(client, wait_job, demo["id"], "bundle", {"run_id": rid, "outputs": ["manifest"],
                                                           "include_checkpoint": True})
    assert f"{rid}/checkpoint.pkl" in _zip(client, ex3).namelist()
    # the record next to the file and the project listing
    rec = json.loads((Path(ex["path"]).parent / "export.json").read_text())
    assert rec["status"] == "ready" and rec["bytes"] == ex["bytes"]
    listing = client.get(f"/api/projects/{demo['id']}/exports").json()
    assert {e["id"] for e in listing} == {ex["id"], ex2["id"], ex3["id"]}
    r = client.post("/api/exports", json={"kind": "bundle", "project_id": demo["id"],
                                          "params": {"run_id": rid, "outputs": ["nope"]}})
    assert r.status_code == 422
    r = client.post("/api/exports", json={"kind": "bundle", "project_id": demo["id"],
                                          "params": {"run_id": rid, "export_id": "ex_mine0000"}})
    assert r.status_code == 422 and "server" in r.text


def test_gis_geotiffs_open_in_rasterio_with_the_run_crs(client, ctx, demo, fixture_run, wait_job, tmp_path):
    import pandas as pd
    import rasterio

    rid, rd = fixture_run
    (rd / "planner").mkdir(exist_ok=True)
    pd.DataFrame({"cell": [0, 5], "role": ["low canopy", "high canopy"], "canopy": [5.0, 60.0],
                  "impervious": [70.0, 20.0]}).to_csv(rd / "planner" / "logger_sites.csv", index=False)
    ex = _export(client, wait_job, demo["id"], "gis", {"run_id": rid, "layers": ["obs", "pred", "fp_canopy"]})
    z = _zip(client, ex)
    top = f"{rid}_gis"
    names = set(z.namelist())
    assert {f"{top}/geotiff/obs.tif", f"{top}/geotiff/pred.tif", f"{top}/geotiff/fp_canopy.tif",
            f"{top}/hexagons.gpkg", f"{top}/logger_sites.csv", f"{top}/README.txt"} <= names
    z.extractall(tmp_path)
    reader = ctx.services["reader"].get(rid)
    g = reader.grid
    obs = np.asarray(__import__("sparc.studio.runs.layers", fromlist=["layer_array"]).layer_array(reader, "obs"),
                     dtype=np.float64)
    with rasterio.open(tmp_path / top / "geotiff" / "obs.tif") as src:
        assert src.crs.to_epsg() == 32619 and src.dtypes == ("float32",)
        assert src.width == g.nx and src.height == g.ny and src.res == pytest.approx((g.dx, g.dx))
        a = src.read(1)
        i = int(np.flatnonzero(np.isfinite(obs))[0])
        x, y = g.x0 + g.ix[i] * g.dx, g.y0 + g.iy[i] * g.dx               # the cell centre (run metres = CRS m)
        r, c = src.index(x, y)
        assert a[r, c] == pytest.approx(obs[i], rel=1e-5)
        assert np.isnan(a).any() or np.isfinite(a).all()
    import pyogrio

    layers = {name for name, _ in pyogrio.list_layers(tmp_path / top / "hexagons.gpkg")}
    assert layers == {"hex_250m", "hex_500m"}
    sites = pd.read_csv(tmp_path / top / "logger_sites.csv")
    assert sites["lon"].iloc[0] == pytest.approx(float(g.lon[0]), abs=1e-6)
    assert sites["lat"].between(-90, 90).all()
    r = client.post("/api/exports", json={"kind": "gis", "project_id": demo["id"],
                                          "params": {"run_id": rid, "layers": ["no_such_layer"]}})
    assert r.status_code == 422


def test_gis_refuses_a_run_without_crs(client, demo, place_run):
    def no_crs(raw):
        raw["data"].pop("crs", None)

    place_run(demo, edit_config=no_crs)
    r = client.post("/api/exports", json={"kind": "gis", "project_id": demo["id"], "params": {"run_id": RUN_ID}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_crs"
    assert client.get(f"/api/projects/{demo['id']}/exports").json() == []


# ---------------------------------------------------------------------------
# results page and report
# ---------------------------------------------------------------------------

def test_page_export_uses_the_packaged_builder(client, demo, fixture_run, wait_job):
    rid, _ = fixture_run
    ex = _export(client, wait_job, demo["id"], "page", {"run_id": rid})
    r = client.get(f"/api/exports/{ex['id']}/download")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.text.startswith("<!doctype html>") and "const DATA = {" in r.text
    assert Path(ex["path"]).name == "results.html"


def _optimize_with_frontier(rd: Path, ratio: float) -> None:
    o = json.loads((rd / "optimize.json").read_text())
    o["pareto"] = {"points": [{"budget": o["budget"], "total_benefit": 100.0},
                              {"budget": 2 * o["budget"], "total_benefit": 100.0 * ratio}]}
    (rd / "optimize.json").write_text(json.dumps(o))


def test_report_preview_sections_and_caveats_from_numbers(client, ctx, demo, fixture_run):
    from sparc.studio.runs.caveats import caveats_for

    rid, rd = fixture_run
    _optimize_with_frontier(rd, 1.2)
    r = client.post(f"/api/projects/{demo['id']}/report/preview",
                    json={"run_id": rid, "sections": ["summary", "caveats", "provenance"]})
    assert r.status_code == 200, r.text
    html = r.json()["html"]
    assert html.startswith("<!doctype html>") and "@media print" in html
    assert 'id="summary"' in html and 'id="caveats"' in html and 'id="provenance"' in html
    assert 'id="accuracy"' not in html and 'id="scenarios"' not in html
    # the summary states the run's own numbers
    m = json.loads((rd / "manifest.json").read_text())
    r2 = m["metrics"]["stacker"]["r2"]
    assert f"explains {100 * r2:.0f}%" in html and "data:image/png;base64," in html
    # caveats: the run's generated caveats plus the frontier read from its numbers - no canned conclusion
    import html as H

    for c in caveats_for(ctx.services["reader"].get(rid)):
        assert H.escape(c) in html
    assert "raises total cooling 1.20×" in html and "diminish sharply" in html
    assert "only slightly" not in html
    _optimize_with_frontier(rd, 1.95)
    html2 = client.post(f"/api/projects/{demo['id']}/report/preview",
                        json={"run_id": rid, "sections": ["caveats"]}).json()["html"]
    assert "raises total cooling 1.95×" in html2 and "barely diminish" in html2
    assert rid in html                                                    # provenance block
    r = client.post(f"/api/projects/{demo['id']}/report/preview", json={"run_id": rid, "sections": ["nope"]})
    assert r.status_code == 422


def test_report_export_html_and_markdown(client, demo, fixture_run, wait_job):
    rid, _ = fixture_run
    sections = ["summary", "accuracy", "validation", "scenarios", "plans", "climate", "equity", "caveats",
                "limitations", "provenance", "findings"]
    ex = _export(client, wait_job, demo["id"], "report", {"run_id": rid, "sections": sections, "format": "html"})
    text = client.get(f"/api/exports/{ex['id']}/download").text
    assert text.startswith("<!doctype html>") and "<svg" in text and "Climate futures" in text
    for s in sections:
        assert f'id="{s}"' in text
    assert "Canopy Increase +10" in text and "No findings are pinned" in text
    ex2 = _export(client, wait_job, demo["id"], "report", {"run_id": rid, "sections": ["summary", "scenarios"],
                                                           "format": "md"})
    md = client.get(f"/api/exports/{ex2['id']}/download").text
    assert md.startswith("# Synthetic city (DEMO)") and "## Scenarios" in md and "| Scenario |" in md
    assert "## Climate" not in md
    r = client.post("/api/exports", json={"kind": "report", "project_id": demo["id"],
                                          "params": {"run_id": rid, "sections": ["scenarios"],
                                                     "result_ids": ["res_nope0000"]}})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["path"] == "params.result_ids"


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------

def test_findings_crud_image_and_export(client, ctx, demo, fixture_run, wait_job):
    rid, _ = fixture_run
    body = {"project_id": demo["id"], "run_id": rid, "view": "runhub.scenarios", "url_state": "?tab=scenarios&s=2",
            "title": "Canopy +10 cools the city", "note_md": "Seen on the **scenarios** tab.",
            "snapshot": {"scenario": "Canopy Increase +10", "mean_delta": -1.0316500390097199, "likely": [-1.58, -0.48]}}
    r = client.post("/api/findings", json=body)
    assert r.status_code == 201, r.text
    f = r.json()
    assert f["id"].startswith("fd_") and f["image_url"] is None and f["position"] == 0
    second = client.post("/api/findings", json={**body, "title": "Second", "snapshot": {"x": 1}}).json()
    assert second["position"] == 1
    # raw image upload, then read it back
    r = client.put(f"/api/findings/{f['id']}/image", content=PNG, headers={"Content-Type": "image/png"})
    assert r.status_code == 200 and r.json() == {"image_url": f"/api/findings/{f['id']}/image"}
    img = client.get(f"/api/findings/{f['id']}/image")
    assert img.status_code == 200 and img.content == PNG and img.headers["content-type"] == "image/png"
    r = client.put(f"/api/findings/{second['id']}/image", content=SVG, headers={"Content-Type": "image/svg+xml"})
    assert r.status_code == 200
    svg = client.get(f"/api/findings/{second['id']}/image")
    assert svg.content == SVG and svg.headers["content-security-policy"] == "sandbox"
    bad = client.put(f"/api/findings/{f['id']}/image", content=b"GIF89a", headers={"Content-Type": "image/gif"})
    assert bad.status_code == 415
    big = client.put(f"/api/findings/{f['id']}/image", content=b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 2 ** 20 + 1),
                     headers={"Content-Type": "image/png"})
    assert big.status_code == 413
    # patch, order, filter
    p = client.patch(f"/api/findings/{second['id']}", json={"position": -1, "title": "First now"}).json()
    assert p["title"] == "First now"
    listing = client.get("/api/findings", params={"project": demo["id"], "run": rid}).json()
    assert [x["id"] for x in listing] == [second["id"], f["id"]]
    assert listing[1]["image_url"] == f"/api/findings/{f['id']}/image"
    mirror = json.loads((Path(demo["dir"]) / "findings" / f"{f['id']}.json").read_text())
    assert mirror["image"] == f"{f['id']}.png" and "image_url" not in mirror
    # export: HTML with the snapshotted numbers and the provenance block
    ex = _export(client, wait_job, demo["id"], "findings", {"format": "html", "ids": [f["id"]]})
    html = client.get(f"/api/exports/{ex['id']}/download").text
    assert "Canopy +10 cools the city" in html and "-1.03165" in html and "<strong>scenarios</strong>" in html
    assert f"Provenance: run {rid}" in html and "Code commit" in html and "Core code SHA-256" in html
    assert "data:image/png;base64," in html and "First now" not in html
    ex2 = _export(client, wait_job, demo["id"], "findings", {"format": "md", "run_id": rid})
    z = _zip(client, ex2)
    md = z.read("findings/findings.md").decode()
    assert "Canopy +10 cools the city" in md and f"images/{f['id']}.png" in md and "Provenance" in md
    assert z.read(f"findings/images/{f['id']}.png") == PNG and f"findings/images/{second['id']}.svg" in z.namelist()
    # delete
    assert client.delete(f"/api/findings/{second['id']}").json() == {"ok": True}
    assert client.get(f"/api/findings/{second['id']}/image").status_code == 404
    assert not (Path(demo["dir"]) / "findings" / f"{second['id']}.json").exists()


def test_bad_images_never_replace_a_good_one(client, demo, fixture_run):
    """A body that is not what its type says is refused before it lands: the finding keeps its image."""
    rid, _ = fixture_run
    f = client.post("/api/findings", json={"project_id": demo["id"], "run_id": rid, "view": "v", "url_state": "",
                                           "title": "t", "snapshot": {}}).json()
    assert client.put(f"/api/findings/{f['id']}/image", content=PNG, headers={"Content-Type": "image/png"}).status_code \
        == 200
    bad = client.put(f"/api/findings/{f['id']}/image", content=b"GIF89a not a png",
                     headers={"Content-Type": "image/png"})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "bad_suffix"
    assert client.get(f"/api/findings/{f['id']}/image").content == PNG
    html_as_svg = client.put(f"/api/findings/{f['id']}/image", content=b"<html><script>alert(1)</script></html>",
                             headers={"Content-Type": "image/svg+xml"})
    assert html_as_svg.status_code == 415
    assert client.get(f"/api/findings/{f['id']}/image").content == PNG
    d = Path(demo["dir"]) / "findings"
    assert sorted(p.name for p in d.iterdir()) == sorted([f"{f['id']}.json", f"{f['id']}.png"])   # no leftovers


def test_other_projects_objects_are_refused(client, demo, fixture_run):
    """Findings, report previews and exports only take objects of their own project."""
    rid, _ = fixture_run
    other = client.post("/api/projects", json={"name": "Other city", "template": "blank"})
    assert other.status_code == 201, other.text
    opid = other.json()["project"]["id"]
    r = client.post("/api/findings", json={"project_id": opid, "run_id": rid, "view": "v", "url_state": "",
                                           "title": "t", "snapshot": {}})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["path"] == "run_id"
    theirs = client.post("/api/findings", json={"project_id": opid, "view": "v", "url_state": "",
                                                "title": "not yours", "snapshot": {"secret": 42}}).json()
    r = client.post(f"/api/projects/{demo['id']}/report/preview",
                    json={"run_id": rid, "sections": ["findings"], "finding_ids": [theirs["id"]]})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["path"] == "finding_ids"
    r = client.post("/api/exports", json={"kind": "findings", "project_id": demo["id"],
                                          "params": {"ids": [theirs["id"]], "format": "html"}})
    assert r.status_code == 422


def test_bundle_leaves_out_links_out_of_the_run(client, demo, fixture_run, wait_job, tmp_path):
    rid, rd = fixture_run
    secret = tmp_path / "secret.txt"
    secret.write_text("not part of the run")
    (rd / "environment.txt").unlink()
    (rd / "environment.txt").symlink_to(secret)
    ex = _export(client, wait_job, demo["id"], "bundle", {"run_id": rid})
    z = _zip(client, ex)
    assert f"{rid}/environment.txt" not in z.namelist() and f"{rid}/manifest.json" in z.namelist()
    assert not any(b"not part of the run" in z.read(n) for n in z.namelist())


def test_reindex_rebuilds_exports_and_findings(client, ctx, demo, fixture_run, wait_job):
    from sparc.studio import db as dbmod

    rid, _ = fixture_run
    f = client.post("/api/findings", json={"project_id": demo["id"], "run_id": rid, "view": "v", "url_state": "",
                                           "title": "kept", "snapshot": {"a": 1}}).json()
    client.put(f"/api/findings/{f['id']}/image", content=PNG, headers={"Content-Type": "image/png"})
    ex = _export(client, wait_job, demo["id"], "bundle", {"run_id": rid, "outputs": ["manifest"]})
    summary = dbmod.reindex(ctx.db, ctx.workspace)
    assert summary["findings"]["findings"] == 1 and summary["exports"]["exports"] == 1
    assert client.get(f"/api/exports/{ex['id']}").json() == ex
    again = client.get("/api/findings", params={"project": demo["id"]}).json()
    assert [x["id"] for x in again] == [f["id"]] and again[0]["snapshot"] == {"a": 1}
    assert client.get(f"/api/findings/{f['id']}/image").content == PNG


# ---------------------------------------------------------------------------
# rows before jobs (packs too), not-ready downloads, deletes
# ---------------------------------------------------------------------------

def test_pack_rows_are_created_before_the_job(client, ctx, demo, fixture_run, wait_job):
    rid, rd = fixture_run
    r = client.post("/api/exports", json={"kind": "plan_pack", "project_id": demo["id"], "params": {"plan_id": "pl_x"}})
    assert r.status_code == 404
    # a comparison whose files are missing: the row exists first, the pack fails, pack_on_finish marks it failed
    cdir = rd / "studio" / "comparisons" / "cmp_test0000"
    cdir.mkdir(parents=True)
    ctx.db.insert("comparisons", {"id": "cmp_test0000", "run_id": rid, "items_json": "[]", "dir": str(cdir),
                                  "summary_json": "{}", "created_utc": "2026-10-02T00:00:00Z"})
    r = client.post("/api/exports", json={"kind": "compare_pack", "project_id": demo["id"],
                                          "params": {"comparison_id": "cmp_test0000"}})
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["export"]["ref"] == "cmp_test0000" and out["export"]["run_id"] == rid
    assert out["job"]["params"] == {"export_id": out["export"]["id"], "comparison_id": "cmp_test0000"}
    wait_job(client, out["job"]["id"], timeout=120)
    ex = client.get(f"/api/exports/{out['export']['id']}").json()
    assert ex["status"] == "failed" and ex["draft"] is False
    assert client.get(f"/api/exports/{ex['id']}/download").json()["error"]["code"] == "not_ready"


def test_download_not_ready_while_queued_and_delete(client, ctx, demo, fixture_run, wait_job):
    rid, _ = fixture_run
    assert client.post("/api/queue/pause").status_code == 200
    r = client.post("/api/exports", json={"kind": "bundle", "project_id": demo["id"],
                                          "params": {"run_id": rid, "outputs": ["manifest"]}})
    ex = r.json()["export"]
    d = client.get(f"/api/exports/{ex['id']}/download")
    assert d.status_code == 409 and d.json()["error"]["code"] == "not_ready"
    assert client.delete(f"/api/exports/{ex['id']}").json()["error"]["code"] == "active"
    client.post(f"/api/jobs/{ex['job_id']}/cancel")
    wait_job(client, ex["job_id"])
    assert client.get(f"/api/exports/{ex['id']}").json()["status"] == "failed"
    assert client.delete(f"/api/exports/{ex['id']}").json() == {"ok": True}
    assert client.get(f"/api/exports/{ex['id']}").status_code == 404
    assert not (Path(demo["dir"]) / "exports" / ex["id"]).exists()
    client.post("/api/queue/resume")


def test_report_reads_results_and_plans_from_their_on_disk_formats(client, ctx, demo, fixture_run):
    """Selected exact results (``results/<id>/summary.json`` + ``cells.parquet``) and plans
    (``plans/<id>/{params,planned,realised}.json`` + ``dose.npy``) appear with their numbers and maps."""
    import pandas as pd

    rid, rd = fixture_run
    n = ctx.services["reader"].get(rid).grid.n
    res_dir = rd / "studio" / "results" / "res_test0001"
    res_dir.mkdir(parents=True)
    (res_dir / "summary.json").write_text(json.dumps({
        "scenario": {"id": "sc_x", "revision": 1, "name": "Shade the hottest blocks"},
        "city": {"estimate": -0.21, "se": 0.04, "lo": -0.29, "hi": -0.13, "confidence": "confident_cools",
                 "phrase": "cools"}, "extrapolated_edited": 0.05,
        "plain": {"headline": "Shading the hottest blocks cools the city by 0.21 °F."}}))
    pd.DataFrame({"id": np.arange(n), "delta": np.linspace(-0.5, 0.0, n).astype("float32")}).to_parquet(
        res_dir / "cells.parquet")
    ctx.db.insert("results", {"id": "res_test0001", "scenario_id": "sc_x", "run_id": rid, "kind": "exact",
                              "dir": str(res_dir), "summary_json": "{}", "created_utc": "2026-10-02T00:00:00Z"})
    pl_dir = rd / "studio" / "plans" / "pl_test0001"
    pl_dir.mkdir(parents=True)
    (pl_dir / "planned.json").write_text(json.dumps({"planned_total": 410.0, "n_cells_treated": 120,
                                                     "mean_dose_treated": 12.5, "gini": 0.7}))
    (pl_dir / "realised.json").write_text(json.dumps({"total": 369.0, "mean_treated": -0.4, "mean_all": -0.05,
                                                      "result_id": "res_test0001"}))
    np.save(pl_dir / "dose.npy", np.where(np.arange(n) % 9 == 0, 10.0, 0.0).astype("float32"))
    ctx.db.insert("plans", {"id": "pl_test0001", "project_id": demo["id"], "run_id": rid, "name": "Canopy budget",
                            "params_json": json.dumps({"lever": "canopy", "budget": 1500}), "summary_json": "{}",
                            "dir": str(pl_dir), "created_utc": "2026-10-02T00:00:00Z"})
    html = client.post(f"/api/projects/{demo['id']}/report/preview",
                       json={"run_id": rid, "sections": ["scenarios", "plans"], "result_ids": ["res_test0001"],
                             "plan_ids": ["pl_test0001"]}).json()["html"]
    assert "Shade the hottest blocks" in html and "−0.21" in html and "−0.29 to −0.13" in html
    assert "Shading the hottest blocks cools the city by 0.21 °F." in html
    assert "Canopy budget" in html and "1,500 on canopy treats 120 cells" in html and "90% of plan" in html
    assert html.count("data:image/png;base64,") == 2                      # the result's ΔT map and the plan's doses


def test_climate_sentences_without_warming_numbers():
    """Projections that carry no warming numbers (an older or partial climate section) are skipped, not fatal."""
    from sparc.studio.exports.narrative import climate_sentences, summary_sentences  # noqa: F401

    assert climate_sentences({"climate": {"projections": [{"label": "SSP2-4.5", "period": "2041-2060"}]}}, "°F") == [
        "This run has no climate projections with warming numbers."]
    m = {"climate": {"n_models": 3, "projections": [
        {"label": "SSP2-4.5", "experiment": "ssp245", "period": "2041-2060", "n_models": 3,
         "warming": {"median": 2.1, "p10": 1.5, "p90": 2.9}},
        {"label": "SSP5-8.5", "experiment": "ssp585", "period": "2081-2100"}]}}
    out = climate_sentences(m, "°F")
    assert out[0].startswith("3 CMIP6 models: under SSP2-4.5 by 2041–2060 summer highs warm by +2.1 °F")
