"""The Providence example with its recorded fast run (SPEC §14.5 item 4; nightly, marked ``slow``).

Needs the recorded runs: ``$SPARC_PROVIDENCE_RUNS`` (a folder holding ``providence_uhi_fast``),
else the checkout's ``output/core/providence``; skipped without them (``output/`` is gitignored).
From a checkout with outputs the example's own "import existing runs" option brings the run in
place. From ``$SPARC_PROVIDENCE_RUNS`` the run folder is first copied into the test's temporary
folder (nothing is ever written next to the recorded run) and imported with pickle trust and the
example project's config.

The journey: every tab of ``providence_uhi_fast`` renders without an error, stale outputs carry
their badge, the engine opens, one exact scenario runs, and the emulator preview the Lab shows
is the server's preview for the same edits (summary and per-cell array agree).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from .conftest import RUN_FULL_MODEL, ROOT, StudioPage, dd_of, lab_open_engine, lab_pick_lever, visit_run_pages, wait_for

pytestmark = pytest.mark.slow
RUN = "providence_uhi_fast"


def _runs_source() -> tuple[Path | None, bool]:
    env = os.environ.get("SPARC_PROVIDENCE_RUNS")
    d = Path(env) if env else ROOT / "output" / "core" / "providence"
    return (d if (d / RUN / "manifest.json").is_file() else None), not env


def _unpack(resp) -> dict:
    out = {}
    for o in json.loads(resp.headers["X-SPARC-Offsets"]):
        dt = np.dtype({"float32": "<f4", "uint8": "u1"}[o["dtype"]])
        out[o["name"]] = np.frombuffer(resp.content[o["offset"]:o["offset"] + dt.itemsize * o["length"]], dtype=dt)
    return out


def test_e2e_providence(studio_server, studio_page, tmp_path):
    src, in_repo = _runs_source()
    if src is None:
        pytest.skip(f"no recorded {RUN} (set SPARC_PROVIDENCE_RUNS)")
    server = studio_server()
    sp: StudioPage = studio_page(server)
    page = sp.page

    # -- open the example (importing the runs in place when this checkout has them) -----------------
    sp.goto("/")
    page.get_by_role("button", name="Open Providence example").click()
    dialog = page.get_by_role("dialog")
    importing = dialog.get_by_label(re.compile(r"Import the existing Providence runs"))
    importing.set_checked(in_repo)
    dialog.get_by_role("button", name="Create").click()
    page.wait_for_url(re.compile(r"/p/p_[a-z0-9]+$"), timeout=180_000)
    pid = page.url.rsplit("/", 1)[1]
    project = server.get(f"/api/projects/{pid}")["project"]
    if in_repo:
        runs = server.get(f"/api/projects/{pid}/runs")["items"]
        rid = next(r["id"] for r in runs if Path(server.get(f"/api/runs/{r['id']}")["header"]["run_dir"]).name == RUN)
    else:
        copy = tmp_path / RUN
        shutil.copytree(src / RUN, copy)
        rid = server.post("/api/runs/import", {"dir": str(copy), "project_id": pid, "config_path": project["config_path"],
                                               "trust_pickles": True})["id"]
    sp.shot("providence project")

    # -- every tab and section renders ------------------------------------------------------------------
    stale_marks = 0
    for label, _href in visit_run_pages(sp, rid, 120):
        alerts = page.locator("main [role=alert]")
        assert not alerts.count(), f"{label}: {alerts.all_inner_texts()}"
        stale_marks += page.locator("main").get_by_text(re.compile(r"\bstale\b", re.I)).count()
        sp.shot(f"providence tab {label}")
    outputs = server.get(f"/api/runs/{rid}/outputs")["outputs"]
    stale = [o["id"] for o in outputs if o["state"] == "stale"]
    assert stale, "the recorded run has no stale output to badge"
    assert stale_marks > 0, f"stale outputs {stale} carry no badge on any tab"

    # -- the engine opens; one exact scenario -----------------------------------------------------------
    sp.goto(f"/r/{rid}/lab")
    lab_open_engine(page)
    engine = wait_for(lambda: (e := server.get(f"/api/runs/{rid}/engine"))["state"] in ("ready", "error") and e, 1200,
                      "the engine to load", interval=1)
    assert engine["state"] == "ready", engine
    lab_pick_lever(page)                                       # step 01 creates the change (whole city)
    page.wait_for_url(re.compile(r"/lab/s/sc_[a-z0-9]+"))
    sid = re.search(r"/lab/s/(sc_[a-z0-9]+)", page.url).group(1)
    row = page.locator("article.edit-row").first
    amount = row.locator("#edit-0-amount")
    amount.fill("10")
    amount.press("Tab")

    # -- the emulator preview the page shows is the server's preview of the same edits ----------------
    preview = page.get_by_role("region", name="Preview", exact=True)
    area = preview.locator("[data-n-edited]")                 # "Area changed" (the count rides on data-n-edited)
    wait_for(lambda: area.count() and int(area.get_attribute("data-n-edited")) > 0, 120, "the preview")
    wait_for(lambda: server.get(f"/api/scenarios/{sid}")["doc"]["edits"]
             and server.get(f"/api/scenarios/{sid}")["doc"]["edits"][0].get("amount") == 10, 30, "the autosave")
    edits = server.get(f"/api/scenarios/{sid}")["doc"]["edits"]
    resp = server.http.post(f"/api/runs/{rid}/preview", json={"edits": edits, "request_seq": 10_000})
    assert resp.status_code == 200, resp.text
    summary = json.loads(resp.headers["X-SPARC-Summary"])
    arrays = _unpack(resp)
    edited = np.unpackbits(arrays["edited"], bitorder="little")[:arrays["delta"].size].astype(bool)
    assert int(edited.sum()) == summary["n_edited"]
    assert float(np.nanmean(arrays["delta"])) == pytest.approx(summary["mean"], abs=1e-6)
    assert float(np.nanmean(arrays["delta"][edited])) == pytest.approx(summary["edited_mean"], abs=1e-6)
    assert int(area.get_attribute("data-n-edited")) == summary["n_edited"]
    # the big number "About −0.42 °F" (where you made changes) carries the full-precision mean
    shown_mean = preview.locator("[data-edited-mean]")
    wait_for(lambda: float(shown_mean.get_attribute("data-edited-mean")) == pytest.approx(summary["edited_mean"], abs=1e-6),
             30, "the preview of the saved edits")
    big = dd_of(preview, "where you made changes").inner_text().replace("−", "-")
    if abs(summary["edited_mean"]) >= 0.005:                   # below that it reads "No noticeable change"
        assert float(re.search(r"-?\d+(\.\d+)?", big).group(0)) == pytest.approx(summary["edited_mean"], abs=0.051)
    sp.shot("providence preview")

    page.get_by_role("button", name=RUN_FULL_MODEL).click()
    headline = page.locator(".result-inspector .plain-result .headline")
    headline.wait_for(timeout=1_200_000)
    assert "likely range" in headline.inner_text()
    sp.shot("providence exact result")
