"""Studies from the browser (SPEC §14.5 item 3; nightly, marked ``slow``).

* A placebo suite with only the ``shift`` kind at coarse 60 m, on the n=96 synthetic city
  (≈1,600 coarse cells; the n=40/48 cities fall under 200 cells at 120 m and skip MGWR tuning):
  launched from the run's Validation tab, tracked in Mission Control (one child run), ending with
  a verdict in the placebo table.
* The project-wide effect benchmark on an n=48 synthetic city, launched from the Studies hub.
"""

from __future__ import annotations

import re

import pytest

from .conftest import StudioPage, wait_for

pytestmark = pytest.mark.slow


def _study_job(server, kind: str) -> dict:
    wait_for(lambda: server.get("/api/jobs", params={"kind": kind, "limit": 1})["items"], 60, f"the {kind} job")
    return server.get("/api/jobs", params={"kind": kind, "limit": 1})["items"][0]


def test_e2e_placebo_shift(studio_server, studio_page):
    server = studio_server()
    pid = server.create_demo("Placebo city", n=96)["id"]
    launched = server.post(f"/api/projects/{pid}/runs", {"mode": "fast"})
    rid = launched["run"]["id"]
    assert server.wait_job(launched["job"]["id"], timeout=900)["status"] == "succeeded"

    sp: StudioPage = studio_page(server)
    page = sp.page
    sp.goto(f"/r/{rid}/validation")
    card = page.locator("[aria-label='Placebo tests']").first
    card.wait_for()
    card.get_by_label(re.compile(r"^Random field")).uncheck()
    card.get_by_label(re.compile(r"^Rotated layers")).uncheck()
    assert card.get_by_label(re.compile(r"^Shifted layers")).is_checked()
    cell = card.get_by_label("Grid for the re-fits")
    cell.fill("60")
    cell.press("Tab")
    sp.shot("placebo form")
    card.get_by_role("button", name="Run placebo tests").click()

    job = _study_job(server, "study.placebo")
    assert job["params"]["kinds"] == ["shift"] and job["params"]["coarse_m"] == 60, job["params"]
    sp.goto(f"/jobs/{job['id']}")
    page.get_by_role("link", name=re.compile(r"^\d{8}-\d{6}-")).first.wait_for(timeout=600_000)   # the child run's row
    sp.shot("placebo mission control")
    done = server.wait_job(job["id"], timeout=3600)
    assert done["status"] == "succeeded", done

    view = server.get(f"/api/studies/{job['study_id']}/view")
    placebo_rows = [r for r in view["rows"] if r.get("placebo")]
    assert placebo_rows and all(r["kind"] == "shift" for r in placebo_rows)
    assert all(r.get("verdict") for r in placebo_rows), placebo_rows
    assert view.get("n_placebos") == len(placebo_rows)
    children = [c for c in view.get("children") or [] if c.get("run_id")]
    assert [c["kind"] for c in children] == ["shift"], view.get("children")
    assert server.get(f"/api/runs/{children[0]['run_id']}")["run"]["status"] == "complete"

    sp.goto(f"/r/{rid}/validation")
    table = page.get_by_role("table", name="Placebo verdicts")
    table.wait_for(timeout=60_000)
    text = table.inner_text()
    assert "shift" in text and "verdict pending" not in text, text
    sp.shot("placebo verdict")


def test_e2e_benchmark(studio_server, studio_page):
    server = studio_server()
    pid = server.create_demo("Benchmark city", n=48)["id"]
    sp: StudioPage = studio_page(server)
    page = sp.page
    sp.goto(f"/p/{pid}/studies")
    card = page.locator("[aria-label='Effect benchmark']").first
    card.wait_for()
    size = card.get_by_label("City size (cells per side)")
    size.fill("48")
    size.press("Tab")
    card.get_by_role("button", name="Run effect benchmark").click()
    job = _study_job(server, "study.benchmark")
    assert job["params"].get("n") == 48, job["params"]
    done = server.wait_job(job["id"], timeout=3600)
    assert done["status"] == "succeeded", done
    view = server.get(f"/api/studies/{job['study_id']}/view")
    assert view.get("runs"), view
    for name, r in view["runs"].items():
        assert "stack" in r and r["models"], (name, r)
    sp.goto(f"/studies/{job['study_id']}")
    page.get_by_text("Effect share recovered on the synthetic city").first.wait_for(timeout=60_000)
    sp.shot("benchmark result")
