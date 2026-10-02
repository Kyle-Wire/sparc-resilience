"""J1–J3 with the real pipeline (SPEC §14.5 item 2): a ``--fast`` run of the n=48 synthetic city,
cancel and resume, then the Scenario Lab end to end.

Cancel and resume. The spec cancels during S2–S3 and expects the resume to show cached stages.
The first checkpoint is written when S2–S3 ends, so a resume after a cancel *during* S2–S3 has
nothing to reuse and starts over. The test therefore cancels twice: during S2–S3 (cancelled
within 30 s, then resumed), and once more after that checkpoint, whose resume shows the stages
it covers as cached before it succeeds.

The Lab part draws a +10 canopy circle (emulator preview), runs it exactly (fold ticks,
plain-language card with the likely range), saves and forks it, compares it with a configured
scenario (paired SE), explores the climate with a changed threshold, moves the budget slider of
a plan (the Pareto curve and the planned totals change) and verifies the plan, then exports a
decision pack and a GeoTIFF.
"""

from __future__ import annotations

import io
import re
import time
import zipfile

from .conftest import StudioPage, dd_of, drag_on, rail_states, wait_for

N_POINTS = 48
CHECKPOINT_KEY = {"S1": "S3", "S2_S3": "S3", "baselines": "baselines", "S4": "S4", "S5": "S5", "climate": "climate",
                  "S6": "S6"}


def _rail(page) -> dict[str, str]:
    return dict(rail_states(page))


def _job_from_url(page) -> str:
    page.wait_for_url(re.compile(r"/jobs/j_[a-z0-9]+$"))
    return page.url.rsplit("/", 1)[1]


def _cancel_when(page, server, jid: str, ready, what: str) -> float:
    """Wait until ``ready(rail)``, press Cancel; returns the seconds until the job ended cancelled."""
    wait_for(lambda: ready(_rail(page)), 300, what, interval=0.2)
    t0 = time.monotonic()
    page.get_by_role("button", name="Cancel", exact=True).click()
    job = server.wait_job(jid, timeout=90)
    assert job["status"] == "cancelled", job
    page.get_by_role("button", name="Resume").wait_for()
    return time.monotonic() - t0


def _newest_job(server, kind: str) -> dict:
    return server.get("/api/jobs", params={"kind": kind, "limit": 1})["items"][0]


def test_e2e_real_fast(studio_server, studio_page):
    t_start = time.monotonic()
    server = studio_server()
    pid = server.create_demo("Fast city", n=N_POINTS)["id"]
    sp: StudioPage = studio_page(server)
    page = sp.page

    # -- launch, cancel during S2–S3, resume ----------------------------------------------------------
    sp.goto(f"/p/{pid}/launch")
    page.get_by_role("button", name="Start fast run").click()
    jid = _job_from_url(page)
    rid = server.job(jid)["run_id"]
    took = _cancel_when(page, server, jid, lambda r: r.get("S2_S3") == "running", "S2–S3 to start")
    assert took < 30, f"the cancel took {took:.0f} s"
    assert server.get(f"/api/runs/{rid}")["run"]["status"] == "cancelled"
    sp.shot("cancelled during S2 S3")

    page.get_by_role("button", name="Resume").click()
    page.wait_for_url(lambda url: not url.endswith(jid))
    jid2 = _job_from_url(page)
    assert server.job(jid2)["run_id"] == rid

    # cancel again once the S2–S3 checkpoint exists (S4 running), so the resume has stages to reuse
    _cancel_when(page, server, jid2, lambda r: r.get("S4") == "running", "S4 to start")
    done = set(server.get(f"/api/runs/{rid}")["checkpoint"]["done"])
    reused = [s for s, key in CHECKPOINT_KEY.items() if key in done]
    assert {"S1", "S2_S3"} <= set(reused), done
    page.get_by_role("button", name="Resume").click()
    page.wait_for_url(lambda url: not url.endswith(jid2))
    jid3 = _job_from_url(page)
    wait_for(lambda: all(_rail(page).get(s) == "cached" for s in reused), 60, f"{reused} shown as cached")
    sp.shot("resumed with cached stages")
    final = server.wait_job(jid3, timeout=600)
    assert final["status"] == "succeeded", final
    wait_for(lambda: _rail(page).get("finish") == "done", 30, "the rail to finish")
    assert all(_rail(page).get(s) == "cached" for s in reused)
    sp.shot("run succeeded")

    # -- the Lab: build the emulator, load the engine ---------------------------------------------------
    sp.goto(f"/r/{rid}/lab")
    page.get_by_role("button", name=re.compile(r"^Build emulator")).click()
    wait_for(lambda: server.get("/api/jobs", params={"kind": "post.emulator", "limit": 1})["items"], 30, "the emulator job")
    emulator = server.wait_job(_newest_job(server, "post.emulator")["id"], timeout=300)
    assert emulator["status"] == "succeeded", emulator
    wait_for(lambda: not page.get_by_role("button", name=re.compile(r"^Build emulator")).count(), 60,
             "the Lab to see the emulator")
    page.get_by_role("button", name="Open engine").click()
    wait_for(lambda: server.get(f"/api/runs/{rid}/engine")["state"] == "ready", 300, "the engine to load", interval=0.5)
    page.get_by_text("engine ready").first.wait_for()
    sp.shot("lab engine ready")

    # -- draw a circle with +10 canopy → preview --------------------------------------------------------
    page.get_by_role("button", name="Add edit").click()
    page.wait_for_url(re.compile(r"/lab/s/sc_[a-z0-9]+"))      # the first autosave creates the scenario
    sid = re.search(r"/lab/s/(sc_[a-z0-9]+)", page.url).group(1)
    page.wait_for_timeout(500)
    stage = page.locator(".map-stage[role=application]").first
    stage.scroll_into_view_if_needed()
    page.get_by_role("button", name="Select a circle").click()
    drag_on(page, stage, (0.5, 0.5), (0.63, 0.5))
    row = page.locator("article.edit-row").first
    row.get_by_text(re.compile(r"^Map: Circle")).wait_for()
    row.get_by_role("button", name="Use", exact=True).click()
    amount = row.locator("#edit-0-amount")
    amount.fill("10")
    amount.press("Tab")
    n_all = server.get(f"/api/runs/{rid}/grid")["n"]

    def circle_count():
        if "circle" not in row.locator(".sel-desc").inner_text().lower():
            return None
        m = re.match(r"([\d,]+) cells", row.get_by_test_id("sel-count").inner_text())
        n = m and int(m.group(1).replace(",", ""))
        return n if n and n != n_all else None     # the all-cells count until the circle is counted

    n_circle = wait_for(circle_count, 30, "the circle's cell count")
    assert 0 < n_circle < n_all
    preview = page.get_by_role("region", name="Preview", exact=True)
    edited = dd_of(preview, "Edited cells")
    wait_for(lambda: edited.count() and edited.inner_text().strip() == f"{n_circle:,}", 60, "the preview of the circle")
    cooled = dd_of(preview, "Mean in edited cells").inner_text()
    assert cooled.startswith("−"), f"a +10 canopy circle should cool in the preview: {cooled}"
    sp.shot("preview of the circle")

    # name it (the draft autosaves)
    page.get_by_label("Name", exact=True).fill("Circle +10 canopy")
    wait_for(lambda: server.get(f"/api/scenarios/{sid}")["doc"]["name"] == "Circle +10 canopy", 30, "the autosave")

    # -- Run exact → fold ticks → plain-language card ---------------------------------------------------
    page.get_by_role("button", name="Run exact").click()
    wait_for(lambda: server.get("/api/jobs", params={"kind": "engine.scenario", "limit": 1})["items"], 30, "the exact job")
    exact = _newest_job(server, "engine.scenario")
    exact = server.wait_job(exact["id"], timeout=300)
    assert exact["status"] == "succeeded", exact
    ticks = [e for e in server.get(f"/api/jobs/{exact['id']}/events", params={"types": "tick", "limit": 500})["events"]
             if e.get("unit") == "engine_pass"]
    n_folds = ticks[-1]["n"]
    assert n_folds >= 2 and [t["k"] for t in ticks] == list(range(1, n_folds + 1)), ticks
    headline = page.locator(".result-inspector .plain-result .headline")
    headline.wait_for(timeout=60_000)
    text = headline.inner_text()
    assert re.search(r"^Cools the edited area by .+ \(likely range .+\)\.$", text), text
    sp.shot("exact result plain card")
    results = [r for r in server.get(f"/api/scenarios/{sid}")["results"] if r["run_id"] == rid and r["kind"] == "exact"]
    assert results and results[-1]["has_folds"]
    result_id = max(results, key=lambda r: r["created_utc"])["id"]

    # -- fork it from the library -------------------------------------------------------------------------
    sp.goto(f"/r/{rid}/lab/library")
    page.get_by_label("More actions for Circle +10 canopy").select_option("fork")
    page.wait_for_url(lambda url: "/lab/s/" in url and sid not in url)
    fork_sid = re.search(r"/lab/s/(sc_[a-z0-9]+)", page.url).group(1)
    fork = server.get(f"/api/scenarios/{fork_sid}")
    assert fork["doc"]["edits"] == server.get(f"/api/scenarios/{sid}")["doc"]["edits"]
    sp.shot("forked scenario")

    # -- compare with a configured scenario: paired SE ------------------------------------------------
    configured = server.get(f"/api/runs/{rid}/scenarios")["configured"]
    other = next(c for c in configured if "canopy" in c["slug"])
    sp.goto(f"/r/{rid}/lab/compare")
    add = page.get_by_label("Add an item")
    add.select_option(f"res:{result_id}")
    add.select_option(f"configured:{other['slug']}")
    pairs = page.get_by_role("table", name=re.compile(r"Pairwise differences"))
    pairs.wait_for(timeout=120_000)
    rerun = page.get_by_role("button", name=re.compile(r"^Re-run"))
    if rerun.count():      # the configured scenario has no per-fold results yet: re-run it exactly
        rerun.first.click()
        wait_for(lambda: server.get("/api/jobs", params={"kind": "engine.rerun_configured", "limit": 1})["items"], 30,
                 "the configured re-run job")
        rerun_job = server.wait_job(_newest_job(server, "engine.rerun_configured")["id"], timeout=300)
        assert rerun_job["status"] == "succeeded", rerun_job
    pair_row = pairs.locator("tbody tr").first
    wait_for(lambda: "paired" in pair_row.inner_text().lower() and "needs exact" not in pair_row.inner_text().lower(), 180,
             "the paired SE of the pair")
    assert re.search(r"[−+]?\d[\d.]* ± \d[\d.]* °F", pair_row.inner_text()), pair_row.inner_text()
    sp.shot("compare paired SE")

    # -- climate explore with a threshold change -------------------------------------------------------
    sp.goto(f"/r/{rid}/lab/climate")
    page.get_by_role("heading", name=re.compile(r"Share of cells at or above")).first.wait_for(timeout=120_000)
    thresholds = page.get_by_label("Thresholds (°F)")
    thresholds.fill("92")
    thresholds.press("Enter")
    page.get_by_role("heading", name="Share of cells at or above 92.0 °F").wait_for(timeout=60_000)
    sp.shot("climate threshold 92")

    # -- plan: the budget slider moves the Pareto curve and the totals; save and verify ---------------
    sp.goto(f"/r/{rid}/lab/plans")
    benefit = page.locator(".kpi", has_text="Planned benefit").first.locator(".value")
    wait_for(lambda: re.search(r"\d", benefit.inner_text()), 60, "the plan preview")
    pareto = page.get_by_role("figure").filter(has=page.get_by_text("Benefit vs budget")).first
    before = (benefit.inner_text(), pareto.inner_html())
    slider = page.get_by_label(re.compile(r"^Budget"))
    lo, hi = float(slider.get_attribute("min") or 0), float(slider.get_attribute("max"))
    slider.fill(str(round(lo + 0.5 * (hi - lo))))
    wait_for(lambda: (benefit.inner_text(), pareto.inner_html()) != before and benefit.inner_text() != before[0], 60,
             "the plan preview to follow the slider")
    sp.shot("plan slider moved")
    page.get_by_label("Name", exact=True).fill("Canopy plan")
    page.get_by_role("button", name="Save plan").click()
    page.wait_for_url(re.compile(r"/lab/plans/pl_[a-z0-9]+"))
    plid = page.url.rsplit("/", 1)[1]
    wait_for(lambda: server.get(f"/api/plans/{plid}").get("realised"), 300, "the plan verification", interval=0.5)
    realised = page.locator(".kpi", has_text="Realised (exact)").first.locator(".value")
    wait_for(lambda: re.search(r"\d", realised.inner_text()), 60, "the realised total on the page")
    sp.shot("plan verified")

    # -- decision pack of the exact result -------------------------------------------------------------
    sp.goto(f"/r/{rid}/lab/s/{sid}")
    page.get_by_role("button", name="Decision pack").click()
    wait_for(lambda: server.get(f"/api/projects/{pid}/exports"), 30, "the export record")
    pack = next(e for e in server.get(f"/api/projects/{pid}/exports") if e["kind"] == "decision_pack")
    wait_for(lambda: server.get(f"/api/exports/{pack['id']}")["status"] in ("ready", "failed"), 300, "the decision pack")
    assert server.get(f"/api/exports/{pack['id']}")["status"] == "ready"
    sp.goto(f"/p/{pid}/exports")
    history = page.get_by_role("table", name="Export history")
    link = history.locator(f"tr[data-export='{pack['id']}']").get_by_role("link", name="Download")
    link.wait_for()
    assert link.get_attribute("href") == f"/api/exports/{pack['id']}/download"
    with page.expect_download() as dl:
        link.click()
    with zipfile.ZipFile(dl.value.path()) as z:
        names = {n.rsplit("/", 1)[-1] for n in z.namelist()}
        assert {"brief.html", "cells.csv", "delta.tif", "README.txt"} <= names, names
        tif = next(n for n in z.namelist() if n.endswith("delta.tif"))
        assert z.read(tif)[:4] in (b"II*\x00", b"MM\x00*")
    sp.shot("decision pack exported")

    # -- GeoTIFF of a map layer ----------------------------------------------------------------------------
    sp.goto(f"/r/{rid}/map")
    with page.expect_download() as dl:
        page.get_by_role("link", name="GeoTIFF").click()
    data = open(dl.value.path(), "rb").read()
    assert data[:4] in (b"II*\x00", b"MM\x00*"), data[:8]
    try:
        import rasterio
    except ImportError:
        rasterio = None
    if rasterio is not None:
        with rasterio.open(io.BytesIO(data)) as ds:
            assert ds.crs is not None and ds.width > 1 and ds.height > 1
    sp.shot("map geotiff")
    assert time.monotonic() - t_start < 15 * 60
