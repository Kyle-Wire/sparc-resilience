"""J1 on the replay runner (SPEC §14.5 item 1): set up, launch, track, explore the outputs.

The server runs with ``SPARC_STUDIO_RUNNER=replay:tests/studio/fixtures/synth_run``: a "fast"
run replays the recorded synthetic run (its events and its output files) instead of fitting
models, so the journey is deterministic and quick. The project is the synthetic demo made with
the fixture's ``n`` and ``seed`` (``FIXTURE.json``), so its data is byte-identical to the data
the fixture was recorded from and the replayed outputs pass the reader's id and fold checks.

The fixture spans 31 s of events. At the ×20 of api.md §14 the whole replay takes under two
seconds, which leaves no time to open a tab or reload mid-run, so this test replays at ×1
(``SPARC_STUDIO_REPLAY_SPEED``).
"""

from __future__ import annotations

import json
import re
import time

from .conftest import SYNTH_RUN, StudioPage, drag_on, rail_states, wait_for, wait_view_ready

REPLAY_SPEED = "1"
SETUP_STEPS = ["Levers", "Physics", "Inputs", "Scenarios", "Analysis", "About"]
STATE_FOR = {"will_run": "done", "skipped": "disabled", "cached": "cached"}


def _fixture() -> tuple[dict, list[dict]]:
    meta = json.loads((SYNTH_RUN / "FIXTURE.json").read_text("utf-8"))
    events = [json.loads(line) for line in (SYNTH_RUN / "events.jsonl").read_text("utf-8").splitlines() if line]
    return meta, events


def _server_log_lines(server, jid: str) -> list[dict]:
    lines, after = [], None
    while True:
        q = {"limit": 1000, **({"after": after} if after is not None else {})}
        page = server.get(f"/api/jobs/{jid}/logs", params=q)
        lines += page["lines"]
        if len(page["lines"]) < 1000 or page["next_cursor"] == after:
            return lines
        after = page["next_cursor"]


def _brush_histogram(page) -> None:
    """Drag across the middle third of the legend histogram."""
    hist = page.locator('[role=group][aria-label*="Drag to select a range"]').first
    hist.wait_for()
    drag_on(page, hist, (0.3, 0.5), (0.62, 0.5))


def test_e2e_replay(studio_server, studio_page):
    t_start = time.monotonic()
    meta, events = _fixture()
    plan = next(e for e in events if e["type"] == "run.plan")["nodes"]
    server = studio_server(runner=f"replay:{SYNTH_RUN}", env={"SPARC_STUDIO_REPLAY_SPEED": REPLAY_SPEED})
    project = server.create_demo("Replay city", n=meta["n"], seed=meta["seed"])
    pid = project["id"]
    sp: StudioPage = studio_page(server)
    page = sp.page
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    # -- setup pages: the data check, then every step --------------------------------------------
    sp.goto(f"/p/{pid}/setup/data")
    page.get_by_role("button", name="Check data").click()
    page.get_by_text("Points kept").wait_for()
    kept = page.locator(".kpi", has_text="Points kept").inner_text()
    assert re.search(r"\d", kept), kept
    sp.shot("setup data check")
    for step in SETUP_STEPS:
        page.get_by_role("tab", name=step).click()
        page.wait_for_url(re.compile(rf"/setup/{step.lower()}$"))
        page.get_by_role("tab", name=step, selected=True).wait_for()
        sp.shot(f"setup {step}")

    # -- launch: the plan graph is the fixture's plan --------------------------------------------
    sp.goto(f"/p/{pid}/launch")
    nodes = page.locator(".plan-node")
    wait_for(lambda: nodes.count() == len(plan), 30, "the plan graph")
    shown = [(n.get_attribute("data-node"), n.get_attribute("data-state")) for n in nodes.all()]
    assert shown == [(n["id"], n["state"]) for n in plan]
    sp.shot("launch plan")
    page.get_by_role("button", name="Start fast run").click()
    page.wait_for_url(re.compile(r"/jobs/j_[a-z0-9]+$"))
    jid = page.url.rsplit("/", 1)[1]
    job = server.job(jid)
    rid = job["run_id"]
    assert job["kind"] == "run.core" and rid

    # -- Mission Control while the replay runs ----------------------------------------------------
    # The rail lists the plan's stages in order; record the order in which each reaches "done".
    page.get_by_role("tab", name=re.compile(r"^Outputs")).click()
    done_order: list[str] = []
    heat_full = False
    states_seen: dict[str, set] = {}
    predictions_live = False
    while True:
        rail = rail_states(page)
        if rail:
            assert [s for s, _ in rail] == [n["id"] for n in plan]
        for sid, st in rail:
            states_seen.setdefault(sid, set()).add(st)
            if st == "done" and sid not in done_order:
                done_order.append(sid)
        cells = page.locator("table.foldgrid td")
        n_cells = cells.count()
        if n_cells and page.locator("table.foldgrid td[data-status=ok]").count() == n_cells:
            heat_full = True
        if page.get_by_role("tabpanel").get_by_text("predictions.parquet").count():
            predictions_live = server.job(jid)["status"] == "running"
            break
        assert time.monotonic() - t_start < 150, "predictions.parquet never appeared in the Outputs feed"
        page.wait_for_timeout(200)
    assert heat_full, "the fold × model heatmap never filled"
    assert predictions_live, "the replay ended before the Outputs feed showed predictions.parquet"
    sp.shot("mission control mid run")

    # Accuracy opens mid-replay (a second tab) and says the run is still going.
    acc = studio_page(server)
    acc.goto(f"/r/{rid}/accuracy")
    acc.page.locator(".runhub-view[data-view=accuracy]").wait_for()
    acc.page.locator(".runhub-view [data-live=true]").wait_for()
    assert server.job(jid)["status"] == "running"
    acc.page.locator(".runhub-view figure, .runhub-view svg, .runhub-view table").first.wait_for()
    acc.shot("accuracy mid run")

    # Reload Mission Control mid-run: the tracker rebuilds from the log and keeps streaming.
    assert server.job(jid)["status"] == "running"
    page.reload()
    page.locator("[aria-label=Stages] .rail-chip").first.wait_for()
    final = server.wait_job(jid, timeout=120)
    assert final["status"] == "succeeded", final

    def all_done():
        rail = rail_states(page)
        for sid, st in rail:
            states_seen.setdefault(sid, set()).add(st)
            if st == "done" and sid not in done_order:
                done_order.append(sid)
        return rail and all(st == STATE_FOR[n["state"]] for (_, st), n in zip(rail, plan))

    wait_for(all_done, 30, "every stage of the rail to settle")
    expected = [n["id"] for n in plan if n["state"] == "will_run"]
    assert done_order == expected, f"stages reached done out of order: {done_order}"
    assert "running" in states_seen["S2_S3"] or "running" in states_seen["S4"]
    sp.shot("mission control done")

    # No gap in the log after the reload: the client's lines are the server's, in order.
    page.get_by_role("tab", name="Logs").click()
    server_lines = _server_log_lines(server, jid)
    count = page.get_by_role("list", name=re.compile(r"^\d+ log lines$"))
    wait_for(lambda: count.get_attribute("aria-label") == f"{len(server_lines)} log lines", 20,
             f"the log list to hold {len(server_lines)} lines")
    page.get_by_role("button", name="Copy", exact=True).click()
    copied = page.evaluate("navigator.clipboard.readText()").splitlines()
    assert [ln.split(": ", 1)[1] for ln in copied] == [ln["msg"] for ln in server_lines]
    cursors = [ln["cursor"] for ln in server_lines]
    assert cursors == sorted(cursors) and len(set(cursors)) == len(cursors)

    # -- every run tab -----------------------------------------------------------------------------
    sp.goto(f"/r/{rid}")
    tabs = page.get_by_role("navigation", name="Run views").get_by_role("link")
    hrefs = [(t.inner_text().split("\n")[0].strip(), t.get_attribute("href")) for t in tabs.all()]
    assert len(hrefs) >= 15
    for label, href in hrefs:
        sp.goto(href)
        wait_view_ready(page, f"{label} tab", 30)
        alerts = page.locator("main [role=alert]")
        assert not alerts.count(), f"{label}: {alerts.all_inner_texts()}"
        sp.shot(f"tab {label}")

    # -- every map layer ---------------------------------------------------------------------------
    sp.goto(f"/r/{rid}/map")
    picker = page.get_by_role("group", name="Choose a map layer")
    picker.wait_for()
    themes = picker.get_by_role("group", name="Map theme").get_by_role("button")
    n_layers = 0
    groups = server.get(f"/api/runs/{rid}/layers")["groups"]
    for i, group in enumerate(groups):
        themes.nth(i).click()
        select = picker.locator("select").last
        for layer in group["layers"]:
            select.select_option(layer["key"])
            page.get_by_role("heading", name=layer["label"], exact=True).first.wait_for()
            n_layers += 1
        if i in (0, len(groups) - 1):
            sp.shot(f"map theme {group['label']}")
    assert n_layers == sum(len(g["layers"]) for g in groups) and n_layers >= 50

    # -- brush the legend histogram → the selection count on the map and in Region stats ----------
    sp.goto(f"/r/{rid}/map?tool=region")
    _brush_histogram(page)
    status = page.get_by_test_id("selection-status")
    wait_for(lambda: re.search(r"([\d,]+) cells", status.inner_text()), 20, "the brushed selection")
    n_sel = int(re.search(r"([\d,]+) cells", status.inner_text()).group(1).replace(",", ""))
    assert 0 < n_sel < server.get(f"/api/runs/{rid}/grid")["n"]
    summary = page.get_by_label("Selection summary")
    summary.wait_for()
    wait_for(lambda: f"{n_sel:,}" in summary.inner_text(), 20, "Region stats to count the selection")
    sp.shot("brushed selection region stats")

    # -- pin a finding → export the findings as HTML -----------------------------------------------
    page.get_by_role("button", name="Pin Region stats to Findings").click()
    wait_for(lambda: server.get("/api/findings", params={"project": pid}), 20, "the pinned finding")
    sp.goto(f"/p/{pid}/findings")
    page.get_by_role("list", name="Findings").get_by_role("listitem").first.wait_for()
    page.get_by_role("button", name="Export HTML").click()
    download = page.get_by_role("group", name="Export findings").get_by_role("link", name="Download")
    download.wait_for(timeout=60_000)
    with page.expect_download() as dl:
        download.click()
    html = open(dl.value.path(), encoding="utf-8").read()
    assert html.lstrip().lower().startswith("<!doctype html") and "Region stats" in html
    sp.shot("findings exported")
    assert time.monotonic() - t_start < 180, f"the replay journey took {time.monotonic() - t_start:.0f} s"
