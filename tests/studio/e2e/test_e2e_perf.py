"""Performance budgets of SPEC §14.5 (nightly, marked ``slow``) on a 54,701-cell synthetic run.

| Check                          | Target              | How                                                        |
|--------------------------------|---------------------|------------------------------------------------------------|
| Layer endpoint, warm           | < 50 ms             | median of warm ``layers/<key>.bin`` fetches, live server   |
| Layer switch                   | < 50 ms             | browser: layer change → legend shows it, next frame drawn  |
| Recolour                       | < 10 ms             | browser: range change → next frame (the colour pass)       |
| Preview                        | < 150 ms            | ``POST /preview`` on the Providence fast run (skipped w/o) |
| Selection resolve              | < 30 ms             | median of ``selection/resolve`` (circle), live server      |
| Run list (200 runs)            | < 100 ms            | median of ``GET /api/runs?limit=200`` with 200 runs        |
| Tailer throughput              | ≥ 2,000 events/s    | read + validate + reduce a 20,000-event log in process     |
| Tracker with 20,000 events     | no frame > 100 ms   | browser long-task observer while Mission Control streams   |
| SPA initial bundle             | ≤ 200 kB gzip       | entry script + preloads + CSS of the built ``index.html``  |

Timings are medians over repeats so one scheduling hiccup on a shared machine does not fail
the night.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import ROOT, SYNTH_RUN, TOKEN, VIEWPORT, StudioPage, rail_states, start_studio, wait_for

pytestmark = pytest.mark.slow

N_CELLS = 54_701
N_EVENTS = 20_000
STATIC = ROOT / "sparc" / "studio" / "static"


def _median_ms(fn, repeat: int = 9) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def _big_run(root: Path) -> tuple[Path, Path]:
    """A finished run of the n=281 synthetic demo city (54,901 cells, the size of Providence's 54,701).

    Only the files the browse and selection paths read are written (predictions and manifest),
    with the ids, coordinates and target the core loader rebuilds from the project's config, so
    the import's id check passes. Returns ``(run_dir, config_path)``.
    """
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data
    from sparc.core.synthetic import write_demo_project

    config = Path(write_demo_project(root / "big_city", n=281, seed=0)["config_path"])
    data = load_core_data(load_core_config(config))
    n = len(data.ids)
    assert abs(n - N_CELLS) < 1000, n
    rng = np.random.default_rng(0)
    target = np.asarray(data.target_raw, float)
    pred = target + rng.normal(0, 0.5, n)
    rd = root / "big_run"
    rd.mkdir(parents=True)
    pd.DataFrame({"id": np.asarray(data.ids), "x_m": np.asarray(data.x, float), "y_m": np.asarray(data.y_coord, float),
                  "fold": rng.integers(0, 5, n), "target": target, "pred": pred, "pi_lo": pred - 1,
                  "pi_hi": pred + 1}).to_parquet(rd / "predictions.parquet")
    (rd / "manifest.json").write_text(json.dumps({
        "name": "big", "created_utc": "2026-10-01T12:00:00+00:00", "fast_mode": False,
        "qa": {"cell_m": 30.0}, "metrics": {"stacker": {"r2": 0.8, "rmse": 0.5}}}))
    return rd, config


def _small_run(root: Path, i: int) -> Path:
    rd = root / f"run_{i:03d}"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({
        "name": f"run {i}", "created_utc": f"2026-09-{1 + i % 28:02d}T{i % 24:02d}:00:00+00:00", "fast_mode": True,
        "metrics": {"stacker": {"r2": 0.5 + i / 1000, "rmse": 1.0}}}))
    return rd


def _many_events_fixture(root: Path) -> Path:
    """The synthetic replay fixture padded to 20,000 events (ticks and info logs) at the start of S4.

    S4 starts ≈19 s into the recording, which leaves the page time to load and settle before the
    burst arrives, so the frames measured are the streaming frames.
    """
    fx = root / "fixture20k"
    shutil.copytree(SYNTH_RUN, fx)
    events = [json.loads(line) for line in (SYNTH_RUN / "events.jsonl").read_text("utf-8").splitlines() if line]
    s4 = next(i for i, e in enumerate(events) if e["type"] == "stage.start" and e.get("stage") == "S4")
    at = next(i for i, e in enumerate(events) if i > s4 and e["type"] == "tick" and e.get("k", 0) < e.get("n", 0))
    tick, nxt = events[at], events[at + 1]
    pad, extra = N_EVENTS - len(events), []
    for j in range(pad):
        ts = tick["ts"] + (nxt["ts"] - tick["ts"]) * (j + 1) / (pad + 1)
        if j % 2:
            extra.append({**tick, "ts": ts})
        else:
            extra.append({k: tick[k] for k in ("v", "lvl", "span", "parent", "path", "ctx", "job", "pid") if k in tick}
                         | {"type": "log", "ts": ts, "lvl": "info", "logger": "sparc.core.perf", "level": "INFO",
                            "msg": f"padding line {j}"})
    events = events[:at + 1] + extra + events[at + 1:]
    for i, e in enumerate(events):
        e["seq"] = i + 1
    (fx / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), "utf-8")
    return fx


@pytest.fixture(scope="module")
def perf_site(tmp_path_factory):
    root = tmp_path_factory.mktemp("perf")
    fixture = _many_events_fixture(root)
    server = start_studio(root / "ws", runner=f"replay:{fixture}", env={"SPARC_STUDIO_REPLAY_SPEED": "2"})
    try:
        big_dir, big_config = _big_run(root)
        big = server.post("/api/runs/import", {"dir": str(big_dir), "config_path": str(big_config)})["id"]
        from sparc.core.synthetic import write_demo_project

        small_config = str(write_demo_project(root / "small_city", n=24, seed=0)["config_path"])
        for i in range(199):
            server.post("/api/runs/import", {"dir": str(_small_run(root / "many", i)), "config_path": small_config})
        yield {"server": server, "big": big, "fixture": fixture, "root": root}
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# server side
# ---------------------------------------------------------------------------

def test_layer_endpoint_warm(perf_site):
    s, rid = perf_site["server"], perf_site["big"]
    assert abs(s.get(f"/api/runs/{rid}/grid")["n"] - N_CELLS) < 1000
    path = f"/api/runs/{rid}/layers/pred.bin"
    assert s.http.get(path).status_code == 200                   # warm the cache
    ms = _median_ms(lambda: s.http.get(path).raise_for_status())
    assert ms < 50, f"warm layer fetch {ms:.1f} ms"


def test_selection_resolve(perf_site):
    s, rid = perf_site["server"], perf_site["big"]
    grid = s.get(f"/api/runs/{rid}/grid")
    centre = [grid["x0_m"] + grid["dx_m"] * grid["nx"] / 2, grid["y0_m"] + grid["dx_m"] * grid["ny"] / 2]
    body = {"selection": {"kind": "circle", "crs": "run_xy_m", "center": centre, "radius_m": 1500}}
    first = s.post(f"/api/runs/{rid}/selection/resolve", body)
    assert first["n_cells"] > 0
    ms = _median_ms(lambda: s.http.post(f"/api/runs/{rid}/selection/resolve", json=body).raise_for_status())
    assert ms < 30, f"selection resolve {ms:.1f} ms"


def test_run_list_200(perf_site):
    s = perf_site["server"]
    assert len(s.get("/api/runs", params={"limit": 200})["items"]) == 200
    ms = _median_ms(lambda: s.http.get("/api/runs", params={"limit": 200}).raise_for_status())
    assert ms < 100, f"run list {ms:.1f} ms"


def test_preview_providence():
    env = os.environ.get("SPARC_PROVIDENCE_RUNS")
    src = Path(env) if env else ROOT / "output" / "core" / "providence"
    run = src / "providence_uhi_fast"
    if not (run / "emulator.npz").is_file():
        pytest.skip("no recorded providence_uhi_fast with an emulator (set SPARC_PROVIDENCE_RUNS)")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sparc-perf-") as tmp:
        copy = Path(tmp) / run.name
        shutil.copytree(run, copy)
        server = start_studio(Path(tmp) / "ws")
        try:
            rid = server.post("/api/runs/import", {"dir": str(copy), "trust_pickles": True})["id"]
            edits = [{"lever": "Pct_Canopy", "mode": "add", "amount": 10,
                      "where": {"kind": "top", "column": "pred:target", "frac": 0.2, "direction": "highest"}}]
            seq = iter(range(1, 1000))

            def preview():
                r = server.http.post(f"/api/runs/{rid}/preview", json={"edits": edits, "request_seq": next(seq)})
                assert r.status_code == 200, r.text

            preview()                                            # loads the emulator and the run context
            ms = _median_ms(preview)
            assert ms < 150, f"preview {ms:.1f} ms"
        finally:
            server.stop()


def test_tailer_throughput(perf_site):
    from sparc.studio.jobs import tracker
    from sparc.studio.jobs.tailer import iter_lines, parse_line

    path = perf_site["fixture"] / "events.jsonl"
    t = time.perf_counter()
    state = tracker.new_state()
    n = 0
    for cursor, raw in iter_lines(path):
        tracker.reduce(state, parse_line(raw), cursor)
        n += 1
    rate = n / (time.perf_counter() - t)
    assert n == N_EVENTS
    assert rate >= 2000, f"tailer + reducer {rate:.0f} events/s"


# ---------------------------------------------------------------------------
# browser side
# ---------------------------------------------------------------------------

LONG_TASKS = """
window.__longest = 0;
window.__frames = [];
const __seen = (list) => {
  for (const e of list.getEntries()) {
    window.__longest = Math.max(window.__longest, e.duration);
    window.__frames.push([Math.round(e.startTime), Math.round(e.duration),
      (e.scripts || []).slice(0, 2).map((s) => (s.sourceFunctionName || s.invoker || "") + "@" + (s.sourceURL || "").split("/").pop() + ":" + Math.round(s.duration))]);
  }
};
try {
  new PerformanceObserver(__seen).observe({type: "long-animation-frame", buffered: true});
} catch (e) {
  new PerformanceObserver(__seen).observe({type: "longtask", buffered: true});
}
"""
RESET = "window.__longest = 0; window.__frames = [];"
SLOWEST = "window.__frames.filter((f) => f[1] > 50).sort((a, b) => b[1] - a[1]).slice(0, 5)"

NEXT_FRAME_AFTER = """async ([selector, value, waitText]) => {
  const el = document.querySelector(selector);
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value").set;
  const t0 = performance.now();
  setter.call(el, value);
  el.dispatchEvent(new Event(el.tagName === "SELECT" ? "change" : "input", {bubbles: true}));
  if (waitText) {
    await new Promise((resolve) => {
      const done = () => [...document.querySelectorAll("h3")].some((h) => h.textContent.trim() === waitText);
      if (done()) return resolve();
      const mo = new MutationObserver(() => { if (done()) { mo.disconnect(); resolve(); } });
      mo.observe(document.body, {subtree: true, childList: true, characterData: true});
    });
  }
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return performance.now() - t0;
}"""


def test_layer_switch_and_recolour(perf_site, studio_page):
    s, rid = perf_site["server"], perf_site["big"]
    sp: StudioPage = studio_page(s)
    page = sp.page
    sp.goto(f"/r/{rid}/map")
    page.get_by_role("application").first.wait_for()
    layers = [layer for g in s.get(f"/api/runs/{rid}/layers")["groups"] for layer in g["layers"]]
    select = "[aria-label='Choose a map layer'] select:last-of-type"
    keys = [layer["key"] for layer in layers[:2]]
    labels = {layer["key"]: layer["label"] for layer in layers}
    for k in keys + keys:                                         # load both once: later switches are warm
        page.select_option(select, k)
        page.get_by_role("heading", name=labels[k], exact=True).first.wait_for()
    switches = [page.evaluate(NEXT_FRAME_AFTER, [select, keys[i % 2], labels[keys[i % 2]]]) for i in range(7)]
    assert statistics.median(switches) < 50, f"layer switch {switches} ms"
    lo = page.get_by_label("Range minimum")
    sel = "input[aria-label='Range minimum']"
    assert lo.count()
    recolours = [page.evaluate(NEXT_FRAME_AFTER, [sel, str(84 + (i % 3) * 0.5), None]) for i in range(7)]
    # two animation frames bound the measurement from below (~33 ms at 60 Hz in headless Chromium),
    # so the colour pass is the time beyond them
    frame = page.evaluate("""async () => { const t0 = performance.now();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); return performance.now() - t0; }""")
    extra = statistics.median(recolours) - frame
    assert extra < 10, f"recolour {recolours} ms against an idle double frame of {frame:.1f} ms"


@pytest.fixture(scope="module")
def streamed_job(perf_site, browser, tmp_path_factory) -> dict:
    """A replayed fast run padded to 20,000 events, watched live in Mission Control.

    Returns the job id and the longest frame while the 20,000-event burst streamed in (Mission
    Control was open and idle before the burst; the long-frame counter was reset then).
    """
    s = perf_site["server"]
    meta = json.loads((SYNTH_RUN / "FIXTURE.json").read_text("utf-8"))
    pid = s.create_demo("Perf city", n=meta["n"], seed=meta["seed"])["id"]
    ctx = browser.new_context(viewport=VIEWPORT)
    try:
        sp = StudioPage(ctx.new_page(), s.base, tmp_path_factory.mktemp("tracker_shots"))
        page = sp.page
        page.add_init_script(LONG_TASKS)
        sp.goto(f"/auth?t={TOKEN}&next=/")
        jid = s.post(f"/api/projects/{pid}/runs", {"mode": "fast"})["job"]["id"]
        sp.goto(f"/jobs/{jid}")
        page.get_by_role("tab", name="Logs").click()
        wait_for(lambda: dict(rail_states(page)).get("S1") == "done", 60, "S1 to finish on the rail")
        page.wait_for_timeout(500)
        page.evaluate(RESET)
        assert dict(rail_states(page)).get("S4") in ("planned", "running"), rail_states(page)
        assert s.wait_job(jid, timeout=600)["status"] == "succeeded"
        count = page.get_by_role("list", name=re.compile(r"^\d+ log lines$"))
        wait_for(lambda: count.count() and int(count.get_attribute("aria-label").split()[0]) > N_EVENTS // 3, 120,
                 "the streamed log lines")
        page.wait_for_timeout(1000)
        live = page.evaluate("window.__longest")
        frames = page.evaluate(SLOWEST)
        sp.assert_no_errors()
        return {"jid": jid, "live_longest": live, "frames": frames}
    finally:
        ctx.close()


def test_tracker_20000_events_streaming(streamed_job):
    assert streamed_job["live_longest"] <= 100, \
        f"longest frame {streamed_job['live_longest']:.0f} ms while streaming; slowest frames {streamed_job['frames']}"


def test_tracker_20000_events_replayed_log(perf_site, streamed_job, studio_page):
    jid = streamed_job["jid"]
    sp: StudioPage = studio_page(perf_site["server"])
    page = sp.page
    page.add_init_script(LONG_TASKS)
    sp.goto("/jobs")                         # open the finished job from Activity, without a page reload
    link = page.locator(f"a[href='/jobs/{jid}']").first
    link.wait_for()
    page.wait_for_timeout(500)
    page.evaluate(RESET)
    link.click()
    page.get_by_role("tab", name="Logs").click()
    count = page.get_by_role("list", name=re.compile(r"^\d+ log lines$"))
    wait_for(lambda: count.count() and int(count.get_attribute("aria-label").split()[0]) > N_EVENTS // 3, 120,
             "the replayed log lines")
    page.wait_for_timeout(1000)
    longest = page.evaluate("window.__longest")
    assert longest <= 100, f"longest frame {longest:.0f} ms while replaying the log; slowest {page.evaluate(SLOWEST)}"


def test_initial_bundle_size():
    html = (STATIC / "index.html").read_text("utf-8")
    refs = re.findall(r'<script[^>]+src="/([^"]+)"', html) + re.findall(r'<link[^>]+rel="modulepreload"[^>]+href="/([^"]+)"', html) \
        + re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="/([^"]+)"', html)
    assert refs, "index.html references no assets"
    size = sum(len(gzip.compress((STATIC / r).read_bytes(), 9)) for r in refs)
    assert size <= 200 * 1024, f"initial bundle {size / 1024:.1f} kB gzip ({refs})"
