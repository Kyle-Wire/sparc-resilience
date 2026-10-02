"""Keyboard use, accessible names and the phone layout (SPEC §14.5 item 5, §12.8).

One server (replay runner) and one replayed run serve every check:

* the shell is a proper HTML document with landmarks; Tab starts at the skip link, which moves
  focus to the main region; the shell's controls are reached in order, each with a name and a
  visible focus ring;
* the map's keyboard cursor moves cell by cell and the aria-live readout says where it is; Enter
  pins the cell inspector;
* every button on the main pages has an accessible name (as the browser computes it);
* at 390 px no page scrolls sideways (tables and tab strips scroll inside their own box).
"""

from __future__ import annotations

import json
import re

import pytest

from .conftest import SYNTH_RUN, start_studio, wait_for

PHONE = {"width": 390, "height": 844}
PROJECT_NAV = ["Overview", "Setup", "Inputs", "Launch", "Runs", "Studies", "Exports", "Findings"]

# Pages checked for names and for the phone layout ({pid}, {rid} and {jid} are filled in).
PAGES = [
    "/", "/projects", "/runs", "/jobs", "/settings", "/findings",
    "/p/{pid}", "/p/{pid}/setup/data", "/p/{pid}/setup/inputs", "/p/{pid}/config", "/p/{pid}/launch", "/p/{pid}/runs",
    "/p/{pid}/studies", "/p/{pid}/exports", "/p/{pid}/findings",
    "/r/{rid}", "/r/{rid}/data", "/r/{rid}/accuracy", "/r/{rid}/distance", "/r/{rid}/influence", "/r/{rid}/response",
    "/r/{rid}/causal", "/r/{rid}/scenarios", "/r/{rid}/climate", "/r/{rid}/budget", "/r/{rid}/validation",
    "/r/{rid}/provenance", "/r/{rid}/map", "/r/{rid}/files", "/r/{rid}/lab", "/r/{rid}/docs", "/r/{rid}/track",
    "/jobs/{jid}",
]

# Pages that still scroll sideways at 390 px. Each is reported to its owner; the strict xfail
# turns into a failure once the page is fixed, so the entry has to be removed then.
WIDE_AT_390 = {
    "/r/{rid}/docs": "the Documents header and document picker rows do not wrap (page 680 px wide; frontend-run-hub)",
    "/r/{rid}/track": "Mission Control's bottom tab strip widens its card and the job picker sizes to its longest "
                      "option (frontend-foundation .stack/.card children need min-width: 0; frontend-tracking)",
    "/jobs/{jid}": "Mission Control's bottom tab strip widens its card (frontend-foundation .stack/.card children "
                   "need min-width: 0)",
}

FOCUSED = """() => {
  const e = document.activeElement;
  const r = e.getBoundingClientRect();
  const cs = getComputedStyle(e);
  const by = (e.getAttribute('aria-labelledby') || '').split(/\\s+/).filter(Boolean)
    .map(id => (document.getElementById(id) || {}).textContent || '').join(' ');
  const label = e.labels && e.labels.length ? e.labels[0].textContent : '';
  const name = (e.getAttribute('aria-label') || by || label || e.innerText || e.getAttribute('title') || '').trim();
  return {tag: e.tagName, id: e.id, name: name.replace(/\\s+/g, ' ').slice(0, 80), w: r.width, h: r.height,
          ring: cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0 || cs.boxShadow !== 'none',
          visible: e.matches(':focus-visible')};
}"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A server with the synthetic demo and one replayed fast run."""
    meta = json.loads((SYNTH_RUN / "FIXTURE.json").read_text("utf-8"))
    server = start_studio(tmp_path_factory.mktemp("a11y") / "ws", runner=f"replay:{SYNTH_RUN}")
    try:
        pid = server.create_demo("Access city", n=meta["n"], seed=meta["seed"])["id"]
        launched = server.post(f"/api/projects/{pid}/runs", {"mode": "fast"})
        jid, rid = launched["job"]["id"], launched["run"]["id"]
        assert server.wait_job(jid, timeout=120)["status"] == "succeeded"
        yield {"server": server, "pid": pid, "rid": rid, "jid": jid}
    finally:
        server.stop()


def _fill(path: str, site: dict) -> str:
    return path.format(pid=site["pid"], rid=site["rid"], jid=site["jid"])


def _settle(page) -> None:
    page.locator("main h1").first.wait_for()
    wait_for(lambda: not page.locator("main .spinner").count(), 30, "the page to load")


def test_keyboard_shell(site, studio_page):
    sp = studio_page(site["server"])
    page = sp.page
    sp.goto(_fill("/p/{pid}", site))
    _settle(page)

    # a proper document with landmarks
    assert page.evaluate("document.doctype && document.doctype.name") == "html"
    assert page.evaluate("document.documentElement.lang") == "en"
    assert page.locator("meta[charset]").count() and page.locator("meta[name=viewport]").count()
    for role in ("banner", "navigation", "main"):
        assert page.get_by_role(role).count(), f"no {role} landmark"

    # Tab starts at the skip link, which moves focus into the main region
    page.keyboard.press("Tab")
    first = page.evaluate(FOCUSED)
    assert first["tag"] == "A" and first["name"] == "Skip to content" and first["visible"] and first["ring"]
    page.keyboard.press("Enter")
    assert page.evaluate("document.activeElement.id") == "main"
    page.keyboard.press("Tab")
    inside = page.evaluate("document.querySelector('main').contains(document.activeElement)")
    assert inside, "the first stop after the skip link is outside the main region"

    # from the top: every stop of the shell is visible, named and ringed; the project nav comes in order
    sp.goto(_fill("/p/{pid}", site))
    _settle(page)
    names = []
    for _ in range(30):
        page.keyboard.press("Tab")
        f = page.evaluate(FOCUSED)
        assert f["w"] > 0 and f["h"] > 0, f"focus on an invisible element: {f}"
        assert f["name"], f"focus on an element without a name: {f}"
        assert f["visible"] and f["ring"], f"no focus ring on {f}"
        names.append(f["name"])
        if page.evaluate("document.querySelector('main').contains(document.activeElement)"):
            break
    else:
        pytest.fail(f"30 Tab presses never reached the main region: {names}")
    nav = [n for n in names if n in PROJECT_NAV]
    assert nav == PROJECT_NAV, f"project navigation out of order: {names}"
    assert "Settings" in names and "Activity" in names
    sp.shot("keyboard focus in main")

    # the command palette opens from the keyboard and closes with Escape
    page.keyboard.press("Control+k")
    dialog = page.get_by_role("dialog")
    dialog.wait_for()
    assert page.evaluate("document.querySelector('[role=dialog]').contains(document.activeElement)")
    sp.shot("command palette")
    page.keyboard.press("Escape")
    dialog.wait_for(state="detached")


def test_map_keyboard_cursor(site, studio_page):
    sp = studio_page(site["server"])
    page = sp.page
    sp.goto(_fill("/r/{rid}/map", site))
    stage = page.get_by_role("application").first
    stage.wait_for()
    live = page.get_by_test_id("map-live").first
    stage.focus()
    readouts = []
    for key in ("ArrowRight", "ArrowRight", "ArrowDown", "Shift+ArrowRight"):
        before = live.inner_text()
        page.keyboard.press(key)
        wait_for(lambda: live.inner_text() and live.inner_text() != before, 10, f"the readout after {key}")
        readouts.append(live.inner_text())
    # "<layer>: <value> · <lat>, <lon> · row <n>" on a cell with data, "No observation · <lat>, <lon>" elsewhere
    where = [re.search(r"(\d+\.\d+°[NS], \d+\.\d+°[EW])", t) for t in readouts]
    assert all(where), readouts
    assert len({w.group(1) for w in where}) == len(readouts), f"the cursor did not move: {readouts}"
    # walk 10 cells at a time until the cursor is on a cell with data: the readout then names the layer value
    sweep = ["Shift+ArrowRight"] * 3 + ["Shift+ArrowUp"] * 3 + ["Shift+ArrowLeft"] * 3 + ["Shift+ArrowDown"] * 6
    for key in sweep:
        if re.match(r"^Observed temperature: .+ · row \d+$", live.inner_text()):
            break
        page.keyboard.press(key)
        page.wait_for_timeout(150)
    assert re.match(r"^Observed temperature: .+ · row \d+$", live.inner_text()), live.inner_text()
    assert live.get_attribute("aria-live") == "polite"
    page.keyboard.press("Enter")
    page.get_by_role("region", name="Pinned cell").wait_for()
    sp.shot("map keyboard cursor pinned")
    page.keyboard.press("Escape")


@pytest.mark.parametrize("path", PAGES)
def test_every_button_has_a_name(site, studio_page, path):
    sp = studio_page(site["server"])
    page = sp.page
    sp.goto(_fill(path, site))
    _settle(page)
    page.wait_for_timeout(500)
    snapshot = page.locator("body").aria_snapshot()
    unnamed = [line.strip() for line in snapshot.splitlines() if re.match(r"^\s*- (button|link)(\s*\[[^\]]*\])*:?\s*$", line)]
    assert page.get_by_role("button").count() > 0
    assert not unnamed, f"{path}: controls without an accessible name: {unnamed}"


@pytest.mark.parametrize("path", [
    pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=WIDE_AT_390[p])) if p in WIDE_AT_390 else p
    for p in PAGES
])
def test_no_horizontal_scroll_at_390(site, studio_page, path):
    sp = studio_page(site["server"], viewport=PHONE)
    page = sp.page
    sp.goto(_fill(path, site))
    _settle(page)
    page.wait_for_timeout(500)
    width, inner = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    if path in ("/p/{pid}", "/r/{rid}", "/r/{rid}/map"):
        sp.shot(f"phone {path}")
    assert width <= inner, f"{path} is {width} px wide at {inner} px"
