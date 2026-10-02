// Findings notebook (SPEC §6.10, api.md §11): reordering sends PATCH position updates (move
// buttons and drag and drop, also under a run filter), "Open" navigates to the stored
// url_state, notes are edited in place, and export posts kind "findings".
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act } from "react";
import { clearResources } from "../../../api/resource";
import type { Finding } from "../../../api/types";
import { ProjectLayout } from "../../../layouts/ProjectLayout";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, typeInto, waitFor, type MockHandler } from "../../../test/render";
import Findings from "../Findings";
import { applyPositions, findingHref, ordered, reorderPatches, snapshotTable } from "../model/findings";
import { exportRow, finding, job, PID, projectDetail, RID } from "../__fixtures__/api";

const OTHER = "20261001-170000-full-cd34";

describe("findings model", () => {
  const all = [finding("a", 0), finding("b", 1), finding("c", 2), finding("d", 3)];

  it("moves a finding by swapping position slots, only patching what changes", () => {
    const ids = all.map((f) => f.id);
    expect(reorderPatches(all, ids, 0, 2)).toEqual([
      { id: "a", position: 2 },
      { id: "b", position: 0 },
      { id: "c", position: 1 },
    ]);
    expect(reorderPatches(all, ids, 3, 2)).toEqual([
      { id: "c", position: 3 },
      { id: "d", position: 2 },
    ]);
    expect(reorderPatches(all, ids, 1, 1)).toEqual([]);
    expect(reorderPatches(all, ids, 0, 9)).toEqual([]);
    expect(ordered(applyPositions(all, reorderPatches(all, ids, 0, 3))).map((f) => f.id)).toEqual(["b", "c", "d", "a"]);
  });

  it("keeps hidden findings in place under a filter", () => {
    // visible: a, c, d (b is another run's): moving d to the top uses a's, c's and d's slots only
    const p = reorderPatches(all, ["a", "c", "d"], 2, 0);
    expect(p).toEqual([
      { id: "a", position: 2 },
      { id: "c", position: 3 },
      { id: "d", position: 0 },
    ]);
    expect(ordered(applyPositions(all, p)).map((f) => f.id)).toEqual(["d", "b", "a", "c"]);
  });

  it("renumbers tied positions before moving", () => {
    const tied = [finding("a", 0), finding("b", 0, { created_utc: "2026-10-01T19:00:00Z" }), finding("c", 0, { created_utc: "2026-10-01T20:00:00Z" })];
    const p = reorderPatches(tied, ["a", "b", "c"], 2, 0);
    expect(ordered(applyPositions(tied, p)).map((f) => f.id)).toEqual(["c", "a", "b"]);
    expect(new Set(applyPositions(tied, p).map((f) => f.position)).size).toBe(3);
  });

  it("opens the stored url_state, or the view filled with the run plus the query", () => {
    expect(findingHref(finding("a", 0))).toBe(`/r/${RID}/accuracy?model=stack`);
    expect(findingHref(finding("a", 0, { url_state: "?layer=pred&sel=rg_1", view: "/r/:rid/map" }))).toBe(`/r/${RID}/map?layer=pred&sel=rg_1`);
    expect(findingHref(finding("a", 0, { url_state: "", view: "/p/:pid/compare", run_id: null }))).toBe(`/p/${PID}/compare`);
    expect(findingHref(finding("a", 0, { url_state: "x=1", view: "accuracy" }))).toBe(`/r/${RID}?x=1`);
    // protocol-relative URLs never leave the app: they are read as a query on the finding's route
    expect(findingHref(finding("a", 0, { url_state: "//example.org/x" }))).toBe(`/r/${RID}/accuracy?//example.org/x`);
    expect(findingHref(finding("a", 0, { url_state: "/\\example.org/x" })).startsWith(`/r/${RID}/accuracy?`)).toBe(true);
  });

  it("reads a chart snapshot as a table", () => {
    expect(snapshotTable(finding("a", 0).snapshot)).toEqual({ columns: ["Model", "R²"], rows: [["stack", 0.81]] });
    expect(snapshotTable({ r2: 0.8, kind: "chart" })).toEqual({ columns: ["Field", "Value"], rows: [["r2", 0.8]] });
    expect(snapshotTable({})).toBeNull();
  });
});

function pageRoutes(list: Finding[], extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  let current = list.map((f) => ({ ...f }));
  return {
    [`GET /api/projects/${PID}`]: { body: projectDetail() },
    "GET /api/findings": () => ({ body: ordered(current) }),
    "GET /api/projects": { body: [] },
    "POST /api/exports": { status: 202, body: { export: exportRow("ex_f", "findings", { status: "running", bytes: null }), job: job("j_f", "export.findings") } },
    "GET /api/exports/ex_f": { body: exportRow("ex_f", "findings", { status: "running", bytes: null }) },
    ...Object.fromEntries(
      list.map((f) => [
        `PATCH /api/findings/${f.id}`,
        ((_u: URL, init: RequestInit) => {
          const patch = JSON.parse(String(init.body)) as Partial<Finding>;
          current = current.map((x) => (x.id === f.id ? { ...x, ...patch } : x));
          return { body: current.find((x) => x.id === f.id) };
        }) as MockHandler,
      ]),
    ),
    ...extra,
  };
}

async function mountPage(list: Finding[], path = `/p/${PID}/findings`, extra: Record<string, MockHandler> = {}) {
  const m = mockFetch(pageRoutes(list, extra));
  navigate(path, { replace: true });
  const r = render(
    <ProjectLayout pid={PID}>
      <Findings />
    </ProjectLayout>,
  );
  await waitFor(() => r.container.querySelectorAll("li[data-finding]").length === list.length || r.container.querySelector("li[data-finding]"), 4000, "findings");
  await flush(2);
  return { ...r, m };
}

const order = (c: HTMLElement) => [...c.querySelectorAll("li[data-finding]")].map((li) => li.getAttribute("data-finding"));

beforeEach(() => useJobs.getState().reset());
afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("Findings page", () => {
  it("lists findings in position order with thumbnails and notes", async () => {
    const { container, m } = await mountPage([
      finding("fd_2", 1, { image_url: "/api/findings/fd_2/image", note_md: "**Hot spot** near the river" }),
      finding("fd_1", 0),
      finding("fd_3", 2, { run_id: OTHER }),
    ]);
    expect(order(container)).toEqual(["fd_1", "fd_2", "fd_3"]);
    expect(m.calls.find((c) => c.url.startsWith("/api/findings"))!.url).toBe(`/api/findings?project=${PID}`);
    const li = container.querySelector('li[data-finding="fd_2"]')!;
    expect(li.querySelector("img")!.getAttribute("src")).toBe("/api/findings/fd_2/image");
    expect(li.querySelector("strong")!.textContent).toBe("Hot spot");
    expect(container.querySelector('li[data-finding="fd_1"]')!.textContent).toContain("No note yet");
    m.restore();
  });

  it("move buttons send PATCH position updates and reorder the list", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0), finding("fd_2", 1), finding("fd_3", 2)]);
    click(container.querySelector('[aria-label="Move “Finding fd_1” down"]'));
    await flush(4);
    const patches = m.calls.filter((c) => c.method === "PATCH");
    expect(patches.map((c) => [c.url, c.body])).toEqual([
      ["/api/findings/fd_1", { position: 1 }],
      ["/api/findings/fd_2", { position: 0 }],
    ]);
    expect(order(container)).toEqual(["fd_2", "fd_1", "fd_3"]);
    expect((container.querySelector('[aria-label="Move “Finding fd_2” up"]') as HTMLButtonElement).disabled).toBe(true);
    m.restore();
  });

  it("drag and drop reorders with PATCH positions", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0), finding("fd_2", 1), finding("fd_3", 2)]);
    const li = (id: string) => container.querySelector(`li[data-finding="${id}"]`)!;
    act(() => {
      li("fd_3").dispatchEvent(new Event("dragstart", { bubbles: true }));
    });
    act(() => {
      li("fd_1").dispatchEvent(new Event("dragover", { bubbles: true, cancelable: true }));
    });
    expect(li("fd_1").getAttribute("data-drop")).toBe("true");
    act(() => {
      li("fd_1").dispatchEvent(new Event("drop", { bubbles: true, cancelable: true }));
    });
    await flush(4);
    expect(order(container)).toEqual(["fd_3", "fd_1", "fd_2"]);
    const patches = m.calls.filter((c) => c.method === "PATCH").map((c) => [c.url, c.body]);
    expect(patches).toEqual([
      ["/api/findings/fd_1", { position: 1 }],
      ["/api/findings/fd_2", { position: 2 }],
      ["/api/findings/fd_3", { position: 0 }],
    ]);
    m.restore();
  });

  it("'Open' navigates to the stored url_state", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0, { url_state: `/r/${RID}/map?layer=pred&sel=rg_7` })]);
    click(container.querySelector('[data-open="fd_1"]'));
    expect(window.location.pathname + window.location.search).toBe(`/r/${RID}/map?layer=pred&sel=rg_7`);
    m.restore();
  });

  it("filters by run through the URL and exports the visible findings", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0), finding("fd_2", 1, { run_id: OTHER }), finding("fd_3", 2)]);
    const sel = container.querySelector<HTMLSelectElement>("#fd-run")!;
    expect([...sel.options].map((o) => o.value)).toEqual(["", RID, OTHER]);
    typeInto(sel, RID);
    await flush(2);
    expect(new URLSearchParams(window.location.search).get("run")).toBe(RID);
    expect(order(container)).toEqual(["fd_1", "fd_3"]);
    // moving under a filter keeps the hidden finding's slot
    click(container.querySelector('[aria-label="Move “Finding fd_3” up"]'));
    await flush(4);
    expect(m.calls.filter((c) => c.method === "PATCH").map((c) => [c.url, c.body])).toEqual([
      ["/api/findings/fd_1", { position: 2 }],
      ["/api/findings/fd_3", { position: 0 }],
    ]);
    click(byText(container, "button", "Export HTML"));
    await flush(4);
    const post = m.calls.find((c) => c.method === "POST" && c.url === "/api/exports")!;
    expect(post.body).toEqual({ kind: "findings", project_id: PID, params: { project_id: PID, run_id: RID, ids: ["fd_3", "fd_1"], format: "html" } });
    expect(useJobs.getState().jobs.j_f).toBeDefined();
    m.restore();
  });

  it("edits a note in place and saves it with PATCH note_md", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0)]);
    click(byText(container, "button", "Edit note"));
    typeInto(container.querySelector('textarea[aria-label^="Note for"]'), "Canopy cools the *east* side most.");
    click(byText(container, "button", "Save"));
    await flush(4);
    expect(m.calls.filter((c) => c.method === "PATCH").map((c) => [c.url, c.body])).toEqual([["/api/findings/fd_1", { note_md: "Canopy cools the *east* side most." }]]);
    expect(container.querySelector('li[data-finding="fd_1"] em')!.textContent).toBe("east");
    m.restore();
  });

  it("deletes a finding after confirmation", async () => {
    const { container, m } = await mountPage([finding("fd_1", 0), finding("fd_2", 1)], `/p/${PID}/findings`, { "DELETE /api/findings/fd_1": { body: { ok: true } } });
    click(byText(container.querySelector('li[data-finding="fd_1"]')!, "button", "Delete"));
    await flush(2);
    click(byText(document.body, ".dialog button", "Delete"));
    await flush(4);
    expect(m.calls.some((c) => c.method === "DELETE" && c.url === "/api/findings/fd_1")).toBe(true);
    expect(order(container)).toEqual(["fd_2"]);
    m.restore();
  });
});
