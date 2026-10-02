// Exports & reports (SPEC §6.7, api.md §10): the report preview renders in a sandboxed iframe
// with no scripts; exports post kind "report" with the selected ids; the bundle sends its
// output checklist and checkpoint toggle (with a size warning); runs without a CRS cannot build
// a GIS pack; the history downloads and deletes.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { ProjectLayout } from "../../../layouts/ProjectLayout";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { fmtBytes } from "../../../theme/format";
import { byText, click, flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import Exports, { ExportHistory, ReportBuilder } from "../Exports";
import { DEFAULT_SECTIONS, inertPreview, orderedSections, previewBody, reportParams, withSelection, type ReportSelection } from "../model/report";
import { exportRow, finding, job, output, PID, projectDetail, RID, study } from "../__fixtures__/api";

const HOSTILE = `<!doctype html><html><head><title>Report</title><script>window.parent.hacked = 1</script></head>
<body onload="steal()"><h1>Demo city report</h1><img src="x.png" onerror="steal()"><a href="javascript:steal()">link</a>
<iframe src="https://example.org"></iframe><svg><script>steal()</script><circle r="2"/></svg></body></html>`;

const likely = { estimate: -0.8, se: 0.1, lo: -1, hi: -0.6, confidence: "confident_cools", phrase: "cools" };

function builderRoutes(extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  return {
    [`GET /api/runs/${RID}/scenarios`]: {
      body: {
        configured: [],
        results: [
          { id: "res_1", scenario_id: "sc_1", run_id: RID, kind: "exact", created_utc: "2026-10-01T15:00:00Z", stale: false, has_folds: true, city: likely, edited: null, frac_extrapolated_edited: null, job_id: null },
          { id: "res_2", scenario_id: null, run_id: RID, kind: "configured", created_utc: "2026-10-01T15:10:00Z", stale: true, has_folds: true, city: likely, edited: null, frac_extrapolated_edited: null, job_id: null },
        ],
      },
    },
    [`GET /api/projects/${PID}/scenarios`]: { body: [{ id: "sc_1", name: "Shade the hottest", revision: 1, parent_id: null, status: "exact", created_utc: "x", updated_utc: "x", tags: [], latest: null }] },
    [`GET /api/runs/${RID}/plans`]: { body: [{ id: "pl_1", name: "Budget 2M", params: {}, planned: {}, realised: null, frontier: null, created_utc: "2026-10-01T16:00:00Z" }] },
    "GET /api/findings": { body: [finding("fd_1", 0), finding("fd_2", 1, { run_id: "other-run" })] },
    [`POST /api/projects/${PID}/report/preview`]: { body: { html: HOSTILE } },
    "POST /api/exports": (_u, init) => {
      const b = JSON.parse(String(init.body)) as { kind: string };
      return { status: 202, body: { export: exportRow("ex_new", b.kind, { status: "running", bytes: null, path: null }), job: job("j_ex", `export.${b.kind}`) } };
    },
    [`GET /api/projects/${PID}/exports`]: { body: [] },
    ...extra,
  };
}

beforeEach(() => useJobs.getState().reset());
afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("report model", () => {
  it("orders sections canonically and ticks plans/findings when ids are picked", () => {
    expect(orderedSections(["provenance", "summary", "plans"])).toEqual(["summary", "plans", "provenance"]);
    let sel: ReportSelection = { runId: RID, sections: [...DEFAULT_SECTIONS], resultIds: [], planIds: [], findingIds: [] };
    sel = withSelection(sel, { planIds: ["pl_1"] });
    sel = withSelection(sel, { findingIds: ["fd_1"] });
    expect(sel.sections).toContain("plans");
    expect(sel.sections).toContain("findings");
    expect(previewBody(sel)).toEqual({ run_id: RID, sections: orderedSections([...DEFAULT_SECTIONS, "plans", "findings"]), result_ids: [], plan_ids: ["pl_1"], finding_ids: ["fd_1"] });
    expect(reportParams(sel, "md").format).toBe("md");
  });

  it("makes the preview inert: no scripts, handlers, javascript: URLs or frames", () => {
    const out = inertPreview(HOSTILE);
    expect(out).not.toMatch(/<script/i);
    expect(out).not.toMatch(/onload|onerror/i);
    expect(out).not.toMatch(/javascript:/i);
    expect(out).not.toMatch(/<iframe/i);
    expect(out).toContain("<h1>Demo city report</h1>");
    expect(out).toContain("<circle");
    expect(out).toMatch(/Content-Security-Policy/);
  });

  it("drops <base> and <meta http-equiv> (a refresh would navigate the frame away), keeping its own CSP", () => {
    const out = inertPreview(
      `<html><head><meta charset="utf-8"><base href="https://example.org/"><meta http-equiv="refresh" content="0;url=https://example.org"></head><body><p>ok</p></body></html>`,
    );
    expect(out).not.toMatch(/<base/i);
    expect(out).not.toMatch(/refresh/i);
    expect(out).toMatch(/charset/);
    expect(out.match(/http-equiv/gi)).toHaveLength(1);
    expect(out).toContain("script-src 'none'");
  });
});

describe("ReportBuilder", () => {
  it("previews in a sandboxed iframe without scripts and exports kind 'report' with the selected ids", async () => {
    const m = mockFetch(builderRoutes());
    const { container } = render(<ReportBuilder pid={PID} rid={RID} />);
    const frame = await waitFor(() => container.querySelector<HTMLIFrameElement>("iframe[data-testid=report-preview]"), 4000, "preview iframe");
    expect(frame.hasAttribute("sandbox")).toBe(true);
    expect(frame.getAttribute("sandbox")).toBe("");
    expect(frame.getAttribute("sandbox")).not.toContain("allow-scripts");
    expect(frame.getAttribute("title")).toBe("Report preview");
    const doc = frame.getAttribute("srcdoc")!;
    expect(doc).toContain("Demo city report");
    expect(doc).not.toMatch(/<script/i);
    expect(doc).not.toMatch(/onload|onerror|javascript:/i);
    const first = m.calls.find((c) => c.url === `/api/projects/${PID}/report/preview`)!;
    expect(first.body).toEqual({ run_id: RID, sections: [...DEFAULT_SECTIONS], result_ids: [], plan_ids: [], finding_ids: [] });

    // results carry their scenario names; only this run's and project-wide findings are offered
    await waitFor(() => byText(container, "label", "Shade the hottest"), 3000, "results");
    expect(byText(container, "label", "Finding fd_2")).toBeNull();
    click(byText(container, "label", "Shade the hottest")!.querySelector("input"));
    click(byText(container, "label", "Budget 2M")!.querySelector("input"));
    click(byText(container, "label", "Finding fd_1")!.querySelector("input"));
    click(byText(container, "label", "Climate")!.querySelector("input")); // untick a section
    await flush(2);
    click(byText(container, "button", "Export HTML"));
    await flush(4);
    const post = m.calls.find((c) => c.method === "POST" && c.url === "/api/exports")!;
    const sections = orderedSections([...DEFAULT_SECTIONS.filter((s) => s !== "climate"), "plans", "findings"]);
    expect(post.body).toEqual({
      kind: "report",
      project_id: PID,
      params: { run_id: RID, sections, result_ids: ["res_1"], plan_ids: ["pl_1"], finding_ids: ["fd_1"], format: "html" },
    });
    expect(useJobs.getState().jobs.j_ex).toBeDefined();
    // the preview follows the selection (debounced)
    await waitFor(() => m.calls.filter((c) => c.url === `/api/projects/${PID}/report/preview`).some((c) => JSON.stringify((c.body as { result_ids: string[] }).result_ids) === '["res_1"]'), 4000, "updated preview");
    click(byText(container, "button", "Export Markdown"));
    await flush(4);
    const md = m.calls.filter((c) => c.method === "POST" && c.url === "/api/exports")[1];
    expect((md.body as { params: { format: string } }).params.format).toBe("md");
    expect(container.textContent).toContain("print stylesheet");
    m.restore();
  });

  it("shows the server's error instead of a preview", async () => {
    const m = mockFetch(builderRoutes({ [`POST /api/projects/${PID}/report/preview`]: { status: 404, body: { error: { code: "output_missing", message: "The run has no manifest" } } } }));
    const { container } = render(<ReportBuilder pid={PID} rid={RID} />);
    await waitFor(() => container.querySelector('[role="alert"]'), 4000, "error");
    expect(container.querySelector('[role="alert"]')!.textContent).toContain("no manifest");
    expect(container.querySelector("iframe")).toBeNull();
    m.restore();
  });
});

describe("ExportHistory", () => {
  it("downloads ready exports, tracks running ones and deletes after confirmation", async () => {
    const m = mockFetch({
      [`GET /api/projects/${PID}/exports`]: {
        body: [
          exportRow("ex_a", "bundle", { created_utc: "2026-10-02T08:00:00Z" }),
          exportRow("ex_b", "report", { status: "running", bytes: null, options: { format: "html" }, created_utc: "2026-10-02T09:00:00Z" }),
          exportRow("ex_c", "decision_pack", { status: "ready", draft: true, created_utc: "2026-10-02T07:00:00Z" }),
        ],
      },
      "DELETE /api/exports/ex_a": { body: { ok: true } },
    });
    const { container } = render(<ExportHistory pid={PID} />);
    await waitFor(() => container.querySelector("tr[data-export]"), 3000, "rows");
    const rows = [...container.querySelectorAll("tr[data-export]")];
    expect(rows.map((r) => r.getAttribute("data-export"))).toEqual(["ex_b", "ex_a", "ex_c"]);
    expect(rows[0].textContent).toContain("Project report (HTML)");
    expect(byText(rows[0], "a", "Track")!.getAttribute("href")).toBe("/jobs/j_ex_b");
    expect(rows[0].querySelector("a[download]")).toBeNull(); // not ready yet
    const dl = rows[1].querySelector("a[download]")!;
    expect(dl.getAttribute("href")).toBe("/api/exports/ex_a/download");
    expect(rows[2].textContent).toContain("draft");
    click(byText(rows[1], "button", "Delete"));
    await flush(2);
    click(byText(document.body, ".dialog button", "Delete"));
    await flush(4);
    expect(m.calls.some((c) => c.method === "DELETE" && c.url === "/api/exports/ex_a")).toBe(true);
    expect(container.querySelector('tr[data-export="ex_a"]')).toBeNull();
    m.restore();
  });
});

describe("Exports page", () => {
  it("explains a disabled GIS pack when the run's grid cannot be read", async () => {
    const m = mockFetch(
      builderRoutes({
        [`GET /api/projects/${PID}`]: { body: projectDetail() },
        [`GET /api/runs/${RID}/outputs`]: { body: { outputs: [], tabs: [] } },
        [`GET /api/runs/${RID}/grid`]: { status: 404, body: { error: { code: "output_missing", message: "The run has no predictions yet" } } },
        [`GET /api/projects/${PID}/studies`]: { body: [] },
      }),
    );
    navigate(`/p/${PID}/exports`, { replace: true });
    const { container } = render(
      <ProjectLayout pid={PID}>
        <Exports />
      </ProjectLayout>,
    );
    const gis = await waitFor(() => container.querySelector('[aria-label="GIS pack"]'), 4000, "gis card");
    await waitFor(() => gis.textContent!.includes("no predictions yet"), 3000, "grid error");
    expect((byText(gis, "button", "Build GIS pack") as HTMLButtonElement).disabled).toBe(true);
    m.restore();
  });

  it("bundles the ticked outputs with the checkpoint (size warning) and disables the GIS pack without a CRS", async () => {
    const m = mockFetch(
      builderRoutes({
        [`GET /api/projects/${PID}`]: { body: projectDetail() },
        [`GET /api/runs/${RID}/outputs`]: {
          body: {
            outputs: [output("metrics", "Metrics", 1200), output("predictions", "Predictions", 800_000), output("climate", "Climate", 0, "missing"), output("checkpoint", "Checkpoint", 350 * 1024 * 1024)],
            tabs: [],
          },
        },
        [`GET /api/runs/${RID}/grid`]: { body: { n: 4, nx: 2, ny: 2, dx_m: 30, x0_m: 0, y0_m: 0, crs: null, coord_scale: 1, has_lonlat: false, bounds_lonlat: null, corners: null, ids_kind: "int", zones: [], n_folds: 5, units: { target: "°F" }, background: null, etag: "g" } },
        [`GET /api/projects/${PID}/studies`]: {
          body: [study("st_ok", "placebo"), study("st_cancelled", "placebo", { status: "cancelled" }), study("st_run", "placebo", { status: "running" }), study("st_sim", "simcheck")],
        },
      }),
    );
    navigate(`/p/${PID}/exports`, { replace: true });
    const { container } = render(
      <ProjectLayout pid={PID}>
        <Exports />
      </ProjectLayout>,
    );
    const bundle = await waitFor(() => container.querySelector('[aria-label="Run bundle"]'), 4000, "bundle card");
    await waitFor(() => byText(bundle, "label", "Predictions"), 3000, "outputs");
    expect(byText(bundle, "label", "Climate")).toBeNull(); // missing outputs are not offered
    expect(byText(bundle, "label", "Checkpoint (")).toBeNull(); // the checkpoint has its own toggle
    click(byText(bundle, "label", "Metrics")!.querySelector("input"));
    click(byText(bundle, "label", "Include the checkpoint")!.querySelector("input"));
    await flush(2);
    expect(bundle.querySelector('[data-testid="checkpoint-warning"]')!.textContent).toContain(fmtBytes(350 * 1024 * 1024));
    click(byText(bundle, "button", "Build bundle"));
    await flush(4);
    const post = m.calls.find((c) => c.method === "POST" && c.url === "/api/exports")!;
    expect(post.body).toEqual({ kind: "bundle", project_id: PID, params: { run_id: RID, outputs: ["predictions"], include_checkpoint: true } });

    const gis = container.querySelector('[aria-label="GIS pack"]')!;
    await waitFor(() => gis.textContent!.includes("no coordinate reference system"), 3000, "crs note");
    expect((byText(gis, "button", "Build GIS pack") as HTMLButtonElement).disabled).toBe(true);

    // an empty `outputs` list would mean "every output" to the server: at least one is required
    click(byText(bundle, "button", "None"));
    await flush(2);
    expect((byText(bundle, "button", "Build bundle") as HTMLButtonElement).disabled).toBe(true);
    expect(bundle.textContent).toContain("Pick at least one output");

    const page = container.querySelector('[aria-label="Standalone results page"]')!;
    // only finished placebo suites can feed the page
    await waitFor(() => page.querySelectorAll("#page-placebo option").length > 1, 3000, "placebo options");
    expect([...page.querySelectorAll<HTMLOptionElement>("#page-placebo option")].map((o) => o.value)).toEqual(["", "st_ok"]);
    click(byText(page, "button", "Build results page"));
    await flush(4);
    const pagePost = m.calls.filter((c) => c.method === "POST" && c.url === "/api/exports").find((c) => (c.body as { kind: string }).kind === "page")!;
    expect(pagePost.body).toEqual({ kind: "page", project_id: PID, params: { run_id: RID } });
    m.restore();
  });
});
