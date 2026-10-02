// Overview header, Docs (rendered markdown, frozen report, Regenerate), Files (tree, preview,
// conversions, dictionary, guarded checkpoint delete), Provenance (environment diff picker,
// Reproduce) and the Breakdown / Relationships / Correlogram tools.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ReactElement } from "react";
import { clearResources } from "../../../api/resource";
import type { DocEntry, FileEntry } from "../../../api/runs";
import type { Job, RunOutputs } from "../../../api/types";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { fmtBytes } from "../../../theme/format";
import { byText, click, flush, mockFetch, render, typeInto, waitFor, type MockHandler } from "../../../test/render";
import { viewFixture } from "../__fixtures__/views";
import Docs from "../Docs";
import Files from "../Files";
import MapTab from "../MapTab";
import Overview from "../Overview";
import Provenance from "../Provenance";
import { gridMeta, mapRoutes, runDetail, runSummary, unframedCharts } from "./helpers";

let seq = 0;
const nextRid = (p: string) => `r_${p}${++seq}`;

function job(id: string, kind: string): Job {
  return {
    id,
    kind,
    lane: "medium",
    executor: "process",
    label: kind,
    status: "queued",
    project_id: "p_1",
    run_id: null,
    study_id: null,
    scenario_id: null,
    parent_job_id: null,
    after_job_id: null,
    priority: 0,
    params: {},
    created_utc: "2026-10-01T22:00:00Z",
    started_utc: null,
    finished_utc: null,
    progress: null,
    eta_s: null,
    eta_lo: null,
    eta_hi: null,
    stage: null,
    current_path: null,
    exit_code: null,
    error: null,
    blocked: null,
    result: null,
    peak_rss_mb: null,
    threads: null,
  };
}

async function mount(path: string, ui: ReactElement, routes: Record<string, MockHandler>, ready: (root: HTMLElement) => unknown) {
  const m = mockFetch(routes);
  navigate(path, { replace: true });
  const r = render(ui);
  await waitFor(() => ready(r.container), 5000, path);
  await flush(3);
  return { ...r, m };
}

beforeEach(() => useJobs.setState({ jobs: {} }));
afterEach(() => clearResources());

describe("Overview", () => {
  it("shows the header, KPIs, the missing-output action and the caveats", async () => {
    const rid = nextRid("ov");
    const { container, m } = await mount(
      `/r/${rid}`,
      <Overview />,
      { [`GET /api/runs/${rid}`]: { body: runDetail(rid) }, [`GET /api/runs/${rid}/views/overview`]: { body: viewFixture("overview") } },
      (c) => c.querySelector('[aria-label="Run key numbers"]'),
    );
    const header = container.querySelector('[aria-label="Run header"]')!.textContent!;
    expect(header).toContain("5a04f44");
    expect(header).toContain("dirty");
    expect(header).toContain("3 × 3 cells at 30 m");
    expect(container.textContent).toContain("DEMO");
    const kpis = container.querySelector('[aria-label="Run key numbers"]')!.textContent!;
    expect(kpis).toContain("89.0%");
    expect(kpis).toContain("target 90.0%");
    expect(kpis).toMatch(/cooler/);
    expect(container.querySelector('[data-output="planner"][data-state="missing"]')!.textContent).toContain("Run planner pack");
    expect(container.querySelector("details.callout[open]")!.textContent).toContain("Generic physics forcing");
    expect(container.textContent).toContain("Daytime (afternoon) model");
    m.restore();
  });
});

describe("Docs", () => {
  const docs: DocEntry[] = [
    // the report is labelled "as of run end" even if the server does not flag it frozen
    { id: "report", file: "report.md", title: "Report", mtime: "2026-10-01T21:21:49Z", present: true, regenerable: false, frozen: false },
    { id: "methods", file: "methods.md", title: "Methods", mtime: "2026-10-01T21:21:49Z", present: true, regenerable: true, frozen: false },
    { id: "uncertainty", file: "uncertainty.md", title: "Uncertainty", mtime: null, present: false, regenerable: false, frozen: false },
  ];

  it("renders markdown with raw HTML escaped, labels the frozen report, and regenerates through the writeup action", async () => {
    const rid = nextRid("docs");
    const posted: unknown[] = [];
    const { container, m } = await mount(
      `/r/${rid}/docs`,
      <Docs />,
      {
        [`GET /api/runs/${rid}/docs`]: { body: docs },
        [`GET /api/runs/${rid}/docs/report`]: { body: { markdown: "# Report\n\nCooling <script>alert(1)</script> **0.41 °F**.", mtime: "2026-10-01T21:21:49Z", frozen: false } },
        [`GET /api/runs/${rid}/docs/methods`]: { body: { markdown: "## Methods\n\nSpatial CV.", mtime: "2026-10-01T21:21:49Z", frozen: false } },
        [`POST /api/runs/${rid}/actions/writeup`]: (_u, init) => {
          posted.push(JSON.parse(String(init.body)));
          return { status: 202, body: job("j_writeup", "post.writeup") };
        },
      },
      (c) => c.querySelector(".md h1"),
    );
    expect(container.querySelector(".md h1")!.textContent).toBe("Report");
    expect(container.querySelector(".md script")).toBeNull();
    expect(container.querySelector(".md")!.textContent).toContain("<script>");
    expect(container.querySelector("article")!.textContent).toContain("as of run end");
    click(byText(container, "button", "Regenerate methods & model card"));
    await flush(4);
    expect(posted).toEqual([{}]);
    expect(useJobs.getState().jobs["j_writeup"]).toBeDefined();
    navigate(`/r/${rid}/docs/methods`);
    await waitFor(() => container.querySelector(".md h2"), 3000, "methods");
    expect(container.querySelector(".md h2")!.textContent).toBe("Methods");
    navigate(`/r/${rid}/docs/uncertainty`);
    await flush(3);
    expect(container.textContent).toContain("produced by the uncertainty report");
    m.restore();
  });
});

describe("Files", () => {
  it("shows the tree with states, previews tables, offers conversions and the dictionary, and guards the checkpoint delete", async () => {
    const rid = nextRid("files");
    const root: FileEntry[] = [
      { name: "planner", relpath: "planner", dir: true, bytes: null, mtime: "2026-10-01T21:21:49Z", output_id: null, state: null, in_manifest: false },
      {
        name: "predictions.parquet",
        relpath: "predictions.parquet",
        dir: false,
        bytes: 182_000,
        mtime: "2026-10-01T21:21:40Z",
        output_id: "predictions",
        state: "present",
        in_manifest: true,
      },
      { name: "causal.json", relpath: "causal.json", dir: false, bytes: 9_000, mtime: "2026-09-30T10:00:00Z", output_id: "causal", state: "stale", in_manifest: false },
    ];
    const outputs: RunOutputs = {
      outputs: [
        {
          id: "predictions",
          label: "Held-out predictions",
          group: "model",
          state: "present",
          produced_by: "stage:S2_S3",
          view: "accuracy",
          formats: ["parquet", "csv", "geojson"],
          files: [{ relpath: "predictions.parquet", bytes: 182_000, mtime: "" }],
          action: null,
        },
        {
          id: "causal",
          label: "Causal audit",
          group: "effects",
          state: "stale",
          produced_by: "stage:S6",
          view: "causal",
          formats: ["json"],
          files: [{ relpath: "causal.json", bytes: 9000, mtime: "" }],
          action: null,
        },
      ],
      tabs: [],
    };
    let deleted = 0;
    const { container, m } = await mount(
      `/r/${rid}/files`,
      <Files />,
      {
        [`GET /api/runs/${rid}`]: { body: runDetail(rid) },
        [`GET /api/runs/${rid}/outputs`]: { body: outputs },
        [`GET /api/runs/${rid}/files`]: (u) => ({
          body:
            u.searchParams.get("path") === "planner"
              ? [
                  {
                    name: "hex_250m.csv",
                    relpath: "planner/hex_250m.csv",
                    dir: false,
                    bytes: 3000,
                    mtime: null,
                    output_id: "planner_hex:250",
                    state: "present",
                    in_manifest: false,
                  },
                ]
              : root,
        }),
        [`GET /api/runs/${rid}/files/table`]: {
          body: {
            columns: [
              { name: "id", dtype: "int64" },
              { name: "pred", dtype: "float64" },
            ],
            rows: [
              [1, 87.5],
              [2, -0.25],
            ],
            n_rows: 1120,
          },
        },
        [`GET /api/runs/${rid}/dictionary`]: {
          body: [
            { output: "predictions.parquet", column: "pred", unit: "degF", sign: null, description: "Held-out prediction" },
            { output: "scenario_deltas.parquet", column: "<scenario>", unit: "degF", sign: "negative = cooler", description: "Change per scenario" },
          ],
        },
        [`DELETE /api/runs/${rid}/checkpoint`]: () => {
          deleted++;
          return { body: { freed_bytes: 1_200_000 } };
        },
      },
      (c) => c.querySelector('tr[data-path="predictions.parquet"]'),
    );
    const row = container.querySelector('tr[data-path="predictions.parquet"]')!;
    expect(row.textContent).toContain("predictions");
    expect(row.textContent).toContain("stage S2_S3");
    expect(row.textContent).toContain("manifest");
    expect(container.querySelector('tr[data-path="causal.json"]')!.textContent).toContain("stale");
    // expand a directory
    click(byText(container, 'tr[data-path="planner"] button', "planner"));
    await waitFor(() => container.querySelector('tr[data-path="planner/hex_250m.csv"]'), 3000, "planner children");
    // preview a parquet file
    click(byText(row, "button", "predictions.parquet"));
    await waitFor(() => container.textContent?.includes("of 1,120 rows"), 3000, "preview");
    expect(container.textContent).toContain("−0.25");
    const links = [...container.querySelectorAll("a[download]")].map((a) => a.getAttribute("href")!);
    expect(links).toContain(`/api/runs/${rid}/files/raw?path=predictions.parquet`);
    expect(links).toContain(`/api/runs/${rid}/files/raw?path=predictions.parquet&as=csv`);
    expect(links).toContain(`/api/runs/${rid}/files/raw?path=predictions.parquet&as=geojson`);
    // dictionary with a filter
    expect(container.textContent).toContain("Held-out prediction");
    typeInto(container.querySelector('input[aria-label="Filter the data dictionary"]'), "scenario");
    expect(container.textContent).not.toContain("Held-out prediction");
    // guarded delete: nothing is deleted before confirming
    click(byText(container, "button", "Delete checkpoint"));
    await flush(2);
    expect(deleted).toBe(0);
    const dialog = document.body.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain("exact scenarios and the emulator build will need a refit");
    click(byText(dialog as HTMLElement, "button", "Delete 1.2 MB"));
    await flush(4);
    expect(deleted).toBe(1);
    m.restore();
  });
});

describe("Files previews", () => {
  it("reads only the start of a large text file, handles an empty one, and disables GeoJSON without a CRS", async () => {
    const rid = nextRid("filesprev");
    const big = "x".repeat(300 * 1024);
    const ranges: (string | null)[] = [];
    const entry = (name: string, bytes: number): FileEntry => ({ name, relpath: name, dir: false, bytes, mtime: null, output_id: null, state: null, in_manifest: false });
    const outputs: RunOutputs = {
      outputs: [
        {
          id: "predictions",
          label: "Held-out predictions",
          group: "model",
          state: "present",
          produced_by: "stage:S2_S3",
          view: "accuracy",
          formats: ["parquet", "csv", "geojson"],
          files: [{ relpath: "predictions.parquet", bytes: 10, mtime: "" }],
          action: null,
        },
      ],
      tabs: [],
    };
    const { container, m } = await mount(
      `/r/${rid}/files`,
      <Files />,
      {
        [`GET /api/runs/${rid}`]: { body: runDetail(rid) },
        [`GET /api/runs/${rid}/outputs`]: { body: outputs },
        [`GET /api/runs/${rid}/grid`]: { body: { ...gridMeta(), crs: null, has_lonlat: false } },
        [`GET /api/runs/${rid}/files`]: { body: [entry("simcheck.jsonl", big.length), entry("empty.json", 0), entry("predictions.parquet", 10)] },
        [`GET /api/runs/${rid}/files/table`]: { body: { columns: [{ name: "id", dtype: "int64" }], rows: [[1]], n_rows: 1 } },
        [`GET /api/runs/${rid}/dictionary`]: { body: [] },
        [`GET /api/runs/${rid}/files/raw`]: (u, init) => {
          ranges.push(new Headers(init.headers).get("Range"));
          // a server without Range support sends the whole file: the client still stops early
          if (u.searchParams.get("path") === "empty.json") return { status: 416, body: { error: { code: "range", message: "unsatisfiable" } } };
          return { raw: big, headers: { "content-type": "text/plain", "Content-Length": String(big.length) } };
        },
      },
      (c) => c.querySelector('tr[data-path="simcheck.jsonl"]'),
    );
    click(byText(container.querySelector('tr[data-path="simcheck.jsonl"]')!, "button", "simcheck.jsonl"));
    await waitFor(() => container.querySelector('[data-truncated="true"]'), 3000, "truncated preview");
    expect(ranges[0]).toBe(`bytes=0-${256 * 1024 - 1}`);
    expect(container.querySelector('[data-truncated="true"]')!.textContent).toContain(`Showing the first ${fmtBytes(256 * 1024)} of ${fmtBytes(big.length)}`);
    expect(container.querySelector("pre")!.textContent!.length).toBe(256 * 1024);
    click(byText(container.querySelector('tr[data-path="empty.json"]')!, "button", "empty.json"));
    await waitFor(() => !container.textContent?.includes("Loading preview"), 3000, "empty preview");
    expect(container.querySelector("pre")!.textContent).toBe("");
    expect(container.querySelector('[data-truncated="true"]')).toBeNull();
    // no CRS: GeoJSON is offered but disabled with the reason; CSV stays a download
    click(byText(container.querySelector('tr[data-path="predictions.parquet"]')!, "button", "predictions.parquet"));
    await waitFor(() => byText(container, '[aria-disabled="true"]', "GeoJSON"), 3000, "disabled GeoJSON");
    expect(byText(container, '[aria-disabled="true"]', "GeoJSON")!.getAttribute("title")).toContain("no CRS");
    expect([...container.querySelectorAll("a[download]")].map((a) => a.getAttribute("href"))).toContain(`/api/runs/${rid}/files/raw?path=predictions.parquet&as=csv`);
    expect([...container.querySelectorAll("a[download]")].some((a) => a.getAttribute("href")!.includes("as=geojson"))).toBe(false);
    m.restore();
  });
});

describe("Provenance", () => {
  it("copies hashes, diffs the environment against another run and starts a reproduction", async () => {
    const rid = nextRid("prov");
    const other = `${rid}_b`;
    let reproduced: unknown = null;
    const { container, m } = await mount(
      `/r/${rid}/provenance`,
      <Provenance />,
      {
        [`GET /api/runs/${rid}`]: { body: runDetail(rid) },
        [`GET /api/runs/${rid}/views/provenance`]: { body: viewFixture("provenance") },
        [`GET /api/runs/${rid}/config`]: {
          body: {
            effective: {},
            raw: {},
            yaml: "name: synthetic_demo\n",
            source: "launch",
            config_dir: "/w/p",
            vs_project_diff: [{ path: "stacker.epochs", run: 200, project: 400 }],
            vs_defaults: [],
          },
        },
        "GET /api/projects/p_1/runs": { body: { items: [runSummary(rid), runSummary(other, { label: "Full run" })], next_cursor: null } },
        [`GET /api/runs/${rid}/environment`]: (u) => ({
          body:
            u.searchParams.get("diff_with") === other
              ? { packages: ["numpy==1.26.4"], diff: { added: ["torch==2.14.0"], removed: [], changed: [{ name: "numpy", a: "1.26.4", b: "2.1.0" }] } }
              : { packages: ["numpy==1.26.4"] },
        }),
        [`POST /api/runs/${rid}/studies/reproduce`]: (_u, init) => {
          reproduced = JSON.parse(String(init.body));
          return { status: 202, body: { study: { id: "st_1", kind: "reproduce", project_id: "p_1" }, job: job("j_repro", "study.reproduce") } };
        },
      },
      (c) => c.querySelector('[aria-label="Hashes"]') && c.querySelector('select[aria-label="Run to diff the environment with"] option + option'),
    );
    expect(container.querySelectorAll('[aria-label^="Copy"]').length).toBeGreaterThanOrEqual(3);
    expect(container.textContent).toContain("stacker.epochs");
    typeInto(container.querySelector('select[aria-label="Run to diff the environment with"]'), other);
    await waitFor(() => container.querySelector('[aria-label="Environment diff"]'), 3000, "environment diff");
    expect(container.querySelector('[aria-label="Environment diff"]')!.textContent).toContain("2.1.0");
    expect(container.textContent).toContain("Only in the other run: torch==2.14.0");
    click(byText(container, "button", "Reproduce this run"));
    await flush(4);
    expect(reproduced).toEqual({});
    expect(useJobs.getState().jobs["j_repro"]).toBeDefined();
    m.restore();
  });
});

describe("analysis tools", () => {
  it("Breakdown, Relationships and Correlogram post their requests and chart the replies", async () => {
    const rid = nextRid("tools");
    const bodies: Record<string, unknown> = {};
    const capture = (name: string, body: unknown) => (_u: URL, init: RequestInit) => {
      bodies[name] = JSON.parse(String(init.body));
      return { body };
    };
    const { container, m } = await mount(
      `/r/${rid}/map?layer=obs&tool=breakdown`,
      <MapTab />,
      {
        ...mapRoutes(rid),
        [`POST /api/runs/${rid}/stats/breakdown`]: capture("breakdown", {
          groups: [
            { label: "1", n: 3, people: null, mean: 81, q: [80.2, 80.5, 81, 81.5, 81.8] },
            { label: "2", n: 4, people: null, mean: 84.5, q: [83.3, 83.8, 84.5, 85.2, 85.7] },
          ],
        }),
        [`POST /api/runs/${rid}/stats/hexbin`]: capture("hexbin", {
          x_edges: [0, 150, 300],
          y_edges: [80, 83, 86],
          counts: [
            [2, 1],
            [1, 3],
          ],
          sel_counts: null,
          spearman: 0.93,
          binned_mean: [
            { x: 75, y: 81 },
            { x: 225, y: 85 },
          ],
        }),
        // the last lag has no pairs: the server sends null there, and the chart leaves it out
        [`POST /api/runs/${rid}/stats/acf`]: capture("acf", { lags_m: [30, 60, 90, 120], acf: [0.8, 0.4, 0.1, null], band_mean: [0, 0, 0, null], band_sd: [0.1, 0.1, 0.1, null] }),
      },
      (c) => c.querySelector('[data-tool="breakdown"] figure.chart-frame'),
    );
    expect(bodies.breakdown).toEqual({ value: "obs", by: { kind: "zone" }, weights: null, stat: "box" });
    expect(container.querySelector('[data-tool="breakdown"] figure')!.getAttribute("data-chart")).toBe("Observed by zone");
    click(byText(container, '[role="tab"]', "Relationships"));
    await waitFor(() => container.querySelector('[data-tool="relationships"] figure.chart-frame'), 3000, "hexbin");
    expect(bodies.hexbin).toEqual({ x: "dist_train_m", y: "obs", bins: 60 });
    expect(container.querySelector('[data-tool="relationships"]')!.textContent).toContain("Spearman ρ = 0.93");
    click(byText(container, '[role="tab"]', "Correlogram"));
    await waitFor(() => container.querySelector('[data-tool="correlogram"] figure.chart-frame'), 3000, "correlogram");
    expect(bodies.acf).toEqual({ layer: "obs", n_perm: 19 });
    expect(container.querySelector('[data-tool="correlogram"] figcaption')!.textContent).toContain("60 m");
    expect(new URLSearchParams(window.location.search).get("tool")).toBe("correlogram");
    expect(unframedCharts(container)).toEqual([]);
    m.restore();
  });
});
