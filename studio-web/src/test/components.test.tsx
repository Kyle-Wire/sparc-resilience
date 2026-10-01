import { afterEach, describe, expect, it } from "vitest";
import { ApiError, errorFromResponse } from "../api/client";
import type { Job, Likely } from "../api/types";
import { filterCommands } from "../components/ui/CommandPalette";
import { ConnectionPill } from "../components/ui/ConnectionPill";
import { collapseContext, Diff, diffLines } from "../components/ui/Diff";
import { EmptyState } from "../components/ui/EmptyState";
import { JobStrip } from "../components/ui/JobStrip";
import { Markdown, renderMarkdown } from "../components/ui/Markdown";
import { NumberField } from "../components/ui/NumberField";
import { confidenceOf, likelySentence, PlainResult } from "../components/ui/PlainResult";
import { ProgressBar } from "../components/ui/ProgressBar";
import { Seg } from "../components/ui/Seg";
import { STATUS, StatusChip } from "../components/ui/StatusChip";
import { Table, tableCsv } from "../components/ui/Table";
import { StreamManager } from "../api/sse";
import { decodeSelection, encodeSelection, isSelectionSpec } from "../stores/selection";
import { useJobs } from "../stores/jobs";
import { useUi } from "../stores/ui";
import { byText, click, flush, key, mockFetch, render, typeInto } from "./render";

afterEach(() => {
  useJobs.getState().reset();
  useUi.setState({ toasts: [] });
});

describe("EmptyState renders ApiError.action as a button", () => {
  it("names the producing stage and runs the remedy", async () => {
    const err = errorFromResponse(404, {
      error: {
        code: "output_missing",
        message: "causal.json is not in this run.",
        detail: { output: "causal", produced_by: "stage:S6" },
        action: { kind: "resume", label: "Resume to compute S6", method: "POST", path: "/api/runs/r1/resume", body: { threads: 2 } },
      },
    });
    const job = { id: "j_9", status: "queued", label: "Resume fast run", kind: "run.core" } as Job;
    const m = mockFetch({ "POST /api/runs/r1/resume": { status: 202, body: job } });
    const { container } = render(<EmptyState error={err} />);
    expect(container.textContent).toContain("Not in this run yet");
    expect(container.textContent).toContain("produced by stage S6");
    const btn = byText(container, "button", "Resume to compute S6")!;
    expect(btn).not.toBeNull();
    click(btn);
    await flush(4);
    m.restore();
    expect(m.calls[0]).toMatchObject({ method: "POST", url: "/api/runs/r1/resume", body: { threads: 2 } });
    expect(useJobs.getState().jobs.j_9?.label).toBe("Resume fast run");
    expect(useUi.getState().toasts.at(-1)).toMatchObject({ title: "Resume fast run started", href: "/jobs/j_9" });
  });
  it("in-app paths navigate instead of posting", () => {
    const err = new ApiError(404, "no_emulator", "No emulator for this run", null, { kind: "open", label: "Open Lab", path: "/r/r1/lab" });
    const { container } = render(<EmptyState error={err} />);
    click(byText(container, "button", "Open Lab"));
    expect(window.location.pathname).toBe("/r/r1/lab");
    window.history.replaceState(null, "", "/");
  });
  it("explains generic errors and plain empty states", () => {
    const r = render(<EmptyState title="No runs yet" body="Launch a run to see it here." />);
    expect(r.container.querySelector('[role="status"]')!.textContent).toContain("No runs yet");
    const net = render(<EmptyState error={new ApiError(0, "network", "Studio server is not reachable")} />);
    expect(net.container.textContent).toContain("Studio server not reachable");
  });
});

describe("Markdown", () => {
  it("escapes raw HTML and neutralises unsafe links and remote images", () => {
    const html = renderMarkdown('# Report\n\n<script>alert(1)</script>\n\n[ok](/r/r1) [bad](javascript:alert(1)) ![x](https://example.com/x.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |');
    expect(html).toContain("<h1>Report</h1>");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain('href="/r/r1"');
    expect(html).not.toContain("javascript:");
    expect(html).not.toContain("<img");
    expect(html).toContain("<table>");
    const { container } = render(<Markdown source="Hello **world**" />);
    expect(container.querySelector(".md strong")!.textContent).toBe("world");
  });

  it("never emits a script URL, however it is encoded", () => {
    const payloads = [
      "[a](javascript:alert(1))",
      "[a](JaVaScRiPt:alert(1))",
      "[a](&#106;avascript:alert(1))",
      "[a](javascript&colon;alert(1))",
      "[a](java&#x09;script:alert(1))",
      "[a](java&#x0A;script:alert(1))",
      "[a](<java\tscript:alert(1)>)",
      "[a](data:text/html;base64,PHNjcmlwdD4=)",
      "[a](vbscript:msgbox(1))",
      "[a]: javascript&colon;alert(1)\n\n[x][a]",
      "![x](javascript&colon;alert(1))",
      "![x](data:image/svg+xml;base64,PHN2Zz4=)",
      "<a href=javascript:alert(1)>x</a>",
    ];
    for (const p of payloads) {
      // Parse the HTML the way the browser will and inspect the resulting URLs.
      const host = document.createElement("div");
      host.innerHTML = renderMarkdown(p);
      for (const el of host.querySelectorAll("[href], [src]")) {
        const url = (el.getAttribute("href") ?? el.getAttribute("src") ?? "").replace(/[\u0000- ]/g, "").toLowerCase();
        expect(url, p).not.toMatch(/^(javascript|vbscript|data:text|data:image\/svg)/);
      }
      expect(host.querySelector("a[href^='java'], script, img[src^='java']"), p).toBeNull();
    }
  });

  it("keeps ordinary links, titles and local images intact", () => {
    const host = document.createElement("div");
    host.innerHTML = renderMarkdown('[docs](https://example.org/a?x=1&y=2 "Read \\"this\\"") [run](/r/r1/accuracy) [mail](mailto:a@b.org) [top](#methods) ![map](figures/map.png "Map")');
    const links = [...host.querySelectorAll("a")].map((a) => [a.getAttribute("href"), a.getAttribute("title")]);
    expect(links).toEqual([
      ["https://example.org/a?x=1&y=2", 'Read "this"'],
      ["/r/r1/accuracy", null],
      ["mailto:a@b.org", null],
      ["#methods", null],
    ]);
    const img = host.querySelector("img")!;
    expect([img.getAttribute("src"), img.getAttribute("alt"), img.getAttribute("title")]).toEqual(["figures/map.png", "map", "Map"]);
    expect(renderMarkdown("[x](//evil.example/p)")).not.toContain("href");
  });
});

describe("Diff", () => {
  it("diffs lines and collapses unchanged runs", () => {
    const a = "a\nb\nc\nd\ne\nf\ng\nh";
    const b = "a\nb\nc\nD\ne\nf\ng\nh\ni";
    const d = diffLines(a, b);
    expect(d.filter((x) => x.op === "del").map((x) => x.text)).toEqual(["d"]);
    expect(d.filter((x) => x.op === "add").map((x) => x.text)).toEqual(["D", "i"]);
    const rows = collapseContext(d, 1);
    expect(rows.some((r) => r.op === "skip")).toBe(true);
    const { container } = render(<Diff a={"x: 1\ny: 2"} b={"x: 1\ny: 3"} />);
    expect(container.querySelector(".del")!.textContent).toContain("y: 2");
    expect(container.querySelector(".add")!.textContent).toContain("y: 3");
    expect(container.querySelector(".add")!.textContent).toContain("added:");
  });
});

describe("PlainResult wording", () => {
  const L = (estimate: number, lo: number | null, hi: number | null): Likely => ({ estimate, se: 0.05, lo, hi, confidence: "unknown", phrase: "" });
  it("is confident it cools iff hi < 0", () => {
    expect(confidenceOf(L(-0.41, -0.52, -0.3))).toBe("confident_cools");
    expect(confidenceOf(L(-0.2, -0.5, 0.1))).toBe("could_be_zero");
    expect(confidenceOf(L(0.2, 0.1, 0.3))).toBe("confident_warms");
    expect(confidenceOf(L(0, null, null))).toBe("unknown");
    expect(likelySentence(L(-0.41, -0.52, -0.3), "degF", "the edited area")).toBe("Cools the edited area by 0.41 °F (likely range 0.30–0.52 °F).");
  });
  it("adds the extrapolation qualifier above 20% and has an expert toggle", () => {
    const { container } = render(<PlainResult likely={L(-0.41, -0.52, -0.3)} unit="degF" subject="the edited area" fracExtrapolated={0.35} expert={{ n: 1284, method: "jackknife over 5 folds" }} />);
    expect(container.textContent).toContain("Confident it cools.");
    expect(container.textContent).toContain("35% of the edited cells are outside");
    click(byText(container, "button", "Expert details"));
    expect(container.textContent).toContain("jackknife over 5 folds");
    const z = render(<PlainResult likely={L(-0.1, -0.3, 0.1)} unit="degF" subject="the city" fracExtrapolated={0.1} />);
    expect(z.container.textContent).toContain("Could be zero.");
    expect(z.container.textContent).not.toContain("outside the conditions");
  });
});

describe("StatusChip", () => {
  it("always carries text and an icon", () => {
    for (const s of ["queued", "running", "succeeded", "failed", "cancelled", "interrupted", "cached", "skipped", "stale", "missing", "blocked"]) {
      const { container } = render(<StatusChip status={s} />);
      const chip = container.querySelector(".status")!;
      expect(chip.querySelector("svg")).not.toBeNull();
      expect(chip.textContent).toBe(STATUS[s].text);
    }
    const { container } = render(<StatusChip status="skipped" text="skipped: no budget" meta="0 s" />);
    expect(container.textContent).toBe("skipped: no budget0 s");
  });
});

describe("Table", () => {
  type Row = { name: string; r2: number | null };
  const rows: Row[] = [
    { name: "mgwr", r2: 0.71 },
    { name: "gam", r2: null },
    { name: "stack", r2: 0.78 },
  ];
  const cols = [
    { key: "name", label: "Model", value: (r: Row) => r.name },
    { key: "r2", label: "R²", align: "right" as const, value: (r: Row) => r.r2 },
  ];
  it("sorts (missing values last) and exports CSV", () => {
    const { container } = render(<Table columns={cols} rows={rows} csvName="models" />);
    const names = () => [...container.querySelectorAll("tbody tr td:first-child")].map((td) => td.textContent);
    click(byText(container, "th button", "R²"));
    expect(names()).toEqual(["mgwr", "stack", "gam"]);
    click(byText(container, "th button", "R²"));
    expect(names()).toEqual(["stack", "mgwr", "gam"]);
    expect(container.querySelector('th[aria-sort="descending"]')).not.toBeNull();
    expect(tableCsv(cols, rows)).toBe("Model,R²\nmgwr,0.71\ngam,\nstack,0.78\n");
  });
  it("virtualises long tables", () => {
    const many = Array.from({ length: 2000 }, (_, i) => ({ name: `m${i}`, r2: i / 2000 }));
    const { container } = render(<Table columns={cols} rows={many} maxHeight={330} />);
    const rendered = container.querySelectorAll("tbody tr:not(.spacer-row)").length;
    expect(rendered).toBeGreaterThan(5);
    expect(rendered).toBeLessThan(60);
  });
});

describe("small controls", () => {
  it("Seg is a pressed-button group with arrow keys", () => {
    let v = "a";
    const { container, rerender } = render(<Seg label="Mode" options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} value={v} onChange={(x) => (v = x)} />);
    expect(container.querySelector('[aria-pressed="true"]')!.textContent).toBe("A");
    key(container.querySelector('[role="group"]'), "ArrowRight");
    expect(v).toBe("b");
    rerender(<Seg label="Mode" options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} value={v} onChange={(x) => (v = x)} />);
    expect(container.querySelector('[aria-pressed="true"]')!.textContent).toBe("B");
  });
  it("NumberField accepts U+2212 and clamps on commit", () => {
    let v: number | null = 0;
    const { container } = render(<NumberField label="Amount" value={v} min={-5} max={5} onChange={(x) => (v = x)} unit="pp" />);
    const input = container.querySelector("input")!;
    typeInto(input, "−7");
    key(input, "Enter");
    expect(v).toBe(-5);
    expect(container.textContent).toContain("pp");
  });
  it("ProgressBar exposes its value", () => {
    const { container } = render(<ProgressBar value={0.42} label="Run" />);
    const bar = container.querySelector('[role="progressbar"]')!;
    expect(bar.getAttribute("aria-valuenow")).toBe("42");
    const ind = render(<ProgressBar value={null} label="Run" />);
    expect(ind.container.querySelector('[role="progressbar"]')!.getAttribute("data-indeterminate")).toBe("true");
  });
  it("filters commands by every word", () => {
    const cmds = [
      { id: "1", label: "Accuracy", group: "This run", run: () => {} },
      { id: "2", label: "Open run full overnight", group: "Run", keywords: "20261001", run: () => {} },
      { id: "3", label: "Theme: dark", group: "Action", run: () => {} },
    ];
    expect(filterCommands(cmds, "run full").map((c) => c.id)).toEqual(["2"]);
    expect(filterCommands(cmds, "2026").map((c) => c.id)).toEqual(["2"]);
    expect(filterCommands(cmds, "").length).toBe(3);
  });
});

describe("JobStrip and ConnectionPill", () => {
  it("JobStrip shows the run's active job with stage, ETA and progress", () => {
    useJobs.getState().setActive([{ id: "j_1", kind: "run.core", lane: "heavy", label: "Full run", status: "running", run_id: "r1", created_utc: "2026-10-01T00:00:00Z", progress: 0.3, stage: "S4", eta_s: 600, eta_lo: 500, eta_hi: 800 } as Job]);
    useJobs.getState().applyGlobal([{ type: "job.progress", gseq: 1, ts: 1, job_id: "j_1", frac: 0.42, eta_s: 720, eta_lo: 600, eta_hi: 900, stage: "S2_S3", path_tail: ["fold 4/5", "mgwr"] }]);
    const { container } = render(<JobStrip runId="r1" />);
    expect(container.querySelector("a")!.getAttribute("href")).toBe("/jobs/j_1");
    expect(container.textContent).toContain("S2_S3");
    expect(container.textContent).toContain("fold 4/5 › mgwr");
    expect(container.textContent).toContain("≈12 min (10–15 min)");
    expect(container.querySelector('[role="progressbar"]')!.getAttribute("aria-valuenow")).toBe("42");
    expect(render(<JobStrip runId="other" />).container.textContent).toBe("");
  });
  it("ConnectionPill reflects the stream state", async () => {
    const m = new StreamManager({ EventSource: class { onopen = null; onerror = null; onmessage = null; addEventListener() {} close() {} } as never, frame: (cb) => cb() });
    const { container } = render(<ConnectionPill manager={m} />);
    expect(container.querySelector(".conn-pill")!.getAttribute("data-state")).toBe("connecting");
    expect(container.textContent).toContain("connecting");
    await flush();
  });
});

describe("selection URL codec", () => {
  it("regions are ids, other specs are compact base64url JSON", () => {
    expect(encodeSelection({ kind: "region", id: "rg_12" })).toBe("rg_12");
    expect(encodeSelection({ kind: "all" })).toBeNull();
    const spec = { op: "and" as const, args: [{ kind: "zones" as const, values: [3] }, { kind: "filter" as const, column: "layer:lc_built", op: ">=" as const, value: 0.5 }] };
    const enc = encodeSelection(spec)!;
    expect(enc.startsWith("s.")).toBe(true);
    expect(enc).toMatch(/^s\.[A-Za-z0-9_-]+$/);
    expect(decodeSelection(enc)).toEqual(spec);
    expect(decodeSelection("rg_12")).toEqual({ kind: "region", id: "rg_12" });
    expect(decodeSelection("s.@@@")).toBeNull();
    expect(isSelectionSpec({ op: "not", arg: { kind: "nonsense" } })).toBe(false);
  });
});
