// Launch (SPEC §3.2, §5.4) with fetch mocked against the api.md fixtures: mode cards show the
// ETA range, peak RAM and disk of each mode's plan; the stage checklist enforces the
// dependency rules and drives the plan request; Start is disabled while a preflight check
// fails with severity "error"; Start launches and opens Mission Control.
import { afterEach, describe, expect, it } from "vitest";
import type { RunPlanBody } from "../../../api/projects";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, waitFor, type MockHandler } from "../../../test/render";
import Launch from "../Launch";
import { job, PID, PREFLIGHT_ERROR, runPlan, runSummary } from "../__fixtures__/api";
import { projectRoutes, renderAt, resetAll } from "./helpers";

afterEach(() => resetAll());

type PlanCall = RunPlanBody;

function launchRoutes(planFor: (b: PlanCall) => ReturnType<typeof runPlan>, calls: PlanCall[], extra: Record<string, MockHandler> = {}) {
  return projectRoutes({
    [`POST /api/projects/${PID}/runs/plan`]: (_u, init) => {
      const b = JSON.parse(String(init.body)) as PlanCall;
      calls.push(b);
      return { body: planFor(b) };
    },
    "GET /api/meta": { body: { modes: [{ id: "fast", label: "Fast", desc: "Quick check" }, { id: "coarse", label: "Coarse", desc: "Coarser cells", default_coarse_m: 60 }, { id: "full", label: "Full", desc: "Everything" }] } },
    ...extra,
  });
}

const startButton = (root: ParentNode) => root.querySelector<HTMLButtonElement>('button[data-start="true"]')!;

describe("launch", () => {
  it("mode cards show each mode's time range, peak RAM and disk; Fast is preselected", async () => {
    const calls: PlanCall[] = [];
    const m = mockFetch(
      launchRoutes(
        (b) => (b.mode === "full" ? runPlan({ est_lo: 6000, est_hi: 6900, est_peak_rss_gb: 3.1, est_disk_gb: 0.53 }) : b.mode === "coarse" ? runPlan({ est_lo: 1320, est_hi: 1680, est_peak_rss_gb: 1.6, est_disk_gb: 0.13 }) : runPlan({ est_lo: 240, est_hi: 360 })),
        calls,
      ),
    );
    const { container } = renderAt(`/p/${PID}/launch`, <Launch />);
    await waitFor(() => container.querySelector('[data-mode="full"] [data-est="time"]')?.textContent?.includes("h"), 5000, "estimates");
    const est = (mode: string, k: string) => container.querySelector(`[data-mode="${mode}"] [data-est="${k}"]`)!.textContent;
    expect(est("fast", "time")).toBe("≈4–6 min");
    expect(est("coarse", "time")).toBe("≈22–28 min");
    expect(est("full", "time")).toBe("≈1 h 40 m–1 h 55 m");
    expect(est("full", "ram")).toBe("≈3.1 GB");
    expect(est("full", "disk")).toBe("≈530 MB");
    expect(container.querySelector('[data-mode="fast"] .mode-pick')!.getAttribute("aria-pressed")).toBe("true");
    expect(container.querySelector('[data-mode="coarse"] .mode-name')!.textContent).toBe("Coarse 60 m");
    const coarse = calls.find((c) => c.mode === "coarse")!;
    expect(coarse.coarse_m).toBe(60);
    expect(calls.find((c) => c.mode === "fast")!.stages).toEqual(["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]);
    // the plan graph is the selected mode's
    expect(container.querySelector('li.plan-node[data-node="S4"] .plan-reason')!.textContent).toContain("required by S6");
    m.restore();
  });

  it("the stage checklist enforces dependencies and replans with the explicit stages", async () => {
    const calls: PlanCall[] = [];
    const m = mockFetch(launchRoutes(() => runPlan(), calls));
    const { container } = renderAt(`/p/${PID}/launch`, <Launch />);
    await waitFor(() => container.querySelector('[data-stage="S4"] input'), 5000, "checklist");
    const box = (id: string) => container.querySelector<HTMLInputElement>(`[data-stage="${id}"] input`)!;
    // full run: S4 and S2–S3 are forced by the later stages
    expect(box("S0").disabled).toBe(true);
    expect(box("S4").disabled).toBe(true);
    expect(container.querySelector('[data-stage="S4"] .stage-reason')!.textContent).toBe("(required by S6)");
    // uncheck S5, S6, S7: S4 becomes a free choice again
    for (const id of ["S5", "S6", "S7"]) click(box(id));
    await flush(3);
    expect(box("S4").disabled).toBe(false);
    expect(box("S4").checked).toBe(true);
    expect(container.querySelector('[data-stage="S2_S3"] .stage-reason')!.textContent).toBe("(required by S4)");
    await waitFor(() => calls.some((c) => c.mode === "fast" && JSON.stringify(c.stages) === JSON.stringify(["S0", "S1", "S2", "S3", "S4"])), 5000, "replan");
    // and back: choosing S6 alone locks S4 again
    click(box("S4"));
    click(box("S6"));
    await flush(3);
    expect(box("S4").disabled).toBe(true);
    expect(box("S4").checked).toBe(true);
    await waitFor(() => calls.some((c) => JSON.stringify(c.stages) === JSON.stringify(["S0", "S1", "S2", "S3", "S6"])), 5000, "replan S6");
    m.restore();
  });

  it("a deep link prefills mode, coarse cell, stages and CV curve (Mission Control's Duplicate with changes)", async () => {
    const calls: PlanCall[] = [];
    const m = mockFetch(launchRoutes(() => runPlan(), calls));
    const { container } = renderAt(`/p/${PID}/launch?mode=coarse&coarse=90&stages=S0,S1,S2_S3,S4&cv=off`, <Launch />);
    await waitFor(() => calls.some((c) => c.mode === "coarse"), 5000, "coarse plan");
    expect(calls.find((c) => c.mode === "coarse")).toMatchObject({ mode: "coarse", coarse_m: 90, stages: ["S0", "S1", "S2", "S3", "S4"], cv_curve: false });
    expect(container.querySelector('[data-mode="coarse"] .mode-pick')!.getAttribute("aria-pressed")).toBe("true");
    expect(container.querySelector('[data-mode="coarse"] .mode-name')!.textContent).toBe("Coarse 90 m");
    expect(container.querySelector<HTMLInputElement>('[data-stage="S6"] input')!.checked).toBe(false);
    m.restore();
  });

  it("?from=<run_id> prefills from that run's launch snapshot (the Re-run with … remedy), explicit parameters win", async () => {
    const calls: PlanCall[] = [];
    const rid = "20261001-120000-coarse-ab12";
    const detail = {
      run: runSummary({ id: rid, label: "first coarse", mode: "coarse", coarse_m: 120, status: "complete" }),
      launch: { project_id: PID, args: { stages: ["S0", "S1", "S2", "S3", "S4", "S5"], fast: false, coarse: 120, cv_curve: false, threads: 3 } },
    };
    const m = mockFetch(launchRoutes(() => runPlan(), calls, { [`GET /api/runs/${rid}`]: { body: detail } }));
    const { container } = renderAt(`/p/${PID}/launch?from=${rid}`, <Launch />);
    await waitFor(() => container.querySelector(`[data-prefill="${rid}"]`), 5000, "prefill note");
    await waitFor(() => calls.some((c) => c.mode === "coarse" && c.coarse_m === 120 && c.threads === 3), 5000, "prefilled plan");
    expect(calls.find((c) => c.mode === "coarse" && c.coarse_m === 120)).toMatchObject({ stages: ["S0", "S1", "S2", "S3", "S4", "S5"], cv_curve: false, threads: 3 });
    expect(container.querySelector('[data-mode="coarse"] .mode-pick')!.getAttribute("aria-pressed")).toBe("true");
    expect(container.querySelector<HTMLInputElement>('[data-stage="S6"] input')!.checked).toBe(false);
    expect(container.querySelector(`[data-prefill="${rid}"]`)!.textContent).toContain("first coarse");
    const q = new URLSearchParams(window.location.search);
    expect(q.get("from")).toBeNull();
    expect(q.get("stages")).toBe("S0,S1,S2_S3,S4,S5");
    expect(q.get("cv")).toBe("off");
    m.restore();
    resetAll();
    // a parameter in the URL is kept over the run's
    const m2 = mockFetch(launchRoutes(() => runPlan(), [], { [`GET /api/runs/${rid}`]: { body: detail } }));
    const r2 = renderAt(`/p/${PID}/launch?from=${rid}&mode=full`, <Launch />);
    await waitFor(() => r2.container.querySelector(`[data-prefill="${rid}"]`), 5000, "prefill note");
    await flush(3);
    expect(r2.container.querySelector('[data-mode="full"] .mode-pick')!.getAttribute("aria-pressed")).toBe("true");
    expect(new URLSearchParams(window.location.search).get("cv")).toBe("off");
    m2.restore();
  });

  it("Start is disabled while preflight has an error, with the action offered", async () => {
    const m = mockFetch(launchRoutes(() => runPlan({ preflight: [...runPlan().preflight, PREFLIGHT_ERROR] }), []));
    const { container } = renderAt(`/p/${PID}/launch`, <Launch />);
    await waitFor(() => container.querySelector('[data-check="data_file"]'), 5000, "preflight");
    expect(startButton(container).disabled).toBe(true);
    const row = container.querySelector('[data-check="data_file"]')!;
    expect(row.getAttribute("data-state")).toBe("error");
    expect(row.textContent).toContain("Blocks the launch");
    expect(row.textContent).toContain("data/city.csv is missing");
    expect(byText(row, "button", "Open the data step")).not.toBeNull();
    expect(container.querySelector(".start-bar")!.textContent).toContain("1 preflight check blocks the launch");
    m.restore();
  });

  it("a warning does not block; config errors do", async () => {
    const m1 = mockFetch(launchRoutes(() => runPlan({ preflight: [{ ...PREFLIGHT_ERROR, severity: "warn" }] }), []));
    const r1 = renderAt(`/p/${PID}/launch`, <Launch />);
    await waitFor(() => r1.container.querySelector('[data-check="data_file"]'), 5000, "preflight");
    expect(startButton(r1.container).disabled).toBe(false);
    r1.unmount();
    m1.restore();
    resetAll();
    const m2 = mockFetch(launchRoutes(() => runPlan({ issues: [{ level: "error", path: "data.target", code: "required", message: "data.target is required" }] }), []));
    const r2 = renderAt(`/p/${PID}/launch`, <Launch />);
    await waitFor(() => r2.container.querySelector(".callout[data-tone='crit']"), 5000, "config errors");
    expect(startButton(r2.container).disabled).toBe(true);
    m2.restore();
  });

  it("Start launches the selected mode with the chain and opens Mission Control", async () => {
    const launched: unknown[] = [];
    const m = mockFetch(
      launchRoutes(() => runPlan(), [], {
        [`POST /api/projects/${PID}/runs`]: (_u, init) => {
          launched.push(JSON.parse(String(init.body)));
          return { status: 202, body: { run: runSummary(), job: job(), chain: [job({ id: "j_chain1", kind: "post.writeup", after_job_id: "j_launch1" })] } };
        },
      }),
    );
    const { container } = renderAt(`/p/${PID}/launch?mode=full`, <Launch />);
    await waitFor(() => !startButton(container)?.disabled, 5000, "start enabled");
    click(byText(container, '[aria-label="Then run"] button', "Methods & model card"));
    click(startButton(container));
    await waitFor(() => window.location.pathname === "/jobs/j_launch1", 5000, "mission control");
    expect(launched[0]).toMatchObject({ mode: "full", stages: ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"], then: ["post.writeup"], cv_curve: null });
    expect(useJobs.getState().jobs.j_launch1?.kind).toBe("run.core");
    expect(useJobs.getState().jobs.j_chain1?.after_job_id).toBe("j_launch1");
    m.restore();
  });
});
