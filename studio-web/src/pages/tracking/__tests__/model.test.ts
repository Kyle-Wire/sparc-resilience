// View models of the tracking pages: rail items, reasons, breadcrumbs, per-stage panel data,
// Gantt rows, log filters, run-view routing and the study child matrices.
import { describe, expect, it } from "vitest";
import { replay } from "../../../stores/tracker";
import liveText from "../__fixtures__/s2s3.sample.jsonl?raw";
import sampleText from "../__fixtures__/events.sample.jsonl?raw";
import {
  advectionDecision,
  artifactTab,
  boardCellChip,
  boardCellHref,
  causalChecklist,
  cvCurvePoints,
  cvPartitions,
  defaultStage,
  doseCurves,
  foldModelGrid,
  ganttRows,
  globMatch,
  matchesLog,
  pathCrumbs,
  railItems,
  reasonText,
  scenarioFeed,
  simcheckGrid,
  verdictStatus,
  warningTab,
} from "../model";
import { makeJob, parseJsonl } from "./events";

const sample = parseJsonl(sampleText);
const live = parseJsonl(liveText);
const base = { v: 1, seq: 0, t_rel: 0, pid: 1, job: "j", lvl: "info", parent: null, ctx: {} };

describe("rail and reasons", () => {
  it("builds eleven rail items with reasons, durations and checkpoint markers", () => {
    const s = replay(sample);
    const items = railItems(s);
    expect(items.map((i) => i.id)).toEqual(["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]);
    const by = Object.fromEntries(items.map((i) => [i.id, i]));
    expect(by.S1.reasonText).toBe("loaded from the checkpoint");
    expect(by.S4.elapsed_s).toBeCloseTo(72.5, 6);
    expect(by.S4.checkpoints[0].done).toEqual(["S3", "S4"]);
    expect(by.S2_S3.checkpoints[0].done).toContain("S3");
    expect(by.S7.reasonText).toBe("no budget configured");
  });

  it("explains every reason in words", () => {
    expect(reasonText("disabled_by_config:cv.distance_curve.enabled")).toBe("disabled in the config (cv.distance_curve.enabled)");
    expect(reasonText("required_by:S6")).toBe("needed by S6");
    expect(reasonText("no_partitions")).toContain("partitions");
    expect(reasonText(null)).toBe("");
  });

  it("opens the running stage by default, else the latest that started", () => {
    expect(defaultStage(replay(live))).toBe("S4");
    expect(defaultStage(replay(sample))).toBe("finish");
  });
});

describe("panel data", () => {
  it("reads the nested cv_curve fold grid, partitions and curve", () => {
    const s = replay(sample);
    const parts = cvPartitions(s);
    expect(parts.map((p) => [p.label, p.active])).toEqual([["250 m blocks", false], ["500 m blocks", true]]);
    const g = foldModelGrid(s, "cv_curve", "500 m blocks");
    expect(g.models).toEqual(["ols", "gam"]);
    expect(g.cells[1][1]).toMatchObject({ rmse: 0.61, r2: 0.65, status: "ok" });
    expect(cvCurvePoints(s)).toEqual([
      { label: "250 m blocks", block_m: 250, r2: 0.74, rmse: 0.53 },
      { label: "500 m blocks", block_m: 500, r2: 0.62, rmse: 0.64 },
    ]);
  });

  it("builds dose curves and the scenario feed", () => {
    const s = replay(sample);
    expect(doseCurves(s)).toEqual([
      {
        lever: "canopy",
        running: false,
        points: [
          { dose: 5, key: "5", mean: 0.21, se: 0.05, frac_extrapolated: 0.1, status: "ok" },
          { dose: 10, key: "10", mean: 0.39, se: 0.09, frac_extrapolated: 0.3, status: "ok" },
        ],
      },
    ]);
    expect(scenarioFeed(s).map((r) => [r.name, r.mean_delta, r.se])).toEqual([
      ["Canopy +5", -0.21, 0.05],
      ["Canopy +10", -0.39, 0.09],
    ]);
  });

  it("decides advection like the core rule (kept only beyond one SE)", () => {
    const P = ["run:r", "stage:S2_S3"];
    const evs = [
      { ...base, type: "stage.start", ts: 1, span: "1:1", path: P, stage: "S2_S3" },
      { ...base, type: "task.start", ts: 2, span: "1:2", path: [...P, "task:advection_check"], name: "advection_check" },
      ...[0.9, 1.0, 1.1].flatMap((d, k) => [
        { ...base, type: "task.start", ts: 3 + k, span: `1:${3 + k}`, path: [...P, "task:advection_check", `task:adv_refit[${k + 1}/3]`], name: "adv_refit", k: k + 1, n: 3, unit: "adv_refit" },
        { ...base, type: "task.end", ts: 3.5 + k, span: `1:${3 + k}`, path: [...P, "task:advection_check", `task:adv_refit[${k + 1}/3]`], name: "adv_refit", k: k + 1, n: 3, unit: "adv_refit", status: "ok", elapsed_s: 0.5, metrics: { heldout_rmse_no_adv: 1.0, heldout_rmse_adv: d } },
      ]),
    ];
    expect(advectionDecision(replay(evs))).toMatchObject({ state: "checking", done: 3, total: 3 });
    const done = replay([...evs, { ...base, type: "task.end", ts: 7, span: "1:2", path: [...P, "task:advection_check"], name: "advection_check", status: "ok", elapsed_s: 5, metrics: {} }]);
    const a = advectionDecision(done);
    expect(a.state).toBe("decided");
    if (a.state === "decided") {
      expect(a.kept).toBe(false); // mean 0 is not below −SE
      expect(a.mean).toBeCloseTo(0, 12);
      expect(a.se).toBeCloseTo(0.1 / Math.sqrt(3), 12);
    }
    expect(advectionDecision(replay(live))).toEqual({ state: "none" });
  });

  it("lists treatments, steps, θ ± SE and audit verdicts for S6", () => {
    const P = ["run:r", "stage:S6"];
    const T = [...P, "task:treatment[1/1]"];
    const evs = [
      { ...base, type: "stage.start", ts: 1, span: "1:1", path: P, stage: "S6" },
      { ...base, type: "task.start", ts: 2, span: "1:2", path: [...P, "task:model_effects[1/1]"], name: "model_effects", key: "canopy", k: 1, n: 1 },
      { ...base, type: "task.end", ts: 3, span: "1:2", path: [...P, "task:model_effects[1/1]"], name: "model_effects", key: "canopy", k: 1, n: 1, status: "ok", elapsed_s: 1, metrics: {} },
      { ...base, type: "task.start", ts: 4, span: "1:3", path: T, name: "treatment", key: "canopy", k: 1, n: 1 },
      { ...base, type: "task.start", ts: 5, span: "1:4", path: [...T, "task:dml"], name: "dml", unit: "causal_step:dml" },
      { ...base, type: "task.end", ts: 6, span: "1:4", path: [...T, "task:dml"], name: "dml", unit: "causal_step:dml", status: "ok", elapsed_s: 1, metrics: {} },
      { ...base, type: "task.start", ts: 7, span: "1:5", path: [...T, "task:dr_curve"], name: "dr_curve", unit: "causal_step:dr" },
      { ...base, type: "tick", ts: 8, span: "1:5", path: [...T, "task:dr_curve"], k: 50, n: 200, unit: "bootstrap", frac: 0.25, label: "" },
      { ...base, type: "warning", ts: 9, span: "1:5", path: [...T, "task:audit"], code: "causal.audit_flag", message: "canopy: own_vs_theta_own — magnitude differs", data: { treatment: "canopy", check: "own_vs_theta_own", verdict: "magnitude differs" } },
      { ...base, type: "metric", ts: 9, span: "1:1", path: P, name: "theta", value: -0.041, unit: null, tags: { treatment: "canopy" } },
      { ...base, type: "metric", ts: 9, span: "1:1", path: P, name: "theta_se", value: 0.007, unit: null, tags: { treatment: "canopy" } },
    ];
    const [row] = causalChecklist(replay(evs));
    expect(row.treatment).toBe("canopy");
    expect(row.steps.model_effects.status).toBe("ok");
    expect(row.steps.dml.status).toBe("ok");
    expect(row.steps.dr_curve).toEqual({ status: "running", frac: 0.25 });
    expect(row.theta).toBe(-0.041);
    expect(row.theta_se).toBe(0.007);
    expect(row.verdicts).toEqual([{ check: "own_vs_theta_own", verdict: "magnitude differs" }]);
    expect(verdictStatus("sign conflict").status).toBe("failed");
    expect(verdictStatus("magnitude differs").status).toBe("stale");
    expect(verdictStatus("consistent").status).toBe("done");
  });
});

describe("timeline, crumbs and logs", () => {
  it("writes the current path as a readable breadcrumb", () => {
    const i = sample.findIndex((e) => e.type === "task.start" && e.name === "base_model" && e.path.includes("task:cv_partition[2/2]"));
    const s = replay(sample.slice(0, i + 1));
    expect(pathCrumbs(s, s.current_path)).toEqual(["cv_curve", "500 m blocks", "fold 1/2", "ols"]);
  });

  it("builds Gantt rows and clips them to a zoom window", () => {
    const s = replay(sample);
    const rows = ganttRows(Object.values(s.spans));
    expect(rows.find((r) => r.label === "run demo")).toBeDefined();
    expect(rows.find((r) => r.label.startsWith("cv_curve"))?.status).toBe("ok");
    const s4 = s.stages!.S4;
    const zoomed = ganttRows(Object.values(s.spans), { from: s4.started_ts!, to: s4.ended_ts! });
    expect(zoomed.some((r) => r.label.startsWith("cv partition"))).toBe(false);
    expect(zoomed.some((r) => r.label.startsWith("dose 2/2"))).toBe(true);
  });

  it("filters log lines like the server", () => {
    const line = { cursor: 1, ts: 1, level: "warning", logger: "warning:qa.window", msg: "Fast window: only 40 points", path: ["run:demo", "stage:S0"] };
    expect(matchesLog(line, { level: "info", logger: "", stage: "", q: "" })).toBe(true);
    expect(matchesLog(line, { level: "error", logger: "", stage: "", q: "" })).toBe(false);
    expect(matchesLog(line, { level: "debug", logger: "warning:", stage: "S0", q: "ONLY 40" })).toBe(true);
    expect(matchesLog(line, { level: "debug", logger: "sparc", stage: "", q: "" })).toBe(false);
    expect(matchesLog(line, { level: "debug", logger: "", stage: "S4", q: "" })).toBe(false);
  });
});

describe("routing to run views", () => {
  it("maps warnings and artifacts to run tabs", () => {
    expect(warningTab("qa.window")).toBe("data");
    expect(warningTab("causal.audit_flag")).toBe("causal");
    expect(warningTab("log.sparc.core", null)).toBeNull();
    expect(warningTab("qa.window", "files")).toBe("files");
    const art = { relpath: "response_canopy.parquet", role: "response_map", bytes: 1, stage: "S4", ts: 1 };
    expect(artifactTab(art)).toBe("response");
    expect(artifactTab(art, [{ files: ["response_*.parquet"], view: "map" }])).toBe("map");
    expect(globMatch("studio/**/summary.json", "studio/results/res_1/summary.json")).toBe(true);
    expect(globMatch("response_*.parquet", "sub/response_a.parquet")).toBe(false);
  });

  it("formats board cells and their links", () => {
    expect(boardCellChip({ state: "done", seconds: 4520, progress: null, reason: null, job_id: null, study_id: null, action: null })).toEqual({ status: "done", text: "done", meta: "1 h 15 m" });
    expect(boardCellChip(undefined).text).toBe("not run");
    expect(boardCellHref("r1", "S4", "stage", { state: "done", seconds: 1, progress: null, reason: null, job_id: null, study_id: null, action: null })).toBe("/r/r1/response");
    expect(boardCellHref("r1", "planner", "post", { state: "done", seconds: 1, progress: null, reason: null, job_id: "j1", study_id: null, action: null })).toBe("/r/r1/planner");
    expect(boardCellHref("r1", "planner", "post", { state: "not_run", seconds: null, progress: null, reason: null, job_id: null, study_id: null, action: null })).toBeNull();
  });
});

describe("study children", () => {
  it("fills the simcheck generator × seed grid from replicate tasks and share metrics", () => {
    const evs = [
      { ...base, type: "task.start", ts: 1, span: "1:1", path: ["task:replicate[physics/0]"], name: "replicate", key: "physics/0", unit: "replicate:physics", ctx: { generator: "physics", seed: 0 } },
      { ...base, type: "task.end", ts: 9, span: "1:1", path: ["task:replicate[physics/0]"], name: "replicate", key: "physics/0", unit: "replicate:physics", status: "ok", elapsed_s: 8, metrics: {} },
      { ...base, type: "metric", ts: 9, span: null, path: [], name: "share", value: 0.92, unit: null, tags: { generator: "physics", seed: 0 } },
      { ...base, type: "task.start", ts: 10, span: "1:2", path: ["task:replicate[null/1]"], name: "replicate", key: "null/1", unit: "replicate:null", ctx: {} },
    ];
    const g = simcheckGrid(replay(evs), makeJob({ params: { design: { physics: 2, null: 2 } } }), { grid: [{ generator: "physics", seed: 1, status: "gate_fail", share: null, seconds: null, gate_attempt: 1 }] });
    expect(g.generators).toEqual(["physics", "null"]);
    expect(g.seeds).toEqual([0, 1]);
    expect(g.cells.get("physics/0")).toMatchObject({ status: "done", share: 0.92, seconds: 8 });
    expect(g.cells.get("null/1")?.status).toBe("running");
    expect(g.cells.get("physics/1")).toMatchObject({ status: "gate_fail", redraw: true });
    expect(g.cells.has("null/0")).toBe(false);
  });
});
