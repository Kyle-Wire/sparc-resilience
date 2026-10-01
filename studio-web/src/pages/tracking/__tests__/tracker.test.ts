// The tracker reducer (stores/tracker.ts): replay of the hand-written fixture, unit accounting,
// warnings dedup, purity, snapshot resume, nested children, ETA and the Python golden.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  applyEvent,
  applyEvents,
  childrenRows,
  foldBatch,
  fromSnapshot,
  metricKey,
  newTrackerState,
  projectionEta,
  pyJsonList,
  replay,
  sha1Hex,
  stageOf,
  toContract,
  toSnapshot,
  type TrackerState,
} from "../../../stores/tracker";
import sampleText from "../__fixtures__/events.sample.jsonl?raw";
import liveText from "../__fixtures__/s2s3.sample.jsonl?raw";
import { makeJob, parseJsonl } from "./events";

const sample = parseJsonl(sampleText);
const live = parseJsonl(liveText);

function deepFreeze<T>(o: T): T {
  if (o && typeof o === "object" && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o as object)) deepFreeze(v);
  }
  return o;
}

/** Deep comparison with a relative float tolerance (the Python golden prints rounded floats). */
function expectClose(a: unknown, b: unknown, path = "$"): void {
  if (typeof a === "number" || typeof b === "number") {
    expect(typeof a, path).toBe("number");
    expect(typeof b, path).toBe("number");
    const x = a as number;
    const y = b as number;
    expect(Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(x), Math.abs(y)), `${path}: ${x} vs ${y}`).toBe(true);
    return;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    expect(Array.isArray(a) && Array.isArray(b), path).toBe(true);
    expect((a as unknown[]).length, path).toBe((b as unknown[]).length);
    (a as unknown[]).forEach((v, i) => expectClose(v, (b as unknown[])[i], `${path}[${i}]`));
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    expect(Object.keys(a).sort(), path).toEqual(Object.keys(b).sort());
    for (const k of Object.keys(a)) expectClose((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`);
    return;
  }
  expect(a, path).toEqual(b);
}

const plain = (s: unknown) => JSON.parse(JSON.stringify(s)) as unknown;

describe("applyEvent replay of events.sample.jsonl", () => {
  const final = replay(sample);

  it("yields the expected stage states and reasons", () => {
    expect(toContract(final).stages).toEqual({
      S0: { state: "done", reason: null },
      S1: { state: "cached", reason: "checkpoint" },
      S2_S3: { state: "cached", reason: "checkpoint" },
      S4: { state: "done", reason: null },
      S5: { state: "done", reason: null },
      S6: { state: "skipped", reason: "no_treatments" },
      S7: { state: "skipped", reason: "no_budget" },
      baselines: { state: "disabled", reason: "disabled_by_config:baselines.enabled" },
      climate: { state: "disabled", reason: "disabled_by_config:climate.enabled" },
      cv_curve: { state: "done", reason: null },
      finish: { state: "done", reason: null },
    });
  });

  it("marks the checkpoint stages cached only after the re-emitted plan and stage.skip", () => {
    const iSkip = sample.findIndex((e) => e.type === "stage.skip" && e.stage === "S2_S3");
    const iPlan2 = sample.findIndex((e, i) => e.type === "run.plan" && i > 1);
    const before = replay(sample.slice(0, iPlan2));
    expect(before.stages!.S2_S3.state).toBe("planned");
    // the first plan counts S2_S3's 2 gam fits plus cv_curve's 4
    expect(before.planned_units["base_fit:gam"]).toBe(6);
    const afterPlan = replay(sample.slice(0, iPlan2 + 1));
    expect(afterPlan.stages!.S2_S3).toMatchObject({ state: "cached", reason: "checkpoint" });
    // cached nodes leave the planned units: only cv_curve's 4 gam fits remain
    expect(afterPlan.planned_units["base_fit:gam"]).toBe(4);
    expect(afterPlan.planned_units.s1_influence).toBeUndefined();
    expect(replay(sample.slice(0, iSkip + 1)).stages!.S2_S3.state).toBe("cached");
  });

  it("has monotonic cost-weighted progress that ends at 1", () => {
    let state = newTrackerState();
    const seen: number[] = [];
    for (const ev of sample) {
      state = applyEvent(state, ev);
      seen.push(state.progress ?? 0);
    }
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1] - 1e-12);
    expect(seen[seen.length - 1]).toBe(1);
    // right after S0 and the re-emitted plan: 0.3 s of s0_load out of 95.5 weighted seconds
    // (cv_curve 26.0 + S4 3×13 + 2.1 + S5 2×13 + 2.1 + s0_load 0.3)
    const iPlan2 = sample.findIndex((e, i) => e.type === "run.plan" && i > 1);
    expect(replay(sample.slice(0, iPlan2 + 1)).progress).toBeCloseTo(0.3 / 95.5, 12);
    // half an engine pass (a k=1/2 tick) counts as half a unit
    const iTick = sample.findIndex((e) => e.type === "tick" && e.span !== null && e.k === 1 && e.path.includes("task:engine_init"));
    const a = replay(sample.slice(0, iTick));
    const b = replay(sample.slice(0, iTick + 1));
    expect((b.progress! - a.progress!) * 95.5).toBeCloseTo(13 * 0.5, 9);
  });

  it("counts planned units completed by task.end, k == n ticks, S0 end and saved checkpoints", () => {
    expect(final.done_units).toEqual({
      s0_load: 1,
      "base_fit:ols": 4,
      "base_fit:gam": 4,
      "stacker_fit:mean": 4,
      "stacker_fit:nnls": 4,
      engine_pass: 5,
      checkpoint_save: 2,
    });
    expect(final.stage_done.cv_curve).toEqual({ "base_fit:ols": 4, "base_fit:gam": 4, "stacker_fit:mean": 4, "stacker_fit:nnls": 4 });
    expect(final.stage_done.S4).toEqual({ engine_pass: 3, checkpoint_save: 1 });
    expect(final.partial).toEqual({});
  });

  it("deduplicates warnings by code and message and counts repeats", () => {
    const rows = Object.values(final.warnings);
    expect(rows).toHaveLength(3);
    const gam = rows.find((w) => w.message === "GAM smoothing reached the upper bound")!;
    expect(gam.count).toBe(2);
    expect(gam.stage).toBe("cv_curve");
    expect(toContract(final).warnings).toEqual([
      { code: "log.sparc.core.base_models", count: 3 },
      { code: "qa.window", count: 1 },
    ]);
  });

  it("keeps one artifact row per path (a rewrite updates it), checkpoints and the sleep gap", () => {
    expect(toContract(final).artifacts).toEqual([
      "cv_curve.json", "influence.json", "manifest.json", "report.md", "response_canopy.parquet", "response_curves.json", "scenario_deltas.parquet", "scenarios.json",
    ]);
    expect(final.artifacts["scenarios.json"]).toMatchObject({ bytes: 3265, stage: "finish" });
    expect(final.checkpoints.map((c) => c.action)).toEqual(["loaded", "saved", "saved"]);
    expect(final.heartbeat_gaps).toEqual([{ from_ts: 1790900015, to_ts: 1790900090 }]);
  });

  it("records metric series, current path, run and job status", () => {
    expect(final.metric_series["cv_row.r2{partition=250 m blocks}"].map((p) => p.value)).toEqual([0.74]);
    expect(final.metrics_latest["candidate_rmse{candidate=nnls}"].value).toBe(0.64);
    expect(final.metrics_latest["heldout_rmse{key=gam,task=base_model}"].value).toBe(0.61);
    expect(final.current_path).toBeNull();
    expect(final.status).toBe("succeeded");
    expect(final.run_status).toBe("succeeded");
    expect(final.run?.resume).toBe(true);
    const mid = replay(sample.slice(0, sample.findIndex((e) => e.type === "task.start" && e.name === "base_model") + 1));
    expect(mid.current_path).toEqual(["run:demo", "stage:cv_curve", "task:cv_partition[1/2]", "task:fold[1/2]", "task:base_model[ols]"]);
    expect(mid.stage).toBe("cv_curve");
    expect(mid.stages!.cv_curve.state).toBe("running");
  });
});

describe("purity and structural sharing", () => {
  it("never mutates its input state", () => {
    const half = deepFreeze(replay(sample.slice(0, 60)));
    const snapshot = JSON.stringify(half);
    const next = applyEvents(half, sample.slice(60));
    expect(JSON.stringify(half)).toBe(snapshot);
    expect(next).not.toBe(half);
  });

  it("keeps untouched containers by identity", () => {
    const a = replay(sample.slice(0, 40));
    const metricEv = sample.find((e, i) => i >= 40 && e.type === "metric")!;
    const b = applyEvent(a, metricEv);
    expect(b.spans).toBe(a.spans);
    expect(b.warnings).toBe(a.warnings);
    expect(b.metrics_latest).not.toBe(a.metrics_latest);
  });
});

describe("snapshot then events after the cursor", () => {
  const full = replay(sample);
  const job = makeJob();

  it("equals applying every event from the first one (server projection attached)", () => {
    for (let cut = 0; cut < sample.length; cut += 7) {
      const at = replay(sample.slice(0, cut + 1));
      const snap = JSON.parse(JSON.stringify(toSnapshot(at, job)));
      expect(snap.cursor).toBe(sample[cut].cursor);
      const { state, reconstructed } = fromSnapshot(snap);
      expect(reconstructed).toBe(false);
      const rest = sample.filter((e) => e.cursor > snap.cursor);
      const resumed = applyEvents(state, rest);
      expect(plain(resumed)).toEqual(plain(full));
      expect(toContract(resumed)).toEqual(toContract(full));
    }
  });

  it("rebuilds counters from the wire fields when the server sends no projection", () => {
    for (let cut = 3; cut < sample.length; cut += 5) {
      const at = replay(sample.slice(0, cut + 1));
      const snap = JSON.parse(JSON.stringify(toSnapshot(at, makeJob({ status: at.status ?? "running" }))));
      delete snap.projection;
      const { state, reconstructed } = fromSnapshot(snap);
      expect(reconstructed).toBe(true);
      // what the snapshot shows is carried over exactly
      expect(toContract(state).stages).toEqual(toContract(at).stages);
      expect(toContract(state).warnings).toEqual(toContract(at).warnings);
      expect(toContract(state).artifacts).toEqual(toContract(at).artifacts);
      // progress is rebuilt from the stages' own progress
      expect(Math.abs((state.progress ?? 0) - (at.progress ?? 0))).toBeLessThan(1e-6);
      const resumed = applyEvents(state, sample.filter((e) => e.cursor > snap.cursor));
      const c = toContract(resumed);
      expect(c.stages).toEqual(toContract(full).stages);
      expect(c.warnings).toEqual(toContract(full).warnings);
      expect(c.artifacts).toEqual(toContract(full).artifacts);
      expect(c.progress).toBe(1);
    }
  });

  it("rebuilds span paths, running spans and the current path", () => {
    const iMgwr = live.findIndex((e) => e.type === "task.start" && e.name === "base_model" && e.key === "mgwr" && e.path.includes("task:fold[2/3]"));
    const at = replay(live.slice(0, iMgwr + 1));
    const snap = JSON.parse(JSON.stringify(toSnapshot(at, makeJob({ id: "j_live" }))));
    delete snap.projection;
    const { state } = fromSnapshot(snap);
    expect(state.current_path).toEqual(at.current_path);
    expect(state.running).toEqual(at.running);
    expect(state.stage).toBe("S2_S3");
  });
});

describe("live batches", () => {
  it("folds events, transient resource samples, job status, log lines and epoch losses", () => {
    const iCand = live.findIndex((e) => e.type === "task.start" && e.name === "stacker_fold" && e.unit === "stacker_fit:residual");
    const entry = {
      jid: "j_live", phase: "live" as const, error: null, job: makeJob({ id: "j_live", status: "running" }), state: replay(live.slice(0, iCand)),
      reconstructed: false, resources: [], recent: [], ended: null, epochs: [], logs: [], logCapped: false,
    };
    const rest = live.slice(iCand);
    const env = { v: 1 as const, seq: 0, t_rel: 0, pid: 1, job: "j_live", lvl: "info" as const, span: null, parent: null, path: [], ctx: {} };
    const batch = [
      ...rest,
      { ...env, type: "resource" as const, ts: 1790950999, rss_mb: 900, cpu_pct: 310, n_procs: 3, threads: 4 },
      { ...env, type: "log" as const, ts: 1790951000, cursor: 999999, logger: "sparc.core.pipeline", level: "INFO", msg: "S4: engine ready" },
      { ...env, type: "job.status" as const, ts: 1790951001, cursor: 1000001, status: "cancelling" as const, exit_code: null, error: null },
    ];
    const next = foldBatch(entry, batch as never);
    expect(next.state).toEqual(applyEvents(entry.state, batch.filter((e) => e.type !== "resource") as never));
    expect(next.resources).toEqual([{ ts: 1790950999, rss_mb: 900, cpu_pct: 310, n_procs: 3, threads: 4 }]);
    expect(next.job?.status).toBe("cancelling");
    expect(next.logs.map((l) => l.msg)).toEqual(["90% interval coverage is 87.4%, below the 0.90 target", "S4: engine ready"]);
    expect(next.epochs).toHaveLength(9);
    expect(next.epochs[0].value).toBeCloseTo(0.38, 9);
    expect(next.recent.length).toBeLessThanOrEqual(200);
    expect(next.recent[next.recent.length - 1].type).toBe("job.status");
  });
});

describe("helpers", () => {
  it("formats metric keys like the server (tags sorted, integers plain)", () => {
    expect(metricKey("candidate_rmse", { candidate: 0.1 })).toBe("candidate_rmse{candidate=0.1}");
    expect(metricKey("x", { b: 2.0, a: true })).toBe("x{a=true,b=2}");
    expect(metricKey("x", {})).toBe("x");
  });

  it("hashes like hashlib.sha1 and dumps lists like json.dumps", () => {
    expect(sha1Hex("")).toBe("da39a3ee5e6b4b0d3255bfef95601890afd80709");
    expect(sha1Hex("abc")).toBe("a9993e364706816aba3e25717850c26c9cd0d89d");
    expect(sha1Hex("a".repeat(1000))).toBe("291e9a6c66994949b57ba5e650361e98fc36b1ba");
    expect(pyJsonList(["run:a", "task:x[ΔT]"])).toBe('["run:a", "task:x[\\u0394T]"]');
    expect(stageOf(["run:a", "stage:cv_curve", "task:fold[1/2]"])).toBe("cv_curve");
  });

  it("attributes nested study runs to children, not to the job's own stages", () => {
    const base = { v: 1, seq: 0, t_rel: 0, pid: 1, job: "j_p", lvl: "info", parent: null, ctx: {} };
    const evs = [
      { ...base, type: "task.start", ts: 10, span: "1:1", path: ["task:placebo_kind[grf]"], ctx: { placebo: "grf" }, name: "placebo_kind", key: "grf", k: null, n: null, unit: null },
      { ...base, type: "run.start", ts: 11, span: "1:2", path: ["task:placebo_kind[grf]", "run:placebo_grf"], ctx: { placebo: "grf" }, name: "placebo_grf", run_meta: { role: "placebo:grf", studio_run_id: "r_child" } },
      { ...base, type: "stage.start", ts: 12, span: "1:3", path: ["task:placebo_kind[grf]", "run:placebo_grf", "stage:S0"], ctx: { placebo: "grf" }, stage: "S0", label: "S0", est_s: null },
      { ...base, type: "metric", ts: 13, span: "1:3", path: ["task:placebo_kind[grf]", "run:placebo_grf", "stage:S0"], ctx: { placebo: "grf" }, name: "stacker_r2", value: 0.4, unit: null, tags: {} },
      { ...base, type: "warning", ts: 13.5, span: "1:3", path: ["task:placebo_kind[grf]", "run:placebo_grf", "stage:S0"], ctx: { placebo: "grf" }, code: "qa.window", message: "small", data: {} },
    ];
    const s = replay(evs);
    expect(s.stages).toBeNull();
    expect(Object.keys(s.children)).toEqual(["placebo:grf"]);
    const child = s.children["placebo:grf"];
    expect(child.state.stages!.S0.state).toBe("running");
    expect(child.run_id).toBe("r_child");
    expect(childrenRows(s)).toEqual([{ job_id: null, run_id: "r_child", key: "placebo:grf", label: "placebo:grf", status: "running", progress: null, metrics: { stacker_r2: 0.4 } }]);
    // warnings of children are counted on the job too, and the span tree holds the child's spans
    expect(toContract(s).warnings).toEqual([{ code: "qa.window", count: 1 }]);
    expect(s.spans["1:3"].path).toEqual(["task:placebo_kind[grf]", "run:placebo_grf", "stage:S0"]);
  });

  it("estimates the remaining time from the plan with in-job means", () => {
    const iMgwr = live.findIndex((e) => e.type === "task.start" && e.name === "base_model" && e.key === "mgwr" && e.path.includes("task:fold[2/3]"));
    const s: TrackerState = replay(live.slice(0, iMgwr + 1));
    const eta = projectionEta(s, { nCells: 54_701, threads: 4 });
    // S2_S3 left: 2 mgwr fits at the in-job mean 150 s, 1 ols + 1 gam (their in-job means), stackers and a save
    expect(eta.stages.S2_S3).toBeGreaterThan(2 * 150);
    expect(eta.eta_lo!).toBeLessThan(eta.eta_s!);
    expect(eta.eta_hi!).toBeGreaterThan(eta.eta_s!);
    expect(projectionEta(replay(sample)).eta_s).toBe(0);
  });
});

describe("Python projection golden (tests/studio/fixtures, foundation selftest)", () => {
  // vitest runs from studio-web/ (npm --prefix studio-web test); the fixtures sit in the repo's tests/
  const dir = resolve(process.cwd(), "..", "tests", "studio", "fixtures");
  const evPath = resolve(dir, "selftest_events.jsonl");
  const goldenPath = resolve(dir, "selftest_projection.golden.json");
  const present = existsSync(evPath) && existsSync(goldenPath);

  it.runIf(present)("reproduces the Python reducer's full state and contract", () => {
    const events = parseJsonl(readFileSync(evPath, "utf8"));
    const golden = JSON.parse(readFileSync(goldenPath, "utf8")) as { state: unknown; contract: unknown };
    const s = replay(events);
    expectClose(plain(s), golden.state);
    expectClose(toContract(s), golden.contract);
  });
});
