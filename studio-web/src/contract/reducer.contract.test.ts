// Cross-language reducer contract (SPEC §14.3): the client tracker (`applyEvent`, `toContract`)
// replays the recorded synthetic run, tests/studio/fixtures/synth_run/events.jsonl, and must
// give the projection the server's `jobs/tracker.reduce` gives.
//
// - On its own (`npm test`) this compares the replay with the committed golden projection,
//   tests/studio/fixtures/reducer_projection.golden.json (written from the Python reducer).
// - With SPARC_CONTRACT_OUT set (tests/studio/test_reducer_contract.py does this), it also
//   writes the projection after every event to that path, and the Python test compares it
//   step by step with its own.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import type { JobEvent } from "../api/types";
import { applyEvent, newTrackerState, replay, toContract, type Contract } from "../stores/tracker";

const FIXTURE = "tests/studio/fixtures/synth_run/events.jsonl";
const GOLDEN = "tests/studio/fixtures/reducer_projection.golden.json";
const TOL = 1e-9;

/** The checkout root: the first folder up from the working directory (studio-web) that holds the fixture. */
function repoRoot(): string {
  let dir = process.cwd();
  for (;;) {
    if (existsSync(join(dir, FIXTURE))) return dir;
    const up = dirname(dir);
    if (up === dir) throw new Error(`${FIXTURE} not found above ${process.cwd()}`);
    dir = up;
  }
}
const REPO = repoRoot();

type CursorEvent = JobEvent & { cursor: number };
type Step = Contract & { cursor: number; type: string };
type Trajectory = { fixture: string; n_events: number; final: Contract; steps: Step[] };
type StageChange = [number, Record<string, [string, string | null]>];
type Golden = {
  fixture: string;
  n_events: number;
  final: Contract;
  progress: [number, number | null][];
  stage_changes: StageChange[];
};

/** The JSONL lines as the server streams them: each event with `cursor` = byte offset of its line. */
function parseJsonl(bytes: Uint8Array): CursorEvent[] {
  const out: CursorEvent[] = [];
  const dec = new TextDecoder();
  let start = 0;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] !== 0x0a) continue;
    const line = dec.decode(bytes.subarray(start, i));
    if (line.trim()) out.push({ ...(JSON.parse(line) as JobEvent), cursor: start });
    start = i + 1;
  }
  return out; // a trailing line without "\n" is still being written: the server skips it too
}

/** The canonical projection after every event, folded one event at a time. */
function trajectory(events: CursorEvent[]): Trajectory {
  let state = newTrackerState();
  const steps: Step[] = [];
  for (const ev of events) {
    state = applyEvent(state, ev);
    steps.push({ cursor: ev.cursor, type: ev.type, ...toContract(state) });
  }
  return { fixture: FIXTURE, n_events: events.length, final: toContract(state), steps };
}

/** The golden's compact form: progress per event and the stage states wherever they change. */
function compact(t: Trajectory): Golden {
  const changes: StageChange[] = [];
  let prev: Contract["stages"] = {};
  for (const s of t.steps) {
    const diff: Record<string, [string, string | null]> = {};
    for (const [sid, st] of Object.entries(s.stages)) {
      const p = prev[sid];
      if (!p || p.state !== st.state || p.reason !== st.reason) diff[sid] = [st.state, st.reason];
    }
    if (Object.keys(diff).length) changes.push([s.cursor, diff]);
    prev = s.stages;
  }
  return {
    fixture: t.fixture,
    n_events: t.n_events,
    final: t.final,
    progress: t.steps.map((s) => [s.cursor, s.progress]),
    stage_changes: changes,
  };
}

function closeTo(a: number | null, b: number | null): boolean {
  if (a === null || b === null) return a === b;
  return Math.abs(a - b) <= TOL;
}

function expectSameContract(got: Contract, want: Contract, where: string): void {
  expect(got.stages, `${where}: stages`).toEqual(want.stages);
  expect(got.done_units, `${where}: done_units`).toEqual(want.done_units);
  expect(got.warnings, `${where}: warnings`).toEqual(want.warnings);
  expect(got.artifacts, `${where}: artifacts`).toEqual(want.artifacts);
  expect(closeTo(got.progress, want.progress), `${where}: progress ${got.progress} vs ${want.progress}`).toBe(true);
}

describe("tracker reducer contract (synthetic run)", () => {
  const events = parseJsonl(readFileSync(join(REPO, FIXTURE)));
  const traj = trajectory(events);
  const out = process.env.SPARC_CONTRACT_OUT;
  if (out) writeFileSync(out, JSON.stringify(traj));

  it("reads every event of the fixture with byte cursors", () => {
    expect(events.length).toBeGreaterThan(100);
    expect(events[0].cursor).toBe(0);
    expect(events.every((e, i) => i === 0 || e.cursor > events[i - 1].cursor)).toBe(true);
    expect(events.at(-1)?.type).toBe("run.end");
  });

  it("matches the golden projection at the end of the run", () => {
    const golden = JSON.parse(readFileSync(join(REPO, GOLDEN), "utf8")) as Golden;
    expect(traj.n_events).toBe(golden.n_events);
    expectSameContract(traj.final, golden.final, "final");
  });

  it("matches the golden progress and stage states after every event", () => {
    const golden = JSON.parse(readFileSync(join(REPO, GOLDEN), "utf8")) as Golden;
    const mine = compact(traj);
    expect(mine.progress.length).toBe(golden.progress.length);
    mine.progress.forEach(([cursor, p], i) => {
      const [gc, gp] = golden.progress[i];
      expect(cursor).toBe(gc);
      expect(closeTo(p, gp), `progress at cursor ${cursor}: ${p} vs ${gp}`).toBe(true);
    });
    expect(mine.stage_changes).toEqual(golden.stage_changes);
  });

  it("folds the same state in one batch as event by event", () => {
    const batch = toContract(replay(events));
    expectSameContract(batch, traj.final, "batch replay");
  });

  it("never lowers progress and ends the succeeded run at 1", () => {
    let last = 0;
    for (const s of traj.steps) {
      if (s.progress === null) continue;
      expect(s.progress, `progress at cursor ${s.cursor}`).toBeGreaterThanOrEqual(last - 1e-12);
      last = s.progress;
    }
    expect(traj.final.progress).toBe(1);
  });
});
