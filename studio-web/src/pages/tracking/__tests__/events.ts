// Test helpers: parse a JSONL event fixture the way the server does (cursor = byte offset of
// each line's first byte), and build a minimal Job for snapshots.
import type { Job, JobEvent } from "../../../api/types";

export type CursorEvent = JobEvent & { cursor: number };

export function parseJsonl(text: string): CursorEvent[] {
  const enc = new TextEncoder();
  const out: CursorEvent[] = [];
  let offset = 0;
  for (const line of text.split("\n")) {
    const bytes = enc.encode(line).length + 1;
    if (line.trim()) out.push({ ...(JSON.parse(line) as JobEvent), cursor: offset });
    offset += bytes;
  }
  return out;
}

export function makeJob(patch: Partial<Job> = {}): Job {
  return {
    id: "j_sample", kind: "run.core", lane: "heavy", executor: "process", label: "Resume demo (fast)", status: "running",
    project_id: "p_demo", run_id: "20261001-101500-fast-ab12", study_id: null, scenario_id: null, parent_job_id: null, after_job_id: null,
    priority: 0, params: { run_id: "20261001-101500-fast-ab12", resume: true }, created_utc: "2026-10-01T10:15:00Z", started_utc: "2026-10-01T10:15:01Z",
    finished_utc: null, progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null, error: null,
    blocked: null, result: null, peak_rss_mb: null, threads: 3, ...patch,
  };
}
