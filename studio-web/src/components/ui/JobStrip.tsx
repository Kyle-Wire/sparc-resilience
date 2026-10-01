import { Link } from "../../router";
import { jobsForRun, useJobs } from "../../stores/jobs";
import { fmtEta, fmtPct } from "../../theme/format";
import { ProgressBar } from "./ProgressBar";
import { StatusChip } from "./StatusChip";

/**
 * Compact progress of one job from stores/jobs (fed by the global `job.progress` events):
 * status, label linking to Mission Control, stage, ETA range and a bar. Give `jobId`, or
 * `runId` to show that run's most important active job. Renders nothing when idle.
 */
export function JobStrip({ jobId, runId }: { jobId?: string; runId?: string }) {
  const jobs = useJobs((s) => s.jobs);
  const progress = useJobs((s) => s.progress);
  const job = jobId ? jobs[jobId] : runId ? jobsForRun(jobs, runId)[0] : undefined;
  if (!job) return null;
  const p = progress[job.id];
  const frac = p?.frac ?? job.progress;
  const stage = p?.stage ?? job.stage;
  const path = p?.path_tail?.length ? p.path_tail.join(" › ") : job.current_path?.slice(-2).join(" › ");
  return (
    <section className="jobstrip" aria-label={`Job ${job.label}`}>
      <StatusChip status={job.status} meta={frac !== null && frac !== undefined ? fmtPct(frac) : undefined} />
      <div style={{ minWidth: 0 }}>
        <Link to={`/jobs/${job.id}`} style={{ fontWeight: 600 }}>
          {job.label || job.kind}
        </Link>
        <span className="cap">
          {stage ? ` · ${stage}` : ""}
          {path ? ` · ${path}` : ""}
        </span>
      </div>
      <span className="cap num">{fmtEta(p?.eta_s ?? job.eta_s, p?.eta_lo ?? job.eta_lo, p?.eta_hi ?? job.eta_hi)}</span>
      <ProgressBar value={job.status === "queued" || job.status === "blocked" ? 0 : frac} label={`${job.label} progress`} />
    </section>
  );
}
