// Mission Control header (SPEC §5.10): project · run label · mode badge · status pill; elapsed and
// ETA with range; cost-weighted progress; current-path breadcrumb; threads and pid; and the job
// actions Cancel, Force stop (after the 90 s grace), Resume, Retry, Duplicate with changes, Open
// outputs and Copy diagnostics.
import { useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { cancelJob, getEvents, getSystem, isFinalStatus, killJob, resumeRun, retryJob } from "../../../api/tracking";
import type { Job, JobEvent, RunSummary } from "../../../api/types";
import { Badge, modeLabel } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { ConfirmDialog } from "../../../components/ui/Dialog";
import { copyText, downloadText } from "../../../components/ui/download";
import { ProgressBar } from "../../../components/ui/ProgressBar";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link, navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { DIAG_EVENTS, toContract, type TrackerState } from "../../../stores/tracker";
import { toast } from "../../../stores/ui";
import { fmtDuration, fmtEta, fmtPct } from "../../../theme/format";
import { FORCE_STOP_GRACE_S, useCancellingSince } from "../hooks";
import { pathCrumbs, workerPid } from "../model";

export type Eta = { eta_s: number | null; eta_lo: number | null; eta_hi: number | null };

const ACTIVE = new Set(["queued", "blocked", "starting", "running"]);

function failMessage(what: string, e: unknown) {
  toast("error", `${what} failed`, { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
}

/** The diagnostics blob: job, the last 200 events, versions and the tracker summary. */
export async function diagnostics(job: Job, state: TrackerState, recent: JobEvent[]): Promise<string> {
  let events: unknown[] = recent;
  if (events.length < DIAG_EVENTS && state.cursor > 0) {
    // Lines are at most 4 KB: this byte window holds at least the last 200 complete lines.
    try {
      const page = await getEvents(job.id, { after: Math.max(-1, state.cursor - DIAG_EVENTS * 4096), limit: 5000 });
      const seen = new Set(recent.map((e) => e.cursor));
      events = [...page.events.filter((e) => !seen.has(e.cursor)), ...recent].slice(-DIAG_EVENTS);
    } catch {
      /* keep what was streamed */
    }
  }
  let versions: unknown = null;
  try {
    const sys = await getSystem();
    versions = { ...sys.versions, web_build: sys.web_build, host_id: sys.host_id, cpu_count: sys.cpu_count, mem_total_gb: sys.mem_total_gb };
  } catch (e) {
    versions = { error: errorMessage(e) };
  }
  return JSON.stringify({ generated_utc: new Date().toISOString(), job, tracker: toContract(state), current_path: state.current_path, cursor: state.cursor, versions, events }, null, 2);
}

export function Header({
  job,
  state,
  run,
  projectName,
  eta,
  nowMs,
  recent,
  onJob,
}: {
  job: Job;
  state: TrackerState;
  run: RunSummary | null;
  projectName: string | null;
  eta: Eta;
  nowMs: number;
  recent: JobEvent[];
  onJob: (job: Job) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmKill, setConfirmKill] = useState(false);
  const since = useCancellingSince(job, typeof state.cancel?.ts === "number" ? state.cancel.ts : null);
  const graceLeft = since !== null ? FORCE_STOP_GRACE_S - (nowMs - since) / 1000 : null;
  const external = job.executor === "external";
  const final = isFinalStatus(job.status);
  const started = job.started_utc ? Date.parse(job.started_utc) : null;
  const ended = job.finished_utc ? Date.parse(job.finished_utc) : null;
  const elapsed = started !== null ? ((ended ?? nowMs) - started) / 1000 : null;
  const progress = job.status === "succeeded" ? 1 : state.progress ?? job.progress;
  const crumbs = pathCrumbs(state, state.current_path ?? job.current_path);
  const pid = workerPid(state);
  const r = state.run;
  const mode = run ? modeLabel(run.mode, run.coarse_m) : r && typeof r.fast === "boolean" ? (r.coarse ? `COARSE ${Math.round(r.coarse)}` : r.fast ? "FAST" : "FULL") : null;
  const isRun = job.kind === "run.core" || job.kind === "run.external";

  const act = async (what: string, fn: () => Promise<Job>, after?: (j: Job) => void) => {
    setBusy(what);
    try {
      const j = await fn();
      useJobs.getState().upsert(j);
      if (after) after(j);
      else onJob(j);
    } catch (e) {
      failMessage(what, e);
    } finally {
      setBusy(null);
    }
  };

  const openOutputs = job.run_id ? `/r/${encodeURIComponent(job.run_id)}` : job.study_id ? `/studies/${encodeURIComponent(job.study_id)}` : job.kind.startsWith("export.") && job.project_id ? `/p/${encodeURIComponent(job.project_id)}/exports` : null;
  const duplicate =
    job.kind === "run.core" && job.project_id && job.run_id
      ? `/p/${encodeURIComponent(job.project_id)}/launch?from=${encodeURIComponent(job.run_id)}`
      : job.kind.startsWith("study.") && job.run_id
        ? `/r/${encodeURIComponent(job.run_id)}/validation`
        : null;

  return (
    <header className="card mc-head" aria-label="Job">
      <div className="mc-title">
        <h1>{job.label || job.kind}</h1>
        {mode ? <Badge tone="accent">{mode}</Badge> : null}
        {run?.demo ? <Badge tone="demo">DEMO</Badge> : null}
        <StatusChip status={job.status} />
        {job.blocked ? <span className="cap">blocked: {job.blocked.reason}</span> : null}
      </div>
      <div className="mc-facts">
        {projectName && job.project_id ? (
          <span>
            Project <Link to={`/p/${encodeURIComponent(job.project_id)}`}>{projectName}</Link>
          </span>
        ) : null}
        {job.run_id ? (
          <span>
            Run <Link to={`/r/${encodeURIComponent(job.run_id)}`}>{run?.label || job.run_id}</Link>
          </span>
        ) : null}
        <span>
          Kind <b>{job.kind}</b>
        </span>
        <span>
          Elapsed <b>{fmtDuration(elapsed)}</b>
        </span>
        {!final ? (
          <span>
            ETA <b>{fmtEta(eta.eta_s, eta.eta_lo, eta.eta_hi)}</b>
          </span>
        ) : null}
        {job.threads ? (
          <span>
            Threads <b>{job.threads}</b>
          </span>
        ) : null}
        {pid !== null ? (
          <span>
            pid <b>{pid}</b>
          </span>
        ) : null}
        {job.exit_code !== null ? (
          <span>
            exit <b>{job.exit_code}</b>
          </span>
        ) : null}
      </div>
      <div className="mc-progress">
        <ProgressBar value={job.status === "queued" || job.status === "blocked" ? 0 : progress} label={`${job.label || job.kind} progress`} tone={job.status === "failed" ? "crit" : job.status === "succeeded" ? "good" : undefined} />
        <span className="num">{fmtPct(progress, 0)}</span>
      </div>
      {crumbs.length ? (
        <ol className="mc-crumbs" aria-label="Current step">
          {crumbs.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ol>
      ) : null}
      {job.error ? (
        <div className="callout" data-tone="crit" role="alert">
          <b>{job.error.type}</b>: {job.error.message}
          {job.error.traceback_tail ? (
            <details>
              <summary className="cap">Traceback</summary>
              <pre className="raw-pre">{job.error.traceback_tail}</pre>
            </details>
          ) : null}
        </div>
      ) : null}
      <div className="mc-actions">
        {ACTIVE.has(job.status) && !external ? (
          <Button variant="danger" icon="stop" busy={busy === "Cancel"} onClick={() => void act("Cancel", () => cancelJob(job.id))}>
            Cancel
          </Button>
        ) : null}
        {external && !final ? <span className="cap">Started from the command line{pid !== null ? ` (pid ${pid})` : ""}: stop it in its terminal.</span> : null}
        {job.status === "cancelling" && graceLeft !== null ? (
          graceLeft <= 0 ? (
            <Button variant="danger" icon="x" busy={busy === "Force stop"} onClick={() => setConfirmKill(true)}>
              Force stop
            </Button>
          ) : (
            <span className="cap" role="status">
              Waiting for a safe point; Force stop is offered in {Math.ceil(graceLeft)} s.
            </span>
          )
        ) : null}
        {final && job.status !== "succeeded" && job.kind === "run.core" && job.run_id ? (
          <Button variant="primary" icon="play" busy={busy === "Resume"} onClick={() => void act("Resume", () => resumeRun(job.run_id!), (j) => navigate(`/jobs/${encodeURIComponent(j.id)}`))}>
            Resume
          </Button>
        ) : null}
        {final && job.status !== "succeeded" && !isRun ? (
          <Button icon="refresh" busy={busy === "Retry"} onClick={() => void act("Retry", () => retryJob(job.id), (j) => navigate(`/jobs/${encodeURIComponent(j.id)}`))}>
            Retry
          </Button>
        ) : null}
        {duplicate ? (
          <Link className="btn" to={duplicate}>
            Duplicate with changes
          </Link>
        ) : null}
        {openOutputs ? (
          <Link className="btn" to={openOutputs}>
            Open outputs
          </Link>
        ) : null}
        <Button
          variant="ghost"
          icon="copy"
          busy={busy === "Copy diagnostics"}
          onClick={async () => {
            setBusy("Copy diagnostics");
            try {
              const text = await diagnostics(job, state, recent);
              if (await copyText(text)) toast("success", "Diagnostics copied", { body: "Job, the last 200 events and versions, as JSON." });
              else {
                downloadText(text, `${job.id}-diagnostics.json`, "application/json");
                toast("info", "Diagnostics downloaded");
              }
            } finally {
              setBusy(null);
            }
          }}
        >
          Copy diagnostics
        </Button>
      </div>
      <ConfirmDialog
        open={confirmKill}
        onClose={() => setConfirmKill(false)}
        title="Force stop this job?"
        confirmLabel="Force stop"
        danger
        busy={busy === "Force stop"}
        onConfirm={() => {
          void act("Force stop", () => killJob(job.id)).then(() => setConfirmKill(false));
        }}
      >
        The worker did not reach a safe point within {FORCE_STOP_GRACE_S} s. Force stop kills its process group, pool workers included. Checkpoints are written atomically, so the run stays consistent with its last checkpoint
        {isRun ? " and can be resumed from it" : ""}.
      </ConfirmDialog>
    </header>
  );
}
