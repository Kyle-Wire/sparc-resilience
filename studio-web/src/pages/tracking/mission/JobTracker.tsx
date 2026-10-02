// Mission Control for one job (SPEC §5.10): header, stage rail, study child matrix, Gantt beside
// the selected stage's live panel, and the bottom tabs. The tracker store loads the snapshot and
// streams the job's events; the global job list supplies status changes and the server ETA.
import { useMemo } from "react";
import type { Job, JobStatus, StageId } from "../../../api/types";
import { Button } from "../../../components/ui/Button";
import { EmptyState } from "../../../components/ui/EmptyState";
import { useProject, useRunDetail } from "../../../layouts/resources";
import { codecs, useUrlState } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { loadEpochHistory, projectionEta, reloadTracker, STAGE_IDS, useJobTracker, useTracker, type TrackerDeps } from "../../../stores/tracker";
import { toast } from "../../../stores/ui";
import { isFinalStatus } from "../../../api/tracking";
import { useMeta, useNow } from "../hooks";
import { defaultStage, railItems } from "../model";
import { BottomTabs } from "./BottomTabs";
import { ChildMatrix } from "./ChildMatrix";
import { GanttPanel } from "./GanttPanel";
import { Header, type Eta } from "./Header";
import { GenericPanel, StagePanel } from "./StagePanels";
import { StageRail } from "./StageRail";
import "../tracking.css";

const RANK: Record<JobStatus, number> = { queued: 0, blocked: 0, starting: 1, running: 2, cancelling: 3, succeeded: 4, failed: 4, cancelled: 4, interrupted: 4 };

/** The fresher of the tracked job and the global job list's copy (status only moves forward). */
export function mergeJob(tracked: Job | null, global: Job | undefined): Job | null {
  if (!tracked) return global ?? null;
  if (!global) return tracked;
  if (RANK[global.status] > RANK[tracked.status]) return { ...tracked, ...global, params: tracked.params, result: global.result ?? tracked.result };
  return { ...tracked, progress: global.progress ?? tracked.progress, eta_s: global.eta_s ?? tracked.eta_s, eta_lo: global.eta_lo ?? tracked.eta_lo, eta_hi: global.eta_hi ?? tracked.eta_hi };
}

export function JobTracker({ jid, deps }: { jid: string; deps?: TrackerDeps }) {
  const entry = useJobTracker(jid, deps);
  const globalJob = useJobs((s) => s.jobs[jid]);
  const globalProgress = useJobs((s) => s.progress[jid]);
  const meta = useMeta();
  const job = mergeJob(entry?.job ?? null, globalJob);
  const live = !!job && !isFinalStatus(job.status);
  const nowMs = useNow(1000, live);
  const run = useRunDetail(job?.run_id ?? null).data?.run ?? null;
  const project = useProject(job?.project_id ?? null).data?.project ?? null;
  const [stageQ, setStageQ] = useUrlState("stage", codecs.optString());
  const state = entry?.state;
  const etaLocal = useMemo(() => (state ? projectionEta(state, { threads: job?.threads ?? null, now: nowMs / 1000 }) : null), [state, job?.threads, nowMs]);

  if (entry?.phase === "error") {
    return (
      <EmptyState error={entry.error} title="Could not load this job">
        <Button icon="refresh" onClick={() => reloadTracker(jid, deps)}>
          Try again
        </Button>
      </EmptyState>
    );
  }
  if (!entry || entry.phase === "loading" || !job || !state) {
    return (
      <p className="cap" role="status">
        Loading the job tracker…
      </p>
    );
  }

  // ETA: the server's calibrated estimate while fresh (1 Hz job.progress), else the snapshot's,
  // else the client's own projection with seed rates and in-job means.
  const serverFresh = globalProgress && nowMs / 1000 - globalProgress.ts < 10 && globalProgress.eta_s !== null;
  const eta: Eta = serverFresh
    ? { eta_s: globalProgress.eta_s, eta_lo: globalProgress.eta_lo, eta_hi: globalProgress.eta_hi }
    : job.eta_s !== null
      ? { eta_s: job.eta_s, eta_lo: job.eta_lo, eta_hi: job.eta_hi }
      : { eta_s: etaLocal?.eta_s ?? null, eta_lo: etaLocal?.eta_lo ?? null, eta_hi: etaLocal?.eta_hi ?? null };
  // Per-stage estimates: the client's split of the remaining time, scaled to the ETA shown above
  // (the server's calibrated total when it has one), so the rail and the header agree.
  const clientTotal = etaLocal?.eta_s ?? null;
  const scale = eta.eta_s !== null && clientTotal ? eta.eta_s / clientTotal : 1;
  const etaStages = Object.fromEntries(Object.entries(etaLocal?.stages ?? {}).map(([k, v]) => [k, v * scale]));
  const hasStages = !!state.stages && Object.keys(state.stages).length > 0;
  const selected: StageId = stageQ && (STAGE_IDS as readonly string[]).includes(stageQ) ? (stageQ as StageId) : defaultStage(state);
  const started = Object.fromEntries(Object.entries(state.stages ?? {}).map(([k, v]) => [k, v.started_ts]));
  const onJob = (j: Job) => useTracker.getState().put(jid, { job: j });

  return (
    <div className="mc">
      <Header job={job} state={state} run={run} projectName={project?.name ?? null} eta={eta} nowMs={nowMs} recent={entry.recent} onJob={onJob} />
      {hasStages ? (
        <section className="card" aria-label="Stages">
          <StageRail items={railItems(state, etaStages)} selected={selected} onSelect={(id) => setStageQ(id === stageQ ? null : id)} nowS={nowMs / 1000} started={started} />
        </section>
      ) : null}
      {job.kind.startsWith("study.") ? <ChildMatrix job={job} state={state} /> : null}
      <div className="mc-body">
        <section className="card" aria-label="Timeline">
          <GanttPanel jid={jid} state={state} logs={entry.logs} nowS={nowMs / 1000} zoom={stageQ} onStage={(s) => (STAGE_IDS as readonly string[]).includes(s) && setStageQ(s)} />
        </section>
        {hasStages ? (
          <StagePanel
            stage={selected}
            state={state}
            runId={job.run_id}
            epochs={entry.epochs}
            onLoadEpochs={async () => {
              const n = await loadEpochHistory(jid);
              if (!n) toast("info", "No epoch losses in this job's log", { body: "They are written only when the job logs at debug level." });
            }}
          />
        ) : (
          <GenericPanel state={state} job={job} />
        )}
      </div>
      <BottomTabs entry={entry} job={job} meta={meta.data} />
    </div>
  );
}
