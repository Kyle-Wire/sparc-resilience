// `/p/:pid` — project overview (SPEC §3.2), the planner's landing page: where the project is on
// its way (the four-stage rail Set up → Run → Explore → Report), the primary call to action
// (Set up data → Launch run → Open Scenario Lab), the readiness spine (ok / warn / missing / n/a,
// each with its one-click action), active jobs and the latest runs with their trust dot. The
// technical Pipeline Status Board lives on the Runs page; accuracy numbers on each run's How
// much to trust this (and the Runs page's comparison columns).
import { useEffect } from "react";
import type { Action, ReadinessRow, RunSummary } from "../../api/types";
import { Badge, DemoBadge, modeLabel } from "../../components/ui/Badge";
import { Card } from "../../components/ui/Card";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { JobStrip } from "../../components/ui/JobStrip";
import { StageRail, type StageRailStage, type StageStatus } from "../../components/ui/StageRail";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { useProjectContext } from "../../layouts/ProjectLayout";
import type { ProjectDetail } from "../../layouts/resources";
import { TRUST_GRADE_LABEL, trustGrade, type TrustGrade } from "../../content/trust";
import { Link, matchRoute, redirectTarget, useRegistry, type Registry } from "../../router";
import { activeJobs, useJobs } from "../../stores/jobs";
import { fmtDate, fmtDuration } from "../../theme/format";
import "./tracking.css";

/** Readiness state → chip status word and text (icon + text, never colour alone). */
const READY_CHIP: Record<ReadinessRow["state"], { status: string; text: string; rail: StageStatus }> = {
  ok: { status: "done", text: "ok", rail: "done" },
  warn: { status: "stale", text: "check", rail: "todo" },
  missing: { status: "missing", text: "missing", rail: "todo" },
  "n/a": { status: "not_requested", text: "n/a", rail: "skipped" },
};

export type Cta = { label: string; to: string; hint: string };

const isFinished = (r: RunSummary) => r.status === "complete" || r.status === "partial" || r.status === "imported";
const isLive = (r: RunSummary) => r.status === "queued" || r.status === "running" || r.status === "external_live";
const needsData = (detail: Pick<ProjectDetail, "readiness">) => detail.readiness.some((r) => (r.key === "data" || r.key === "columns") && r.state === "missing");
const runName = (r: RunSummary) => r.label || r.id;

/** The one primary button of the overview: Set up data → Launch (first) run → Open Scenario Lab. */
export function primaryCta(detail: Pick<ProjectDetail, "project" | "readiness" | "runs">): Cta {
  const pid = encodeURIComponent(detail.project.id);
  if (needsData(detail)) return { label: "Set up data", to: `/p/${pid}/setup/data`, hint: "Upload the temperature points and map their columns." };
  const runs = detail.runs;
  if (!runs.length) return { label: "Launch first run", to: `/p/${pid}/launch`, hint: "Fast mode takes a few minutes and sets up everything else." };
  const usable = (r: RunSummary) => isFinished(r) && (r.checkpoint_bytes ?? 0) > 0;
  const lab = runs.find((r) => r.id === detail.project.active_run_id && usable(r)) ?? runs.find(usable);
  if (lab) return { label: "Open Scenario Lab", to: `/r/${encodeURIComponent(lab.id)}/lab`, hint: `Design scenarios on ${runName(lab)}.` };
  // Say what is true of the runs there are: a finished run's results are there to explore.
  const finished = runs.find(isFinished);
  if (finished)
    return {
      label: "Launch run",
      to: `/p/${pid}/launch`,
      hint: `${runName(finished)} is ready to explore, but it saved no checkpoint: designing new scenarios needs a run that does.`,
    };
  const live = runs.find(isLive);
  if (live) return { label: "Launch run", to: `/p/${pid}/launch`, hint: `${runName(live)} is still running: its results appear as each stage finishes.` };
  return { label: "Launch run", to: `/p/${pid}/launch`, hint: "No run has finished yet." };
}

/**
 * The project's four stages (Set up → Run → Explore → Report): Set up is done once the data and
 * its columns are in, Run once a run has finished (running while one is going, failed when the
 * latest stopped and none finished), and Explore is current once there are results to open.
 */
export function projectStages(detail: Pick<ProjectDetail, "project" | "readiness" | "runs">): { stages: StageRailStage[]; current: string } {
  const pid = encodeURIComponent(detail.project.id);
  const runs = detail.runs;
  const setUp = !needsData(detail);
  const finished = runs.find((r) => r.id === detail.project.active_run_id && isFinished(r)) ?? runs.find(isFinished) ?? null;
  const live = runs.find(isLive) ?? null;
  const latest = runs[0] ?? null;
  const runStatus: StageStatus = finished ? "done" : live ? "running" : latest ? "failed" : "todo";
  const score = detail.project.readiness_score;
  const stages: StageRailStage[] = [
    { id: "setup", label: "Set up", status: setUp ? "done" : "todo", href: `/p/${pid}/setup/data`, meta: `${score.done} of ${score.total} ready` },
    {
      id: "run",
      label: "Run",
      status: runStatus,
      href: runs.length ? `/p/${pid}/runs` : `/p/${pid}/launch`,
      meta: finished ? `${runName(finished)} finished` : live ? `${runName(live)} is running` : latest ? `${runName(latest)} stopped` : "No run yet",
    },
    {
      id: "explore",
      label: "Explore",
      status: "todo",
      href: finished ? `/r/${encodeURIComponent(finished.id)}` : undefined,
      meta: finished ? "Results, map, heat risk and scenarios" : "Once a run finishes",
    },
    { id: "report", label: "Report", status: "todo", href: `/p/${pid}/exports`, meta: "Reports, exports and notes" },
  ];
  const current = !setUp ? "setup" : !finished ? "run" : "explore";
  return { stages, current };
}

const GRADE_ORDER: Record<TrustGrade, number> = { good: 3, fair: 2, limited: 1 };

/** A run's trust dot and grade word (the same rule as How much to trust this); a dash when not graded. */
function TrustCell({ run }: { run: RunSummary }) {
  const g = trustGrade({ r2: run.r2, coverage: run.coverage })?.grade ?? null;
  if (!g) return <span className="cap">—</span>;
  return (
    <span className="row" style={{ gap: 6, flexWrap: "nowrap" }} data-trust={g}>
      <span className="trust-dot" data-grade={g} aria-hidden="true" />
      {TRUST_GRADE_LABEL[g]}
    </span>
  );
}

/** Labels of server readiness actions that name a run tab which no longer exists (phase 2), in the new words. */
const MOVED_ACTION_LABELS: Record<string, string> = { "Open validation": "See the checks" };

/**
 * A readiness action with its link at the page's current home: an "open" action whose path is a
 * legacy URL (`/r/{rid}/validation`, …) goes straight to where its redirect leads (`/r/{rid}/trust`),
 * and a label that names a moved tab gets its new words. Every other action is returned as is.
 */
export function currentAction(action: Action, reg: Registry): Action {
  if (action.kind !== "open" || !action.path || !action.path.startsWith("/")) return action;
  const hashAt = action.path.indexOf("#");
  const base = hashAt >= 0 ? action.path.slice(0, hashAt) : action.path;
  const qAt = base.indexOf("?");
  const pathname = qAt >= 0 ? base.slice(0, qAt) : base;
  const m = matchRoute(reg.compiled, pathname);
  const to = m?.route.redirect ? redirectTarget(m.route, m.params, { search: qAt >= 0 ? base.slice(qAt) : "", hash: hashAt >= 0 ? action.path.slice(hashAt) : "" }) : null;
  if (!to) return action;
  return { ...action, path: to, label: MOVED_ACTION_LABELS[action.label] ?? action.label };
}

/** The readiness spine on the numbered stage rail, filled up to the last ready step. */
function Spine({ rows }: { rows: ReadinessRow[] }) {
  const reg = useRegistry();
  return (
    <StageRail
      className="readiness-rail"
      label="Readiness"
      stages={rows.map((r) => {
        const chip = READY_CHIP[r.state] ?? READY_CHIP.missing;
        return {
          id: r.key,
          status: chip.rail,
          label: (
            <span className="row" style={{ gap: 8 }}>
              <StatusChip status={chip.status} text={chip.text} />
              <b>{r.label}</b>
            </span>
          ),
          meta: r.detail,
          aside: r.action ? <ActionButton action={currentAction(r.action, reg)} size="small" variant={r.state === "missing" ? "primary" : "default"} /> : null,
        };
      })}
    />
  );
}

const RUN_COLS: Column<RunSummary>[] = [
  { key: "status", label: "Status", value: (r) => r.status, render: (r) => <StatusChip status={r.status} /> },
  { key: "label", label: "Run", value: (r) => r.label ?? r.id, render: (r) => <Link to={`/r/${encodeURIComponent(r.id)}`}>{r.label || r.id}</Link> },
  { key: "mode", label: "Mode", value: (r) => r.mode, render: (r) => <Badge tone="mode">{modeLabel(r.mode, r.coarse_m)}</Badge> },
  { key: "created", label: "Started", value: (r) => r.created_utc, render: (r) => fmtDate(r.created_utc) },
  { key: "duration", label: "Duration", align: "right", value: (r) => r.duration_s, render: (r) => fmtDuration(r.duration_s) },
  {
    key: "trust",
    label: "Trust",
    value: (r) => {
      const g = trustGrade({ r2: r.r2, coverage: r.coverage })?.grade;
      return g ? GRADE_ORDER[g] : null;
    },
    render: (r) => <TrustCell run={r} />,
  },
];

export default function ProjectOverview() {
  const ctx = useProjectContext();
  const jobs = useJobs((s) => s.jobs);
  const detail = ctx?.detail;
  // Jobs the project reports as active join the global list (its strips read from there).
  useEffect(() => {
    if (!detail) return;
    const known = useJobs.getState().jobs;
    for (const j of detail.active_jobs) if (!known[j.id]) useJobs.getState().upsert(j);
  }, [detail]);
  if (!ctx) return null;
  if (!detail) return <p className="cap">Loading the project…</p>;
  const p = detail.project;
  const cta = primaryCta(detail);
  const journey = projectStages(detail);
  const active = activeJobs(jobs).filter((j) => j.project_id === p.id);
  return (
    <div className="stack">
      <header className="page-head">
        <div className="row">
          <h1>{p.name}</h1>
          {p.demo ? <DemoBadge /> : null}
        </div>
        <div className="row">
          <Link className="btn primary" to={cta.to}>
            {cta.label}
          </Link>
          <span className="cap">{cta.hint}</span>
        </div>
      </header>
      <StageRail className="project-stages" stages={journey.stages} current={journey.current} orientation="horizontal" label="Project stages" />
      <div className="grid2" style={{ alignItems: "start" }}>
        <Card title="Readiness" eyebrow={`${p.readiness_score.done} of ${p.readiness_score.total} ready`}>
          {detail.readiness.length ? <Spine rows={detail.readiness} /> : <p className="cap">No readiness checks reported.</p>}
        </Card>
        <div className="stack">
          <Card title="Active jobs">
            {active.length ? (
              <div className="stack" style={{ gap: 8 }}>
                {active.map((j) => (
                  <JobStrip key={j.id} jobId={j.id} />
                ))}
              </div>
            ) : (
              <p className="cap">No jobs running for this project.</p>
            )}
          </Card>
          <Card title="Latest runs" actions={<Link to={`/p/${encodeURIComponent(p.id)}/runs`}>All runs</Link>}>
            {detail.runs.length ? (
              <Table columns={RUN_COLS} rows={detail.runs.slice(0, 5)} rowKey={(r) => r.id} caption="Latest runs" />
            ) : (
              <EmptyState title="No runs yet" body="Launch a run to see its stages here." />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
