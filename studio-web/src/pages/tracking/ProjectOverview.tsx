// `/p/:pid` — project overview (SPEC §3.2): readiness spine (ok / warn / missing / n/a, each with
// its one-click action), the Pipeline Status Board, latest runs, active jobs and the primary
// call to action (Set up data → Launch run → Open Scenario Lab).
import { useEffect } from "react";
import type { ReadinessRow, RunSummary } from "../../api/types";
import { Badge, DemoBadge, modeLabel } from "../../components/ui/Badge";
import { Card } from "../../components/ui/Card";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { JobStrip } from "../../components/ui/JobStrip";
import { StageRail, type StageStatus } from "../../components/ui/StageRail";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { useProjectContext } from "../../layouts/ProjectLayout";
import type { ProjectDetail } from "../../layouts/resources";
import { Link } from "../../router";
import { activeJobs, useJobs } from "../../stores/jobs";
import { fmtDate, fmtDuration, fmtNum } from "../../theme/format";
import { StatusBoard } from "./StatusBoard";
import "./tracking.css";

/** Readiness state → chip status word and text (icon + text, never colour alone). */
const READY_CHIP: Record<ReadinessRow["state"], { status: string; text: string; rail: StageStatus }> = {
  ok: { status: "done", text: "ok", rail: "done" },
  warn: { status: "stale", text: "check", rail: "todo" },
  missing: { status: "missing", text: "missing", rail: "todo" },
  "n/a": { status: "not_requested", text: "n/a", rail: "skipped" },
};

export type Cta = { label: string; to: string; hint: string };

/** The one primary button of the overview: Set up data → Launch (first) run → Open Scenario Lab. */
export function primaryCta(detail: Pick<ProjectDetail, "project" | "readiness" | "runs">): Cta {
  const pid = encodeURIComponent(detail.project.id);
  const needsData = detail.readiness.some((r) => (r.key === "data" || r.key === "columns") && r.state === "missing");
  if (needsData) return { label: "Set up data", to: `/p/${pid}/setup/data`, hint: "Upload the temperature points and map their columns." };
  const runs = detail.runs;
  if (!runs.length) return { label: "Launch first run", to: `/p/${pid}/launch`, hint: "Fast mode takes a few minutes and sets up everything else." };
  const usable = (r: RunSummary) => (r.status === "complete" || r.status === "partial" || r.status === "imported") && (r.checkpoint_bytes ?? 0) > 0;
  const lab = runs.find((r) => r.id === detail.project.active_run_id && usable(r)) ?? runs.find(usable);
  if (lab) return { label: "Open Scenario Lab", to: `/r/${encodeURIComponent(lab.id)}/lab`, hint: `Design scenarios on ${lab.label || lab.id}.` };
  return { label: "Launch run", to: `/p/${pid}/launch`, hint: "No finished run with a checkpoint yet." };
}

/** The readiness spine on the numbered stage rail, filled up to the last ready step. */
function Spine({ rows }: { rows: ReadinessRow[] }) {
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
          aside: r.action ? <ActionButton action={r.action} size="small" variant={r.state === "missing" ? "primary" : "default"} /> : null,
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
  { key: "r2", label: "R²", align: "right", value: (r) => r.r2, render: (r) => fmtNum(r.r2, 3) },
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
      <Card title="Pipeline Status Board" eyebrow="every run × every stage, post-run action and study">
        <StatusBoard pid={p.id} />
      </Card>
    </div>
  );
}
