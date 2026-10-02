// One Validation-tab card per study kind (SPEC §8): status (not run / queued / running / done /
// stale vs run / failed), the row's cost estimate and headline, attach/detach for studies that
// feed the uncertainty report, the launch form (disabled with the server's missing requirements
// listed) and the kind's result chart.
import type { ReactNode } from "react";
import {
  ATTACHABLE_KINDS,
  hasStudyView,
  KIND_LABELS,
  type LaunchableKind,
  type StudyKind,
  type StudyState,
  type StudyStatusRow,
} from "../../../api/studies";
import { Card } from "../../../components/ui/Card";
import { ActionButton } from "../../../components/ui/EmptyState";
import { JobStrip } from "../../../components/ui/JobStrip";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link } from "../../../router";
import { fmtRelative } from "../../../theme/format";
import { AttachToggle, estimateText } from "./common";
import type { FormContext } from "./KindForms";
import { formFromParams } from "../model/params";
import { LaunchPanel } from "./LaunchPanel";
import { BaselinesResult, EmulatorResult, LiteratureResult, PlannerResult, UncertaintyResult, WriteupResult } from "./RunViews";
import { StudyViewPanel } from "./StudyViews";

/** What each card is for (shown under its title). */
export const KIND_INTROS: Record<StudyKind, string> = {
  baselines: "Is the stack better than simpler models? Refits reference baselines on the same folds and compares held-out error.",
  planner: "Exposure, hot days, equity, plantable space, hexagon summaries and logger sites from the people and land-cover layers.",
  emulator: "A fast stand-in for the exact engine, validated on random patches; enables the Lab's instant preview.",
  uncertainty: "Layers estimation, specification (multiverse), attribution (simulation check) and causal intervals per scenario.",
  writeup: "Rewrites the methods and the model card from the merged manifest (the run report stays as written).",
  placebo: "Re-fits with fake or displaced layers: a placebo should show no effect, the real layers should.",
  simcheck: "Plants known effects in synthetic targets on this city and checks how much the model recovers, and how often its intervals cover the truth.",
  multiverse: "Re-fits under defensible alternative choices (CV blocks, forcing, models) to see whether the answers keep their sign.",
  reproduce: "Re-runs the stages from the run's launch snapshot in a new run and checks the numbers match within tolerance.",
  literature: "Published cooling ranges for comparable levers next to this run's estimates (computed with the response stage).",
  benchmark: "Plants a known canopy effect in a synthetic city and measures the share each model recovers (project-wide, no run needed).",
};

const STATE_TEXT: Record<StudyState, string> = {
  not_run: "not run",
  queued: "queued",
  running: "running",
  done: "done",
  stale: "stale vs run",
  failed: "failed",
};

/** Kinds whose status row names a study whose params a new launch can start from. */
const STUDY_PARAMS_KINDS = new Set<StudyKind>(["placebo", "simcheck", "multiverse", "reproduce", "benchmark"]);

/** Whether the card has something to show as a result for this row. */
export function hasResult(row: StudyStatusRow): boolean {
  if (row.kind === "literature") return row.state !== "not_run";
  if (hasStudyView(row.kind)) return !!row.study_id;
  return row.state === "done" || row.state === "stale";
}

function Result({ row, rid, units }: { row: StudyStatusRow; rid: string; units: string }) {
  const live = row.state === "running" || row.state === "queued";
  if (hasStudyView(row.kind) && row.study_id) return <StudyViewPanel kind={row.kind} studyId={row.study_id} units={units} live={live} />;
  switch (row.kind) {
    case "baselines":
      return <BaselinesResult rid={rid} headline={row.headline} />;
    case "planner":
      return <PlannerResult rid={rid} />;
    case "emulator":
      return <EmulatorResult rid={rid} />;
    case "uncertainty":
      return <UncertaintyResult rid={rid} />;
    case "writeup":
      return <WriteupResult rid={rid} />;
    case "literature":
      return <LiteratureResult rid={rid} />;
    default:
      return null;
  }
}

/** Kinds launched from a card (literature comes with the response stage, no launch of its own). */
function launchable(kind: StudyKind): kind is LaunchableKind {
  return kind !== "literature";
}

export type StudyCardProps = {
  row: StudyStatusRow;
  rid: string;
  pid: string | null;
  units: string;
  ctx: FormContext;
  /** Open the launch form initially (deep link `#<kind>`). */
  open?: boolean;
  extra?: ReactNode;
};

export function StudyCard({ row, rid, pid, units, ctx, open, extra }: StudyCardProps) {
  const label = KIND_LABELS[row.kind] ?? row.kind;
  const est = estimateText(row.estimate);
  const active = row.state === "running" || row.state === "queued";
  const attachable = ATTACHABLE_KINDS.includes(row.kind) && !!row.study_id && row.attached !== null;
  const wide = row.kind === "simcheck" || row.kind === "multiverse" || row.kind === "placebo";
  const blocked = !row.requirements.ok;
  // "Run again" starts from the latest study's settings (its custom variants, design, kinds…).
  const prior = row.study_id && STUDY_PARAMS_KINDS.has(row.kind) ? ctx.studies.find((s) => s.id === row.study_id) ?? null : null;
  return (
    <Card
      as="article"
      className="vt-card"
      id={`study-${row.kind}`}
      data-kind={row.kind}
      data-state={row.state}
      data-wide={wide || undefined}
      aria-label={label}
      eyebrow={est ? `Estimated ${est}` : undefined}
      title={
        <>
          {label}
          <StatusChip status={row.state} text={STATE_TEXT[row.state] ?? row.state} title={row.updated_utc ? `Updated ${fmtRelative(row.updated_utc)}` : undefined} />
        </>
      }
      actions={
        <>
          {attachable ? <AttachToggle studyId={row.study_id!} runId={rid} attached={!!row.attached} label={label} /> : null}
          {row.study_id ? (
            <Link to={`/studies/${encodeURIComponent(row.study_id)}`} className="btn small ghost">
              Study page
            </Link>
          ) : null}
        </>
      }
    >
      <p className="cap" style={{ margin: 0 }}>
        {KIND_INTROS[row.kind]}
      </p>
      {row.headline ? <p className="vt-headline">{row.headline}</p> : null}
      {row.job_id && active ? (
        <div className="stack" style={{ gap: 4 }}>
          <JobStrip jobId={row.job_id} />
          <Link to={`/jobs/${encodeURIComponent(row.job_id)}`}>Track in Mission Control →</Link>
        </div>
      ) : null}
      {row.state === "failed" && row.job_id ? (
        <p className="cap">
          The last attempt failed: see its <Link to={`/jobs/${encodeURIComponent(row.job_id)}`}>log and error</Link>.
        </p>
      ) : null}
      {hasResult(row) ? <Result row={row} rid={rid} units={units} /> : null}
      {extra}
      {launchable(row.kind) ? (
        <details className="vt-launch" open={open || row.state === "not_run" || undefined}>
          <summary>
            {row.state === "not_run" ? "Set up and run" : "Run again"}
            {blocked ? <span className="cap"> · {row.requirements.missing.length || "some"} requirement{row.requirements.missing.length === 1 ? "" : "s"} missing</span> : null}
          </summary>
          {prior ? <p className="cap">Starts from the settings of the last study ({prior.id}).</p> : null}
          <LaunchPanel
            key={prior?.id ?? "new"}
            kind={row.kind}
            rid={rid}
            pid={pid}
            requirements={row.requirements}
            ctx={{ ...ctx, runId: rid }}
            initial={prior ? formFromParams(row.kind as LaunchableKind, prior.params) : undefined}
          />
        </details>
      ) : row.action ? (
        <div className="sx-actions">
          <ActionButton action={row.action} size="small" variant="default" />
        </div>
      ) : null}
    </Card>
  );
}
