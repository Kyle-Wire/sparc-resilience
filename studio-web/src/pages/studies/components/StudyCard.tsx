// One Validation-tab card per study kind (SPEC §8): status (not run / queued / running / done /
// stale vs run / failed), the row's cost estimate and headline, attach/detach for studies that
// feed the uncertainty report, the launch form (disabled with the server's missing requirements
// listed) and the kind's result chart.
// Plain words (phase 4): each card says what the check does in plain words, with an ⓘ at the end
// of that line giving the check's plain name (glossary) and its technical name; the card's status
// explains itself in a tooltip (no native title).
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
import { InfoTip } from "../../../components/ui/InfoTip";
import { JobStrip } from "../../../components/ui/JobStrip";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link } from "../../../router";
import { fmtRelative } from "../../../theme/format";
import { TipChip } from "../../runhub/common";
import { AttachToggle, KIND_TERMS, estimateText } from "./common";
import type { FormContext } from "./KindForms";
import { formFromParams } from "../model/params";
import { LaunchPanel } from "./LaunchPanel";
import { BaselinesResult, EmulatorResult, LiteratureResult, PlannerResult, UncertaintyResult, WriteupResult } from "./RunViews";
import { StudyViewPanel } from "./StudyViews";

/** What each card is for (shown under its title). */
export const KIND_INTROS: Record<StudyKind, string> = {
  baselines: "Does the full model beat simpler ones? Scores simple reference models on the same test areas and compares their typical error.",
  planner: "Who is exposed, hot days, how evenly the heat falls, open planting space, neighbourhood summaries and where to put sensors, from the people and land-cover layers.",
  emulator: "Builds the quick estimate: a fast stand-in for the full model, checked on random patches. It powers the Lab's instant preview.",
  uncertainty: "Adds up every source of doubt for each scenario: the model fit, other model versions, the simulation check and the cause-and-effect check.",
  writeup: "Rewrites the methods and the model card from the run's records (the run report stays as written).",
  placebo: "Refits with fake or shuffled map layers: a fake change should show no effect, the real layers should.",
  simcheck: "Plants known effects in made-up temperatures for this city and checks how much of them the model recovers, and how often its likely ranges contain them.",
  multiverse: "Refits with other reasonable choices (test-area size, weather inputs, models) to see whether the answers keep their direction.",
  reproduce: "Re-runs the analysis from the run's launch settings in a new run and checks the numbers match.",
  literature: "Published cooling for similar changes next to this run's estimates (worked out with the response curves).",
  benchmark: "Plants a known tree-cover effect in a made-up city and measures how much of it each model recovers (project-wide, no run needed).",
};

const STATE_TEXT: Record<StudyState, string> = {
  not_run: "not run",
  queued: "queued",
  running: "running",
  done: "done",
  stale: "out of date",
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
          <TipChip tip={row.updated_utc ? `Updated ${fmtRelative(row.updated_utc)}${row.state === "stale" ? ". The run has changed since this check ran." : ""}` : null}>
            <StatusChip status={row.state} text={STATE_TEXT[row.state] ?? row.state} />
          </TipChip>
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
        {KIND_TERMS[row.kind] ? <InfoTip term={KIND_TERMS[row.kind]!} technical={label} /> : null}
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
