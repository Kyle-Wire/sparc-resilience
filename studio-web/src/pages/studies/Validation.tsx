// Validation (SPEC §8): one card per study kind with its status, cost estimate, requirements,
// launch form and result chart, plus "Truth vs recovered" for synthetic demo projects. Rows
// come from `GET /api/runs/{rid}/studies` and are refreshed by `study.updated` and the run's
// job events. The run's home for it is How much to trust this (`/r/:rid/trust#validation`),
// which lists the verdicts (`IndependentChecks`) and keeps the cards (`StudyCards`) behind a
// disclosure; the default export is the complete page.
import { useEffect } from "react";
import { useRunScenarios } from "../../api/lab";
import { useView } from "../../api/runs";
import {
  KIND_LABELS,
  STUDY_KINDS,
  useProjectStudies,
  useRunStudies,
  useThreadsHeavy,
  type StudyKind,
  type StudyState,
  type StudyStatusRow,
} from "../../api/studies";
import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { useRunContext } from "../../layouts/RunLayout";
import { Link, runHref, useRoute } from "../../router";
import { StudyCard } from "./components/StudyCard";
import { TruthCard } from "./components/RunViews";
import "./studies.css";

/** A not-run row for a kind the server did not list (older servers), so every card shows. */
function placeholderRow(kind: StudyKind): StudyStatusRow {
  return { kind, state: "not_run", study_id: null, job_id: null, updated_utc: null, headline: null, estimate: null, attached: null, action: null, requirements: { ok: true, missing: [] } };
}

/** Every kind once, in the card order, from the server's rows. */
export function cardRows(rows: readonly StudyStatusRow[] | undefined): StudyStatusRow[] {
  const byKind = new Map((rows ?? []).map((r) => [r.kind, r]));
  return STUDY_KINDS.map((k) => byKind.get(k) ?? placeholderRow(k));
}

/**
 * The study kinds whose result is a verdict on the run (Independent checks). The planner
 * pack, the emulator and the write-up are tools, and the uncertainty report has its own
 * section on How much to trust this.
 */
export const CHECK_KINDS: readonly StudyKind[] = ["baselines", "placebo", "simcheck", "multiverse", "reproduce", "literature", "benchmark"];

export type CheckVerdict = { kind: StudyKind; label: string; state: StudyState; line: string };

const STATE_LINE: Record<StudyState, string> = {
  not_run: "Not run yet.",
  queued: "Queued.",
  running: "Running now.",
  done: "Done.",
  stale: "Out of date: the run has changed since this check.",
  failed: "The last attempt failed.",
};

/** One line per independent check, in card order: the server's headline, else what its state means. */
export function checkVerdicts(rows: readonly StudyStatusRow[] | undefined): CheckVerdict[] {
  return cardRows(rows)
    .filter((r) => CHECK_KINDS.includes(r.kind))
    .map((r) => {
      const head = r.headline?.trim();
      const line = head ? (r.state === "stale" ? `${head} (out of date: the run has changed since)` : head) : STATE_LINE[r.state] ?? r.state;
      return { kind: r.kind, label: KIND_LABELS[r.kind] ?? r.kind, state: r.state, line };
    });
}

/** The independent checks' verdicts, one row each, linking to the check's card (`#study-<kind>`). */
export function IndependentChecks({ rid }: { rid: string }) {
  const rows = useRunStudies(rid || null);
  if (rows.error && !rows.data) return <EmptyState error={rows.error} title="Could not load the independent checks" />;
  if (!rows.data)
    return (
      <p className="cap" role="status">
        <span className="spinner" aria-hidden="true" /> Loading the independent checks…
      </p>
    );
  const verdicts = checkVerdicts(rows.data);
  const done = verdicts.filter((v) => v.state === "done" || v.state === "stale").length;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <p className="cap" style={{ margin: 0 }}>
        {done ? `${done} of ${verdicts.length} checks have results.` : "None of these checks has been run on this run yet."}
      </p>
      <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 6 }} aria-label="Independent checks">
        {verdicts.map((v) => (
          <li key={v.kind} className="row" data-check-kind={v.kind} data-state={v.state} style={{ alignItems: "baseline", flexWrap: "nowrap" }}>
            <StatusChip status={v.state} text={v.state === "not_run" ? "not run" : v.state === "stale" ? "out of date" : undefined} />
            <span>
              <strong>{v.label}</strong> {v.line}{" "}
              <Link to={`${runHref(rid, "trust")}#study-${v.kind}`} className="cap">
                {v.state === "not_run" ? "Run it" : "Details"}
              </Link>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Every study card of the run (and Truth vs recovered on demo runs), with their launch forms.
 * A `#study-<kind>` hash scrolls to that card and opens its form.
 */
export function StudyCards() {
  const { params, hash } = useRoute();
  const rid = params.rid ?? "";
  const run = useRunContext();
  const pid = run?.detail?.run.project_id ?? null;
  const demo = !!(run?.detail?.run.demo || run?.detail?.header.demo);
  const rows = useRunStudies(rid || null);
  const studies = useProjectStudies(pid);
  const overview = useView(rid || null, "overview");
  const scenarios = useRunScenarios(rid || null);
  const threadsHeavy = useThreadsHeavy();
  const units = overview.data?.units.target ?? "";
  const focus = hash.startsWith("#study-") ? hash.slice(7) : null;

  // Deep link `#study-<kind>`: bring that card into view once the cards exist.
  useEffect(() => {
    if (!focus || !rows.data) return;
    document.getElementById(`study-${focus}`)?.scrollIntoView?.({ block: "start" });
  }, [focus, rows.data]);

  const ctx = {
    studies: studies.data ?? [],
    threadsHeavy,
    packages: (scenarios.data?.configured ?? []).map((c) => c.name),
    runId: rid,
  };

  return (
    <>
      {rows.error && !rows.data ? <EmptyState error={rows.error} title="Could not load the study status" /> : null}
      {!rows.data && !rows.error ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Loading study status…
        </p>
      ) : null}
      {demo ? (
        <Card title="Truth vs recovered" eyebrow="Synthetic demo" aria-label="Truth vs recovered" className="vt-card">
          <p className="cap" style={{ margin: 0 }}>
            This project's city is synthetic, so the planted truth is known. Each row compares it with what the run recovered.
          </p>
          <TruthCard rid={rid} />
        </Card>
      ) : null}
      {rows.data ? (
        <div className="vt-cards">
          {cardRows(rows.data).map((r) => (
            <StudyCard key={r.kind} row={r} rid={rid} pid={pid} units={units} ctx={ctx} open={focus === r.kind} />
          ))}
        </div>
      ) : null}
    </>
  );
}

export default function Validation() {
  return (
    <section className="sx-page" aria-labelledby="validation-title">
      <header className="stack" style={{ gap: 4 }}>
        <h2 id="validation-title" style={{ margin: 0 }}>
          Validation
        </h2>
        <p className="sx-intro">
          Checks that tell you how far to trust this run: reference baselines, placebo layers, simulation checks with planted effects, a multiverse of alternative choices, an exact
          reproduction and the uncertainty report that combines them. Each study runs against the run's launch snapshot, never the current project config.
        </p>
      </header>
      <StudyCards />
    </section>
  );
}
