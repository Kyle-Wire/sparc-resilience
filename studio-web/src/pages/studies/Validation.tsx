// Validation run tab (`/r/:rid/validation`, group Trust; SPEC §8): one card per study kind with
// its status, cost estimate, requirements, launch form and result chart, plus "Truth vs
// recovered" for synthetic demo projects. Rows come from `GET /api/runs/{rid}/studies` and are
// refreshed by `study.updated` and the run's job events.
import { useEffect } from "react";
import { useRunScenarios } from "../../api/lab";
import { useView } from "../../api/runs";
import { STUDY_KINDS, useProjectStudies, useRunStudies, useThreadsHeavy, type StudyKind, type StudyStatusRow } from "../../api/studies";
import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/EmptyState";
import { useRunContext } from "../../layouts/RunLayout";
import { useRoute } from "../../router";
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

export default function Validation() {
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
    </section>
  );
}
