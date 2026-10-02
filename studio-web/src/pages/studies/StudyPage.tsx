// Study page (`/studies/:stid`, SPEC §3.2, §5.11, §8): what the study is and where it writes,
// its live job, its child runs (each linking to its run and tracker), the result view for its
// kind, attach/detach, resume, delete, and for simulation checks a merge across studies
// (`POST /api/studies/simcheck/merge`, inline).
import { useEffect, useState } from "react";
import { errorMessage, isApiError } from "../../api/client";
import { invalidate } from "../../api/resource";
import { useView } from "../../api/runs";
import {
  ATTACHABLE_KINDS,
  deleteStudy,
  hasStudyView,
  KIND_LABELS,
  mergeSimcheck,
  resumeStudy,
  useProjectStudies,
  useStudy,
  type SimcheckMerge,
  type Study,
  type StudyKind,
} from "../../api/studies";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { JobStrip } from "../../components/ui/JobStrip";
import { Markdown } from "../../components/ui/Markdown";
import { StatusChip } from "../../components/ui/StatusChip";
import { downloadText, fileSlug } from "../../components/ui/download";
import { Link, navigate, useRoute } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { fmtDate, fmtDateTime, fmtDuration, fmtNum } from "../../theme/format";
import { announceJob, AttachToggle, Check, runLabel } from "./components/common";
import { BiasCorrectionCard, GeneratorTable, StudyViewPanel } from "./components/StudyViews";
import "./studies.css";

const enc = encodeURIComponent;

const ACTIVE = new Set(["queued", "starting", "running", "cancelling", "blocked"]);
const DONE = new Set(["done", "succeeded", "complete"]);

/** A study whose job ended without finishing can be resumed (children and replicates already done are reused). */
export function canResume(s: Pick<Study, "status" | "origin">): boolean {
  return s.origin === "studio" && !ACTIVE.has(s.status) && !DONE.has(s.status);
}

function ChildRuns({ study }: { study: Study }) {
  if (!study.children.length) return <p className="cap">No child runs{ACTIVE.has(study.status) ? " yet" : ""}.</p>;
  return (
    <div className="tablewrap">
      <table className="tbl" aria-label="Child runs">
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Status</th>
            <th scope="col" className="r">
              R²
            </th>
            <th scope="col" className="r">
              Time
            </th>
            <th scope="col">Open</th>
          </tr>
        </thead>
        <tbody>
          {study.children.map((c) => (
            <tr key={c.id} data-child={c.id}>
              <td>{runLabel(c)}</td>
              <td>
                <StatusChip status={c.status} />
              </td>
              <td className="r num">{fmtNum(c.r2, 3)}</td>
              <td className="r num">{fmtDuration(c.duration_s)}</td>
              <td>
                <span className="row" style={{ gap: 10 }}>
                  <Link to={`/r/${enc(c.id)}`}>Run</Link>
                  <Link to={`/r/${enc(c.id)}/track`}>Tracker</Link>
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SimcheckMergePanel({ study }: { study: Study }) {
  const all = useProjectStudies(study.project_id);
  const others = (all.data ?? []).filter((s) => s.kind === "simcheck" && s.id !== study.id);
  const [picked, setPicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SimcheckMerge | null>(null);
  const merge = async () => {
    setBusy(true);
    try {
      setResult(await mergeSimcheck([study.id, ...picked]));
    } catch (e) {
      toast("error", "Could not merge the simulation checks", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card title="Merge simulation checks" eyebrow="Across studies" aria-label="Merge simulation checks">
      <p className="cap" style={{ margin: 0 }}>
        Pools the replicates of several simulation-check studies (for example a quick first pass and a later continuation) into one summary. Nothing is written.
      </p>
      {others.length ? (
        <fieldset className="sx-checks column" aria-label="Studies to merge with this one">
          <legend className="cap">Merge this study with</legend>
          {others.map((s) => (
            <Check key={s.id} checked={picked.includes(s.id)} onChange={(on) => setPicked(on ? [...picked, s.id] : picked.filter((x) => x !== s.id))}>
              {typeof s.summary?.label === "string" ? s.summary.label : s.id} · {s.status} · {fmtDate(s.created_utc)}
              {s.target_run_id ? ` · run ${s.target_run_id}` : ""}
            </Check>
          ))}
        </fieldset>
      ) : (
        <p className="cap">No other simulation-check study in this project.</p>
      )}
      <div className="sx-actions">
        <Button busy={busy} disabled={!picked.length} onClick={() => void merge()}>
          {picked.length ? `Merge ${picked.length + 1} studies` : "Pick a study to merge with"}
        </Button>
      </div>
      {result ? (
        <div className="stack" aria-label="Merged summary">
          <p className="cap">
            {result.summary.n_rows ?? "?"} replicates, {result.summary.n_errors ?? 0} errors.
          </p>
          {result.summary.generators ? <GeneratorTable generators={result.summary.generators} /> : null}
          {result.summary.bias_correction ? <BiasCorrectionCard bc={result.summary.bias_correction} /> : null}
          <details>
            <summary className="cap">Merged summary as Markdown</summary>
            <Markdown source={result.markdown} />
          </details>
          <div>
            <Button size="small" icon="download" onClick={() => downloadText(result.markdown, `simcheck-merged-${fileSlug(study.id)}.md`, "text/markdown")}>
              Download Markdown
            </Button>
          </div>
        </div>
      ) : null}
    </Card>
  );
}

function Header({ study }: { study: Study }) {
  const [busy, setBusy] = useState<"resume" | "delete" | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [withFiles, setWithFiles] = useState(false);
  const label = KIND_LABELS[study.kind as StudyKind] ?? study.kind;
  const attachable = ATTACHABLE_KINDS.includes(study.kind as StudyKind) && !!study.target_run_id;
  const resume = async () => {
    setBusy("resume");
    try {
      const job = await resumeStudy(study.id);
      announceJob(job, `${label} (resume)`);
      invalidate(`study:${study.id}`);
    } catch (e) {
      toast("error", "Could not resume the study", { body: errorMessage(e), action: isApiError(e) && e.action ? e.action : undefined });
    } finally {
      setBusy(null);
    }
  };
  const remove = async () => {
    setBusy("delete");
    try {
      await deleteStudy(study.id, withFiles && study.origin === "studio");
      toast("success", `${label} deleted`);
      invalidate("studies");
      if (study.target_run_id) invalidate(`run:${study.target_run_id}:studies`);
      navigate(`/p/${enc(study.project_id)}/studies`);
    } catch (e) {
      toast("error", isApiError(e) && e.code === "active" ? "Cancel the study's job before deleting it" : "Could not delete the study", { body: errorMessage(e) });
    } finally {
      setBusy(null);
      setConfirm(false);
    }
  };
  return (
    <header className="page-head">
      <p className="eyebrow">
        <Link to={`/p/${enc(study.project_id)}/studies`}>Studies</Link> › {study.id}
      </p>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <h1 style={{ margin: 0 }}>{typeof study.summary?.label === "string" ? study.summary.label : label}</h1>
          <StatusChip status={study.status} />
          {study.origin === "imported" ? <Badge>imported</Badge> : null}
          {study.stale_vs.length ? <StatusChip status="stale" text="stale vs run" /> : null}
        </div>
        <div className="sx-actions">
          {attachable ? <AttachToggle studyId={study.id} runId={study.target_run_id!} attached={study.attached_runs.includes(study.target_run_id!)} label={label} /> : null}
          {canResume(study) ? (
            <Button icon="play" busy={busy === "resume"} onClick={() => void resume()}>
              Resume
            </Button>
          ) : null}
          <Button variant="danger" onClick={() => setConfirm(true)} disabled={ACTIVE.has(study.status)} title={ACTIVE.has(study.status) ? "Cancel the job first" : undefined}>
            Delete
          </Button>
        </div>
      </div>
      <div className="sx-meta" role="group" aria-label="Study details">
        <span>
          Kind <b>{label}</b>
        </span>
        <span>
          Run{" "}
          {study.target_run_id ? (
            <Link to={`/r/${enc(study.target_run_id)}/validation`}>
              <b>{study.target_run_id}</b>
            </Link>
          ) : (
            <b>project-wide</b>
          )}
        </span>
        <span>
          Started <b>{fmtDateTime(study.created_utc)}</b>
        </span>
        <span>
          Updated <b>{fmtDateTime(study.updated_utc)}</b>
        </span>
      </div>
      <p className="cap" style={{ margin: 0 }}>
        Writes to <span className="sx-mono">{study.out_dir}</span>
      </p>
      <ConfirmDialog
        open={confirm}
        onClose={() => setConfirm(false)}
        onConfirm={() => void remove()}
        title={`Delete ${label}?`}
        confirmLabel="Delete"
        danger
        busy={busy === "delete"}
      >
        <p>The study is removed from Studio{study.attached_runs.length ? ` and detached from ${study.attached_runs.length} run(s)` : ""}.</p>
        {study.origin === "studio" ? (
          <Check checked={withFiles} onChange={setWithFiles}>
            Also delete its folder on disk and the child runs it holds
          </Check>
        ) : (
          <p className="cap">The imported folder on disk is never touched; only Studio's record of it goes.</p>
        )}
      </ConfirmDialog>
    </header>
  );
}

export default function StudyPage() {
  const { params } = useRoute();
  const stid = params.stid ?? "";
  const res = useStudy(stid || null);
  const study = res.data;
  const setContext = useUi((s) => s.setContext);
  const overview = useView(study?.target_run_id ?? null, "overview");
  const units = overview.data?.units.target ?? "";
  useEffect(() => {
    if (study) setContext({ projectId: study.project_id, runId: null });
  }, [study, setContext]);

  if (res.error && !res.data) return <EmptyState error={res.error} title={isApiError(res.error) && res.error.status === 404 ? "Study not found" : undefined} />;
  if (!study) return <p className="cap">Loading study…</p>;
  const kind = study.kind;
  const workers = typeof study.params.workers === "number" ? study.params.workers : 1;
  const summary = study.summary ?? {};
  const headline = typeof summary.headline === "string" ? summary.headline : null;
  return (
    <section className="sx-page" aria-label={`Study ${study.id}`}>
      <Header study={study} />
      {headline ? <p className="vt-headline">{headline}</p> : null}
      {study.job_id && ACTIVE.has(study.status) ? (
        <Card title="Live" eyebrow={`Job ${study.job_id}`}>
          <JobStrip jobId={study.job_id} />
          <p className="cap" style={{ margin: 0 }}>
            Replicates and child runs fill in below as they finish. <Link to={`/jobs/${enc(study.job_id)}`}>Open Mission Control →</Link>
          </p>
        </Card>
      ) : study.job_id ? (
        <p className="cap">
          Last job: <Link to={`/jobs/${enc(study.job_id)}`}>{study.job_id}</Link> (logs, timings and resources).
        </p>
      ) : null}
      {study.stale_vs.length ? (
        <div className="callout" role="note">
          The checkpoint of {study.stale_vs.length === 1 ? "run" : "runs"}{" "}
          {study.stale_vs.map((r, i) => (
            <span key={r}>
              {i ? ", " : ""}
              <Link to={`/r/${enc(r)}`}>{r}</Link>
            </span>
          ))}{" "}
          changed since this study ran. Its results describe the earlier fit; run it again to compare like with like.
        </div>
      ) : null}
      {hasStudyView(kind) ? (
        <Card title="Results" eyebrow={KIND_LABELS[kind]}>
          <StudyViewPanel
            kind={kind}
            studyId={study.id}
            units={units}
            workers={workers}
            childRunId={kind === "reproduce" ? study.children[0]?.id ?? null : null}
            empty={<p className="cap">No results yet.</p>}
            live={ACTIVE.has(study.status)}
          />
        </Card>
      ) : null}
      {kind === "simcheck" ? <SimcheckMergePanel study={study} /> : null}
      <Card title="Child runs" eyebrow={`${study.children.length} run${study.children.length === 1 ? "" : "s"}`}>
        <ChildRuns study={study} />
      </Card>
      {study.attached_runs.length ? (
        <Card title="Attached to" eyebrow="Feeds the uncertainty report of">
          <ul className="sx-list">
            {study.attached_runs.map((r) => (
              <li key={r} className="row" style={{ justifyContent: "space-between" }}>
                <Link to={`/r/${enc(r)}/uncertainty`}>{r}</Link>
                <AttachToggle studyId={study.id} runId={r} attached label={KIND_LABELS[kind as StudyKind] ?? kind} />
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
      <details className="card">
        <summary>Parameters and summary</summary>
        <pre className="mv-json">{JSON.stringify({ params: study.params, summary: study.summary }, null, 2)}</pre>
      </details>
    </section>
  );
}
