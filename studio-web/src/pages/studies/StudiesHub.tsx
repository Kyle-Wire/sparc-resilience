// Studies hub (`/p/:pid/studies`, SPEC §3.2, §8): the per-run status matrix of post-run actions
// and studies (from the project Status Board), the project's studies list filtered by kind, and
// the project-wide effect benchmark launch (it needs no run).
import { useEffect, useState } from "react";
import { KIND_LABELS, useProjectStudies, useStatusBoard, useThreadsHeavy, type BoardCell, type Study, type StudyKind } from "../../api/studies";
import type { RunSummary } from "../../api/types";
import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { useProjectContext } from "../../layouts/ProjectLayout";
import { Link, codecs, useRoute, useUrlState } from "../../router";
import { fmtDate, fmtPct, fmtRelative } from "../../theme/format";
import { runLabel } from "./components/common";
import { LaunchPanel } from "./components/LaunchPanel";
import "./studies.css";

const enc = encodeURIComponent;

/** Board columns the hub shows: the baselines stage, post-run actions and studies. */
export function hubColumns<C extends { id: string; group: string }>(columns: readonly C[]): C[] {
  return columns.filter((c) => c.group === "post" || c.group === "study" || c.id === "baselines");
}

/** Where a matrix cell leads: the study page, the running job, else the run's Validation card. */
export function cellHref(rid: string, column: string, cell: BoardCell | undefined): string {
  if (cell?.study_id) return `/studies/${enc(cell.study_id)}`;
  if (cell?.job_id && cell.state === "running") return `/jobs/${enc(cell.job_id)}`;
  return `/r/${enc(rid)}/validation#study-${column}`;
}

function cellText(cell: BoardCell): string | undefined {
  if (cell.state === "running") {
    if (cell.reason === "queued" || cell.reason === "blocked") return cell.reason;
    return cell.progress !== null ? `running ${fmtPct(cell.progress)}` : undefined;
  }
  return undefined;
}

/** Running cells carry progress that no global event updates: the matrix refetches this often while one runs. */
export const MATRIX_REFRESH_MS = 10_000;

function Matrix({ pid }: { pid: string }) {
  const board = useStatusBoard(pid);
  const running = !!board.data?.rows.some((r) => Object.values(r.cells).some((c) => c.state === "running"));
  const reload = board.reload;
  useEffect(() => {
    if (!running) return;
    const h = setInterval(() => void reload(), MATRIX_REFRESH_MS);
    return () => clearInterval(h);
  }, [running, reload]);
  if (board.error && !board.data) return <EmptyState error={board.error} title="Could not load the status matrix" />;
  if (!board.data) return <p className="cap">Loading…</p>;
  const cols = hubColumns(board.data.columns);
  const rows = board.data.rows.filter((r) => r.run.origin !== "study_child");
  if (!rows.length)
    return (
      <EmptyState title="No runs yet" body="Studies run against a finished run. Launch one first.">
        <Link to={`/p/${enc(pid)}/launch`} className="btn primary">
          Launch a run
        </Link>
      </EmptyState>
    );
  return (
    <div className="tablewrap">
      <table className="hub-matrix" aria-label="Studies by run">
        <thead>
          <tr>
            <th scope="col">Run</th>
            {cols.map((c) => (
              <th key={c.id} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ run, cells }) => (
            <tr key={run.id} data-run={run.id}>
              <th scope="row">
                <Link to={`/r/${enc(run.id)}/validation`}>{runLabel(run)}</Link>
              </th>
              {cols.map((c) => {
                const cell = cells[c.id];
                if (!cell) return <td key={c.id} className="cap">—</td>;
                return (
                  <td key={c.id} data-col={c.id} data-state={cell.state}>
                    <Link to={cellHref(run.id, c.id, cell)} title={cell.reason ?? undefined} aria-label={`${c.label} for ${run.label || run.id}: ${cell.state.replace(/_/g, " ")}${cell.reason ? ` (${cell.reason})` : ""}`}>
                      <StatusChip status={cell.state === "running" && (cell.reason === "queued" || cell.reason === "blocked") ? cell.reason : cell.state} text={cellText(cell)} />
                    </Link>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const KIND_FILTERS: (StudyKind | "all")[] = ["all", "placebo", "simcheck", "multiverse", "reproduce", "benchmark"];

function StudyList({ studies, runs }: { studies: Study[]; runs: Map<string, RunSummary> }) {
  const [kind, setKind] = useUrlState("kind", codecs.enum(KIND_FILTERS, "all"));
  const shown = studies.filter((s) => kind === "all" || s.kind === kind).sort((a, b) => b.created_utc.localeCompare(a.created_utc));
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row">
        <label htmlFor="hub-kind">Kind</label>
        <select id="hub-kind" value={kind} onChange={(e) => setKind(e.target.value as StudyKind | "all")}>
          {KIND_FILTERS.map((k) => (
            <option key={k} value={k}>
              {k === "all" ? "All kinds" : KIND_LABELS[k]}
            </option>
          ))}
        </select>
        <span className="cap">
          {shown.length} of {studies.length}
        </span>
      </div>
      {shown.length ? (
        <div className="tablewrap">
          <table className="tbl" aria-label="Project studies">
            <thead>
              <tr>
                <th scope="col">Study</th>
                <th scope="col">Kind</th>
                <th scope="col">Run</th>
                <th scope="col">Status</th>
                <th scope="col">Attached to</th>
                <th scope="col" className="r">
                  Child runs
                </th>
                <th scope="col">Started</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => {
                const target = s.target_run_id ? runs.get(s.target_run_id) : undefined;
                const label = typeof s.summary?.label === "string" ? s.summary.label : s.id;
                return (
                  <tr key={s.id} data-study={s.id}>
                    <td>
                      <Link to={`/studies/${enc(s.id)}`}>{label}</Link>
                      {s.origin === "imported" ? <span className="cap"> · imported</span> : null}
                    </td>
                    <td>{KIND_LABELS[s.kind as StudyKind] ?? s.kind}</td>
                    <td>{s.target_run_id ? <Link to={`/r/${enc(s.target_run_id)}/validation`}>{target ? runLabel(target) : s.target_run_id}</Link> : <span className="cap">project</span>}</td>
                    <td>
                      <span className="row" style={{ gap: 4 }}>
                        <StatusChip status={s.status} />
                        {s.stale_vs.length ? <StatusChip status="stale" text={`stale vs ${s.stale_vs.length} run${s.stale_vs.length === 1 ? "" : "s"}`} /> : null}
                      </span>
                    </td>
                    <td>{s.attached_runs.length ? `${s.attached_runs.length} run${s.attached_runs.length === 1 ? "" : "s"}` : <span className="cap">—</span>}</td>
                    <td className="r num">{s.children.length}</td>
                    <td title={s.created_utc}>{fmtRelative(s.created_utc)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="cap">{studies.length ? "No study of this kind." : "No studies yet: start one from a run's Validation tab."}</p>
      )}
    </div>
  );
}

export default function StudiesHub() {
  const { params } = useRoute();
  const pid = params.pid ?? "";
  const project = useProjectContext();
  const studies = useProjectStudies(pid || null);
  const threadsHeavy = useThreadsHeavy();
  const [benchKey, setBenchKey] = useState(0);
  const runs = new Map((project?.detail?.runs ?? []).map((r) => [r.id, r]));
  const lastBench = (studies.data ?? []).filter((s) => s.kind === "benchmark").sort((a, b) => b.created_utc.localeCompare(a.created_utc))[0] ?? null;
  return (
    <section className="sx-page" aria-labelledby="hub-title">
      <header className="page-head">
        <h1 id="hub-title">Studies</h1>
        <p className="sx-intro">
          Post-run actions and validation studies for every run of {project?.detail?.project.name ?? "this project"}. Open a cell to see its study, its live job, or the run's
          Validation card to start it.
        </p>
      </header>
      <Card title="Status by run" eyebrow="Matrix">
        <Matrix pid={pid} />
      </Card>
      <Card title="Project studies" eyebrow={studies.data ? `${studies.data.length} in total` : undefined}>
        {studies.error && !studies.data ? <EmptyState error={studies.error} /> : studies.data ? <StudyList studies={studies.data} runs={runs} /> : <p className="cap">Loading…</p>}
      </Card>
      <Card title="Effect benchmark" eyebrow="Project-wide" aria-label="Effect benchmark">
        <p className="cap" style={{ margin: 0 }}>
          Plants a known canopy effect in a synthetic city and measures the share of it each model recovers. It needs no run.{" "}
          {lastBench ? (
            <>
              Last run: <Link to={`/studies/${enc(lastBench.id)}`}>{fmtDate(lastBench.created_utc)}</Link> ({lastBench.status}).
            </>
          ) : (
            "None has run in this project yet."
          )}
        </p>
        <LaunchPanel key={benchKey} kind="benchmark" rid={null} pid={pid} ctx={{ studies: studies.data ?? [], threadsHeavy }} onLaunched={() => setBenchKey((k) => k + 1)} />
      </Card>
    </section>
  );
}
