// Exports & reports (`/p/:pid/exports`, SPEC §6.7): the report builder (section picker, results,
// plans and findings selection, live preview in a sandboxed iframe, export as HTML or Markdown,
// print hint), the run bundle (output checklist, checkpoint toggle with its size), the GIS pack,
// the standalone results page, and the export history with download and delete. Every export is
// `POST /api/exports` (the server creates the row and the `export.<kind>` job).
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { errorMessage, isApiError } from "../../api/client";
import { createExport, deleteExport, EXPORT_LABELS, exportDownloadUrl, previewReport, REPORT_SECTIONS, useExports, type Export, type ExportKind, type ExportParams } from "../../api/exports";
import { useFindings } from "../../api/findings";
import { usePlans, useRunScenarios, useScenarios } from "../../api/lab";
import { invalidate, mutate } from "../../api/resource";
import { useGridMeta } from "../../api/runs";
import { useProjectStudies } from "../../api/studies";
import type { RunSummary } from "../../api/types";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { useProjectContext } from "../../layouts/ProjectLayout";
import { useRunOutputs } from "../../layouts/resources";
import { codecs, Link, useRoute, useUrlState } from "../../router";
import { toast } from "../../stores/ui";
import { fmtBytes, fmtDate, fmtRelative, fmtSignedValue } from "../../theme/format";
import { announceJob, Check, runLabel } from "./components/common";
import {
  bundleOutputs,
  bundleParams,
  CHECKPOINT_WARN_BYTES,
  DEFAULT_SECTIONS,
  inertPreview,
  outputBytes,
  previewBody,
  reportParams,
  sortedExports,
  withExport,
  withSelection,
  type ReportSelection,
} from "./model/report";
import "./studies.css";

const enc = encodeURIComponent;

/** Study statuses of a finished study (`succeeded` from the job, `done`/`complete` on older rows). */
const FINISHED_STUDY = new Set(["succeeded", "done", "complete"]);

/** Create an export, put it at the top of the history and announce its job. */
async function startExport<K extends ExportKind>(pid: string, kind: K, params: ExportParams[K]): Promise<Export | null> {
  try {
    const r = await createExport(pid, kind, params);
    mutate<Export[]>(`project:${pid}:exports`, (prev) => withExport(prev, r.export));
    announceJob(r.job, EXPORT_LABELS[kind]);
    invalidate(`project:${pid}:exports`);
    return r.export;
  } catch (e) {
    toast("error", `Could not start the ${EXPORT_LABELS[kind].toLowerCase()}`, { body: errorMessage(e), action: isApiError(e) && e.action ? e.action : undefined });
    return null;
  }
}

// ---------------------------------------------------------------- report builder

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function PickList<T extends { id: string }>({ legend, items, picked, onChange, render, empty }: { legend: string; items: T[]; picked: string[]; onChange: (ids: string[]) => void; render: (t: T) => ReactNode; empty: string }) {
  return (
    <fieldset aria-label={legend}>
      <legend>
        {legend} {items.length ? <span className="cap">({picked.length} of {items.length})</span> : null}
      </legend>
      {items.length ? (
        <>
          <div className="sx-actions">
            <Button size="small" variant="ghost" onClick={() => onChange(items.map((i) => i.id))}>
              All
            </Button>
            <Button size="small" variant="ghost" onClick={() => onChange([])}>
              None
            </Button>
          </div>
          <div className="sx-checks column rb-pick">
            {items.map((it) => (
              <Check key={it.id} checked={picked.includes(it.id)} onChange={(on) => onChange(on ? [...picked, it.id] : picked.filter((x) => x !== it.id))}>
                {render(it)}
              </Check>
            ))}
          </div>
        </>
      ) : (
        <span className="cap">{empty}</span>
      )}
    </fieldset>
  );
}

export function ReportBuilder({ pid, rid }: { pid: string; rid: string }) {
  const [sel, setSel] = useState<ReportSelection>({ runId: rid, sections: [...DEFAULT_SECTIONS], resultIds: [], planIds: [], findingIds: [] });
  const results = useRunScenarios(rid);
  const scenarios = useScenarios(pid, { run: rid }, rid);
  const plans = usePlans(rid);
  const findings = useFindings(pid);
  // Another run (only a run change resets the picks): its results and plans differ, and only the
  // project-wide findings stay on offer.
  useEffect(() => {
    const projectWide = new Set((findings.data ?? []).filter((f) => f.run_id === null).map((f) => f.id));
    setSel((s) => (s.runId === rid ? s : { ...s, runId: rid, resultIds: [], planIds: [], findingIds: s.findingIds.filter((id) => projectWide.has(id)) }));
  }, [rid]);
  const [preview, setPreview] = useState<{ html: string | null; error: string | null; loading: boolean }>({ html: null, error: null, loading: false });
  const [busy, setBusy] = useState<"html" | "md" | null>(null);

  const names = useMemo(() => new Map((scenarios.data ?? []).map((s) => [s.id, s.name])), [scenarios.data]);
  const resultItems = results.data?.results ?? [];
  const runFindings = (findings.data ?? []).filter((f) => f.run_id === rid || f.run_id === null);
  const body = useDebounced(JSON.stringify(previewBody(sel)), 500);

  useEffect(() => {
    const b = JSON.parse(body) as ReturnType<typeof previewBody>;
    if (!b.sections.length) {
      setPreview({ html: null, error: null, loading: false });
      return;
    }
    const ctrl = new AbortController();
    setPreview((p) => ({ ...p, loading: true }));
    previewReport(pid, b, ctrl.signal).then(
      (r) => setPreview({ html: inertPreview(r.html), error: null, loading: false }),
      (e: unknown) => {
        if (isApiError(e) && e.code === "aborted") return;
        setPreview({ html: null, error: errorMessage(e), loading: false });
      },
    );
    return () => ctrl.abort();
  }, [pid, body]);

  const doExport = async (format: "html" | "md") => {
    setBusy(format);
    await startExport(pid, "report", reportParams(sel, format));
    setBusy(null);
  };

  return (
    <div className="rb">
      <div className="rb-side">
        <fieldset aria-label="Sections">
          <legend>Sections</legend>
          <div className="sx-checks column">
            {REPORT_SECTIONS.map((s) => (
              <Check
                key={s.id}
                checked={sel.sections.includes(s.id)}
                onChange={(on) => setSel((cur) => withSelection(cur, { sections: on ? [...cur.sections, s.id] : cur.sections.filter((x) => x !== s.id) }))}
              >
                {s.label} <small>{s.hint}</small>
              </Check>
            ))}
          </div>
        </fieldset>
        <PickList
          legend="Exact results"
          items={resultItems}
          picked={sel.resultIds}
          onChange={(ids) => setSel((cur) => withSelection(cur, { resultIds: ids }))}
          render={(r) => (
            <>
              {(r.scenario_id && names.get(r.scenario_id)) || r.kind} · {fmtSignedValue(r.city.estimate)} {r.stale ? <Badge tone="warn">stale</Badge> : null} <small>{fmtDate(r.created_utc)}</small>
            </>
          )}
          empty="No exact results on this run yet (Scenario Lab)."
        />
        <PickList
          legend="Budget plans"
          items={plans.data ?? []}
          picked={sel.planIds}
          onChange={(ids) => setSel((cur) => withSelection(cur, { planIds: ids }))}
          render={(p) => (
            <>
              {p.name} {p.realised ? <small>verified</small> : <small>unverified</small>}
            </>
          )}
          empty="No budget plans on this run."
        />
        <PickList
          legend="Findings"
          items={runFindings}
          picked={sel.findingIds}
          onChange={(ids) => setSel((cur) => withSelection(cur, { findingIds: ids }))}
          render={(f) => (
            <>
              {f.title} <small>{f.run_id ? "this run" : "project"}</small>
            </>
          )}
          empty="No pinned findings for this run."
        />
        <div className="sx-actions">
          <Button variant="primary" icon="download" busy={busy === "html"} disabled={!sel.sections.length} onClick={() => void doExport("html")}>
            Export HTML
          </Button>
          <Button icon="download" busy={busy === "md"} disabled={!sel.sections.length} onClick={() => void doExport("md")}>
            Export Markdown
          </Button>
        </div>
        <p className="cap" role="note">
          For a PDF, open the exported HTML in a browser and print it: the report carries a print stylesheet (A4 pages, a page break before each section, charts kept whole,
          links written out).
        </p>
      </div>
      <div className="rb-preview">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <strong>Preview</strong>
          {preview.loading ? (
            <span className="cap" role="status">
              <span className="spinner" aria-hidden="true" /> Updating…
            </span>
          ) : null}
        </div>
        {preview.error ? <div className="callout" data-tone="crit" role="alert">{preview.error}</div> : null}
        {!sel.sections.length ? <p className="cap">Pick at least one section.</p> : null}
        {preview.html !== null ? (
          <iframe className="rb-frame" title="Report preview" sandbox="" referrerPolicy="no-referrer" srcDoc={preview.html} data-testid="report-preview" />
        ) : !preview.error && sel.sections.length ? (
          <p className="cap">Rendering the preview…</p>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- bundle, GIS pack, page

function BundleCard({ pid, run }: { pid: string; run: RunSummary }) {
  const outs = useRunOutputs(run.id);
  const choices = useMemo(() => bundleOutputs(outs.data?.outputs ?? []), [outs.data]);
  const [picked, setPicked] = useState<string[] | null>(null);
  const [ckpt, setCkpt] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => setPicked(null), [run.id]);
  const chosen = picked ?? choices.map((o) => o.id);
  const ckEntry = (outs.data?.outputs ?? []).find((o) => o.id === "checkpoint");
  const ckBytes = ckEntry && ckEntry.state !== "missing" ? outputBytes(ckEntry) || run.checkpoint_bytes : run.checkpoint_bytes;
  const total = choices.filter((o) => chosen.includes(o.id)).reduce((a, o) => a + outputBytes(o), 0) + (ckpt ? ckBytes ?? 0 : 0);
  const go = async () => {
    setBusy(true);
    await startExport(pid, "bundle", bundleParams(run.id, chosen, ckpt));
    setBusy(false);
  };
  return (
    <Card title="Run bundle" eyebrow="ZIP" aria-label="Run bundle">
      <p className="cap" style={{ margin: 0 }}>
        The selected outputs with a contents manifest and a README giving units and sign conventions.
      </p>
      {outs.error && !outs.data ? <EmptyState error={outs.error} /> : null}
      {choices.length ? (
        <details>
          <summary className="cap">
            Outputs ({chosen.length} of {choices.length})
          </summary>
          <div className="sx-actions">
            <Button size="small" variant="ghost" onClick={() => setPicked(choices.map((o) => o.id))}>
              All
            </Button>
            <Button size="small" variant="ghost" onClick={() => setPicked([])}>
              None
            </Button>
          </div>
          <div className="sx-checks column rb-pick" role="group" aria-label="Outputs to include">
            {choices.map((o) => (
              <Check key={o.id} checked={chosen.includes(o.id)} onChange={(on) => setPicked(on ? [...chosen, o.id] : chosen.filter((x) => x !== o.id))}>
                {o.label} <small>{fmtBytes(outputBytes(o))}</small>
                {o.state === "stale" ? <Badge tone="warn">stale</Badge> : null}
              </Check>
            ))}
          </div>
        </details>
      ) : outs.data ? (
        <p className="cap">This run has no outputs to bundle yet.</p>
      ) : null}
      <Check checked={ckpt} onChange={setCkpt} disabled={!ckBytes}>
        Include the checkpoint{ckBytes ? ` (${fmtBytes(ckBytes)})` : " (none)"}
      </Check>
      {ckpt && ckBytes && ckBytes >= CHECKPOINT_WARN_BYTES ? (
        <p className="callout" role="note" data-testid="checkpoint-warning">
          The checkpoint adds {fmtBytes(ckBytes)} to the ZIP. It is only useful to resume or open this run in another Studio with the same code; leave it out to share results.
        </p>
      ) : null}
      <div className="sx-actions">
        {/* An empty `outputs` list means "every output" to the server, so at least one is required. */}
        <Button icon="download" busy={busy} disabled={!chosen.length} onClick={() => void go()}>
          Build bundle
        </Button>
        {!chosen.length && choices.length ? <span className="cap">Pick at least one output.</span> : total ? <span className="cap">≈ {fmtBytes(total)} before compression</span> : null}
      </div>
    </Card>
  );
}

function GisCard({ pid, run }: { pid: string; run: RunSummary }) {
  const grid = useGridMeta(run.id);
  const [busy, setBusy] = useState(false);
  const noCrs = grid.data ? !grid.data.crs : false;
  return (
    <Card title="GIS pack" eyebrow="GeoTIFF + GeoPackage" aria-label="GIS pack">
      <p className="cap" style={{ margin: 0 }}>
        Every layer as a GeoTIFF in the data's CRS, the hexagon summaries as GeoPackage, and the logger sites and pairs with longitude and latitude.
      </p>
      {noCrs ? (
        <p className="callout" role="note">
          This run has no coordinate reference system, so GeoTIFF, GeoPackage and lon/lat cannot be written. Set the data CRS in Setup → Data and run again.
        </p>
      ) : grid.error && !grid.data ? (
        <p className="callout" role="note">
          The run's grid could not be read, so its layers cannot be exported yet: {errorMessage(grid.error)}
        </p>
      ) : null}
      <div className="sx-actions">
        <Button
          icon="layers"
          busy={busy}
          disabled={noCrs || !grid.data}
          onClick={async () => {
            setBusy(true);
            await startExport(pid, "gis", { run_id: run.id });
            setBusy(false);
          }}
        >
          Build GIS pack
        </Button>
      </div>
    </Card>
  );
}

function PageCard({ pid, run }: { pid: string; run: RunSummary }) {
  const studies = useProjectStudies(pid);
  // Finished placebo suites only: a cancelled or interrupted one has no placebo.json to read.
  const placebo = (studies.data ?? []).filter((s) => s.kind === "placebo" && FINISHED_STUDY.has(s.status));
  const [study, setStudy] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <Card title="Standalone results page" eyebrow="results.html" aria-label="Standalone results page">
      <p className="cap" style={{ margin: 0 }}>
        One self-contained HTML file with the maps, charts and caveats of this run, readable offline.
      </p>
      <div className="field">
        <label htmlFor="page-placebo">Placebo results to include</label>
        <select id="page-placebo" value={study} onChange={(e) => setStudy(e.target.value)}>
          <option value="">The run's own (if attached)</option>
          {placebo.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} · {fmtDate(s.created_utc)}
              {s.target_run_id === run.id ? " · this run" : ""}
            </option>
          ))}
        </select>
      </div>
      <div className="sx-actions">
        <Button
          icon="file"
          busy={busy}
          onClick={async () => {
            setBusy(true);
            await startExport(pid, "page", study ? { run_id: run.id, placebo_study: study } : { run_id: run.id });
            setBusy(false);
          }}
        >
          Build results page
        </Button>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------- history

const STATUS_OF: Record<Export["status"], string> = { running: "running", ready: "ready", failed: "failed" };

export function ExportHistory({ pid }: { pid: string }) {
  const res = useExports(pid);
  const [del, setDel] = useState<Export | null>(null);
  const [busy, setBusy] = useState(false);
  const list = sortedExports(res.data ?? []);
  const remove = async (e: Export) => {
    setBusy(true);
    try {
      await deleteExport(e.id);
      mutate<Export[]>(`project:${pid}:exports`, (prev) => (prev ?? []).filter((x) => x.id !== e.id));
      toast("success", `${EXPORT_LABELS[e.kind as ExportKind] ?? e.kind} deleted`);
    } catch (err) {
      toast("error", "Could not delete the export", { body: errorMessage(err) });
    } finally {
      setBusy(false);
      setDel(null);
    }
  };
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  if (!res.data) return <p className="cap">Loading…</p>;
  if (!list.length) return <p className="cap">No exports yet.</p>;
  return (
    <>
      <div className="tablewrap">
        <table className="tbl" aria-label="Export history">
          <thead>
            <tr>
              <th scope="col">Export</th>
              <th scope="col">Run</th>
              <th scope="col">Created</th>
              <th scope="col">Status</th>
              <th scope="col" className="r">
                Size
              </th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {list.map((e) => {
              const fmt = typeof e.options.format === "string" ? ` (${e.options.format.toUpperCase()})` : "";
              return (
                <tr key={e.id} data-export={e.id} data-status={e.status}>
                  <td>
                    {EXPORT_LABELS[e.kind as ExportKind] ?? e.kind}
                    {fmt}
                    {e.draft ? (
                      <>
                        {" "}
                        <Badge tone="warn" title="Built from a preview or an unverified plan">
                          draft
                        </Badge>
                      </>
                    ) : null}
                  </td>
                  <td>{e.run_id ? <Link to={`/r/${enc(e.run_id)}`}>{e.run_id}</Link> : <span className="cap">—</span>}</td>
                  <td title={e.created_utc}>{fmtRelative(e.created_utc)}</td>
                  <td>
                    <StatusChip status={STATUS_OF[e.status] ?? e.status} />
                  </td>
                  <td className="r num">{e.bytes !== null ? fmtBytes(e.bytes) : "—"}</td>
                  <td>
                    <span className="row" style={{ gap: 6 }}>
                      {e.status === "ready" ? (
                        <a className="btn small" href={exportDownloadUrl(e.id)} download>
                          Download
                        </a>
                      ) : e.status === "running" ? (
                        <Link to={`/jobs/${enc(e.job_id)}`} className="btn small ghost">
                          Track
                        </Link>
                      ) : (
                        <Link to={`/jobs/${enc(e.job_id)}`} className="btn small ghost">
                          Why it failed
                        </Link>
                      )}
                      <Button size="small" variant="ghost" onClick={() => setDel(e)} aria-label={`Delete ${EXPORT_LABELS[e.kind as ExportKind] ?? e.kind} ${e.id}`}>
                        Delete
                      </Button>
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ConfirmDialog open={!!del} onClose={() => setDel(null)} onConfirm={() => del && void remove(del)} title="Delete this export?" confirmLabel="Delete" danger busy={busy}>
        <p>The file is removed from the project's exports folder. The run and its outputs are not touched.</p>
      </ConfirmDialog>
    </>
  );
}

// ---------------------------------------------------------------- page

export default function Exports() {
  const { params } = useRoute();
  const pid = params.pid ?? "";
  const project = useProjectContext();
  const runs = (project?.detail?.runs ?? []).filter((r) => r.origin !== "study_child");
  const [runParam, setRun] = useUrlState("run", codecs.optString());
  const fallback = project?.detail?.project.active_run_id ?? runs[0]?.id ?? null;
  const rid = runParam && runs.some((r) => r.id === runParam) ? runParam : fallback;
  const run = runs.find((r) => r.id === rid) ?? null;
  return (
    <section className="sx-page" aria-labelledby="exports-title">
      <header className="page-head">
        <h1 id="exports-title">Exports &amp; reports</h1>
        <p className="sx-intro">Build a project report from a run, or export its outputs. Every export is a tracked job; the files land in the project's exports folder and are listed below.</p>
      </header>
      {project?.detail && !runs.length ? (
        <EmptyState title="No runs yet" body="Reports and exports are built from a run.">
          <Link to={`/p/${enc(pid)}/launch`} className="btn primary">
            Launch a run
          </Link>
        </EmptyState>
      ) : null}
      {runs.length ? (
        <div className="row">
          <label htmlFor="exports-run">Run</label>
          <select id="exports-run" value={rid ?? ""} onChange={(e) => setRun(e.target.value || null)}>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {runLabel(r)}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      {run ? (
        <>
          <Card title="Report builder" eyebrow="Project report">
            <ReportBuilder pid={pid} rid={run.id} />
          </Card>
          <div className="grid2">
            <BundleCard pid={pid} run={run} />
            <GisCard pid={pid} run={run} />
            <PageCard pid={pid} run={run} />
          </div>
        </>
      ) : null}
      <Card title="Export history" eyebrow="This project">
        <ExportHistory pid={pid} />
      </Card>
    </section>
  );
}
