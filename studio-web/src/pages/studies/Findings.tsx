// Findings notebook (`/p/:pid/findings` and `/findings`, SPEC §6.10): pinned views in their
// order, each with its image, its markdown note (edited in place), the numbers that were on
// screen, and "Open" back at the stored URL. Reordering (drag, or the move buttons) sends PATCH
// position updates; the run filter lives in the URL; export writes Markdown (ZIP with images)
// or self-contained HTML through `POST /api/exports` (kind "findings").
import { useState, type DragEvent } from "react";
import { errorMessage } from "../../api/client";
import { createExport, exportDownloadUrl, useExport } from "../../api/exports";
import { deleteFinding, patchFinding, useFindings, type Finding } from "../../api/findings";
import { mutate } from "../../api/resource";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { IconButton } from "../../components/ui/IconButton";
import { Markdown } from "../../components/ui/Markdown";
import { StatusChip } from "../../components/ui/StatusChip";
import { useProjects } from "../../layouts/resources";
import { codecs, Link, navigate, useRegistry, useRoute, useUrlState } from "../../router";
import { toast } from "../../stores/ui";
import { fmtRelative } from "../../theme/format";
import { announceJob } from "./components/common";
import { applyPositions, findingHref, findingRuns, ordered, reorderPatches, snapshotTable } from "./model/findings";
import "./studies.css";

const enc = encodeURIComponent;

const cacheKey = (pid: string | null) => (pid ? `project:${pid}:findings` : "findings:all");

function SnapshotView({ snapshot }: { snapshot: Record<string, unknown> }) {
  const t = snapshotTable(snapshot);
  if (!t) return null;
  return (
    <details>
      <summary className="cap">Numbers on screen when pinned</summary>
      <div className="tablewrap">
        <table className="tbl" aria-label="Snapshot">
          <thead>
            <tr>
              {t.columns.map((c, i) => (
                <th key={i} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.slice(0, 200).map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => (
                  <td key={j} className={typeof v === "number" ? "r num" : undefined}>
                    {v === null ? "—" : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {t.rows.length > 200 ? <p className="cap">First 200 of {t.rows.length} rows.</p> : null}
    </details>
  );
}

type ItemProps = {
  f: Finding;
  index: number;
  count: number;
  reorderable: boolean;
  projectName: string | null;
  onMove: (from: number, to: number) => void;
  onPatched: (f: Finding) => void;
  onDelete: (f: Finding) => void;
  drag: { from: number | null; over: number | null };
  setDrag: (d: { from: number | null; over: number | null }) => void;
};

function FindingItem({ f, index, count, reorderable, projectName, onMove, onPatched, onDelete, drag, setDrag }: ItemProps) {
  const registry = useRegistry();
  // The route's title ("Accuracy") for the stored route pattern; unknown views show as stored.
  const viewTitle = registry.routes.find((r) => r.path === f.view)?.title ?? f.view;
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(f.title);
  const [note, setNote] = useState(f.note_md);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    const patch: { title?: string; note_md?: string } = {};
    if (title.trim() && title !== f.title) patch.title = title.trim();
    if (note !== f.note_md) patch.note_md = note;
    if (!Object.keys(patch).length) {
      setEditing(false);
      return;
    }
    setBusy(true);
    try {
      onPatched(await patchFinding(f.id, patch));
      setEditing(false);
    } catch (e) {
      toast("error", "Could not save the note", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const dnd = reorderable
    ? {
        draggable: !editing,
        onDragStart: (e: DragEvent) => {
          e.dataTransfer?.setData("text/plain", f.id);
          if (e.dataTransfer) e.dataTransfer.effectAllowed = "move";
          setDrag({ from: index, over: index });
        },
        onDragOver: (e: DragEvent) => {
          if (drag.from === null) return;
          e.preventDefault();
          if (drag.over !== index) setDrag({ ...drag, over: index });
        },
        onDrop: (e: DragEvent) => {
          e.preventDefault();
          if (drag.from !== null && drag.from !== index) onMove(drag.from, index);
          setDrag({ from: null, over: null });
        },
        onDragEnd: () => setDrag({ from: null, over: null }),
      }
    : {};
  return (
    <li
      className="card tight fd-item"
      data-finding={f.id}
      data-dragging={drag.from === index || undefined}
      data-drop={drag.from !== null && drag.over === index && drag.from !== index ? true : undefined}
      aria-label={f.title}
      {...dnd}
    >
      <div className="fd-handle">
        {reorderable ? (
          <>
            <span className="grip" aria-hidden="true" title="Drag to reorder">
              ⠿
            </span>
            <IconButton icon="chevronUp" size="small" label={`Move “${f.title}” up`} disabled={index === 0} onClick={() => onMove(index, index - 1)} />
            <IconButton icon="chevronDown" size="small" label={`Move “${f.title}” down`} disabled={index === count - 1} onClick={() => onMove(index, index + 1)} />
          </>
        ) : null}
        <span className="cap num">{index + 1}</span>
      </div>
      <div className="fd-body">
        {editing ? (
          <input type="text" aria-label="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
        ) : (
          <h3>{f.title}</h3>
        )}
        <div className="sx-meta">
          {projectName ? <span>{projectName}</span> : null}
          {f.run_id ? (
            <span>
              Run <Link to={`/r/${enc(f.run_id)}`}>{f.run_id}</Link>
            </span>
          ) : null}
          <span title={f.view}>{viewTitle}</span>
          <span title={f.created_utc}>pinned {fmtRelative(f.created_utc)}</span>
        </div>
        {editing ? (
          <div className="fd-note">
            <textarea aria-label={`Note for ${f.title} (Markdown)`} value={note} onChange={(e) => setNote(e.target.value)} placeholder="What does this show? Markdown is fine." />
            {note.trim() ? (
              <details>
                <summary className="cap">Preview</summary>
                <Markdown source={note} />
              </details>
            ) : null}
          </div>
        ) : f.note_md.trim() ? (
          <Markdown source={f.note_md} className="fd-note-view" />
        ) : (
          <p className="cap" style={{ margin: 0 }}>
            No note yet.
          </p>
        )}
        <SnapshotView snapshot={f.snapshot} />
        <div className="sx-actions">
          <Button size="small" icon="external" onClick={() => navigate(findingHref(f))} data-open={f.id}>
            Open
          </Button>
          {editing ? (
            <>
              <Button size="small" variant="primary" busy={busy} onClick={() => void save()}>
                Save
              </Button>
              <Button
                size="small"
                variant="ghost"
                onClick={() => {
                  setTitle(f.title);
                  setNote(f.note_md);
                  setEditing(false);
                }}
              >
                Cancel
              </Button>
            </>
          ) : (
            <Button
              size="small"
              onClick={() => {
                setTitle(f.title);
                setNote(f.note_md);
                setEditing(true);
              }}
            >
              Edit note
            </Button>
          )}
          <Button size="small" variant="ghost" onClick={() => onDelete(f)}>
            Delete
          </Button>
        </div>
      </div>
      {f.image_url ? (
        <a href={f.image_url} target="_blank" rel="noopener noreferrer" title="Open the image">
          <img className="fd-thumb" src={f.image_url} alt={`Pinned image: ${f.title}`} loading="lazy" />
        </a>
      ) : null}
    </li>
  );
}

function ExportBar({ pid, runFilter, visible }: { pid: string; runFilter: string | null; visible: Finding[] }) {
  const [busy, setBusy] = useState<"md" | "html" | null>(null);
  const [last, setLast] = useState<{ id: string; job: string } | null>(null);
  const exp = useExport(last?.id ?? null, last?.job ?? null);
  const go = async (format: "md" | "html") => {
    setBusy(format);
    try {
      const params = runFilter ? { project_id: pid, run_id: runFilter, ids: visible.map((f) => f.id), format } : { project_id: pid, format };
      const r = await createExport(pid, "findings", params);
      announceJob(r.job, format === "md" ? "Findings (Markdown)" : "Findings (HTML)");
      setLast({ id: r.export.id, job: r.job.id });
      mutate(`export:${r.export.id}`, r.export);
    } catch (e) {
      toast("error", "Could not export the findings", { body: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };
  const e = exp.data;
  return (
    <div className="sx-actions" aria-label="Export findings" role="group">
      <Button icon="download" busy={busy === "md"} disabled={!visible.length} onClick={() => void go("md")}>
        Export Markdown (ZIP)
      </Button>
      <Button icon="download" busy={busy === "html"} disabled={!visible.length} onClick={() => void go("html")}>
        Export HTML
      </Button>
      {e ? (
        <span className="row" style={{ gap: 6 }}>
          <StatusChip status={e.status} />
          {e.status === "ready" ? (
            <a className="btn small" href={exportDownloadUrl(e.id)} download>
              Download
            </a>
          ) : null}
          <Link to={`/p/${enc(pid)}/exports`}>All exports</Link>
        </span>
      ) : null}
    </div>
  );
}

export default function Findings() {
  const { params } = useRoute();
  const routePid = params.pid ?? null;
  const [projectParam, setProjectParam] = useUrlState("project", codecs.optString());
  const pid = routePid ?? projectParam;
  const [runFilter, setRunFilter] = useUrlState("run", codecs.optString());
  const res = useFindings(pid);
  const projects = useProjects();
  const [drag, setDrag] = useState<{ from: number | null; over: number | null }>({ from: null, over: null });
  const [del, setDel] = useState<Finding | null>(null);
  const [deleting, setDeleting] = useState(false);
  const key = cacheKey(pid);
  const all = ordered(res.data ?? []);
  const visible = runFilter ? all.filter((f) => f.run_id === runFilter) : all;
  const runs = findingRuns(all);
  // Positions are per project: reordering needs one project (always true under /p/:pid).
  const reorderable = pid !== null && visible.length > 1;
  const names = new Map((projects.data ?? []).map((p) => [p.id, p.name]));

  const move = async (from: number, to: number) => {
    const patches = reorderPatches(all, visible.map((f) => f.id), from, to);
    if (!patches.length) return;
    mutate<Finding[]>(key, (prev) => applyPositions(prev ?? [], patches));
    try {
      for (const p of patches) await patchFinding(p.id, { position: p.position });
    } catch (e) {
      toast("error", "Could not save the new order", { body: errorMessage(e) });
      void res.reload();
    }
  };
  const patched = (f: Finding) => mutate<Finding[]>(key, (prev) => (prev ?? []).map((x) => (x.id === f.id ? f : x)));
  const remove = async (f: Finding) => {
    setDeleting(true);
    try {
      await deleteFinding(f.id);
      mutate<Finding[]>(key, (prev) => (prev ?? []).filter((x) => x.id !== f.id));
    } catch (e) {
      toast("error", "Could not delete the finding", { body: errorMessage(e) });
    } finally {
      setDeleting(false);
      setDel(null);
    }
  };

  return (
    <section className="sx-page" aria-labelledby="findings-title">
      <header className="page-head">
        <h1 id="findings-title">Findings</h1>
        <p className="sx-intro">
          Views pinned with “Pin to Findings”: each keeps the numbers that were on screen, so an export reproduces exactly what you saw. Add notes, put them in order, and open any
          of them where it was pinned.
        </p>
      </header>
      <Card>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="row">
            {routePid === null ? (
              <>
                <label htmlFor="fd-project">Project</label>
                <select
                  id="fd-project"
                  value={projectParam ?? ""}
                  onChange={(e) => {
                    setProjectParam(e.target.value || null);
                    setRunFilter(null);
                  }}
                >
                  <option value="">All projects</option>
                  {(projects.data ?? []).map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </>
            ) : null}
            <label htmlFor="fd-run">Run</label>
            <select id="fd-run" value={runFilter ?? ""} onChange={(e) => setRunFilter(e.target.value || null)}>
              <option value="">All runs</option>
              {runs.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
            <span className="cap">
              {visible.length} of {all.length}
            </span>
          </div>
          {pid ? <ExportBar pid={pid} runFilter={runFilter} visible={visible} /> : <span className="cap">Pick a project to reorder or export.</span>}
        </div>
      </Card>
      {res.error && !res.data ? <EmptyState error={res.error} /> : null}
      {res.data && !all.length ? (
        <EmptyState title="No findings yet" body="Use the Pin button on any chart to keep it here with the numbers it shows." />
      ) : null}
      {res.data && all.length && !visible.length ? <p className="cap">No findings for this run.</p> : null}
      {visible.length ? (
        <ol className="fd-list" aria-label="Findings">
          {visible.map((f, i) => (
            <FindingItem
              key={f.id}
              f={f}
              index={i}
              count={visible.length}
              reorderable={reorderable}
              projectName={routePid === null ? names.get(f.project_id) ?? f.project_id : null}
              onMove={(a, b) => void move(a, b)}
              onPatched={patched}
              onDelete={setDel}
              drag={drag}
              setDrag={setDrag}
            />
          ))}
        </ol>
      ) : null}
      <ConfirmDialog open={!!del} onClose={() => setDel(null)} onConfirm={() => del && void remove(del)} title="Delete this finding?" confirmLabel="Delete" danger busy={deleting}>
        <p>“{del?.title}” and its image are removed from the notebook. Exports made earlier keep their copy.</p>
      </ConfirmDialog>
    </section>
  );
}
