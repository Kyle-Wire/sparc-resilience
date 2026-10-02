// Files (SPEC §6.4 Files): the run directory as a tree with size, mtime, catalog id, state
// badge and producer; raw downloads (Range supported by the server) and conversions
// (parquet → CSV with lon/lat, JSON → CSV for tabular sections, md → HTML); previews of
// parquet/CSV (first 200 rows), markdown, JSON and text; the data dictionary; and the
// checkpoint card with a guarded delete.
import { useEffect, useMemo, useState } from "react";
import { errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import type { CheckpointInfo, Conversion, DictionaryRow, FileEntry, FileText } from "../../api/runs";
import { deleteCheckpoint, fetchFileText, fileRawUrl, useFileTable, useGridMeta, useRunDetailFull, useRunDictionary, useRunFiles } from "../../api/runs";
import type { OutputEntry } from "../../api/types";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { Markdown } from "../../components/ui/Markdown";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table } from "../../components/ui/Table";
import { runIsLive, useRunOutputs } from "../../layouts/resources";
import { codecs, useUrlState } from "../../router";
import { toast } from "../../stores/ui";
import { fmtBytes, fmtDateTime, fmtDuration } from "../../theme/format";
import { Block, LiveBanner, useRid } from "./common";
import { OUTPUT_STATE_TEXT, cellText, outputStateStatus, producedByText } from "./format";

const ext = (p: string) => (/\.([a-z0-9]+)$/i.exec(p)?.[1] ?? "").toLowerCase();

/** Conversions offered for a file (api.md §6.1 `files/raw?as=`). */
export function conversionsFor(path: string, formats: string[] = []): Conversion[] {
  const e = ext(path);
  const out = new Set<Conversion>();
  if (e === "parquet") {
    out.add("csv");
    if (formats.includes("geojson")) out.add("geojson");
  }
  if (e === "json") out.add("csv");
  if (e === "md") out.add("html");
  if (e === "csv" && formats.includes("geojson")) out.add("geojson");
  for (const f of formats) if ((f === "csv" || f === "geojson" || f === "html" || f === "json") && f !== e) out.add(f);
  return [...out];
}

const CONV_LABEL: Record<Conversion, string> = { csv: "CSV", json: "JSON", html: "HTML", geojson: "GeoJSON" };

type Ctx = { rid: string; byOutput: Map<string, OutputEntry>; selected: string | null; select: (p: string) => void };

function FileRow({ e, depth, ctx }: { e: FileEntry; depth: number; ctx: Ctx }) {
  const [open, setOpen] = useState(false);
  const out = e.output_id ? ctx.byOutput.get(e.output_id) : undefined;
  return (
    <>
      <tr className={ctx.selected === e.relpath ? "hl" : undefined} data-path={e.relpath}>
        <td style={{ paddingLeft: 8 + depth * 16 }}>
          {e.dir ? (
            <button type="button" className="btn small ghost" aria-expanded={open} onClick={() => setOpen(!open)}>
              <Icon name={open ? "chevronDown" : "chevronRight"} /> <Icon name="folder" /> {e.name}
            </button>
          ) : (
            <button type="button" className="btn small ghost" aria-pressed={ctx.selected === e.relpath} onClick={() => ctx.select(e.relpath)}>
              <Icon name="file" /> {e.name}
            </button>
          )}
        </td>
        <td className="r num">{e.dir ? "" : fmtBytes(e.bytes)}</td>
        <td className="num cap">{fmtDateTime(e.mtime)}</td>
        <td className="mono cap">{e.output_id ?? ""}</td>
        <td>{e.state ? <StatusChip status={outputStateStatus(e.state)} text={OUTPUT_STATE_TEXT[e.state] ?? e.state} /> : null}</td>
        <td className="cap">
          {out ? producedByText(out.produced_by) : ""}
          {e.in_manifest ? (
            <>
              {" "}
              <Badge title="Referenced by manifest.json">manifest</Badge>
            </>
          ) : null}
        </td>
      </tr>
      {e.dir && open ? <DirRows path={e.relpath} depth={depth + 1} ctx={ctx} /> : null}
    </>
  );
}

function DirRows({ path, depth, ctx }: { path: string | null; depth: number; ctx: Ctx }) {
  const files = useRunFiles(ctx.rid, path);
  if (files.error)
    return (
      <tr>
        <td colSpan={6}>
          <EmptyState error={files.error} />
        </td>
      </tr>
    );
  if (!files.data)
    return (
      <tr>
        <td colSpan={6} className="cap" style={{ paddingLeft: 8 + depth * 16 }}>
          Loading…
        </td>
      </tr>
    );
  const sorted = [...files.data].sort((a, b) => Number(b.dir) - Number(a.dir) || a.name.localeCompare(b.name));
  return (
    <>
      {sorted.map((e) => (
        <FileRow key={e.relpath} e={e} depth={depth} ctx={ctx} />
      ))}
    </>
  );
}

function TablePreview({ rid, path }: { rid: string; path: string }) {
  const t = useFileTable(rid, path, 200);
  if (t.error) return <EmptyState error={t.error} />;
  if (!t.data) return <p className="cap">Loading preview…</p>;
  const cols = t.data.columns;
  type Row = { i: number; cells: unknown[] };
  return (
    <div className="stack" style={{ gap: 6 }}>
      <p className="cap">
        First {Math.min(t.data.rows.length, 200).toLocaleString("en-US")} of {t.data.n_rows.toLocaleString("en-US")} rows · {cols.length} columns
      </p>
      <Table<Row>
        caption={`Preview of ${path}`}
        csvName={null}
        rowKey={(r) => r.i}
        maxHeight={420}
        columns={cols.map((c, j) => ({
          key: `c${j}`,
          label: c.name,
          align: /int|float|double|decimal/.test(c.dtype) ? "right" : "left",
          value: (r: Row) => r.cells[j] as string | number | null,
          render: (r: Row) => cellText(r.cells[j], 4),
        }))}
        rows={t.data.rows.map((cells, i) => ({ i, cells }))}
      />
    </div>
  );
}

/** Bytes of a text file a preview reads (the rest stays on disk; Download has it all). */
export const PREVIEW_BYTES = 256 * 1024;

function TextPreview({ rid, path }: { rid: string; path: string }) {
  const [state, setState] = useState<{ file: FileText | null; error: unknown }>({ file: null, error: null });
  useEffect(() => {
    const ctrl = new AbortController();
    setState({ file: null, error: null });
    fetchFileText(rid, path, ctrl.signal, PREVIEW_BYTES).then(
      (file) => !ctrl.signal.aborted && setState({ file, error: null }),
      (error: unknown) => !ctrl.signal.aborted && setState({ file: null, error }),
    );
    return () => ctrl.abort();
  }, [rid, path]);
  if (state.error) return <EmptyState error={state.error} />;
  if (state.file === null) return <p className="cap">Loading preview…</p>;
  const { text, truncated, bytes } = state.file;
  const note = truncated ? (
    <p className="cap" data-truncated="true">
      Showing the first {fmtBytes(PREVIEW_BYTES)}
      {bytes !== null ? ` of ${fmtBytes(bytes)}` : ""}; download the file for the rest.
    </p>
  ) : null;
  if (ext(path) === "md")
    return (
      <div className="stack" style={{ gap: 6 }}>
        {note}
        <Markdown source={text} />
      </div>
    );
  let shown = text;
  if (ext(path) === "json" && !truncated) {
    try {
      shown = JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      /* keep the raw text */
    }
  }
  return (
    <div className="stack" style={{ gap: 6 }}>
      {note}
      <pre className="mono" style={{ maxHeight: 480, overflow: "auto", background: "var(--chip)", padding: 10, borderRadius: 8, fontSize: "0.8rem" }}>
        {shown}
      </pre>
    </div>
  );
}

/** Why a conversion is unavailable for this run, or null (SPEC §6.7: no CRS, no GeoJSON). */
export function conversionBlocked(c: Conversion, noCrs: boolean): string | null {
  return c === "geojson" && noCrs ? "This run has no CRS, so GeoJSON (lon/lat) is unavailable" : null;
}

function Preview({ rid, path, entry, noCrs }: { rid: string; path: string; entry: OutputEntry | undefined; noCrs: boolean }) {
  const e = ext(path);
  const convs = conversionsFor(path, entry?.formats ?? []);
  return (
    <Block
      title={path}
      actions={
        <>
          <a className="btn small" href={fileRawUrl(rid, path)} download>
            <Icon name="download" /> Download
          </a>
          {convs.map((c) => {
            const blocked = conversionBlocked(c, noCrs);
            return blocked ? (
              <span key={c} className="btn small ghost" aria-disabled="true" title={blocked}>
                {CONV_LABEL[c]}
              </span>
            ) : (
              <a
                key={c}
                className="btn small ghost"
                href={fileRawUrl(rid, path, c)}
                download
                title={c === "csv" && e === "parquet" ? (noCrs ? "CSV with the id column (no CRS, so no lon/lat)" : "CSV with id and lon/lat columns") : undefined}
              >
                {CONV_LABEL[c]}
              </a>
            );
          })}
        </>
      }
    >
      {e === "parquet" || e === "csv" ? (
        <TablePreview rid={rid} path={path} />
      ) : e === "md" || e === "json" || e === "txt" || e === "jsonl" || e === "yml" || e === "yaml" ? (
        <TextPreview rid={rid} path={path} />
      ) : (
        <p className="cap">No preview for .{e || "?"} files; download it instead.</p>
      )}
    </Block>
  );
}

function Dictionary({ rid }: { rid: string }) {
  const d = useRunDictionary(rid);
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (d.data ?? []).filter((r) => !s || `${r.output} ${r.column} ${r.description}`.toLowerCase().includes(s));
  }, [d.data, q]);
  if (d.error) return <EmptyState error={d.error} />;
  if (!d.data) return <p className="cap">Loading the data dictionary…</p>;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <input type="search" placeholder="Filter columns…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter the data dictionary" />
      <Table<DictionaryRow>
        caption="Data dictionary"
        csvName="data-dictionary"
        rowKey={(r) => `${r.output}:${r.column}`}
        maxHeight={420}
        columns={[
          { key: "output", label: "File", value: (r) => r.output },
          { key: "column", label: "Column", value: (r) => r.column, render: (r) => <span className="mono">{r.column}</span> },
          { key: "unit", label: "Unit", value: (r) => r.unit },
          { key: "sign", label: "Sign", value: (r) => r.sign ?? "" },
          { key: "desc", label: "Meaning", value: (r) => r.description },
        ]}
        rows={rows}
      />
    </div>
  );
}

export function CheckpointCard({ rid, cp }: { rid: string; cp: CheckpointInfo }) {
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const del = async () => {
    setBusy(true);
    try {
      const r = await deleteCheckpoint(rid);
      toast("success", `Checkpoint deleted, ${fmtBytes(r.freed_bytes)} freed`);
      setConfirm(false);
      invalidate(`run:${rid}`);
    } catch (e) {
      toast("error", "Could not delete the checkpoint", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const snap = cp.matches_snapshot;
  // a field is null when the checkpoint sidecar cannot tell (an older checkpoint)
  const mismatched = snap ? (["data", "code", "config"] as const).filter((k) => snap[k] === false) : [];
  const unknown = snap ? (["data", "code", "config"] as const).filter((k) => snap[k] === null) : [];
  return (
    <Block
      title="Checkpoint"
      id="checkpoint"
      actions={
        cp.present ? (
          <Button variant="danger" size="small" icon="x" onClick={() => setConfirm(true)}>
            Delete checkpoint
          </Button>
        ) : undefined
      }
    >
      {cp.present ? (
        <dl className="kv" aria-label="Checkpoint">
          <dt>Size</dt>
          <dd>{fmtBytes(cp.bytes)}</dd>
          <dt>Saved</dt>
          <dd>{fmtDateTime(cp.saved_utc)}</dd>
          <dt>Stages done</dt>
          <dd>{cp.done.join(", ") || "—"}</dd>
          <dt>Matches launch snapshot</dt>
          <dd>{snap ? (mismatched.length ? `no: ${mismatched.join(", ")} changed` : unknown.length ? `cannot tell for ${unknown.join(", ")}` : "yes") : "—"}</dd>
          {cp.changed_sections.length ? (
            <>
              <dt>Changed sections</dt>
              <dd>{cp.changed_sections.join(", ")}</dd>
            </>
          ) : null}
          <dt>Resume</dt>
          <dd>{cp.resumable ? `reuses ${cp.reuses.join(" + ") || "nothing"}${cp.saves_s ? `, saves ≈${fmtDuration(cp.saves_s)}` : ""}` : (cp.reason ?? "not resumable")}</dd>
        </dl>
      ) : (
        <p className="cap">No checkpoint: {cp.reason ?? "this run has none"}. Exact scenarios and the emulator need one.</p>
      )}
      <ConfirmDialog
        open={confirm}
        onClose={() => setConfirm(false)}
        onConfirm={() => void del()}
        title="Delete the checkpoint?"
        confirmLabel={`Delete ${fmtBytes(cp.bytes)}`}
        danger
        busy={busy}
      >
        <p>
          The checkpoint holds the fitted models. Without it, exact scenarios and the emulator build will need a refit, and an interrupted run can no longer resume. Outputs already
          written stay.
        </p>
      </ConfirmDialog>
    </Block>
  );
}

export default function Files() {
  const rid = useRid();
  const outputs = useRunOutputs(rid);
  // Unknown until the grid is readable (mid-run before S0): keep conversions offered then.
  const gridMeta = useGridMeta(rid);
  const noCrs = gridMeta.data ? !gridMeta.data.crs : false;
  const detail = useRunDetailFull(rid);
  const [selected, setSelected] = useUrlState("f", codecs.optString());
  const byOutput = useMemo(() => new Map((outputs.data?.outputs ?? []).map((o) => [o.id, o])), [outputs.data]);
  const byFile = useMemo(() => {
    const m = new Map<string, OutputEntry>();
    for (const o of outputs.data?.outputs ?? []) for (const f of o.files) m.set(f.relpath, o);
    return m;
  }, [outputs.data]);
  const ctx: Ctx = { rid, byOutput, selected, select: (p) => setSelected(p) };
  return (
    <section className="stack" aria-labelledby="files-title">
      <h2 id="files-title" style={{ margin: 0 }}>
        Files
      </h2>
      {runIsLive(detail.data?.run.status) ? <LiveBanner>This run is still running: files marked "being written" are not final yet.</LiveBanner> : null}
      <Block title="Run directory">
        <div className="tablewrap" style={{ maxHeight: 520, overflowY: "auto" }}>
          <table className="tbl" aria-label="Run files">
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col" className="r">
                  Size
                </th>
                <th scope="col">Modified</th>
                <th scope="col">Catalog id</th>
                <th scope="col">State</th>
                <th scope="col">Produced by</th>
              </tr>
            </thead>
            <tbody>
              <DirRows path={null} depth={0} ctx={ctx} />
            </tbody>
          </table>
        </div>
      </Block>
      {selected ? (
        <Preview rid={rid} path={selected} entry={byFile.get(selected)} noCrs={noCrs} />
      ) : (
        <p className="cap">Select a file to preview it, download it or convert it.</p>
      )}
      <Block title="Data dictionary">
        <Dictionary rid={rid} />
      </Block>
      {detail.data?.checkpoint ? <CheckpointCard rid={rid} cp={detail.data.checkpoint} /> : null}
    </section>
  );
}
