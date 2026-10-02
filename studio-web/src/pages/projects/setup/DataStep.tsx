// Setup step "data" (SPEC §9.3, J2): upload the point table (streamed PUT with progress),
// its header preview, the suggested mapping (target, id, x/y, zone, coord unit, CRS with an
// EPSG search and a centroid inset), units, background, subsample/coarse, join tables, and
// "Check data": S0 inline with QA flags as badges, grid stats and a preview raster of any
// column from the preview binaries.
import { useMemo, useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { isNumericDtype, uploadFile, useFileInspect, useProjectFiles, type FileInspect, type FileKind, type UploadResult } from "../../../api/projects";
import { invalidate } from "../../../api/resource";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { ConfirmDialog } from "../../../components/ui/Dialog";
import { EmptyState } from "../../../components/ui/EmptyState";
import { FileDrop } from "../../../components/ui/FileDrop";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { NumberField } from "../../../components/ui/NumberField";
import { Pill } from "../../../components/ui/Pill";
import { ProgressBar } from "../../../components/ui/ProgressBar";
import { Seg } from "../../../components/ui/Seg";
import { Table, type Column } from "../../../components/ui/Table";
import { MapView } from "../../../map/MapView";
import { toast } from "../../../stores/ui";
import { fmtBytes, fmtBytesOf, fmtCount, fmtInt, fmtNum, fmtPct, fmtSig, fmtValue } from "../../../theme/format";
import { CfgNumber, CfgSelect, CfgText, IssueLines, useCfg, type Opt } from "../components/cfg";
import { EpsgField } from "../components/EpsgField";
import { StepIssues } from "../components/StepIssues";
import { getList, getString } from "../model/dotted";
import { stepIssues } from "../model/steps";
import { suggestedValue, suggestionConfidence } from "../model/suggest";
import { previewGroups, previewLoader, runDataCheck, suggestMapping, usePreviewGrid } from "./actions";
import { rawKey } from "./store";

const COORD_UNITS: Opt[] = [
  { value: "m", label: "metres" },
  { value: "us_survey_foot", label: "US survey feet" },
  { value: "ft", label: "international feet" },
];

const TARGET_UNITS: Opt[] = [
  { value: "degF", label: "°F" },
  { value: "degC", label: "°C" },
  { value: "K", label: "K" },
];

/** Upload one file into the project with progress; a name clash asks before replacing. */
export function useUpload(pid: string, kind: FileKind, onDone: (r: UploadResult) => void) {
  const [progress, setProgress] = useState<{ name: string; sent: number; total: number; error?: string; done?: boolean } | null>(null);
  const [clash, setClash] = useState<File | null>(null);
  const start = async (file: File, overwrite = false) => {
    setProgress({ name: file.name, sent: 0, total: file.size });
    try {
      const r = await uploadFile(pid, kind, file, {
        overwrite,
        onProgress: (sent, total) => setProgress((p) => (p ? { ...p, sent, total: total || p.total } : p)),
      });
      setProgress((p) => (p ? { ...p, sent: p.total, done: true } : p));
      invalidate(`project:${pid}:files`);
      onDone(r);
    } catch (e) {
      if (e instanceof ApiError && e.code === "exists") {
        setProgress(null);
        setClash(file);
        return;
      }
      setProgress((p) => (p ? { ...p, error: errorMessage(e) } : p));
    }
  };
  const view = (
    <>
      {progress ? (
        <div className="stack" style={{ gap: 4 }}>
          <div className="row cap">
            <span className="mono">{progress.name}</span>
            <span className="spacer" />
            <span>{progress.error ? `Failed: ${progress.error}` : progress.done ? `Uploaded ${fmtBytes(progress.total)}` : fmtBytesOf(progress.sent, progress.total)}</span>
          </div>
          <ProgressBar value={progress.total ? progress.sent / progress.total : null} label={`Uploading ${progress.name}`} tone={progress.error ? "crit" : progress.done ? "good" : undefined} />
        </div>
      ) : null}
      <ConfirmDialog
        open={!!clash}
        title="Replace the existing file?"
        confirmLabel="Replace"
        danger
        onClose={() => setClash(null)}
        onConfirm={() => {
          const f = clash;
          setClash(null);
          if (f) void start(f, true);
        }}
      >
        <p>
          The project already has <span className="mono">{clash?.name}</span>. Replacing it changes the data of future runs; existing runs keep their launch snapshot.
        </p>
      </ConfirmDialog>
    </>
  );
  return { start, view };
}

function InspectTable({ inspect }: { inspect: FileInspect }) {
  const cols: Column<FileInspect["columns"][number]>[] = [
    { key: "name", label: "Column", render: (r) => <span className="mono">{r.name}</span> },
    { key: "dtype", label: "Type" },
    { key: "n_null", label: "Nulls", align: "right", value: (r) => r.n_null, render: (r) => fmtInt(r.n_null) },
    { key: "min", label: "Min", align: "right", value: (r) => r.min, render: (r) => fmtSig(r.min, 5) },
    { key: "max", label: "Max", align: "right", value: (r) => r.max, render: (r) => fmtSig(r.max, 5) },
    { key: "n_unique", label: "Unique", align: "right", value: (r) => r.n_unique, render: (r) => fmtInt(r.n_unique) },
    {
      key: "sample",
      label: "Sample",
      sortable: false,
      value: (r) => r.sample.slice(0, 3).map(String).join(", "),
      render: (r) => <span className="cap mono">{r.sample.slice(0, 3).map((v) => (v === null ? "∅" : String(v))).join(", ")}</span>,
    },
  ];
  return (
    <div className="stack" style={{ gap: 6 }}>
      <span className="cap">
        {inspect.n_rows_exact ? fmtCount(inspect.n_rows, "row") : `≈${fmtCount(inspect.n_rows, "row")} (estimated)`} · {fmtCount(inspect.columns.length, "column")}
      </span>
      <Table columns={cols} rows={inspect.columns} rowKey={(r) => r.name} caption="Header preview" csvName="header-preview" />
    </div>
  );
}

function CheckResults({ pid }: { pid: string }) {
  const c = useCfg();
  const d = c.draft;
  const check = d.check;
  const inspect = useFileInspect(pid, getString(d.raw, "data.path") ?? getString(d.effective, "data.path"));
  const token = check?.preview_token || null;
  const grid = usePreviewGrid(pid, token);
  const groups = useMemo(() => (check ? previewGroups(check, d.raw, inspect.data) : []), [check, d.raw, inspect.data]);
  const load = useMemo(() => (token ? previewLoader(pid, token) : null), [pid, token]);
  const [layer, setLayer] = useState<string | null>(null);
  if (d.checkError) return <EmptyState error={d.checkError} title="The data check failed" />;
  if (!check) return <p className="cap">Run the check to see QA flags, the grid and a preview of every column.</p>;
  const stale = d.checkKey !== rawKey(d.raw);
  const units = getString(d.raw, "data.target_units") ?? getString(d.effective, "data.target_units") ?? "degF";
  const clipped = Object.entries(check.clipped).filter(([, n]) => n > 0);
  return (
    <div className="stack check-results">
      {stale ? (
        <div className="callout" data-tone="info" role="status">
          The settings changed since this check. Run it again to refresh.
        </div>
      ) : null}
      <KpiRow label="Data check">
        <Kpi label="Points kept" value={fmtInt(check.n_points)} note={`of ${fmtInt(check.n_input)} rows · ${fmtInt(check.n_dropped)} dropped`} tone={check.n_dropped > 0 ? "warn" : undefined} />
        <Kpi label="Grid" value={`${fmtInt(check.grid.nx)} × ${fmtInt(check.grid.ny)}`} note={`cell ${fmtNum(check.grid.cell_m, 1)} m · ${fmtNum(check.extent_m[0] / 1000, 1)} × ${fmtNum(check.extent_m[1] / 1000, 1)} km`} />
        <Kpi label="Fill fraction" value={fmtPct(check.grid.fill_fraction, 1)} note={`${fmtNum(check.grid.collisions, 2)} collisions per cell`} />
        <Kpi label="Background" value={fmtValue(check.background.value, units, 2)} note={check.background.source || undefined} />
        <Kpi label="Noise floor" value={check.noise_floor === null ? "—" : fmtValue(check.noise_floor, units, 2)} note={check.noise_floor === null ? "target not rounded" : "rounding noise of the target"} />
      </KpiRow>
      {check.flags.length ? (
        <div className="row" aria-label="QA flags">
          {check.flags.map((f) => (
            <Pill key={f.code} tone={f.severity === "warn" ? "warn" : "accent"} icon={f.severity === "warn" ? "alert" : "info"} title={f.code}>
              {f.message}
            </Pill>
          ))}
        </div>
      ) : (
        <Pill tone="good" icon="check">
          No QA flags
        </Pill>
      )}
      {clipped.length ? (
        <p className="cap">
          Clipped by qa.clip: {clipped.map(([k, n]) => `${k} ${fmtInt(n)}`).join(" · ")}
        </p>
      ) : null}
      {check.columns_missing.length ? (
        <div className="callout" data-tone="crit" role="alert">
          Columns named in the config but missing from the data: <span className="mono">{check.columns_missing.join(", ")}</span>
        </div>
      ) : null}
      {check.coarse ? <p className="cap">Coarse preview: {Object.entries(check.coarse).map(([k, v]) => `${k} ${String(v)}`).join(" · ")}</p> : null}
      {grid.error ? (
        <EmptyState error={grid.error} title="Preview grid unavailable" body="Preview binaries expire after 30 minutes: run the check again." />
      ) : grid.data && load && groups.length ? (
        <MapView grid={grid.data} groups={groups} loadLayer={load} layerKey={layer} onLayerChange={setLayer} height={360} title="Data preview" tools={["pan"]} />
      ) : (
        <p className="cap">
          <span className="spinner" aria-hidden="true" /> Loading the preview grid…
        </p>
      )}
    </div>
  );
}

function JoinsEditor({ pid, columns }: { pid: string; columns: Opt[] }) {
  const c = useCfg();
  const joins = getList<Record<string, unknown>>(c.draft.raw, "data.join");
  const files = useProjectFiles(pid);
  const fileOpts: Opt[] = (files.data ?? []).map((f) => ({ value: f.path, label: `${f.path} (${f.kind})` }));
  const up = useUpload(pid, "join", (r) => c.set(["data", "join", joins.length], { path: r.path }));
  return (
    <div className="stack" style={{ gap: 8 }}>
      {joins.length === 0 ? <p className="cap">No join tables. Extra predictor tables (e.g. open-data features) are merged by id.</p> : null}
      {joins.map((_, i) => (
        <div key={i} className="join-row">
          <CfgSelect path={`data.join.${i}.path`} label={`Join ${i + 1}: table`} options={fileOpts} missingNote="not among the project files" />
          <CfgSelect path={`data.join.${i}.key`} label="Data id column" options={columns} none="data.id (default)" />
          <CfgText path={`data.join.${i}.right_key`} label="Table id column" placeholder="id" />
          <Button size="small" variant="ghost" icon="x" onClick={() => c.set("data.join", joins.filter((__, k) => k !== i))}>
            Remove
          </Button>
        </div>
      ))}
      <div className="row">
        <Button size="small" icon="plus" onClick={() => c.set(["data", "join", joins.length], { path: fileOpts[0]?.value ?? "" })}>
          Add a join
        </Button>
      </div>
      <FileDrop label="Upload a join table" hint="CSV or parquet with an id column." accept=".csv,.parquet" onFiles={(f) => f[0] && void up.start(f[0])} />
      {up.view}
      <IssueLines issues={c.issues("data.join")} />
    </div>
  );
}

function Background({ columns }: { columns: Opt[] }) {
  const c = useCfg();
  const v = c.value("data.background");
  const mode: "median" | "number" | "column" = typeof v === "number" ? "number" : v === "median" || v === undefined || v === null ? "median" : "column";
  return (
    <div data-cfg-path="data.background" className="field">
      <span className="field-label">Background (ΔT reference)</span>
      <Seg
        label="Background"
        size="small"
        value={mode}
        options={[
          { value: "median", label: "Field median" },
          { value: "number", label: "A number" },
          { value: "column", label: "A column" },
        ]}
        onChange={(m) =>
          c.set("data.background", m === "median" ? "median" : m === "number" ? Number((c.draft.check?.background.value ?? 0).toFixed(1)) : columns[0]?.value ?? "median")
        }
      />
      {mode === "number" ? <NumberField label="Background value" value={typeof v === "number" ? v : null} onChange={(n) => c.set("data.background", n ?? "median")} /> : null}
      {mode === "column" ? (
        <select aria-label="Background column" value={String(v)} onChange={(e) => c.set("data.background", e.target.value)}>
          {columns.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : null}
      <span className="hint">Every ΔT is relative to this value.</span>
      <IssueLines issues={c.issues("data.background")} />
    </div>
  );
}

export function DataStep() {
  const c = useCfg();
  const pid = c.pid;
  const d = c.draft;
  const files = useProjectFiles(pid);
  const dataPath = (c.value("data.path") as string | null) ?? null;
  const inspect = useFileInspect(pid, dataPath);
  const grid = usePreviewGrid(pid, d.check?.preview_token || null);
  const up = useUpload(pid, "data", (r) => {
    c.set("data.path", r.path);
    toast("success", `Uploaded ${r.path}`, { body: r.inspect ? `${fmtCount(r.inspect.n_rows, "row")}, ${fmtCount(r.inspect.columns.length, "column")}` : undefined });
    void suggestMapping(pid, r.path);
  });
  const allCols: Opt[] = (inspect.data?.columns ?? []).map((col) => ({ value: col.name, label: `${col.name}${isNumericDtype(col.dtype) ? "" : ` (${col.dtype})`}` }));
  const numCols: Opt[] = (inspect.data?.columns ?? []).filter((col) => isNumericDtype(col.dtype)).map((col) => ({ value: col.name, label: col.name }));
  const dataFiles: Opt[] = (files.data ?? []).filter((f) => f.kind === "data").map((f) => ({ value: f.path, label: `${f.path} · ${fmtBytes(f.bytes)}` }));
  const s = d.suggestion;
  const bounds = grid.data?.meta.bounds_lonlat;
  const centroid = bounds ? { lon: (bounds[0] + bounds[2]) / 2, lat: (bounds[1] + bounds[3]) / 2 } : null;
  const checking = d.checking;
  const conf = (p: string) => suggestionConfidence(s, p);
  return (
    <div className="stack setup-step" data-step="data">
      <StepIssues issues={stepIssues("data", d.issues)} />
      <Card title="Point table" eyebrow="1 · Upload">
        <FileDrop label="Drop a CSV or parquet of street-level temperatures" hint="Streamed to the project's data folder (CSV or parquet, up to the upload limit in Settings)." accept=".csv,.parquet" onFiles={(f) => f[0] && void up.start(f[0])} />
        {up.view}
        <div className="grid2">
          <CfgSelect path="data.path" label="Data file" options={dataFiles} missingNote="not among the project files" hint="The point table the runs read (relative to the project folder)." />
          <div className="field">
            <span className="field-label">Mapping</span>
            <div className="row">
              <Button size="small" icon="bolt" disabled={!dataPath} onClick={() => dataPath && void suggestMapping(pid, dataPath)}>
                Suggest mapping
              </Button>
              <Button size="small" variant="ghost" disabled={!dataPath} onClick={() => dataPath && void suggestMapping(pid, dataPath, { overwrite: true })}>
                Apply every suggestion
              </Button>
            </div>
            <span className="hint">Empty fields are pre-filled from the column names and values; "Apply every suggestion" replaces fields already set.</span>
          </div>
        </div>
      </Card>

      <Card title="Header preview" eyebrow="2 · Columns">
        {!dataPath ? (
          <p className="cap">Upload or choose a data file first.</p>
        ) : inspect.error ? (
          <EmptyState error={inspect.error} title="Could not read the file" />
        ) : inspect.data ? (
          <InspectTable inspect={inspect.data} />
        ) : (
          <p className="cap" role="status">
            <span className="spinner" aria-hidden="true" /> Reading the header…
          </p>
        )}
      </Card>

      <Card title="Column mapping" eyebrow="3 · Mapping" className="mapping-form">
        <div className="grid3">
          <CfgSelect path="data.target" label="Temperature (target)" options={numCols} confidence={conf("data.target")} hint={suggestHint(s, "data.target", c.value("data.target"))} />
          <CfgSelect path="data.target_units" label="Target units" options={TARGET_UNITS} />
          <CfgSelect path="data.id" label="Id column" options={allCols} none="none (rows are numbered)" confidence={conf("data.id")} hint={suggestHint(s, "data.id", c.value("data.id"))} />
          <CfgSelect path="data.x" label="X coordinate" options={numCols} confidence={conf("data.x")} hint={suggestHint(s, "data.x", c.value("data.x"))} />
          <CfgSelect path="data.y" label="Y coordinate" options={numCols} confidence={conf("data.y")} hint={suggestHint(s, "data.y", c.value("data.y"))} />
          <CfgSelect path="data.coord_unit" label="Coordinate unit" options={COORD_UNITS} confidence={conf("data.coord_unit")} hint={suggestHint(s, "data.coord_unit", c.value("data.coord_unit")) ?? "Feet vs metres follows from the lattice spacing."} />
          <CfgSelect path="data.zone" label="Zone column" options={allCols} none="none" confidence={conf("data.zone")} hint={suggestHint(s, "data.zone", c.value("data.zone")) ?? "Neighbourhood codes (reporting only)."} />
        </div>
        <EpsgField hint={s?.crs_guess ?? null} centroid={centroid} confidence={conf("data.crs")} />
      </Card>

      <Card title="Further settings" eyebrow="4 · Options">
        <div className="grid3">
          <Background columns={numCols} />
          <CfgNumber path="data.subsample" label="Subsample" hint="Keep only this many points around the centre (smoke tests)." step={1000} min={1} />
          <CfgNumber path="data.coarse_m" label="Coarse cell size" unit="m" hint="Aggregate onto cells of this size over the full extent." min={1} />
          <CfgNumber path="data.cell_m" label="Grid cell size" unit="m" hint="Default: inferred from the point lattice." min={0.1} />
          <CfgText path="data.reproject_to" label="Reproject to" placeholder="EPSG:32619" hint="Reproject x/y to this CRS (metres)." />
        </div>
      </Card>

      <Card title="Join tables" eyebrow="5 · Joins">
        <JoinsEditor pid={pid} columns={allCols} />
      </Card>

      <Card
        title="Check data"
        eyebrow="6 · S0 inline"
        actions={
          <Button variant="primary" icon="play" busy={checking} disabled={!dataPath} onClick={() => void runDataCheck(pid)}>
            Check data
          </Button>
        }
      >
        <CheckResults pid={pid} />
      </Card>
    </div>
  );
}

function suggestHint(s: Parameters<typeof suggestedValue>[0], path: string, current: unknown): string | undefined {
  const v = suggestedValue(s, path);
  if (v === null || v === current) return undefined;
  return `Suggested: ${v}`;
}
