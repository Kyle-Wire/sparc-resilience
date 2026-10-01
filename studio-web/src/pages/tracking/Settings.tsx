// `/settings` (SPEC §10.10): workspace, thread budget and slots, engine budget and idle timeout,
// job and app options, watch roots, the cache and storage managers with guarded deletes, the
// network check and about (versions, build).
import { useEffect, useState, type ReactNode } from "react";
import { ApiError, errorMessage } from "../../api/client";
import { invalidate, useResource } from "../../api/resource";
import {
  deleteCache,
  deleteRunData,
  getSettings,
  getStorage,
  getSystem,
  netcheck,
  putSettings,
  type NetcheckResult,
  type Settings as SettingsT,
  type StorageInfo,
} from "../../api/tracking";
import type { Issue } from "../../api/types";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { Dialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { Field, issuesAt } from "../../components/ui/Field";
import { NumberField } from "../../components/ui/NumberField";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { Link } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { fmtBytes, fmtDateTime, fmtNum } from "../../theme/format";
import "./tracking.css";

type NumKey = "thread_budget" | "threads_heavy" | "engine_threads" | "heavy_slots" | "medium_slots" | "network_slots" | "engine_max_runs" | "engine_mem_budget_gb" | "engine_idle_min" | "upload_max_gb";

/** Keys whose draft differs from the saved settings (the PUT body). */
export function settingsPatch(saved: SettingsT, draft: SettingsT): Partial<SettingsT> {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(draft) as (keyof SettingsT)[]) {
    if (JSON.stringify(draft[k]) !== JSON.stringify(saved[k])) out[k] = draft[k];
  }
  return out as Partial<SettingsT>;
}

/** Client-side checks mirrored from the server's validation (it stays authoritative). */
export function settingsIssues(d: SettingsT): Issue[] {
  const out: Issue[] = [];
  if (d.threads_heavy + d.engine_threads > d.thread_budget + 1) {
    out.push({ level: "error", path: "threads_heavy", code: "budget", message: `A heavy job (${d.threads_heavy}) plus the engine (${d.engine_threads}) would use more than the budget of ${d.thread_budget} threads (one over is allowed).` });
  }
  for (const k of ["thread_budget", "threads_heavy", "engine_threads", "heavy_slots", "medium_slots", "network_slots", "engine_max_runs"] as const) {
    if (!(d[k] >= 1)) out.push({ level: "error", path: k, code: "min", message: "Must be at least 1." });
  }
  if (!(d.engine_mem_budget_gb > 0)) out.push({ level: "error", path: "engine_mem_budget_gb", code: "min", message: "Must be above 0 GB." });
  return out;
}

function NumSetting({ k, label, hint, unit, draft, set, issues, step = 1 }: { k: NumKey; label: string; hint?: ReactNode; unit?: string; draft: SettingsT; set: (patch: Partial<SettingsT>) => void; issues: Issue[]; step?: number }) {
  return (
    <Field label={label} hint={hint} issues={issuesAt(issues, k)}>
      {(id, describedBy, invalid) => (
        <NumberField id={id} aria-describedby={describedBy} aria-invalid={invalid || undefined} value={draft[k]} min={k === "engine_mem_budget_gb" || k === "upload_max_gb" ? 0.1 : 1} step={step} unit={unit} onChange={(v) => v !== null && set({ [k]: v } as Partial<SettingsT>)} />
      )}
    </Field>
  );
}

function Toggle({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}

function SettingsForm() {
  const res = useResource<SettingsT>("settings", (s) => getSettings(s), { tags: ["settings"] });
  const sys = useResource("system", (s) => getSystem(s), { tags: ["storage"] });
  const [draft, setDraft] = useState<SettingsT | null>(null);
  const [serverIssues, setServerIssues] = useState<Issue[]>([]);
  const [saving, setSaving] = useState(false);
  const [root, setRoot] = useState("");
  const setUiNotifications = useUi((s) => s.setNotifications);
  useEffect(() => {
    if (res.data) setDraft(res.data);
  }, [res.data]);
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  if (!draft || !res.data) return <p className="cap">Loading settings…</p>;
  const saved = res.data;
  const set = (patch: Partial<SettingsT>) => {
    setDraft({ ...draft, ...patch });
    setServerIssues([]);
  };
  const local = settingsIssues(draft);
  const issues = [...local, ...serverIssues];
  const patch = settingsPatch(saved, draft);
  const dirty = Object.keys(patch).length > 0;
  const save = async () => {
    setSaving(true);
    setServerIssues([]);
    try {
      const next = await putSettings(patch);
      setDraft(next);
      await res.reload();
      invalidate("settings");
      if ("notifications" in patch) setUiNotifications(next.notifications);
      toast("success", "Settings saved");
    } catch (e) {
      if (e instanceof ApiError && e.validationErrors.length) {
        setServerIssues(e.validationErrors.map((v) => ({ level: "error", path: v.path.replace(/^body\./, ""), code: v.code, message: v.message })));
      }
      toast("error", "Settings not saved", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };
  const cpu = sys.data?.cpu_count;
  return (
    <div className="stack">
      <Card title="Threads and slots" eyebrow={cpu ? `${cpu} CPU threads on this machine` : undefined}>
        <div className="settings-grid">
          <NumSetting k="thread_budget" label="Thread budget" hint="All concurrent work shares it." draft={draft} set={set} issues={issues} />
          <NumSetting k="threads_heavy" label="Threads per heavy job" hint="Runs, studies, emulator builds." draft={draft} set={set} issues={issues} />
          <NumSetting k="engine_threads" label="Engine threads" hint="Reduced to 1 while a heavy job runs." draft={draft} set={set} issues={issues} />
          <NumSetting k="heavy_slots" label="Heavy slots" draft={draft} set={set} issues={issues} />
          <NumSetting k="medium_slots" label="Medium slots" hint="Post-run actions and exports." draft={draft} set={set} issues={issues} />
          <NumSetting k="network_slots" label="Network slots" hint="Input downloads." draft={draft} set={set} issues={issues} />
        </div>
      </Card>
      <Card title="Scenario engine">
        <div className="settings-grid">
          <NumSetting k="engine_max_runs" label="Runs kept loaded" draft={draft} set={set} issues={issues} />
          <NumSetting k="engine_mem_budget_gb" label="Memory budget" unit="GB" step={0.5} draft={draft} set={set} issues={issues} />
          <NumSetting k="engine_idle_min" label="Unload after idle" unit="min" draft={draft} set={set} issues={issues} />
        </div>
      </Card>
      <Card title="Jobs and app">
        <div className="settings-grid">
          <Field label="Delete finished job logs after" hint="Empty keeps them until you delete a job in Activity." issues={issuesAt(issues, "keep_job_logs_days")}>
            {(id, d) => <NumberField id={id} aria-describedby={d} value={draft.keep_job_logs_days} nullable min={1} step={1} unit="days" onChange={(v) => set({ keep_job_logs_days: v })} />}
          </Field>
          <NumSetting k="upload_max_gb" label="Largest upload" unit="GB" step={0.5} draft={draft} set={set} issues={issues} />
        </div>
        <div className="stack" style={{ gap: 6 }}>
          <Toggle checked={draft.auto_uncertainty} onChange={(v) => set({ auto_uncertainty: v })}>
            Rebuild uncertainty envelopes when an attached study finishes
          </Toggle>
          <Toggle checked={draft.offline} onChange={(v) => set({ offline: v })}>
            Offline: hide actions that need the network
          </Toggle>
          <Toggle checked={draft.notifications} onChange={(v) => set({ notifications: v })}>
            Browser notification when a job ends (asks for permission)
          </Toggle>
        </div>
      </Card>
      <Card title="Watch roots" eyebrow="folders scanned every 10 s for command-line runs">
        {draft.watch_roots.length ? (
          <ul className="roots">
            {draft.watch_roots.map((r) => (
              <li key={r}>
                <span>{r}</span>
                <Button size="small" variant="ghost" onClick={() => set({ watch_roots: draft.watch_roots.filter((x) => x !== r) })} aria-label={`Remove watch root ${r}`}>
                  Remove
                </Button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="cap">No watch roots. Command-line runs elsewhere can still be imported.</p>
        )}
        <div className="row">
          <input type="text" aria-label="Folder to watch" placeholder="/path/to/output/core" value={root} onChange={(e) => setRoot(e.target.value)} style={{ minWidth: "18em" }} />
          <Button
            size="small"
            disabled={!root.trim() || draft.watch_roots.includes(root.trim())}
            onClick={() => {
              set({ watch_roots: [...draft.watch_roots, root.trim()] });
              setRoot("");
            }}
          >
            Add folder
          </Button>
        </div>
        {issuesAt(issues, "watch_roots").map((i, k) => (
          <p key={k} className="cap" style={{ color: "var(--crit-ink)" }}>
            {i.message}
          </p>
        ))}
      </Card>
      <div className="row save-bar">
        <Button variant="primary" disabled={!dirty || local.some((i) => i.level === "error")} busy={saving} onClick={() => void save()}>
          Save settings
        </Button>
        <Button disabled={!dirty || saving} onClick={() => setDraft(saved)}>
          Discard changes
        </Button>
        {dirty ? <span className="cap">{Object.keys(patch).length} unsaved change{Object.keys(patch).length === 1 ? "" : "s"}</span> : null}
      </div>
    </div>
  );
}

type RunStore = StorageInfo["runs"][number];
type DeleteAsk = { kind: "cache"; name: string; bytes: number } | { kind: "run"; run: RunStore; what: "checkpoint" | "outputs" | "all" };

function StorageManager() {
  const res = useResource<StorageInfo>("storage", (s) => getStorage(s), { tags: ["storage", "runs"] });
  const [ask, setAsk] = useState<DeleteAsk | null>(null);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  const st = res.data;
  if (!st) return <p className="cap">Measuring the workspace…</p>;
  const runCols: Column<RunStore>[] = [
    { key: "run", label: "Run", value: (r) => r.label ?? r.run_id, render: (r) => <Link to={`/r/${encodeURIComponent(r.run_id)}`}>{r.label || r.run_id}</Link> },
    { key: "outputs", label: "Outputs", align: "right", value: (r) => r.outputs_bytes, render: (r) => fmtBytes(r.outputs_bytes) },
    { key: "ckpt", label: "Checkpoint", align: "right", value: (r) => r.checkpoint_bytes, render: (r) => fmtBytes(r.checkpoint_bytes) },
    {
      key: "actions",
      label: "Free space",
      sortable: false,
      render: (r) => (
        <span className="row" style={{ gap: 4 }}>
          <Button size="small" variant="ghost" disabled={!r.checkpoint_bytes} onClick={() => setAsk({ kind: "run", run: r, what: "checkpoint" })}>
            Checkpoint
          </Button>
          <Button size="small" variant="ghost" onClick={() => setAsk({ kind: "run", run: r, what: "outputs" })}>
            Outputs
          </Button>
          <Button size="small" variant="danger" onClick={() => setAsk({ kind: "run", run: r, what: "all" })}>
            Everything
          </Button>
        </span>
      ),
    },
  ];
  const needTyped = ask?.kind === "run" && ask.what === "all";
  const typedOk = !needTyped || (ask?.kind === "run" && typed.trim() === ask.run.run_id);
  const close = () => {
    setAsk(null);
    setTyped("");
  };
  const confirm = async () => {
    if (!ask || !typedOk) return;
    setBusy(true);
    try {
      const r = ask.kind === "cache" ? await deleteCache(ask.name) : await deleteRunData(ask.run.run_id, ask.what);
      toast("success", `Freed ${fmtBytes(r.freed_bytes)}`);
      invalidate("storage");
      if (ask.kind === "run") invalidate(`run:${ask.run.run_id}`);
      await res.reload();
      setAsk(null);
      setTyped("");
    } catch (e) {
      toast("error", "Nothing was deleted", { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="stack">
      <div className="kv-inline">
        <span>
          Workspace <b>{fmtBytes(st.workspace_bytes)}</b>
        </span>
        <span>
          Free disk <b>{fmtBytes(st.free_bytes)}</b>
        </span>
        <span>
          Job logs <b>{fmtBytes(st.jobs_bytes)}</b>
        </span>
        <span>
          Studies <b>{fmtBytes(st.studies.reduce((a, s) => a + s.bytes, 0))}</b> in {st.studies.length}
        </span>
      </div>
      <Card variant="flat" title="Download cache" eyebrow="re-fetched on demand">
        {st.cache.length ? (
          <table className="checklist" aria-label="Cache files">
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col" className="r">
                  Size
                </th>
                <th scope="col">Modified</th>
                <th scope="col">
                  <span className="sr-only">Delete</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {st.cache.map((c) => (
                <tr key={c.name}>
                  <td className="mono">{c.name}</td>
                  <td className="num r">{fmtBytes(c.bytes)}</td>
                  <td>{fmtDateTime(c.mtime)}</td>
                  <td>
                    <Button size="small" variant="ghost" onClick={() => setAsk({ kind: "cache", name: c.name, bytes: c.bytes })} aria-label={`Delete cache file ${c.name}`}>
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="cap">The cache is empty.</p>
        )}
      </Card>
      <Card variant="flat" title="Runs">
        <Table columns={runCols} rows={st.runs} rowKey={(r) => r.run_id} caption="Run storage" initialSort={{ key: "outputs", dir: "desc" }} empty="No runs stored." />
      </Card>
      <Dialog
        open={!!ask}
        onClose={close}
        busy={busy}
        title={ask?.kind === "cache" ? `Delete ${ask.name}?` : ask?.kind === "run" ? `Delete the ${ask.what === "all" ? "whole run" : ask.what} of ${ask.run.label || ask.run.run_id}?` : ""}
        footer={
          <>
            <Button onClick={close} disabled={busy}>
              Cancel
            </Button>
            <Button variant="danger" busy={busy} disabled={!typedOk} onClick={() => void confirm()}>
              Delete
            </Button>
          </>
        }
      >
        {ask?.kind === "cache" ? (
          <p className="prose">{fmtBytes(ask.bytes)} are freed. Studio downloads the file again the next time a job needs it.</p>
        ) : ask?.kind === "run" ? (
          <div className="stack">
            <p className="prose">
              {ask.what === "checkpoint"
                ? `The checkpoint (${fmtBytes(ask.run.checkpoint_bytes)}) is removed: the run can no longer be resumed, and the Scenario Lab cannot open it.`
                : ask.what === "outputs"
                  ? `The run's output files (${fmtBytes(ask.run.outputs_bytes)}) are removed; Studio's own records of the run are kept.`
                  : "The run directory, its outputs, checkpoint and Studio records are removed. This cannot be undone."}
            </p>
            {needTyped ? (
              <label className="field">
                <span className="field-label">Type the run id ({ask.run.run_id}) to confirm</span>
                <input type="text" value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" data-autofocus />
              </label>
            ) : null}
          </div>
        ) : null}
      </Dialog>
    </div>
  );
}

function NetCheck() {
  const [res, setRes] = useState<NetcheckResult | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="stack">
      <div className="row">
        <Button
          icon="wifi"
          busy={busy}
          onClick={async () => {
            setBusy(true);
            try {
              setRes(await netcheck());
            } catch (e) {
              toast("error", "Network check failed", { body: errorMessage(e) });
            } finally {
              setBusy(false);
            }
          }}
        >
          Check the data hosts
        </Button>
        <span className="cap">Results are cached for 10 minutes per host.</span>
      </div>
      {res ? (
        <table className="checklist" aria-label="Network check">
          <thead>
            <tr>
              <th scope="col">Host</th>
              <th scope="col">Reachable</th>
              <th scope="col" className="r">
                Time
              </th>
              <th scope="col">Error</th>
            </tr>
          </thead>
          <tbody>
            {res.results.map((r) => (
              <tr key={r.host}>
                <td className="mono">{r.host}</td>
                <td>
                  <StatusChip status={r.ok ? "done" : "failed"} text={r.ok ? "reachable" : "unreachable"} />
                </td>
                <td className="num r">{r.ms !== null ? `${fmtNum(r.ms, 0)} ms` : "—"}</td>
                <td className="cap">{r.error ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  );
}

function About() {
  const sys = useResource("system", (s) => getSystem(s), { tags: ["storage"] });
  if (sys.error && !sys.data) return <EmptyState error={sys.error} />;
  const d = sys.data;
  if (!d) return <p className="cap">Loading…</p>;
  const rows: [string, string][] = [
    ["Workspace", d.workspace],
    ["Workspace size", fmtBytes(d.workspace_bytes)],
    ["Machine", `${d.cpu_model} · ${d.cpu_count} threads · ${fmtNum(d.mem_total_gb, 1)} GB RAM`],
    ["Free now", `${fmtNum(d.mem_available_gb, 1)} GB RAM · ${fmtNum(d.disk_free_gb, 1)} GB disk`],
    ["Host id", d.host_id],
    ["SPARC", d.versions.sparc],
    ["Python", d.versions.python],
    ["numpy / pandas", `${d.versions.numpy} / ${d.versions.pandas}`],
    ["torch", d.versions.torch ?? "not installed"],
    ["FastAPI", d.versions.fastapi],
    ["Web build", d.web_build ? `${d.web_build.src_sha256.slice(0, 12)} · Vite ${d.web_build.vite} · React ${d.web_build.react}` : "development server"],
  ];
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div className="kv-row" key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function Settings() {
  return (
    <div className="stack">
      <header className="page-head">
        <h1>Settings</h1>
        <p className="cap">Workspace-wide: thread budget, engine, watch roots, storage and network.</p>
      </header>
      <SettingsForm />
      <Card title="Storage">
        <StorageManager />
      </Card>
      <Card title="Network check">
        <NetCheck />
      </Card>
      <Card title="About">
        <About />
      </Card>
    </div>
  );
}
