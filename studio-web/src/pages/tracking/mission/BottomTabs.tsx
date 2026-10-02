// Mission Control bottom tabs (SPEC §5.10): Logs (virtualised, level/logger/stage/text filters,
// follow-tail, copy, downloads, raw stderr), Warnings (grouped by code, linked to the output view),
// Outputs (artifact feed linked to run tabs, even mid-run), Checkpoints, Resources (RSS, CPU and
// process sparklines, peak, free RAM and disk, the 80% memory banner) and Config & provenance.
import { useEffect, useMemo, useRef, useState } from "react";
import { useResource } from "../../../api/resource";
import {
  getLogs,
  getResources,
  getRunConfig,
  getRunDetail,
  getRunProvenance,
  getSystem,
  isFinalStatus,
  logsRawUrl,
  type LogLine,
  type RawStream,
  type ResourceRow,
} from "../../../api/tracking";
import type { Job, Meta } from "../../../api/types";
import { Sparkline } from "../../../charts";
import { Button } from "../../../components/ui/Button";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Icon } from "../../../components/ui/Icon";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { Select } from "../../../components/ui/Select";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Table, type Column } from "../../../components/ui/Table";
import { Tabs } from "../../../components/ui/Tabs";
import { VirtualList } from "../../../components/ui/VirtualList";
import { copyText } from "../../../components/ui/download";
import { codecs, Link, useUrlState } from "../../../router";
import { toast } from "../../../stores/ui";
import { LOG_KEEP, STAGE_IDS, type TrackerEntry, type TrackerState } from "../../../stores/tracker";
import { fmtBytes, fmtDateTime, fmtNum } from "../../../theme/format";
import { artifactTab, matchesLog, warningTab, type LogFilter } from "../model";

export const BOTTOM_TABS = ["logs", "warnings", "outputs", "checkpoints", "resources", "config"] as const;
export type BottomTab = (typeof BOTTOM_TABS)[number];

function clock(ts: number): string {
  const d = new Date(ts * 1000);
  const p = (x: number) => String(x).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// ---------------------------------------------------------------- logs

const LEVEL_OPTIONS = [
  { value: "debug", label: "debug and up" },
  { value: "info", label: "info and up" },
  { value: "warning", label: "warnings and errors" },
  { value: "error", label: "errors only" },
] as const;

const LOG_PAGE = 5000;
/** Lines the Logs tab keeps (the newest); the downloads hold everything. */
export const LOG_VIEW_MAX = 50_000;

/**
 * Every line of a job's log that passes the filters (`GET /api/jobs/{jid}/logs`, page after
 * page), keeping the newest LOG_VIEW_MAX. `next_cursor` is where the server stopped reading.
 */
export async function loadLogLines(jid: string, f: LogFilter, signal?: AbortSignal, fetchLogs: typeof getLogs = getLogs): Promise<{ lines: LogLine[]; next_cursor: number; truncated: boolean }> {
  let after: number | null = null;
  let lines: LogLine[] = [];
  let truncated = false;
  let next = -1;
  for (;;) {
    const page = await fetchLogs(jid, { after, level: f.level, logger: f.logger || undefined, stage: f.stage || undefined, q: f.q || undefined, limit: LOG_PAGE }, signal);
    lines = lines.concat(page.lines);
    if (lines.length > LOG_VIEW_MAX) {
      lines = lines.slice(-LOG_VIEW_MAX);
      truncated = true;
    }
    next = page.next_cursor;
    if (page.lines.length < LOG_PAGE || page.next_cursor === after) break;
    after = page.next_cursor;
  }
  return { lines, next_cursor: next, truncated };
}

function LogsTab({ entry }: { entry: TrackerEntry }) {
  const jid = entry.jid;
  const [level, setLevel] = useUrlState("lvl", codecs.enum(["debug", "info", "warning", "error"] as const, "info"));
  const [logger, setLogger] = useUrlState("logger", codecs.string(""));
  const [stage, setStage] = useUrlState("lstage", codecs.string(""));
  const [q, setQ] = useUrlState("q", codecs.string(""));
  const [qText, setQText] = useState(q);
  const [follow, setFollow] = useState(true);
  const filter: LogFilter = { level, logger, stage, q };
  useEffect(() => {
    const h = setTimeout(() => qText !== q && setQ(qText), 300);
    return () => clearTimeout(h);
  }, [qText]);
  const key = `job:${jid}:logs:${level}|${logger}|${stage}|${q}`;
  const res = useResource(key, (s) => loadLogLines(jid, filter, s), { tags: [`job:${jid}`], keepPrevious: true });
  const server = res.data?.lines;
  const edge = Math.max(res.data?.next_cursor ?? -1, server?.length ? server[server.length - 1].cursor : -1);
  const lines: LogLine[] = useMemo(() => [...(server ?? []), ...entry.logs.filter((l) => l.cursor > edge && matchesLog(l, filter))], [server, edge, entry.logs, level, logger, stage, q]);
  // The entry keeps only the newest LOG_KEEP live lines: once lines after the page's read have
  // been dropped, read the log again so the list has no hole.
  const gap = !!res.data && entry.logs.length >= LOG_KEEP && entry.logs[0].cursor > edge;
  const reload = res.reload;
  useEffect(() => {
    if (gap) void reload();
  }, [gap, reload]);
  const scroller = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (follow && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight;
  }, [lines.length, follow]);
  const [stderr, setStderr] = useState<string | null>(null);
  const loadStderr = async () => {
    try {
      const r = await fetch(logsRawUrl(jid, "stderr"), { credentials: "same-origin" });
      setStderr(r.ok ? await r.text() : r.status === 404 ? "" : `Could not read stderr (HTTP ${r.status})`);
    } catch {
      setStderr("Could not read stderr: the Studio server is not reachable");
    }
  };
  const text = () => lines.map((l) => `${new Date(l.ts * 1000).toISOString()} ${l.level.toUpperCase()} ${l.logger}: ${l.msg}`).join("\n");
  const downloads: { stream: RawStream; label: string }[] = [
    { stream: "events", label: "events.jsonl" },
    { stream: "stdout", label: "stdout.log" },
    { stream: "stderr", label: "stderr.log" },
    { stream: "text", label: "log.txt" },
  ];
  return (
    <div className="stack">
      {entry.logCapped ? (
        <div className="callout" role="note">
          This job's event log passed 200 MB, so debug lines are kept on disk only: download events.jsonl to see them.
        </div>
      ) : null}
      <div className="filters">
        <div className="field">
          <label htmlFor="log-level">Level</label>
          <Select id="log-level" options={LEVEL_OPTIONS} value={level} onChange={(v) => setLevel(v)} />
        </div>
        <div className="field">
          <label htmlFor="log-logger">Logger</label>
          <input id="log-logger" type="text" value={logger} placeholder="e.g. sparc.core or warning:" onChange={(e) => setLogger(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="log-stage">Stage</label>
          <Select id="log-stage" options={[{ value: "", label: "All stages" }, ...STAGE_IDS.map((s) => ({ value: s, label: s }))]} value={stage} onChange={(v) => setStage(v)} />
        </div>
        <div className="field">
          <label htmlFor="log-q">Text</label>
          <input id="log-q" type="search" value={qText} placeholder="Search messages" onChange={(e) => setQText(e.target.value)} />
        </div>
        <Button size="small" icon={follow ? "pause" : "play"} aria-pressed={follow} onClick={() => setFollow((f) => !f)}>
          Follow
        </Button>
        <Button
          size="small"
          icon="copy"
          onClick={async () => {
            const ok = await copyText(text());
            toast(ok ? "success" : "warning", ok ? `${lines.length} log lines copied` : "Copy failed: use a download");
          }}
        >
          Copy
        </Button>
      </div>
      {res.data?.truncated ? (
        <p className="mc-note">Showing the newest {LOG_VIEW_MAX.toLocaleString("en")} matching lines; the downloads below hold every line.</p>
      ) : null}
      {res.error && !res.data ? (
        <EmptyState error={res.error} />
      ) : lines.length ? (
        <div className="loglist">
          <VirtualList
            count={lines.length}
            rowHeight={20}
            height={Math.min(360, 20 * lines.length + 4)}
            label={`${lines.length} log lines`}
            scrollRef={(el) => (scroller.current = el)}
            onScroll={(_top, atEnd) => !atEnd && follow && setFollow(false)}
            getKey={(i) => `${lines[i].cursor}:${i}`}
            renderRow={(i) => {
              const l = lines[i];
              return (
                <div className="logline" title={l.msg}>
                  <span className="ts">{clock(l.ts)}</span>
                  <span className="lvl" data-level={l.level}>
                    {l.level}
                  </span>
                  <span className="logger">{l.logger}</span>
                  <span className="msg">{l.msg}</span>
                </div>
              );
            }}
          />
        </div>
      ) : (
        <p className="mc-note">{res.loading ? "Loading log lines…" : "No log lines match these filters."}</p>
      )}
      <div className="row cap">
        <span>Download:</span>
        {downloads.map((d) => (
          <a key={d.stream} className="btn small ghost" href={logsRawUrl(jid, d.stream)} download>
            <Icon name="download" /> {d.label}
          </a>
        ))}
      </div>
      <details onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && stderr === null && void loadStderr()}>
        <summary className="cap">Raw stderr</summary>
        {stderr === null ? (
          <p className="mc-note">Loading…</p>
        ) : stderr === "" ? (
          <p className="mc-note">Nothing was written to stderr.</p>
        ) : (
          <pre className="raw-pre">{stderr}</pre>
        )}
        <Button size="small" variant="ghost" icon="refresh" onClick={() => void loadStderr()}>
          Reload
        </Button>
      </details>
    </div>
  );
}

// ---------------------------------------------------------------- warnings

function WarningsTab({ state, runId, meta, onShowLogs }: { state: TrackerState; runId: string | null; meta: Meta | undefined; onShowLogs: (code: string) => void }) {
  const groups = new Map<string, { code: string; count: number; rows: { message: string; count: number; stage: string | null }[] }>();
  for (const w of Object.values(state.warnings)) {
    const g = groups.get(w.code) ?? { code: w.code, count: 0, rows: [] };
    g.count += w.count;
    g.rows.push({ message: w.message, count: w.count, stage: w.stage });
    groups.set(w.code, g);
  }
  const list = [...groups.values()].sort((a, b) => b.count - a.count || a.code.localeCompare(b.code));
  if (!list.length) return <p className="mc-note">No warnings so far.</p>;
  const labelOf = (code: string) => meta?.warning_codes.find((c) => c.code === code);
  return (
    <ul className="spine" aria-label="Warnings by code">
      {list.map((g) => {
        const info = labelOf(g.code);
        const tab = warningTab(g.code, info?.view ?? null);
        return (
          <li key={g.code}>
            <div>
              <StatusChip status="stale" text={g.code} meta={`×${g.count}`} />
              {info?.label ? <div className="cap">{info.label}</div> : null}
            </div>
            <ul className="stack" style={{ gap: 2, margin: 0, paddingLeft: "1.1em" }}>
              {g.rows.map((r, i) => (
                <li key={i} className="cap" style={{ color: "var(--ink-2)" }}>
                  {r.message || "(no message)"}
                  {r.count > 1 ? ` ×${r.count}` : ""}
                  {r.stage ? ` · ${r.stage}` : ""}
                </li>
              ))}
            </ul>
            <div className="row">
              {tab && runId ? <Link to={`/r/${encodeURIComponent(runId)}/${tab}`}>Open {tab}</Link> : null}
              <Button size="small" variant="ghost" onClick={() => onShowLogs(g.code)}>
                In logs
              </Button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

// ---------------------------------------------------------------- outputs and checkpoints

function OutputsTab({ state, runId, meta }: { state: TrackerState; runId: string | null; meta: Meta | undefined }) {
  const rows = Object.values(state.artifacts).sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
  const cols: Column<(typeof rows)[number]>[] = [
    {
      key: "relpath",
      label: "File",
      render: (a) => {
        const tab = artifactTab(a, meta?.output_catalog ?? null);
        return tab && runId ? <Link to={`/r/${encodeURIComponent(runId)}/${tab}`}>{a.relpath}</Link> : <span className="mono">{a.relpath}</span>;
      },
    },
    { key: "role", label: "Role" },
    { key: "bytes", label: "Size", align: "right", value: (a) => a.bytes, render: (a) => fmtBytes(a.bytes) },
    { key: "stage", label: "Stage" },
    { key: "ts", label: "Written", value: (a) => a.ts, render: (a) => (a.ts ? clock(a.ts) : "—") },
  ];
  return <Table columns={cols} rows={rows} rowKey={(a) => a.relpath} caption="Files written by this job" empty="No files written yet." csvName={null} />;
}

function CheckpointsTab({ state }: { state: TrackerState }) {
  if (!state.checkpoints.length) return <p className="mc-note">No checkpoint saved or loaded by this job.</p>;
  const tone: Record<string, string> = { saved: "done", loaded: "cached", mismatch: "stale" };
  return (
    <table className="checklist" aria-label="Checkpoints">
      <thead>
        <tr>
          <th scope="col">Action</th>
          <th scope="col">Done set</th>
          <th scope="col" className="r">
            Size
          </th>
          <th scope="col">When</th>
        </tr>
      </thead>
      <tbody>
        {state.checkpoints.map((c, i) => (
          <tr key={i}>
            <td>
              <StatusChip status={tone[c.action] ?? "queued"} text={c.action} />
            </td>
            <td className="mono">{c.done.join(", ") || "—"}</td>
            <td className="num r">{fmtBytes(c.bytes)}</td>
            <td>{c.ts ? fmtDateTime(c.ts) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- resources

/**
 * The samples to draw: the job's stored history (one per 10 s since it started, read once) before
 * the first sample the tracker holds, then the tracker's (the snapshot's last 15 min and the live
 * 2 s samples). Without the history a job opened after it ended, or more than 15 min into it,
 * would show nothing, or only its tail.
 */
export function resourceSeries(history: readonly ResourceRow[] | undefined, tracked: readonly ResourceRow[]): ResourceRow[] {
  const first = tracked.length ? tracked[0].ts : Infinity;
  return [...(history ?? []).filter((r) => r.ts < first), ...tracked];
}

function ResourcesTab({ entry, job }: { entry: TrackerEntry; job: Job }) {
  const sys = useResource("system", (s) => getSystem(s), { tags: ["storage"] });
  const started = job.started_utc ? Date.parse(job.started_utc) / 1000 : null;
  const history = useResource(started !== null && Number.isFinite(started) ? `job:${job.id}:resources-since-start` : null, (s) => getResources(job.id, started!, s), { tags: [`job:${job.id}`] });
  const rs = useMemo(() => resourceSeries(history.data, entry.resources), [history.data, entry.resources]);
  const final = isFinalStatus(job.status);
  if (!rs.length) {
    if (history.loading) return <p className="mc-note">Loading resource samples…</p>;
    return <p className="mc-note">{final ? "No resource samples were recorded for this job." : "Resource samples arrive every 2 s while the job runs."}</p>;
  }
  const last = rs[rs.length - 1];
  const peak = Math.max(job.peak_rss_mb ?? 0, ...rs.map((r) => r.rss_mb));
  const availMb = sys.data ? sys.data.mem_available_gb * 1024 : null;
  const usable = availMb !== null ? availMb + last.rss_mb : null;
  // The memory banner is about a live process (free memory is read now).
  const high = !final && usable !== null && last.rss_mb > 0.8 * usable;
  const xs = rs.map((r) => r.ts - rs[0].ts);
  return (
    <div className="stack">
      {high ? (
        <div className="callout" data-tone="crit" role="alert">
          Memory use is {fmtBytes(last.rss_mb * 1e6)}, above 80% of the {fmtBytes((usable ?? 0) * 1e6)} this job can use. It may run out of memory.
        </div>
      ) : null}
      <KpiRow label="Resources">
        <Kpi label={final ? "Memory (RSS), last sample" : "Memory (RSS)"} value={fmtNum(last.rss_mb / 1024, 2)} unit="GB" note={`peak ${fmtNum(peak / 1024, 2)} GB`} tone={high ? "crit" : undefined} />
        <Kpi label="CPU" value={fmtNum(last.cpu_pct, 0)} unit="%" note={job.threads ? `${job.threads} threads allowed` : undefined} />
        <Kpi label="Processes" value={fmtNum(last.n_procs, 0)} />
        <Kpi label="Free RAM" value={sys.data ? fmtNum(sys.data.mem_available_gb, 1) : "—"} unit="GB" note={sys.data ? `of ${fmtNum(sys.data.mem_total_gb, 1)} GB` : undefined} />
        <Kpi label="Free disk" value={sys.data ? fmtNum(sys.data.disk_free_gb, 1) : "—"} unit="GB" />
      </KpiRow>
      <div className="grid3">
        <Sparkline title="Memory (RSS)" values={rs.map((r) => r.rss_mb)} x={xs} units="MB" unit="MB" decimals={0} area ref={!final && usable !== null ? 0.8 * usable : undefined} pin={false} caption={!final && usable !== null ? "dashed line: 80% of usable memory" : undefined} />
        <Sparkline title="CPU" values={rs.map((r) => r.cpu_pct)} x={xs} units="%" unit="%" decimals={0} pin={false} />
        <Sparkline title="Processes" values={rs.map((r) => r.n_procs)} x={xs} units="processes" unit="processes" decimals={0} pin={false} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- config & provenance

function ConfigTab({ job }: { job: Job }) {
  const rid = job.run_id;
  const cfg = useResource(rid ? `run:${rid}:config` : null, (s) => getRunConfig(rid!, s), { tags: rid ? [`run:${rid}`] : [] });
  const prov = useResource(rid ? `run:${rid}:provenance` : null, (s) => getRunProvenance(rid!, s), { tags: rid ? [`run:${rid}`] : [] });
  const detail = useResource(rid ? `run:${rid}:detail-lite` : null, (s) => getRunDetail(rid!, s), { tags: rid ? [`run:${rid}`] : [] });
  const git = (prov.data?.git ?? null) as Record<string, unknown> | null;
  return (
    <div className="stack">
      <details open={!rid}>
        <summary className="cap">Job parameters</summary>
        <pre className="raw-pre">{JSON.stringify(job.params, null, 2)}</pre>
      </details>
      {rid ? (
        <>
          <details>
            <summary className="cap">Launch snapshot (launch.json)</summary>
            {detail.data?.launch ? <pre className="raw-pre">{JSON.stringify(detail.data.launch, null, 2)}</pre> : <p className="mc-note">{detail.loading ? "Loading…" : "No launch snapshot (an imported run)."}</p>}
          </details>
          <details>
            <summary className="cap">Effective config{cfg.data ? ` (from ${cfg.data.source})` : ""}</summary>
            {cfg.data ? <pre className="raw-pre">{cfg.data.yaml}</pre> : cfg.error ? <EmptyState error={cfg.error} /> : <p className="mc-note">Loading…</p>}
          </details>
          <details open>
            <summary className="cap">Hashes and code version</summary>
            {prov.data ? (
              <dl className="kv">
                {git ? (
                  <div className="kv-row">
                    <dt>git</dt>
                    <dd>
                      {String(git.commit ?? git.sha ?? "—").slice(0, 12)}
                      {git.dirty ? " (uncommitted changes)" : ""}
                    </dd>
                  </div>
                ) : null}
                {Object.entries(prov.data.hashes).map(([k, v]) => (
                  <div className="kv-row" key={k}>
                    <dt>{k}</dt>
                    <dd className="mono">{v}</dd>
                  </div>
                ))}
              </dl>
            ) : prov.error ? (
              <EmptyState error={prov.error} />
            ) : (
              <p className="mc-note">Loading…</p>
            )}
          </details>
        </>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- tabs

export function BottomTabs({ entry, job, meta }: { entry: TrackerEntry; job: Job; meta: Meta | undefined }) {
  const [tab, setTab] = useUrlState("tab", codecs.enum(BOTTOM_TABS, "logs"));
  const [, setLogger] = useUrlState("logger", codecs.string(""));
  const state = entry.state;
  const nWarn = Object.values(state.warnings).reduce((a, w) => a + w.count, 0);
  const items = [
    { id: "logs" as const, label: "Logs" },
    { id: "warnings" as const, label: "Warnings", badge: nWarn ? <span className="muted num">({nWarn})</span> : undefined },
    { id: "outputs" as const, label: "Outputs", badge: Object.keys(state.artifacts).length ? <span className="muted num">({Object.keys(state.artifacts).length})</span> : undefined },
    { id: "checkpoints" as const, label: "Checkpoints" },
    { id: "resources" as const, label: "Resources" },
    { id: "config" as const, label: "Config & provenance" },
  ];
  return (
    <section className="card" aria-label="Job details">
      <Tabs items={items} value={tab} onChange={setTab} label="Job details">
        {tab === "logs" ? (
          <LogsTab entry={entry} />
        ) : tab === "warnings" ? (
          <WarningsTab
            state={state}
            runId={job.run_id}
            meta={meta}
            onShowLogs={(code) => {
              setLogger(`warning:${code}`);
              setTab("logs");
            }}
          />
        ) : tab === "outputs" ? (
          <OutputsTab state={state} runId={job.run_id} meta={meta} />
        ) : tab === "checkpoints" ? (
          <CheckpointsTab state={state} />
        ) : tab === "resources" ? (
          <ResourcesTab entry={entry} job={job} />
        ) : (
          <ConfigTab job={job} />
        )}
      </Tabs>
      {entry.reconstructed && !entry.replaying && entry.phase === "live" ? (
        <p className="mc-note">Progress before this page opened is rebuilt from the server's summary, because this job's event log could not be replayed here (it is very large, or not available); the figures may differ slightly from the server's until the next stage ends.</p>
      ) : null}
    </section>
  );
}
