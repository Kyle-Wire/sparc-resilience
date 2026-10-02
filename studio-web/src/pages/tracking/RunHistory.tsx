// `/runs` and `/p/:pid/runs` — run history (SPEC §3.2, §5.12): status, label, mode, started,
// duration, R², RMSE, coverage, scenarios, checkpoint size, studies, commit (dirty flag) and
// origin; filters with saved filter sets; select two runs → Compare; and the stage-duration
// history chart grouped by commit (GET /api/timings) to spot performance regressions.
import { useEffect, useMemo, useState } from "react";
import { errorMessage } from "../../api/client";
import { useResource } from "../../api/resource";
import { getTimings, listRuns, type RunListQuery, type StageHistoryRow } from "../../api/tracking";
import type { Page, RunSummary } from "../../api/types";
import { Bars } from "../../charts";
import { Badge, modeLabel } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/EmptyState";
import { Select } from "../../components/ui/Select";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { codecs, Link, setQuery, useRoute, useUrlState } from "../../router";
import { readLocal, toast, writeLocal } from "../../stores/ui";
import { fmtBytes, fmtDate, fmtDuration, fmtNum, fmtPct } from "../../theme/format";
import { STAGE_IDS } from "../../stores/tracker";
import "./tracking.css";

type SavedFilter = { name: string; query: Record<string, string> };
const SAVED_KEY = "runs:saved-filters";
const FILTER_KEYS = ["status", "mode", "origin", "q", "sort"] as const;

const STATUS_OPTIONS = ["", "running", "complete", "partial", "failed", "cancelled", "interrupted", "external_live", "imported", "queued"] as const;
const MODE_OPTIONS = ["", "fast", "coarse", "full", "custom"] as const;
const ORIGIN_OPTIONS = ["", "studio", "imported", "study_child", "reproduction", "external_live"] as const;
const SORT_OPTIONS = ["created_desc", "duration", "r2"] as const;

/** Compare link for two selected runs (same project only), or the reason it is unavailable. */
export function compareTarget(selected: RunSummary[]): { href: string | null; reason: string | null } {
  if (selected.length !== 2) return { href: null, reason: "Select two runs to compare" };
  const [a, b] = selected;
  if (!a.project_id || a.project_id !== b.project_id) return { href: null, reason: "Runs from different projects cannot be compared" };
  return { href: `/p/${encodeURIComponent(a.project_id)}/compare?a=${encodeURIComponent(a.id)}&b=${encodeURIComponent(b.id)}`, reason: null };
}

/**
 * Stage-duration history as stacked bars: one bar per run, grouped by commit (runs of the same
 * commit are adjacent, commits in order of their newest run), one series per stage.
 */
export function stageHistoryChart(rows: StageHistoryRow[], stage: string | null): { categories: string[]; series: { id: string; label: string; values: (number | null)[] }[] } {
  const runs: { run_id: string; label: string; commit: string | null }[] = [];
  const seen = new Set<string>();
  for (const r of rows) {
    if (seen.has(r.run_id)) continue;
    seen.add(r.run_id);
    runs.push({ run_id: r.run_id, label: r.label || r.run_id, commit: r.git_commit });
  }
  const commits: (string | null)[] = [];
  for (const r of runs) if (!commits.includes(r.commit)) commits.push(r.commit);
  runs.sort((a, b) => commits.indexOf(a.commit) - commits.indexOf(b.commit));
  const stages = stage ? [stage] : STAGE_IDS.filter((s) => rows.some((r) => r.stage === s));
  const categories = runs.map((r) => `${r.commit ? r.commit.slice(0, 7) : "no commit"} · ${r.label}`);
  // run → stage → seconds (the first row of a pair wins, as the newest comes first)
  const seconds = new Map<string, Map<string, number>>();
  for (const r of rows) {
    const per = seconds.get(r.run_id) ?? new Map<string, number>();
    if (!per.has(r.stage)) per.set(r.stage, r.seconds);
    seconds.set(r.run_id, per);
  }
  const series = stages.map((s) => ({
    id: s,
    label: s,
    values: runs.map((run) => seconds.get(run.run_id)?.get(s) ?? null),
  }));
  return { categories, series };
}

function StageHistory({ pid }: { pid: string | null }) {
  const [stage, setStage] = useUrlState("stage", codecs.string(""));
  const res = useResource(`timings:${pid ?? "all"}`, (s) => getTimings({ project: pid ?? undefined, limit: 2000 }, s), { tags: ["runs"] });
  const rows = res.data?.stage_history;
  const chart = useMemo(() => (rows?.length ? stageHistoryChart(rows, stage || null) : null), [rows, stage]);
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  if (!rows?.length || !chart) return <p className="cap">{res.loading ? "Loading timings…" : "No stage timings recorded yet."}</p>;
  const { categories, series } = chart;
  return (
    <div className="stack">
      <div className="field" style={{ maxWidth: 260 }}>
        <label htmlFor="hist-stage">Stage</label>
        <Select id="hist-stage" value={stage} onChange={setStage} options={[{ value: "", label: "All stages (stacked)" }, ...STAGE_IDS.map((s) => ({ value: s, label: s }))]} />
      </div>
      <Bars
        title="Stage duration by run"
        caption="Runs grouped by code commit; a longer bar for the same stage and grid size after a commit points at a performance regression"
        categories={categories}
        series={series}
        orientation="h"
        mode={stage ? "grouped" : "stacked"}
        valueLabel="Duration"
        unit="s"
        categoryLabel="Commit · run"
        decimals={1}
        pin={{ view: "runs:stage-history" }}
      />
    </div>
  );
}

export default function RunHistory() {
  const { params } = useRoute();
  const pid = params.pid ?? null;
  const [status, setStatus] = useUrlState("status", codecs.enum(STATUS_OPTIONS, ""));
  const [mode, setMode] = useUrlState("mode", codecs.enum(MODE_OPTIONS, ""));
  const [origin, setOrigin] = useUrlState("origin", codecs.enum(ORIGIN_OPTIONS, ""));
  const [q, setQ] = useUrlState("q", codecs.string(""));
  const [sort, setSort] = useUrlState("sort", codecs.enum(SORT_OPTIONS, "created_desc"));
  const [qText, setQText] = useState(q);
  const [selected, setSelected] = useState<string[]>([]);
  const [saved, setSaved] = useState<SavedFilter[]>(() => readLocal<SavedFilter[]>(SAVED_KEY, []));
  const [saveName, setSaveName] = useState("");
  useEffect(() => {
    const h = setTimeout(() => qText !== q && setQ(qText), 300);
    return () => clearTimeout(h);
  }, [qText]);
  const query: RunListQuery = { status: status || undefined, mode: mode || undefined, origin: origin || undefined, q: q || undefined, sort, limit: 100 };
  const key = `runs:history:${pid ?? "all"}:${JSON.stringify(query)}`;
  const first = useResource<Page<RunSummary>>(key, (s) => listRuns(pid, query, s), { tags: pid ? ["runs", `project:${pid}`] : ["runs"], keepPrevious: true });
  const [more, setMore] = useState<{ key: string; items: RunSummary[]; next: string | null } | null>(null);
  useEffect(() => setMore(null), [key]);
  const items = useMemo(() => [...(first.data?.items ?? []), ...(more?.key === key ? more.items : [])], [first.data, more, key]);
  const next = more?.key === key ? more.next : first.data?.next_cursor ?? null;
  const selRuns = selected.map((id) => items.find((r) => r.id === id)).filter((r): r is RunSummary => !!r);
  const cmp = compareTarget(selRuns);

  const toggle = (id: string) =>
    setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-2)));

  const cols: Column<RunSummary>[] = [
    {
      key: "sel",
      label: "Compare",
      sortable: false,
      render: (r) => <input type="checkbox" aria-label={`Select ${r.label || r.id} for comparison`} checked={selected.includes(r.id)} onChange={() => toggle(r.id)} />,
    },
    { key: "status", label: "Status", value: (r) => r.status, render: (r) => <StatusChip status={r.status} /> },
    { key: "label", label: "Label", value: (r) => r.label ?? r.id, render: (r) => <Link to={`/r/${encodeURIComponent(r.id)}`}>{r.label || r.id}</Link> },
    { key: "mode", label: "Mode", value: (r) => r.mode, render: (r) => <Badge tone="accent">{modeLabel(r.mode, r.coarse_m)}</Badge> },
    { key: "created", label: "Started", value: (r) => r.created_utc, render: (r) => fmtDate(r.created_utc) },
    { key: "duration", label: "Duration", align: "right", value: (r) => r.duration_s, render: (r) => fmtDuration(r.duration_s) },
    { key: "r2", label: "R²", align: "right", value: (r) => r.r2, render: (r) => fmtNum(r.r2, 3) },
    { key: "rmse", label: "RMSE", align: "right", value: (r) => r.rmse, render: (r) => fmtNum(r.rmse, 3) },
    { key: "coverage", label: "Coverage", align: "right", value: (r) => r.coverage, render: (r) => fmtPct(r.coverage, 1) },
    { key: "scenarios", label: "Scenarios", align: "right", value: (r) => r.n_scenarios },
    { key: "ckpt", label: "Checkpoint", align: "right", value: (r) => r.checkpoint_bytes, render: (r) => fmtBytes(r.checkpoint_bytes) },
    { key: "studies", label: "Studies", value: (r) => r.studies.join(", "), render: (r) => (r.studies.length ? r.studies.join(", ") : "—") },
    {
      key: "commit",
      label: "Commit",
      value: (r) => r.git_commit,
      render: (r) => (r.git_commit ? <span className="mono" title={r.git_dirty ? "uncommitted changes when it ran" : undefined}>{r.git_commit.slice(0, 7)}{r.git_dirty ? " (dirty)" : ""}</span> : "—"),
    },
    { key: "origin", label: "Origin", value: (r) => r.origin },
  ];

  const values: Record<(typeof FILTER_KEYS)[number], string> = { status, mode, origin, q, sort };
  const current: Record<string, string> = Object.fromEntries(FILTER_KEYS.filter((k) => values[k] && !(k === "sort" && values[k] === "created_desc")).map((k) => [k, values[k]]));
  const persist = (list: SavedFilter[]) => {
    setSaved(list);
    writeLocal(SAVED_KEY, list);
  };

  return (
    <div className="stack">
      <header className="page-head">
        <h1>{pid ? "Runs" : "Run history"}</h1>
        <p className="cap">Every run{pid ? " of this project" : ""}, newest first. Select two to compare them.</p>
      </header>
      <Card>
        <div className="filters">
          <div className="field">
            <label htmlFor="runs-status">Status</label>
            <Select id="runs-status" value={status} onChange={setStatus} options={STATUS_OPTIONS.map((s) => ({ value: s, label: s ? s.replace(/_/g, " ") : "Any status" }))} />
          </div>
          <div className="field">
            <label htmlFor="runs-mode">Mode</label>
            <Select id="runs-mode" value={mode} onChange={setMode} options={MODE_OPTIONS.map((s) => ({ value: s, label: s || "Any mode" }))} />
          </div>
          <div className="field">
            <label htmlFor="runs-origin">Origin</label>
            <Select id="runs-origin" value={origin} onChange={setOrigin} options={ORIGIN_OPTIONS.map((s) => ({ value: s, label: s ? s.replace(/_/g, " ") : "Any origin" }))} />
          </div>
          <div className="field">
            <label htmlFor="runs-q">Search</label>
            <input id="runs-q" type="search" value={qText} placeholder="Label or run id" onChange={(e) => setQText(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="runs-sort">Sort</label>
            <Select id="runs-sort" value={sort} onChange={setSort} options={[{ value: "created_desc", label: "Newest first" }, { value: "duration", label: "Duration" }, { value: "r2", label: "R²" }]} />
          </div>
        </div>
        <div className="row">
          <span className="cap">Saved filters:</span>
          {saved.length ? null : <span className="cap">none</span>}
          {saved.map((f) => (
            <span key={f.name} className="chip">
              <button
                type="button"
                className="x"
                style={{ fontWeight: 600 }}
                onClick={() => {
                  setQText(f.query.q ?? "");
                  setQuery(Object.fromEntries(FILTER_KEYS.map((k) => [k, f.query[k] ?? null])));
                }}
              >
                {f.name}
              </button>
              <button type="button" className="x" aria-label={`Remove saved filter ${f.name}`} onClick={() => persist(saved.filter((x) => x.name !== f.name))}>
                ×
              </button>
            </span>
          ))}
          <input type="text" aria-label="Name for the current filters" placeholder="Name these filters" value={saveName} onChange={(e) => setSaveName(e.target.value)} style={{ width: "12em" }} />
          <Button
            size="small"
            disabled={!saveName.trim() || !Object.keys(current).length}
            onClick={() => {
              persist([...saved.filter((x) => x.name !== saveName.trim()), { name: saveName.trim(), query: current }]);
              setSaveName("");
            }}
          >
            Save filters
          </Button>
        </div>
      </Card>
      <Card
        title="Runs"
        actions={
          cmp.href ? (
            <Link className="btn primary" to={cmp.href}>
              Compare selected
            </Link>
          ) : (
            <Button disabled title={cmp.reason ?? undefined}>
              Compare selected
            </Button>
          )
        }
      >
        {selRuns.length ? <p className="cap">{cmp.reason ?? `Comparing ${selRuns.map((r) => r.label || r.id).join(" and ")}`}</p> : null}
        {first.error && !first.data ? (
          <EmptyState error={first.error} />
        ) : (
          <Table columns={cols} rows={items} rowKey={(r) => r.id} caption="Runs" highlight={(r) => selected.includes(r.id)} empty={first.loading ? "Loading…" : "No runs match these filters."} csvName="runs" />
        )}
        {next ? (
          <Button
            size="small"
            onClick={async () => {
              try {
                const page = await listRuns(pid, { ...query, cursor: next });
                setMore({ key, items: [...(more?.key === key ? more.items : []), ...page.items], next: page.next_cursor });
              } catch (e) {
                toast("error", "Could not load more runs", { body: errorMessage(e) });
              }
            }}
          >
            Load more
          </Button>
        ) : null}
      </Card>
      <Card title="Stage-duration history" eyebrow="performance across runs">
        <StageHistory pid={pid} />
      </Card>
    </div>
  );
}
