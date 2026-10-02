// Launch (`/p/:pid/launch`, SPEC §3.2, §5.4, J1 step 5, J2 step 7): mode cards (Fast /
// Coarse M / Full) with ETA range, peak RAM and checkpoint disk from `/runs/plan`; the stage
// checklist with core's dependency rules; the CV-curve toggle, threads and a "then run"
// chain; the PlanGraph of the chosen mode; the preflight list with one-click actions; and
// Start, which launches the run and opens Mission Control. Start stays disabled while a
// preflight check fails with severity "error" or the config has errors. `?from=<run_id>`
// prefills mode, stages, CV curve and threads from that run's launch snapshot (the remedy
// links of stages a finished run did not compute); explicit query parameters win.
import { useEffect, useMemo, useState } from "react";
import { ApiError, errorMessage } from "../../api/client";
import {
  blockingPreflight,
  getRunLaunch,
  launchRun,
  planRun,
  THEN_ACTIONS,
  useMetaInfo,
  useStudioSettings,
  type RunLaunchSource,
  type RunMode,
  type RunPlan,
  type RunPlanBody,
  type ThenAction,
} from "../../api/projects";
import { useResource } from "../../api/resource";
import { Badge, DemoBadge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { Chips } from "../../components/ui/Chips";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { NumberField } from "../../components/ui/NumberField";
import { Seg } from "../../components/ui/Seg";
import { useProject } from "../../layouts/resources";
import { codecs, Link, navigate, setQuery, useRoute, useUrlState } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtBytes, fmtDuration, fmtDurationRange, fmtNum } from "../../theme/format";
import { PlanGraph } from "./components/PlanGraph";
import { ALL_CHOICES, checklistRows, launchQueryFrom, parseSelection, requestStages, toggleStage, type ChecklistId } from "./model/stages";
import "./projects.css";

const MODES: readonly RunMode[] = ["fast", "coarse", "full"];

const MODE_TEXT: Record<RunMode, { label: string; desc: string }> = {
  fast: { label: "Fast", desc: "Quick check: fewer models and tuning steps; minutes, not hours." },
  coarse: { label: "Coarse", desc: "Full pipeline on coarser cells: a faithful preview." },
  full: { label: "Full", desc: "Every model at full resolution: the run decisions rest on." },
};

const THEN_LABEL: Record<ThenAction, string> = {
  "post.baselines": "Reference baselines",
  "post.planner": "Planner pack",
  "post.emulator": "Scenario emulator",
  "post.uncertainty": "Uncertainty report",
  "post.writeup": "Methods & model card",
};

type CvChoice = "config" | "on" | "off";

/** The plan/launch body for a mode and the shared choices. */
export function planBody(mode: RunMode, o: { coarse_m: number; selection: readonly ChecklistId[]; cv: CvChoice; threads: number | null }): RunPlanBody {
  return {
    mode,
    ...(mode === "coarse" ? { coarse_m: o.coarse_m } : {}),
    stages: requestStages(o.selection),
    cv_curve: o.cv === "config" ? null : o.cv === "on",
    ...(o.threads ? { threads: o.threads } : {}),
  };
}

/** `POST /runs/plan` for one body; keyed by the config version so a config edit replans. */
function usePlan(pid: string, configVersion: number | null, body: RunPlanBody) {
  const key = configVersion === null ? null : `plan:${pid}:v${configVersion}:${JSON.stringify(body)}`;
  return useResource<RunPlan>(key, (s) => planRun(pid, body, s));
}

function ModeCard({
  mode,
  plan,
  selected,
  onSelect,
  coarseM,
  onCoarse,
  desc,
}: {
  mode: RunMode;
  plan: ReturnType<typeof usePlan>;
  selected: boolean;
  onSelect: () => void;
  coarseM: number;
  onCoarse: (v: number) => void;
  desc: string;
}) {
  const p = plan.data;
  const label = mode === "coarse" ? `Coarse ${coarseM} m` : MODE_TEXT[mode].label;
  return (
    <div className="mode-card" data-mode={mode} data-selected={selected || undefined}>
      <button type="button" className="mode-pick" aria-pressed={selected} onClick={onSelect}>
        <span className="mode-name">{label}</span>
        <span className="cap">{desc}</span>
        <dl className="kv mode-est">
          <div className="kv-row">
            <dt>Time</dt>
            <dd className="num" data-est="time">
              {p ? `≈${fmtDurationRange(p.est_lo, p.est_hi)}` : plan.error ? "—" : "…"}
            </dd>
          </div>
          <div className="kv-row">
            <dt>Peak RAM</dt>
            <dd className="num" data-est="ram">
              {p ? `≈${fmtNum(p.est_peak_rss_gb, 1)} GB` : plan.error ? "—" : "…"}
            </dd>
          </div>
          <div className="kv-row">
            <dt>Checkpoint disk</dt>
            <dd className="num" data-est="disk">
              {p ? `≈${fmtBytes(p.est_disk_gb * 1e9)}` : plan.error ? "—" : "…"}
            </dd>
          </div>
        </dl>
      </button>
      {mode === "coarse" ? (
        <div className="row">
          <NumberField label="Coarse cell size" value={coarseM} onChange={(v) => v !== null && onCoarse(v)} min={30} max={1000} step={10} unit="m" />
        </div>
      ) : null}
    </div>
  );
}

function Preflight({ plan, onAction }: { plan: RunPlan; onAction: () => void }) {
  const rows = plan.preflight;
  if (!rows.length) return <p className="cap">No preflight checks.</p>;
  return (
    <ul className="preflight" aria-label="Preflight checks">
      {rows.map((r, i) => {
        const state = r.ok ? "ok" : r.severity;
        return (
          <li key={i} data-state={state} data-check={r.check}>
            <Icon name={r.ok ? "check" : r.severity === "error" ? "x" : r.severity === "warn" ? "alert" : "info"} />
            <span className="pf-text">
              <b>{r.ok ? "OK" : r.severity === "error" ? "Blocks the launch" : r.severity === "warn" ? "Warning" : "Note"}</b> · {r.message}
            </span>
            {r.action && !r.ok ? <ActionButton action={r.action} size="small" variant="default" onDone={onAction} /> : null}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * `?from=<run_id>`: once that run's detail loads, fill the query parameters the URL does not
 * already set and the thread count from its launch snapshot, then drop `from` (the URL state
 * carries the choices from there). Returns the source run for the "prefilled from" note.
 */
function usePrefillFrom(query: URLSearchParams, setThreads: (n: number | null) => void) {
  const from = query.get("from");
  const source = useResource<RunLaunchSource>(from ? `launch-from:${from}` : null, (s) => getRunLaunch(from!, s));
  const [applied, setApplied] = useState<{ rid: string; label: string; snapshot: boolean } | null>(null);
  useEffect(() => {
    const d = source.data;
    if (!from || !d || d.run.id !== from) return;
    const args = d.launch?.args ?? null;
    if (args) {
      const q = launchQueryFrom(args);
      const patch: Record<string, string | null> = { from: null };
      for (const [k, v] of Object.entries(q)) if (!query.has(k) && v !== null && !(k === "mode" && v === "fast")) patch[k] = v;
      if (typeof args.threads === "number" && args.threads > 0) setThreads(args.threads);
      setQuery(patch);
    } else setQuery({ from: null });
    setApplied({ rid: d.run.id, label: d.run.label || d.run.id, snapshot: !!args });
  }, [from, source.data]); // the query and setter are read at apply time only
  return { applied, loading: !!from && source.loading, error: from ? source.error : null, from };
}

export default function Launch() {
  const { params, query } = useRoute();
  const pid = params.pid ?? "";
  const project = useProject(pid || null);
  const meta = useMetaInfo();
  const settings = useStudioSettings();
  const [mode, setMode] = useUrlState("mode", codecs.enum(MODES, "fast"));
  const defaultCoarse = meta.data?.modes.find((m) => m.id === "coarse")?.default_coarse_m ?? 60;
  const [coarseQ, setCoarseQ] = useUrlState("coarse", codecs.float(null));
  const coarseM = coarseQ ?? defaultCoarse;
  const [stagesQ, setStagesQ] = useUrlState("stages", codecs.list());
  const selection = stagesQ.length ? parseSelection(stagesQ) : [...ALL_CHOICES];
  const [cv, setCv] = useUrlState("cv", codecs.enum<CvChoice>(["config", "on", "off"], "config"));
  const [threads, setThreads] = useState<number | null>(null);
  const prefill = usePrefillFrom(query, setThreads);
  const [then, setThen] = useState<ThenAction[]>([]);
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [starting, setStarting] = useState(false);

  const configVersion = project.data ? project.data.config_version : null;
  const shared = { coarse_m: coarseM, selection, cv, threads };
  const bodies = useMemo(() => Object.fromEntries(MODES.map((m) => [m, planBody(m, shared)])) as Record<RunMode, RunPlanBody>, [JSON.stringify(shared)]);
  const plans = {
    fast: usePlan(pid, configVersion, bodies.fast),
    coarse: usePlan(pid, configVersion, bodies.coarse),
    full: usePlan(pid, configVersion, bodies.full),
  };
  const plan = plans[mode];
  const p = plan.data;
  const blocking = blockingPreflight(p);
  const configErrors = (p?.issues ?? []).filter((i) => i.level === "error");
  const canStart = !!p && !plan.loading && blocking.length === 0 && configErrors.length === 0 && !starting;
  const rows = checklistRows(selection);
  const setSelection = (next: ChecklistId[]) => setStagesQ(next.length === ALL_CHOICES.length ? [] : next);

  const start = async () => {
    setStarting(true);
    try {
      const r = await launchRun(pid, { ...bodies[mode], ...(label.trim() ? { label: label.trim() } : {}), ...(notes.trim() ? { notes: notes.trim() } : {}), ...(then.length ? { then } : {}) });
      const st = useJobs.getState();
      st.upsert(r.job);
      for (const j of r.chain ?? []) st.upsert(j);
      toast("success", `${r.run.label || r.run.id} started`, { href: `/jobs/${r.job.id}`, linkLabel: "Mission Control" });
      navigate(`/jobs/${encodeURIComponent(r.job.id)}`);
    } catch (e) {
      if (e instanceof ApiError && e.code === "preflight_failed") {
        toast("error", "The launch preflight failed", { body: e.message, action: e.action ?? undefined });
        void plan.reload();
      } else toast("error", "Could not start the run", { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
    } finally {
      setStarting(false);
    }
  };

  const proj = project.data?.project;
  const modeDesc = (m: RunMode) => meta.data?.modes.find((x) => x.id === m)?.desc ?? MODE_TEXT[m].desc;

  return (
    <div className="stack launch-page">
      <header className="page-head">
        <p className="eyebrow">Launch</p>
        <div className="row">
          <h1>{proj ? `Launch a run of ${proj.name}` : "Launch a run"}</h1>
          {proj?.demo ? <DemoBadge /> : null}
          {configVersion !== null ? <Badge title="Config version this plan uses">config v{configVersion}</Badge> : null}
          <span className="spacer" />
          <Link className="btn small ghost" to={`/p/${encodeURIComponent(pid)}/setup/data`}>
            Setup
          </Link>
          <Link className="btn small ghost" to={`/p/${encodeURIComponent(pid)}/config`}>
            Config
          </Link>
        </div>
      </header>

      {prefill.applied ? (
        <div className="callout" data-tone="info" role="status" data-prefill={prefill.applied.rid}>
          {prefill.applied.snapshot ? (
            <>
              Prefilled from run <Link to={`/r/${encodeURIComponent(prefill.applied.rid)}`}>{prefill.applied.label}</Link>: its mode, stages, CV curve and threads. Enable what it skipped, then start a new run.
            </>
          ) : (
            <>
              Run <Link to={`/r/${encodeURIComponent(prefill.applied.rid)}`}>{prefill.applied.label}</Link> has no launch snapshot to prefill from; the choices below are the defaults.
            </>
          )}
        </div>
      ) : prefill.error ? (
        <div className="callout" data-tone="warn" role="alert">
          Could not read run {prefill.from} to prefill from: {errorMessage(prefill.error)}
        </div>
      ) : prefill.loading ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Reading run {prefill.from}…
        </p>
      ) : null}

      <section className="mode-cards" aria-label="Run mode">
        {MODES.map((m) => (
          <ModeCard key={m} mode={m} plan={plans[m]} selected={mode === m} onSelect={() => setMode(m)} coarseM={coarseM} onCoarse={(v) => setCoarseQ(v)} desc={modeDesc(m)} />
        ))}
      </section>

      <div className="launch-grid">
        <div className="stack">
          <Card title="Stages" eyebrow="What runs">
            <ul className="stage-list" aria-label="Stages">
              {rows.map((r) => (
                <li key={r.id} data-stage={r.id} data-locked={r.locked || undefined}>
                  <label>
                    <input type="checkbox" checked={r.checked} disabled={r.locked} onChange={(e) => setSelection(toggleStage(selection, r.id, e.target.checked))} />
                    <span className="stage-text">
                      <b>{r.label}</b>
                      <span className="cap">{r.desc}</span>
                    </span>
                    {r.reason ? <span className="cap stage-reason">({r.reason})</span> : null}
                  </label>
                </li>
              ))}
            </ul>
            <p className="cap">Baselines and the skill-vs-distance curve run with S2–S3; the climate stage runs with S5 when enabled.</p>
          </Card>
          <Card title="Options" eyebrow="How it runs">
            <div className="field">
              <span className="field-label">Skill-vs-distance curve</span>
              <Seg
                size="small"
                label="CV curve"
                value={cv}
                options={[
                  { value: "config", label: "as configured" },
                  { value: "on", label: "on" },
                  { value: "off", label: "off" },
                ]}
                onChange={setCv}
              />
              <span className="hint">Re-fits the models on smaller blocks: adds about as much time again per partition.</span>
            </div>
            <div className="field">
              <span className="field-label">Threads</span>
              <NumberField
                label="Threads"
                value={threads ?? p?.threads ?? null}
                onChange={setThreads}
                min={1}
                max={settings.data?.thread_budget ?? undefined}
                step={1}
                unit={settings.data ? `of ${settings.data.thread_budget}` : null}
              />
            </div>
            <div className="field">
              <span className="field-label">Then run</span>
              <Chips
                label="Then run"
                items={THEN_ACTIONS.map((t) => ({ value: t, label: THEN_LABEL[t] }))}
                selected={then}
                onToggle={(t, on) => setThen(on ? THEN_ACTIONS.filter((x) => x === t || then.includes(x)) : then.filter((x) => x !== t))}
              />
              <span className="hint">Queued after the run succeeds, in this order.</span>
            </div>
            <div className="grid2">
              <label className="field">
                <span className="field-label">Label</span>
                <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. first fast run" aria-label="Run label" />
              </label>
              <label className="field">
                <span className="field-label">Notes</span>
                <input value={notes} onChange={(e) => setNotes(e.target.value)} aria-label="Run notes" />
              </label>
            </div>
          </Card>
        </div>
        <div className="stack">
          <Card title="Plan" eyebrow={`${MODE_TEXT[mode].label} · what will run`}>
            {plan.error ? (
              <EmptyState error={plan.error} title="Could not plan this run" />
            ) : p ? (
              <PlanGraph nodes={p.nodes} total={{ lo: p.est_lo, hi: p.est_hi }} />
            ) : (
              <p className="cap" role="status">
                <span className="spinner" aria-hidden="true" /> Planning…
              </p>
            )}
            {p?.resumable?.resumable ? (
              <div className="callout" data-tone="info">
                A checkpoint of an earlier run could be reused: {p.resumable.reuses.join(", ")}
                {p.resumable.saves_s ? ` (saves ≈${fmtDuration(p.resumable.saves_s)})` : ""}.
              </div>
            ) : null}
          </Card>
          <Card title="Preflight" eyebrow="Checks before start">
            {p ? <Preflight plan={p} onAction={() => void plan.reload()} /> : null}
            {configErrors.length ? (
              <div className="callout" data-tone="crit" role="alert">
                The config has {configErrors.length} {configErrors.length === 1 ? "error" : "errors"}: {configErrors.slice(0, 3).map((i) => `${i.path}: ${i.message}`).join("; ")}.{" "}
                <Link to={`/p/${encodeURIComponent(pid)}/setup/data`}>Fix them in Setup</Link>.
              </div>
            ) : null}
            {p?.network_hosts.length ? <p className="cap">This run contacts {p.network_hosts.join(", ")}.</p> : null}
          </Card>
          <div className="start-bar">
            <div className="stack" style={{ gap: 2 }}>
              <strong>{p ? `≈${fmtDurationRange(p.est_lo, p.est_hi)}` : "—"}</strong>
              <span className="cap">
                {p ? `peak RAM ≈${fmtNum(p.est_peak_rss_gb, 1)} GB · checkpoint ≈${fmtBytes(p.est_disk_gb * 1e9)}` : ""}
              </span>
            </div>
            <span className="spacer" />
            {!canStart && p && (blocking.length || configErrors.length) ? (
              <span className="cap" role="status">
                {blocking.length ? `${blocking.length} preflight ${blocking.length === 1 ? "check blocks" : "checks block"} the launch` : "Fix the config errors to start"}
              </span>
            ) : null}
            <Button variant="primary" icon="play" busy={starting} disabled={!canStart} onClick={() => void start()} data-start="true">
              Start {MODE_TEXT[mode].label.toLowerCase()} run
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
