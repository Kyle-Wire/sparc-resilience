// Study child matrices above the generic tracker (SPEC §5.11): placebo columns (mini rail, child
// run link, verdict chip), multiverse rows (status, R², seconds, scenario × variant heatmap and
// the priority-stability τ / Jaccard chart), the simcheck generator × seed grid (share coloured
// around 1, pending grey, error red, a dot for a gate redraw; per-generator strip with IQR; ETA =
// mean seconds × remaining ÷ workers) and the reproduce child rail with its checklist. Live data
// comes from the job's events (nested runs and replicate tasks); the study view adds verdicts,
// checks and the priority agreement the events do not carry.
import { useResource } from "../../../api/resource";
import { getStudyView, type StudyView } from "../../../api/tracking";
import type { Job } from "../../../api/types";
import { Bars, BoxStrip, Heatmap } from "../../../charts";
import { Card } from "../../../components/ui/Card";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link } from "../../../router";
import { useDark } from "../../../stores/ui";
import type { TrackerState } from "../../../stores/tracker";
import { fmtDuration, fmtNum, fmtPct } from "../../../theme/format";
import { rampColor } from "../../../theme/palette";
import { childCells, priorityStability, railItems, simcheckGrid, type ChildCell } from "../model";
import { StageRail } from "./StageRail";

function childStatus(c: ChildCell | undefined, viewStatus?: string | null): string {
  if (c) return c.status === "succeeded" ? "done" : c.status;
  return viewStatus ?? "planned";
}

function verdictChip(verdict: string | null | undefined) {
  if (!verdict) return <StatusChip status="planned" text="verdict pending" />;
  const v = verdict.toLowerCase();
  const status = v.includes("fail") || v.includes("leak") || v.includes("spurious") ? "failed" : v.includes("pass") || v.includes("ok") || v.includes("clean") ? "done" : "stale";
  return <StatusChip status={status} text={verdict} />;
}

function MiniRail({ state, childKey }: { state: TrackerState; childKey: string }) {
  const child = state.children[childKey];
  if (!child?.state.stages) return null;
  return <StageRail mini items={railItems(child.state)} label={`${child.label} stages`} />;
}

function Placebo({ job, state, view }: { job: Job; state: TrackerState; view: StudyView | undefined }) {
  const cells = childCells(state);
  const kinds = (Array.isArray(job.params.kinds) ? (job.params.kinds as string[]) : null) ?? ["grf", "shift", "rotate"];
  return (
    <div className="matrix-cols">
      {kinds.map((k) => {
        const c = cells.find((x) => x.key === `placebo:${k}`);
        const v = view?.children?.find((x) => x.kind === k);
        const rid = c?.run_id ?? v?.run_id ?? null;
        return (
          <Card key={k} variant="tight" title={`Placebo: ${k}`} actions={<StatusChip status={childStatus(c, v?.status)} meta={c?.progress !== null && c?.progress !== undefined && c.status === "running" ? fmtPct(c.progress) : undefined} />}>
            <MiniRail state={state} childKey={`placebo:${k}`} />
            <div className="row">
              {verdictChip(v?.verdict)}
              {rid ? <Link to={`/r/${encodeURIComponent(rid)}`}>Child run</Link> : null}
            </div>
          </Card>
        );
      })}
    </div>
  );
}

function Multiverse({ job, state, view }: { job: Job; state: TrackerState; view: StudyView | undefined }) {
  const cells = childCells(state).filter((c) => c.key.startsWith("variant:"));
  const names = [...new Set([...(Array.isArray(job.params.variants) ? (job.params.variants as string[]) : []), ...(view?.variants ?? []).map((v) => v.name), ...cells.map((c) => c.key.slice(8))])];
  const spans = Object.values(state.spans).filter((sp) => sp.name === "variant");
  const rows = names.map((name) => {
    const c = cells.find((x) => x.key === `variant:${name}`);
    const v = view?.variants?.find((x) => x.name === name);
    const sp = spans.find((x) => x.key === name);
    const r2 = typeof c?.metrics.stacker_r2 === "number" ? (c.metrics.stacker_r2 as number) : v?.r2 ?? null;
    const status = sp ? (sp.status === "ok" ? "done" : sp.status === "error" ? "failed" : sp.status) : childStatus(c, v?.status === "done" ? "done" : v?.status);
    const deltas: Record<string, number> = {};
    for (const [k, val] of Object.entries(c?.metrics ?? {})) {
      const m = /^scenario\.mean_delta\{scenario=(.*)\}$/.exec(k);
      if (m && typeof val === "number") deltas[m[1]] = val;
    }
    return { name, label: v?.label ?? name, status, r2, seconds: sp?.elapsed_s ?? v?.seconds ?? null, run_id: c?.run_id ?? v?.run_id ?? null, deltas };
  });
  const scenarios = [...new Set(rows.flatMap((r) => Object.keys(r.deltas)))];
  const stability = priorityStability(view?.priority);
  const labelOf = (name: string) => rows.find((r) => r.name === name)?.label ?? name;
  const stable = stability.filter((r) => r.tau !== null && r.jaccard !== null && r.tau >= 0.6 && r.jaccard >= 0.6).length;
  return (
    <div className="stack">
      <div className="tablewrap">
        <table className="checklist" aria-label="Multiverse variants">
          <thead>
            <tr>
              <th scope="col">Variant</th>
              <th scope="col">Status</th>
              <th scope="col" className="r">
                R²
              </th>
              <th scope="col" className="r">
                Time
              </th>
              <th scope="col">Run</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td>{r.label}</td>
                <td>
                  <StatusChip status={r.status} />
                </td>
                <td className="num r">{fmtNum(r.r2, 3)}</td>
                <td className="num r">{fmtDuration(r.seconds)}</td>
                <td>{r.run_id ? <Link to={`/r/${encodeURIComponent(r.run_id)}`}>open</Link> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {scenarios.length ? (
        <Heatmap
          title="Scenario effect by variant"
          caption="Mean Δ per configured scenario; cells fill in as variants finish"
          rows={rows.map((r) => r.label)}
          cols={scenarios}
          values={rows.map((r) => scenarios.map((s) => r.deltas[s] ?? null))}
          scale="div"
          center={0}
          valueLabel="Mean Δ"
          rowLabel="Variant"
          colLabel="Scenario"
          pin={false}
        />
      ) : null}
      {stability.length ? (
        <Bars
          title="Priority stability by variant"
          caption={`${stable} of ${stability.length} finished variant${stability.length === 1 ? "" : "s"} keep the baseline's priority map (τ and overlap both at least 0.6); median over levers`}
          categories={stability.map((r) => labelOf(r.variant))}
          series={[
            { id: "tau", label: "Kendall's τ vs baseline", values: stability.map((r) => r.tau) },
            { id: "jaccard", label: "Top-decile overlap (Jaccard)", values: stability.map((r) => r.jaccard) },
          ]}
          orientation="h"
          mode="grouped"
          valueLabel="Agreement with the baseline map"
          categoryLabel="Variant"
          decimals={2}
          domain={[Math.min(0, ...stability.map((r) => r.tau ?? 0)), 1]}
          pin={false}
        />
      ) : null}
    </div>
  );
}

function quantile(xs: number[], q: number): number {
  const v = [...xs].sort((a, b) => a - b);
  const pos = q * (v.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, v.length - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

function Simcheck({ job, state, view }: { job: Job; state: TrackerState; view: StudyView | undefined }) {
  const dark = useDark();
  const grid = simcheckGrid(state, job, view ?? null);
  const done = [...grid.cells.values()].filter((c) => c.status === "done");
  const total = grid.generators.reduce((a, g) => a + grid.seeds.filter((s) => grid.cells.has(`${g}/${s}`) || s < Number((job.params.design as Record<string, number> | undefined)?.[g] ?? 0)).length, 0);
  const secs = done.map((c) => c.seconds).filter((v): v is number => v !== null);
  const workers = Math.max(1, Number(job.params.workers ?? 1));
  const eta = view?.eta_s ?? (secs.length ? ((secs.reduce((a, b) => a + b, 0) / secs.length) * Math.max(0, total - done.length)) / workers : null);
  const groups = grid.generators.map((g) => {
    const xs = done.filter((c) => c.generator === g && c.share !== null).map((c) => c.share as number);
    return { label: g, n: xs.length, points: xs, q: xs.length ? ([Math.min(...xs), quantile(xs, 0.25), quantile(xs, 0.5), quantile(xs, 0.75), Math.max(...xs)] as [number, number, number, number, number]) : null };
  });
  const lo = Math.min(0, ...done.map((c) => c.share ?? 1));
  const hi = Math.max(2, ...done.map((c) => c.share ?? 1));
  const span = Math.max(1 - lo, hi - 1) || 1;
  return (
    <div className="stack">
      <div className="mc-facts">
        <span>
          Replicates <b>{done.length}</b> of <b>{total}</b>
        </span>
        <span>
          ETA <b>{eta !== null ? `≈${fmtDuration(eta)}` : "—"}</b> ({workers} worker{workers === 1 ? "" : "s"})
        </span>
      </div>
      <div className="tablewrap">
        <table className="simgrid" aria-label="Generator by seed grid (effect share; 1 = recovered exactly)">
          <tbody>
            {grid.generators.map((g) => (
              <tr key={g}>
                <th scope="row">{g}</th>
                {grid.seeds.map((s) => {
                  const c = grid.cells.get(`${g}/${s}`);
                  const status = c?.status ?? "pending";
                  const t = c?.share !== null && c?.share !== undefined ? Math.min(1, Math.max(0, 0.5 + (0.5 * (c.share - 1)) / span)) : null;
                  const fill = status === "done" && t !== null ? rampColor("div", t, dark) : undefined;
                  const label = `${g}, seed ${s}: ${status === "done" ? `share ${fmtNum(c?.share ?? null, 2)}` : status.replace("_", " ")}${c?.redraw ? ", gate redraw" : ""}`;
                  return (
                    <td key={s} data-status={status} style={fill ? { background: fill } : undefined} title={label} aria-label={label}>
                      {c?.redraw ? "•" : status === "error" ? "!" : status === "running" ? "…" : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {groups.some((g) => g.n) ? (
        <BoxStrip title="Effect share per generator" caption="Recovered ÷ planted effect per replicate (1 = exact); box = IQR" groups={groups} valueLabel="Effect share" groupLabel="Generator" whiskerLabel="min–max" pin={false} />
      ) : null}
    </div>
  );
}

function Reproduce({ state, view }: { state: TrackerState; view: StudyView | undefined }) {
  const key = Object.keys(state.children)[0];
  const child = key ? state.children[key] : null;
  return (
    <div className="stack">
      {child ? <MiniRail state={state} childKey={key} /> : <p className="mc-note">The reproduction run appears once it starts.</p>}
      {child?.run_id ? <Link to={`/r/${encodeURIComponent(child.run_id)}`}>Reproduction run</Link> : null}
      {view?.checks?.length ? (
        <table className="checklist" aria-label="Reproduction checks">
          <thead>
            <tr>
              <th scope="col">Check</th>
              <th scope="col">Kind</th>
              <th scope="col">Result</th>
              <th scope="col">Detail</th>
            </tr>
          </thead>
          <tbody>
            {view.checks.map((c) => (
              <tr key={c.check}>
                <td>{c.check}</td>
                <td>{c.hard ? "hard" : "soft"}</td>
                <td>
                  <StatusChip status={c.ok ? "done" : c.hard ? "failed" : "stale"} text={c.ok ? "matches" : "differs"} />
                </td>
                <td className="cap">{c.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mc-note">The checklist (cv design, R² per model, scenario effects, inputs, code and versions) fills in when the reproduction finishes.</p>
      )}
    </div>
  );
}

/** The child matrix of a study job, or nothing for other kinds. */
export function ChildMatrix({ job, state }: { job: Job; state: TrackerState }) {
  const sid = job.study_id;
  const view = useResource(sid ? `study:${sid}:view` : null, (s) => getStudyView(sid!, s), { tags: sid ? [`study:${sid}`] : [] });
  const kind = job.kind.replace(/^study\./, "");
  let body;
  if (kind === "placebo") body = <Placebo job={job} state={state} view={view.data} />;
  else if (kind === "multiverse") body = <Multiverse job={job} state={state} view={view.data} />;
  else if (kind === "simcheck") body = <Simcheck job={job} state={state} view={view.data} />;
  else if (kind === "reproduce") body = <Reproduce state={state} view={view.data} />;
  else return null;
  return (
    <Card title={`Study: ${kind}`} eyebrow="Child runs" actions={sid ? <Link to={`/studies/${encodeURIComponent(sid)}`}>Study page</Link> : null}>
      {body}
    </Card>
  );
}
