// Compare runs (SPEC §6.8, `/p/:pid/compare?a=&b=`): provenance chips (same data, config,
// code, grid), the config diff tree, metrics with deltas, stage timings side by side,
// scenario effects with likely ranges, climate and causal summaries, the environment diff,
// outputs only in A or B; and, when the grids match, a difference map for any common layer
// plus Kendall τ / top-decile Jaccard of the priority (footprint) layers. When the grids
// differ (or the server answers 409 grid_mismatch) the difference tools are disabled with
// the reason.
import { useMemo, useState } from "react";
import { ApiError } from "../../api/client";
import { useResource } from "../../api/resource";
import type { CompareRuns } from "../../api/runs";
import { getCompareLayer, postPriority, useCompareRuns, useGridMeta, useProjectRuns } from "../../api/runs";
import type { LayerGroup, LayerMeta, RunSummary } from "../../api/types";
import { Bars, DotRange, type DotRangeRow } from "../../charts";
import { Badge, modeLabel } from "../../components/ui/Badge";
import { EmptyState } from "../../components/ui/EmptyState";
import { Pill } from "../../components/ui/Pill";
import { Table } from "../../components/ui/Table";
import { MapView } from "../../map/MapView";
import { runLayerLoader, useRunGrid, useRunLayers } from "../../map/data";
import { Link, codecs, useRoute, useUrlState } from "../../router";
import { fmtDateTime, fmtNum, fmtSigned, unitLabel } from "../../theme/format";
import { Block } from "./common";
import { agreementWords, cellText, humanize, likelyText } from "./format";

const runName = (r: RunSummary) => `${r.label || r.id} · ${modeLabel(r.mode, r.coarse_m)} · ${fmtDateTime(r.created_utc)}`;

function SameChip({ label, v }: { label: string; v: boolean | null }) {
  if (v === null) return <Pill title="Not recorded for one of the runs">{label}: unknown</Pill>;
  return (
    <Pill tone={v ? "good" : "warn"} icon={v ? "check" : "alert"}>
      {v ? `same ${label}` : `different ${label}`}
    </Pill>
  );
}

/** Group config differences by their top-level section (the diff tree). */
export function configTree(diff: CompareRuns["config_diff"]): { section: string; rows: CompareRuns["config_diff"] }[] {
  const m = new Map<string, CompareRuns["config_diff"]>();
  for (const d of diff) {
    const section = d.path.split(/[.[]/)[0] || d.path;
    m.set(section, [...(m.get(section) ?? []), d]);
  }
  return [...m.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([section, rows]) => ({ section, rows }));
}

const show = (v: unknown) => (v === undefined || v === null ? "—" : typeof v === "string" ? v : JSON.stringify(v));

/** Scalar leaves of a nested summary object ("a.b.c" → value), for the climate/causal blocks. */
export function flattenScalars(o: unknown, prefix = "", out: [string, string | number | boolean | null][] = [], depth = 0): [string, string | number | boolean | null][] {
  if (o === null || typeof o !== "object") {
    out.push([prefix, o as string | number | boolean | null]);
    return out;
  }
  if (depth > 4 || out.length > 80) return out;
  for (const [k, v] of Object.entries(o as Record<string, unknown>)) {
    const p = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object") flattenScalars(v, p, out, depth + 1);
    else out.push([p, v as string | number | boolean | null]);
  }
  return out;
}

function Summary({ title, a, b }: { title: string; a: unknown; b: unknown }) {
  // Server summaries are {a: …, b: …} objects or a single object with per-run values.
  const left = flattenScalars(a);
  const right = new Map(flattenScalars(b));
  const keys = [...new Set([...left.map(([k]) => k), ...right.keys()])];
  const lmap = new Map(left);
  if (!keys.length) return null;
  return (
    <Block title={title}>
      <Table
        caption={title}
        csvName={title}
        rowKey={(r) => r}
        columns={[
          { key: "k", label: "Quantity", value: (k) => k, render: (k) => <span className="mono cap">{k}</span> },
          { key: "a", label: "Run A", value: (k) => (lmap.get(k) ?? null) as string | number | null, render: (k) => cellText(lmap.get(k)) },
          { key: "b", label: "Run B", value: (k) => (right.get(k) ?? null) as string | number | null, render: (k) => cellText(right.get(k)) },
        ]}
        rows={keys}
      />
    </Block>
  );
}

function splitSummary(s: Record<string, unknown> | null): [unknown, unknown] {
  if (!s) return [null, null];
  if ("a" in s || "b" in s) return [s.a ?? null, s.b ?? null];
  return [s, null];
}

function ScenarioEffects({ rows, unit }: { rows: CompareRuns["scenarios"]; unit: string }) {
  const dot: DotRangeRow[] = [];
  for (const r of rows) {
    if (r.a) dot.push({ id: `${r.name}:a`, label: `${r.name} · A`, est: r.a.estimate, lo: r.a.lo, hi: r.a.hi });
    if (r.b) dot.push({ id: `${r.name}:b`, label: `${r.name} · B`, est: r.b.estimate, lo: r.b.lo, hi: r.b.hi, muted: true });
  }
  if (!dot.length) return <p className="cap">No configured scenarios in common.</p>;
  const both = rows.filter((r) => r.a && r.b);
  const moved = both.filter((r) => r.a && r.b && r.a.lo !== null && r.a.hi !== null && (r.b.estimate < r.a.lo || r.b.estimate > r.a.hi));
  return (
    <DotRange
      title="Scenario effects in both runs"
      units={`${unit} (negative = cooler)`}
      rows={dot}
      valueLabel="City-mean ΔT"
      unit={unit}
      signed
      caption={`${both.length} scenarios in both runs; ${moved.length ? `B's estimate falls outside A's likely range for ${moved.map((r) => r.name).join(", ")}` : "every B estimate lies inside A's likely range"}. Grey rows are run B.`}
    />
  );
}

function prefixed(m: LayerMeta, p: "a" | "b" | "diff", label: string): LayerMeta {
  return { ...m, key: `${p}:${m.key}`, label };
}

/** Difference map (B − A) for a common layer, with swipe between A and B. */
function DifferenceMap({ a, b, sameGrid }: { a: string; b: string; sameGrid: boolean }) {
  const la = useRunLayers(sameGrid ? a : null);
  const lb = useRunLayers(sameGrid ? b : null);
  const grid = useRunGrid(sameGrid ? a : null);
  const common = useMemo(() => {
    const keysB = new Set((lb.data ?? []).flatMap((g) => g.layers.map((l) => l.key)));
    return (la.data ?? []).flatMap((g) => g.layers.filter((l) => keysB.has(l.key) && l.scale !== "cat"));
  }, [la.data, lb.data]);
  const [key, setKey] = useState<string | null>(null);
  const cur = common.find((l) => l.key === key) ?? common.find((l) => l.key === "pred") ?? common[0] ?? null;
  const diff = useResource<Float32Array>(sameGrid && cur ? `compare:${a}:${b}:layer:${cur.key}` : null, (s) => getCompareLayer(a, b, cur!.key, s), {
    tags: [`run:${a}`, `run:${b}`],
  });
  const mismatch = diff.error instanceof ApiError && diff.error.code === "grid_mismatch" ? diff.error : null;
  const metaB = useMemo(() => (lb.data ?? []).flatMap((g) => g.layers).find((l) => l.key === cur?.key) ?? null, [lb.data, cur]);
  const groups = useMemo<LayerGroup[]>(() => {
    if (!cur || !metaB) return [];
    const d: LayerMeta = {
      ...prefixed(cur, "diff", `${cur.label}: B − A`),
      scale: "div",
      center: 0,
      zero_blank: false,
      labels: null,
      sign_note: "B − A",
      stats: { n: cur.stats.n, lo: null, hi: null, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null },
    };
    return [
      { id: "diff", label: "Difference", layers: [d] },
      { id: "runs", label: "Each run", layers: [prefixed(cur, "a", `${cur.label} (A)`), prefixed(metaB, "b", `${metaB.label} (B)`)] },
    ];
  }, [cur, metaB]);
  const loadA = useMemo(() => (grid.data ? runLayerLoader(a, grid.data.meta.etag) : null), [a, grid.data]);
  const loadB = useMemo(() => runLayerLoader(b), [b]);
  const load = useMemo(
    () => async (m: LayerMeta) => {
      const [p, k] = [m.key.slice(0, m.key.indexOf(":")), m.key.slice(m.key.indexOf(":") + 1)];
      if (p === "diff") {
        if (diff.data) return diff.data;
        return getCompareLayer(a, b, k);
      }
      if (p === "a" && loadA && cur) return loadA({ ...m, key: k });
      if (p === "b" && metaB) return loadB({ ...metaB });
      throw new Error(`Unknown layer ${m.key}`);
    },
    [a, b, diff.data, loadA, loadB, cur, metaB],
  );
  const [layerKey, setLayerKey] = useState<string | null>(null);

  if (!sameGrid)
    return (
      <div className="callout" role="note" data-disabled="difference-map">
        <strong>Difference map unavailable:</strong> the two runs have different grids (cell size or cell ids differ), so a cell-by-cell difference and priority agreement are not
        defined. Compare their maps side by side from each run's Map tab instead.
      </div>
    );
  if (mismatch)
    return (
      <div className="callout" role="note" data-disabled="difference-map">
        <strong>Difference map unavailable:</strong> {mismatch.message || "the runs' grids do not match"} (grid_mismatch).
      </div>
    );
  if (la.error || lb.error || grid.error) return <EmptyState error={la.error ?? lb.error ?? grid.error} />;
  if (!la.data || !lb.data || !grid.data) return <p className="cap">Loading the layers of both runs…</p>;
  if (!common.length || !cur) return <p className="cap">The runs share no numeric layer.</p>;
  if (diff.error) return <EmptyState error={diff.error} />;
  if (!diff.data) return <p className="cap">Computing the difference…</p>;
  const shown = layerKey && groups.some((g) => g.layers.some((l) => l.key === layerKey)) ? layerKey : groups[0].layers[0].key;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <label className="row cap">
        Layer
        <select value={cur.key} onChange={(e) => setKey(e.target.value)} aria-label="Common layer">
          {common.map((l) => (
            <option key={l.key} value={l.key}>
              {l.label}
            </option>
          ))}
        </select>
      </label>
      <MapView grid={grid.data} groups={groups} loadLayer={load} layerKey={shown} onLayerChange={setLayerKey} allowDiff height={460} title="Run comparison" />
    </div>
  );
}

function PriorityAgreement({ a, b, sameGrid, layers }: { a: string; b: string; sameGrid: boolean; layers: string[] }) {
  const res = useResource(
    sameGrid && layers.length ? `compare:${a}:${b}:priority:${layers.join(",")}` : null,
    async () => Promise.all(layers.map(async (l) => ({ layer: l, r: await postPriority(a, b, l) }))),
    { tags: [`run:${a}`, `run:${b}`] },
  );
  if (!sameGrid)
    return (
      <p className="cap" data-disabled="priority">
        Priority agreement needs the same grid.
      </p>
    );
  if (!layers.length) return <p className="cap">No footprint (priority) layers in common.</p>;
  if (res.error instanceof ApiError && res.error.code === "grid_mismatch")
    return (
      <p className="cap" data-disabled="priority">
        Priority agreement unavailable: {res.error.message} (grid_mismatch).
      </p>
    );
  if (res.error) return <EmptyState error={res.error} />;
  if (!res.data) return <p className="cap">Ranking cells in both runs…</p>;
  return (
    <Table
      caption="Priority agreement"
      csvName="priority-agreement"
      rowKey={(r) => r.layer}
      columns={[
        { key: "layer", label: "Priority layer", value: (r) => r.layer, render: (r) => <span className="mono">{r.layer}</span> },
        { key: "tau", label: "Kendall τ", align: "right", value: (r) => r.r.kendall_tau, render: (r) => fmtSigned(r.r.kendall_tau, 2) },
        { key: "j", label: "Top-decile Jaccard", align: "right", value: (r) => r.r.top_decile_jaccard, render: (r) => fmtNum(r.r.top_decile_jaccard, 2) },
        { key: "n", label: "Cells", align: "right", value: (r) => r.r.n },
        { key: "w", label: "Reading", value: (r) => agreementWords(r.r.kendall_tau) },
      ]}
      rows={res.data}
    />
  );
}

export default function Compare() {
  const { params } = useRoute();
  const pid = params.pid ?? null;
  const runs = useProjectRuns(pid);
  const [a, setA] = useUrlState("a", codecs.optString());
  const [b, setB] = useUrlState("b", codecs.optString());
  const cmp = useCompareRuns(a, b);
  const la = useRunLayers(cmp.data?.same.grid ? a : null);
  const lb = useRunLayers(cmp.data?.same.grid ? b : null);
  const priority = useMemo(() => {
    const kb = new Set((lb.data ?? []).flatMap((g) => g.layers.map((l) => l.key)));
    return (la.data ?? []).flatMap((g) => g.layers.map((l) => l.key)).filter((k) => k.startsWith("fp_") && kb.has(k));
  }, [la.data, lb.data]);
  const items = runs.data?.items ?? [];
  const c = cmp.data;
  const meta = useGridMeta(a);
  const unit = unitLabel(meta.data?.units.target ?? "");
  const picker = (label: string, value: string | null, set: (v: string | null) => void) => (
    <label className="row cap">
      {label}
      <select value={value ?? ""} onChange={(e) => set(e.target.value || null)} aria-label={`Run ${label}`}>
        <option value="">Choose a run…</option>
        {items.map((r) => (
          <option key={r.id} value={r.id}>
            {runName(r)}
          </option>
        ))}
      </select>
    </label>
  );
  return (
    <section className="stack" aria-labelledby="compare-title">
      <header className="stack" style={{ gap: 6 }}>
        <h2 id="compare-title" style={{ margin: 0 }}>
          Compare runs
        </h2>
        <div className="row">
          {picker("A", a, setA)}
          {picker("B", b, setB)}
          <button
            type="button"
            className="btn small ghost"
            disabled={!a || !b}
            onClick={() => {
              const [x, y] = [a, b];
              setA(y);
              setB(x);
            }}
          >
            Swap
          </button>
        </div>
      </header>
      {!a || !b ? <p className="cap">Choose two runs of this project to compare (or select two on the Runs page).</p> : null}
      {cmp.error ? <EmptyState error={cmp.error} /> : null}
      {a && b && !c && !cmp.error ? <p className="cap">Comparing…</p> : null}
      {c ? (
        <>
          <div className="row" aria-label="Provenance">
            <SameChip label="data" v={c.same.data} />
            <SameChip label="config" v={c.same.config} />
            <SameChip label="code" v={c.same.code} />
            <SameChip label="grid" v={c.same.grid} />
          </div>
          <div className="grid2">
            {[c.a, c.b].map((r, i) => (
              <Block key={r.id} title={`${i ? "B" : "A"}: ${r.label || r.id}`} actions={<Link to={`/r/${encodeURIComponent(r.id)}`}>Open</Link>}>
                <p className="cap">
                  <Badge tone="accent">{modeLabel(r.mode, r.coarse_m)}</Badge> {r.status} · {fmtDateTime(r.created_utc)} · {r.n_points ?? "?"} cells{" "}
                  {r.demo ? <Badge tone="demo">DEMO</Badge> : null}
                </p>
              </Block>
            ))}
          </div>
          <Block title="Config differences">
            {c.config_diff.length ? (
              <div className="stack" style={{ gap: 4 }} aria-label="Config diff tree">
                {configTree(c.config_diff).map((sec) => (
                  <details key={sec.section} open={c.config_diff.length <= 12}>
                    <summary>
                      <strong>{sec.section}</strong> <span className="cap">({sec.rows.length})</span>
                    </summary>
                    <Table
                      caption={`Config differences in ${sec.section}`}
                      csvName={null}
                      rowKey={(r) => r.path}
                      columns={[
                        { key: "path", label: "Setting", value: (r) => r.path, render: (r) => <span className="mono">{r.path}</span> },
                        { key: "a", label: "A", value: (r) => show(r.a) },
                        { key: "b", label: "B", value: (r) => show(r.b) },
                      ]}
                      rows={sec.rows}
                    />
                  </details>
                ))}
              </div>
            ) : (
              <p className="cap">Same effective configuration (defaults hidden).</p>
            )}
          </Block>
          <div className="grid2">
            <Block title="Metrics">
              <Table
                caption="Metrics"
                csvName="compare-metrics"
                rowKey={(r) => r.key}
                columns={[
                  { key: "key", label: "Metric", value: (r) => humanize(r.key) },
                  { key: "a", label: "A", align: "right", value: (r) => r.a, render: (r) => fmtNum(r.a, 3) },
                  { key: "b", label: "B", align: "right", value: (r) => r.b, render: (r) => fmtNum(r.b, 3) },
                  { key: "d", label: "B − A", align: "right", value: (r) => r.delta, render: (r) => fmtSigned(r.delta, 3) },
                ]}
                rows={c.metrics}
              />
            </Block>
            <Bars
              title="Stage timings"
              units="seconds"
              categories={c.timings.map((t) => t.stage)}
              orientation="h"
              series={[
                { id: "a", label: "A", values: c.timings.map((t) => t.a) },
                { id: "b", label: "B", values: c.timings.map((t) => t.b) },
              ]}
              valueLabel="Time"
              unit="s"
              decimals={1}
              caption={(() => {
                const ta = c.timings.reduce((s, t) => s + (t.a ?? 0), 0);
                const tb = c.timings.reduce((s, t) => s + (t.b ?? 0), 0);
                return `A took ${fmtNum(ta, 0)} s in total, B ${fmtNum(tb, 0)} s.`;
              })()}
            />
          </div>
          <ScenarioEffects rows={c.scenarios} unit={unit} />
          {c.scenarios.length ? (
            <Block title="Scenario effects">
              <Table
                caption="Scenario effects"
                csvName="compare-scenarios"
                rowKey={(r) => r.name}
                columns={[
                  { key: "n", label: "Scenario", value: (r) => r.name },
                  { key: "a", label: "A", value: (r) => r.a?.estimate ?? null, render: (r) => likelyText(r.a, unit) },
                  { key: "b", label: "B", value: (r) => r.b?.estimate ?? null, render: (r) => likelyText(r.b, unit) },
                ]}
                rows={c.scenarios}
              />
            </Block>
          ) : null}
          <div className="grid2">
            {(() => {
              const [ca, cb] = splitSummary(c.climate);
              return ca || cb ? (
                <Summary title="Climate" a={ca} b={cb} />
              ) : (
                <Block title="Climate">
                  <p className="cap">No climate stage in either run.</p>
                </Block>
              );
            })()}
            {(() => {
              const [ca, cb] = splitSummary(c.causal);
              return ca || cb ? (
                <Summary title="Causal" a={ca} b={cb} />
              ) : (
                <Block title="Causal">
                  <p className="cap">No causal stage in either run.</p>
                </Block>
              );
            })()}
          </div>
          <Block title="Environment">
            {!c.environment.added.length && !c.environment.removed.length && !c.environment.changed.length ? (
              <p className="cap">Identical package lists.</p>
            ) : (
              <div className="stack" style={{ gap: 6 }}>
                {c.environment.changed.length ? (
                  <Table
                    caption="Changed packages"
                    csvName="compare-environment"
                    rowKey={(r) => r.name}
                    columns={[
                      { key: "n", label: "Package", value: (r) => r.name },
                      { key: "a", label: "A", value: (r) => r.a },
                      { key: "b", label: "B", value: (r) => r.b },
                    ]}
                    rows={c.environment.changed}
                  />
                ) : null}
                {c.environment.added.length ? <p className="cap">Only in B: {c.environment.added.join(", ")}</p> : null}
                {c.environment.removed.length ? <p className="cap">Only in A: {c.environment.removed.join(", ")}</p> : null}
              </div>
            )}
          </Block>
          <Block title="Outputs">
            <p className="cap">Only in A: {c.outputs.a_only.join(", ") || "none"}</p>
            <p className="cap">Only in B: {c.outputs.b_only.join(", ") || "none"}</p>
          </Block>
          <Block title="Difference map">
            <DifferenceMap a={c.a.id} b={c.b.id} sameGrid={c.same.grid} />
          </Block>
          <Block title="Priority agreement">
            <PriorityAgreement a={c.a.id} b={c.b.id} sameGrid={c.same.grid} layers={priority} />
          </Block>
        </>
      ) : null}
    </section>
  );
}
