// Compare (`/r/:rid/lab/compare?items=`, SPEC §7.12): 2–4 of exact results, configured
// scenarios, plans and the baseline. Small multiples of each item's ΔT on one locked
// diverging scale; the A − B difference map with swipe; a per-cell scatter; the KPI table
// with pairwise differences and the uncertainty of each difference (the paired SE; or a "needs a
// full model run" chip); regions, equity, exposure and cooling per cost; the compare pack export.
// Every pair is A − B with A the earlier item (the server's pair `a`), as in its difference layer
// and SE(A − B). Plain words (phase 4): the glossary's terms, with an ⓘ; no native titles.
import { useEffect, useMemo, useState } from "react";
import { errorMessage, getBin } from "../../api/client";
import { mutate } from "../../api/resource";
import {
  createComparison,
  exportPack,
  getResultLayer,
  rerunConfigured,
  useComparison,
  usePlans,
  useRunScenarios,
  useScenarios,
  type Comparison,
  type ItemRef,
  type PairedLikely,
} from "../../api/lab";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { HexbinScatter } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { InfoTip } from "../../components/ui/InfoTip";
import { Table } from "../../components/ui/Table";
import { term } from "../../content/glossary";
import { useRunDetail } from "../../layouts/resources";
import { GridCanvas } from "../../map/GridCanvas";
import { MapView } from "../../map/MapView";
import { quantiles, type Domain } from "../../map/domain";
import { runLayerLoader, useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import type { GridData } from "../../map/grid";
import { useMapView } from "../../map/useMapView";
import { codecs, useRoute, useUrlState } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { fmtNum, fmtPct, fmtSigned, unitLabel } from "../../theme/format";
import { LabFrame } from "./components/LabFrame";
import { Tip } from "./components/Tip";
import { decodeItemRef, encodeItemRef, itemLayerKey, sameItem } from "./model/doc";
import { syntheticMeta } from "./model/layers";
import { TRAY_MAX, trayRefs, useTray } from "./model/tray";

/** "full model" in sentences, and the button that runs it (glossary). */
const FULL = term("exact_engine").label.toLowerCase();
const RUN_FULL = term("run_exact").label;
/** The item groups of the picker. */
const GROUPS = { results: `${term("exact_engine").label} results`, configured: term("configured_scenario").label, plans: "Plans", baseline: "Baseline" } as const;

/** One locked diverging scale for every panel: symmetric about 0 at the largest |2–98%| value. */
export function lockedDomain(arrays: (ArrayLike<number> | null)[]): Domain {
  let span = 0;
  for (const a of arrays) {
    if (!a) continue;
    const [p2, p98] = quantiles(a, [0.02, 0.98]);
    span = Math.max(span, Math.abs(p2 ?? 0), Math.abs(p98 ?? 0));
  }
  span = span || 1e-6;
  return { kind: "div", lo: -span, hi: span, center: 0, mult: 1, zeroBlank: false, nCat: 0 };
}

/** "−0.12 ± 0.03 °F (paired)" */
export function pairText(l: PairedLikely, unit: string): string {
  const se = l.se !== null ? ` ± ${fmtNum(1.96 * l.se, 3)}` : "";
  return `${fmtSigned(l.estimate, 3)}${se} ${unitLabel(unit)}`;
}

function useItemValues(rid: string, grid: GridData | undefined, refs: ItemRef[], catalog: LayerMeta[]) {
  const [values, setValues] = useState<Record<string, Float32Array | null>>({});
  const key = refs.map(encodeItemRef).join(",");
  useEffect(() => {
    if (!grid) return;
    let live = true;
    const load = runLayerLoader(rid, grid.meta.etag);
    Promise.all(
      refs.map(async (r): Promise<[string, Float32Array | null]> => {
        const k = encodeItemRef(r);
        try {
          if (r.kind === "baseline") return [k, new Float32Array(grid.n)];
          if (r.kind === "result") return [k, await getResultLayer(r.id, "delta")];
          const lk = itemLayerKey(r)!;
          const meta = catalog.find((m) => m.key === lk);
          if (meta) return [k, (await load(meta)) as Float32Array];
          return [k, (await getBin<Float32Array>(`/api/runs/${encodeURIComponent(rid)}/layers/${encodeURIComponent(lk)}.bin`, "float32")).data];
        } catch {
          return [k, null];
        }
      }),
    ).then((pairs) => live && setValues(Object.fromEntries(pairs)));
    return () => {
      live = false;
    };
    // `key` stands for refs
  }, [rid, grid, key, catalog]);
  return values;
}

function SmallMaps({ grid, items, values, unit }: { grid: GridData; items: { ref: ItemRef; label: string }[]; values: Record<string, Float32Array | null>; unit: string }) {
  const view = useMapView();
  const dark = useUi((s) => s.dark);
  const domain = useMemo(() => lockedDomain(items.map((i) => values[encodeItemRef(i.ref)] ?? null)), [items, values]);
  return (
    <section className="stack" aria-label="Small multiples">
      <h3>Temperature change for each item (same colour scale)</h3>
      <p className="cap">
        Blue is cooler, red warmer; every panel uses ±{fmtNum(domain.hi, 3)} {unitLabel(unit)}. Panning or zooming one panel moves all of them.
      </p>
      <div className="mini-maps">
        {items.map((i) => {
          const v = values[encodeItemRef(i.ref)];
          return (
            <figure key={encodeItemRef(i.ref)}>
              <GridCanvas grid={grid} values={v ?? null} domain={v ? domain : null} dark={dark} view={view} label={`Temperature change map: ${i.label}`} height={240} />
              <figcaption>{i.label}{v === null ? " (no place-by-place map)" : ""}</figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
}

function ItemPicker({ rid, pid, refs, onChange }: { rid: string; pid: string | null; refs: ItemRef[]; onChange: (r: ItemRef[]) => void }) {
  const runSc = useRunScenarios(rid);
  const plans = usePlans(rid);
  const scenarios = useScenarios(pid, { run: rid }, rid);
  const tray = trayRefs(useTray((s) => s.byRun[rid]));
  const names = useMemo(() => new Map((scenarios.data ?? []).flatMap((s) => (s.latest ? [[s.latest.id, s.name] as const] : []))), [scenarios.data]);
  const options: { ref: ItemRef; label: string; group: string }[] = [
    ...(runSc.data?.results ?? []).filter((r) => r.kind === "exact").map((r) => ({ ref: { kind: "result", id: r.id } as ItemRef, label: names.get(r.id) ?? `Result ${r.id}`, group: GROUPS.results })),
    ...(runSc.data?.configured ?? []).map((c) => ({ ref: { kind: "configured", slug: c.slug } as ItemRef, label: c.label || c.name, group: GROUPS.configured })),
    ...(plans.data ?? []).map((p) => ({ ref: { kind: "plan", id: p.id } as ItemRef, label: p.name, group: GROUPS.plans })),
    { ref: { kind: "baseline" }, label: "Baseline (no change)", group: GROUPS.baseline },
  ];
  const free = options.filter((o) => !refs.some((r) => sameItem(r, o.ref)));
  return (
    <div className="row">
      <select
        aria-label="Add an item"
        value=""
        disabled={refs.length >= TRAY_MAX}
        onChange={(e) => {
          const r = decodeItemRef(e.target.value);
          if (r) onChange([...refs, r]);
        }}
      >
        <option value="">{refs.length >= TRAY_MAX ? "Up to 4 items" : "Add an item…"}</option>
        {[GROUPS.results, GROUPS.configured, GROUPS.plans, GROUPS.baseline].map((g) => (
          <optgroup key={g} label={g}>
            {free
              .filter((o) => o.group === g)
              .map((o) => (
                <option key={encodeItemRef(o.ref)} value={encodeItemRef(o.ref)}>
                  {o.label}
                </option>
              ))}
          </optgroup>
        ))}
      </select>
      {tray.length >= 2 ? (
        <Button size="small" variant="ghost" onClick={() => onChange(tray.map((t) => t.ref))}>
          Use the compare tray ({tray.length})
        </Button>
      ) : null}
    </div>
  );
}

function CompareBody({ rid, pid, cmp, grid, catalog, unit }: { rid: string; pid: string | null; cmp: Comparison; grid: GridData; catalog: LayerMeta[]; unit: string }) {
  const values = useItemValues(rid, grid, cmp.items.map((i) => i.ref), catalog);
  const [pair, setPair] = useState(0);
  const [busy, setBusy] = useState(false);
  const u = unitLabel(unit);
  const p = cmp.pairs[Math.min(pair, cmp.pairs.length - 1)];
  const label = (i: number) => cmp.items[i]?.label ?? `#${i + 1}`;
  const pairLabel = (q: { a: number; b: number }) => `${label(q.a)} − ${label(q.b)}`;
  const groups = useMemo<LayerGroup[]>(() => {
    const diff = cmp.pairs.flatMap((q) => (q.layer_key ? [syntheticMeta({ key: q.layer_key, label: pairLabel(q), unit, sign_note: "A − B; negative = A cooler than B" }, null)] : []));
    const items = cmp.items.map((i) => syntheticMeta({ key: `item:${encodeItemRef(i.ref)}`, label: i.label, unit, sign_note: "negative = cooler" }, null));
    return [
      { id: "diff", label: "Differences", layers: diff },
      { id: "items", label: "Items", layers: items },
    ];
    // labels come from cmp
  }, [cmp, unit]);
  const load = useMemo(
    () =>
      async (m: LayerMeta): Promise<LayerValues> => {
        if (m.key.startsWith("item:")) return values[m.key.slice(5)] ?? new Float32Array(grid.n).fill(NaN);
        return (await getBin<Float32Array>(`/api/runs/${encodeURIComponent(rid)}/layers/${encodeURIComponent(m.key)}.bin`, "float32")).data;
      },
    [values, rid, grid.n],
  );
  const needs = new Set(cmp.needs_exact.map(encodeItemRef));
  const a = p ? values[encodeItemRef(cmp.items[p.a].ref)] : null;
  const b = p ? values[encodeItemRef(cmp.items[p.b].ref)] : null;
  const exportCompare = async () => {
    if (!pid) return;
    setBusy(true);
    try {
      const r = await exportPack(pid, "compare_pack", { comparison_id: cmp.id });
      useJobs.getState().upsert(r.job);
      toast("info", "Building the compare pack", { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", "Could not start the compare pack", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const rerun = async (ref: ItemRef) => {
    if (ref.kind !== "configured") return;
    try {
      const job = await rerunConfigured(rid, ref.slug);
      useJobs.getState().upsert(job);
      toast("info", `Running the ${FULL} again, for the ${term("paired_se").label.toLowerCase()}`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", `Could not start the ${FULL} run`, { body: errorMessage(e) });
    }
  };
  const regionNames = [...new Set(cmp.pairs.flatMap((q) => Object.keys(q.regions)))];
  const planned = cmp.items.filter((i) => i.per_cell === false);
  const equityGroups = Object.entries(cmp.equity);
  return (
    <div className="stack">
      {cmp.needs_exact.length ? (
        <div className="row" role="note">
          <span className="cap">
            The {term("paired_se").label.toLowerCase()} needs the {FULL}&apos;s result for each test area of:
            <InfoTip term="paired_se" />
          </span>
          {cmp.needs_exact.map((r) => {
            const it = cmp.items.find((i) => sameItem(i.ref, r));
            return (
              <span key={encodeItemRef(r)} className="row" style={{ gap: 4 }}>
                <Badge tone="warn">
                  needs a {FULL} run: {it?.label ?? encodeItemRef(r)}
                </Badge>
                {r.kind === "configured" ? (
                  <Button size="small" variant="ghost" onClick={() => void rerun(r)}>
                    {RUN_FULL} again
                  </Button>
                ) : null}
              </span>
            );
          })}
        </div>
      ) : null}
      {planned.length ? (
        <p className="cap" role="note">
          {planned.map((i) => i.label).join(", ")}: not checked yet. A plan&apos;s {term("open_loop").label.toLowerCase()} is the cooling each treated place brings to its whole neighbourhood ({u}·places), not the change at that place,
          so only the citywide average (the total over all places), the cost and the cooling per cost are compared; where the changes are, areas, who benefits, exposure and the difference maps need the plan checked
          with the {FULL}.
          <InfoTip term="closed_loop" />
        </p>
      ) : null}
      <div className="row">
        <span className="spacer" />
        <Button size="small" icon="download" busy={busy} disabled={!pid} onClick={() => void exportCompare()}>
          Compare pack
        </Button>
      </div>
      <Table
        caption="Items"
        csvName="compare-items"
        rowKey={(i) => encodeItemRef(i.ref)}
        columns={[
          { key: "label", label: "Item", value: (i) => i.label },
          { key: "city", label: "Citywide", unit: u, align: "right", value: (i) => i.city.estimate, render: (i) => pairText({ ...i.city, paired: false }, unit) },
          { key: "edited", label: term("where_changed").label, unit: u, align: "right", value: (i) => i.edited?.estimate ?? null, render: (i) => (i.edited ? pairText({ ...i.edited, paired: false }, unit) : "—") },
          { key: "cost", label: "Cost", unit: term("cost_units").label.toLowerCase(), tip: <InfoTip term="cost_units" />, align: "right", value: (i) => i.cost, render: (i) => fmtNum(i.cost, 0) },
          { key: "cpc", label: "Cooling per unit of cost", align: "right", value: (i) => cmp.cooling_per_cost[encodeItemRef(i.ref)] ?? cmp.cooling_per_cost[i.label] ?? null, render: (i) => fmtNum(cmp.cooling_per_cost[encodeItemRef(i.ref)] ?? cmp.cooling_per_cost[i.label] ?? null, 5) },
          { key: "folds", label: "Can be compared", tip: <InfoTip term="paired_se" />, value: (i) => (i.has_folds ? "yes" : "no"), render: (i) => (needs.has(encodeItemRef(i.ref)) || !i.has_folds ? <Badge tone="warn">needs a {FULL} run</Badge> : "yes") },
        ]}
        rows={cmp.items}
      />
      <p className="cap" data-paired-note="true">
        Each difference reads A − B, ± its {term("paired_se").label.toLowerCase()} (95%).
        <InfoTip term="paired_se" />
      </p>
      <Table
        caption="Pairwise differences (A − B)"
        csvName="compare-pairs"
        rowKey={(q) => `${q.a}__${q.b}`}
        highlight={(q) => q === p}
        onRowClick={(q) => setPair(cmp.pairs.indexOf(q))}
        columns={[
          { key: "pair", label: "Pair (A − B)", value: (q) => pairLabel(q) },
          {
            key: "city",
            label: "Citywide difference",
            unit: u,
            align: "right",
            value: (q) => q.city.estimate,
            render: (q) => (
              <span>
                {pairText(q.city, unit)}{" "}
                {q.city.paired ? (
                  <Tip tip={term("paired_se").tip} focusable>
                    <Badge tone="accent">paired</Badge>
                  </Tip>
                ) : (
                  <Badge tone="warn">needs a {FULL} run</Badge>
                )}
              </span>
            ),
          },
          { key: "conf", label: "Confidence", value: (q) => q.city.phrase },
          ...regionNames.map((name) => ({
            key: `r_${name}`,
            label: name,
            unit: u,
            align: "right" as const,
            value: (q: Comparison["pairs"][number]) => q.regions[name]?.estimate ?? null,
            render: (q: Comparison["pairs"][number]) => (q.regions[name] ? `${pairText(q.regions[name], unit)}${q.regions[name].paired ? "" : " (unpaired)"}` : "—"),
          })),
        ]}
        rows={cmp.pairs}
      />
      <SmallMaps grid={grid} items={cmp.items} values={values} unit={unit} />
      {p ? (
        <div className="grid2">
          <section className="stack">
            <h3>Difference map: {pairLabel(p)}</h3>
            {p.layer_key ? (
              <>
                <p className="cap">Negative (purple) where {label(p.a)} is cooler than {label(p.b)}.</p>
                <MapView grid={grid} groups={groups} loadLayer={load} layerKey={p.layer_key} height={420} title={`${label(p.a)} minus ${label(p.b)}`} />
              </>
            ) : (
              <p className="cap">No difference map: a plan not checked yet has no change place by place. Check it with the {FULL} to map it.</p>
            )}
          </section>
          {a && b ? (
            <HexbinScatter
              title="Temperature change, place by place"
              points={{ x: a, y: b }}
              xLabel={label(p.a)}
              yLabel={label(p.b)}
              xUnit={u}
              yUnit={u}
              diagonal
              caption="Places above the diagonal warm more (or cool less) under B than under A."
            />
          ) : (
            <p className="cap">Place-by-place values are not available for both items of this pair.</p>
          )}
        </div>
      ) : null}
      {equityGroups.length ? (
        <Table
          caption="Who benefits, side by side"
          csvName="compare-equity"
          rowKey={(r) => r[0]}
          columns={[
            { key: "group", label: "Group", value: (r) => r[0] },
            ...Object.keys(equityGroups[0][1]).map((k) => ({ key: k, label: k, align: "right" as const, value: (r: [string, Record<string, number | null>]) => r[1][k] ?? null, render: (r: [string, Record<string, number | null>]) => fmtSigned(r[1][k], 3) })),
          ]}
          rows={equityGroups}
        />
      ) : null}
      {cmp.exposure.length ? (
        <Table
          caption="Exposure today and in the futures"
          csvName="compare-exposure"
          columns={Object.keys(cmp.exposure[0]).map((k) => ({
            key: k,
            label: k.replace(/_/g, " "),
            value: (r: Record<string, unknown>) => (typeof r[k] === "number" ? (r[k] as number) : typeof r[k] === "string" ? (r[k] as string) : r[k] == null ? null : JSON.stringify(r[k])),
            render: (r: Record<string, unknown>) => (typeof r[k] === "number" ? (Math.abs(r[k] as number) <= 1 && /share/.test(k) ? fmtPct(r[k] as number) : fmtNum(r[k] as number, 2)) : String(r[k] ?? "—")),
          }))}
          rows={cmp.exposure}
        />
      ) : null}
    </div>
  );
}

export default function Compare() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
  const detail = useRunDetail(rid);
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = detail.data?.run.project_id ?? ctxPid;
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const [itemsRaw, setItemsRaw] = useUrlState("items", codecs.list());
  const [cid, setCid] = useUrlState("cid", codecs.optString());
  const refs = useMemo(() => itemsRaw.map(decodeItemRef).filter((r): r is ItemRef => r !== null), [itemsRaw]);
  const cmp = useComparison(cid);
  const [creating, setCreating] = useState<{ key: string; error: string | null } | null>(null);
  const key = refs.map(encodeItemRef).join(",");
  const catalog = useMemo(() => (layers.data ?? []).flatMap((g) => g.layers), [layers.data]);

  // A comparison for the current item set: the one in the URL when it matches, else a new one.
  const matches = !!cmp.data && cmp.data.items.map((i) => encodeItemRef(i.ref)).join(",") === key;
  useEffect(() => {
    if (refs.length < 2 || matches || creating?.key === key) return;
    // A comparison named in the URL is fetched first; a new one is made only when it does not
    // match the items (or cannot be loaded).
    if (cid && !cmp.data && !cmp.error) return;
    setCreating({ key, error: null });
    createComparison(rid, refs).then(
      (c) => {
        mutate(`comparison:${c.id}`, c);
        setCid(c.id);
        setCreating(null);
      },
      (e: unknown) => setCreating({ key, error: errorMessage(e) }),
    );
    // `key` stands for refs
  }, [key, matches, cid, cmp.data, cmp.error, rid]);

  if (grid.error) return <EmptyState error={grid.error} />;
  const unit = grid.data?.meta.units.target ?? "";
  return (
    <LabFrame rid={rid} title="Compare">
      <div className="row">
        {refs.map((r) => (
          <span key={encodeItemRef(r)} className="chip">
            {cmp.data?.items.find((i) => sameItem(i.ref, r))?.label ?? encodeItemRef(r)}
            <button type="button" className="x" aria-label={`Remove ${encodeItemRef(r)}`} onClick={() => setItemsRaw(itemsRaw.filter((x) => x !== encodeItemRef(r)))}>
              ×
            </button>
          </span>
        ))}
        <ItemPicker rid={rid} pid={pid} refs={refs} onChange={(r) => setItemsRaw(r.map(encodeItemRef))} />
      </div>
      {refs.length < 2 ? <p className="cap">Choose 2 to 4 items: {FULL} results, scenarios set up with this run, plans or the baseline.</p> : null}
      {creating?.error ? <p className="callout" data-tone="crit">{creating.error}</p> : null}
      {refs.length >= 2 && cmp.data && matches && grid.data ? (
        <CompareBody rid={rid} pid={pid} cmp={cmp.data} grid={grid.data} catalog={catalog} unit={unit} />
      ) : refs.length >= 2 && !creating?.error ? (
        <p className="cap" role="status">
          Comparing…
        </p>
      ) : null}
    </LabFrame>
  );
}
