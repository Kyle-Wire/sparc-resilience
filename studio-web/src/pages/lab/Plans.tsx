// Budget plans (`/r/:rid/lab/plans`, `/r/:rid/lab/plans/:plid`; SPEC §7.10, §2 J7).
// - The params form (lever, budget on a log slider, scalar or column cost, plantable/region cap,
//   minimum dose, objective, equity source and focus, Pareto multipliers) previews the planned
//   allocation inline (< 1 s), at most once per 150 ms while the slider moves.
// - A saved plan shows its Pareto curve (with the closed-loop frontier once verified), the
//   dose map, Verify / Verify frontier (exact engine), planned vs realised with the spillover
//   label, Plan → scenario, the field kit (ranked cells, logger sites, before/after pairs;
//   CSV and GeoJSON rendered in the browser) and the plan pack export.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "../../api/client";
import { base64ToBytes, viewOf } from "../../api/binary";
import { invalidate } from "../../api/resource";
import {
  createPlan,
  deletePlan,
  exportPack,
  getFieldKit,
  getPlanLayer,
  planToScenario,
  previewPlan,
  useLevers,
  usePlan,
  usePlans,
  useRegions,
  verifyPlan,
  type EquitySource,
  type FieldKit,
  type Lever,
  type Plan,
  type PlanParams,
  type PlanPreview,
} from "../../api/lab";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { Pareto } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { JobStrip } from "../../components/ui/JobStrip";
import { Kpi, KpiRow } from "../../components/ui/Kpi";
import { NumberField } from "../../components/ui/NumberField";
import { Seg } from "../../components/ui/Seg";
import { Slider } from "../../components/ui/Slider";
import { Table } from "../../components/ui/Table";
import { downloadText } from "../../components/ui/download";
import { useRunDetail } from "../../layouts/resources";
import { MapView } from "../../map/MapView";
import { useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import type { GridData } from "../../map/grid";
import { Link, navigate, useRoute } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { LabFrame, labHref } from "./components/LabFrame";
import { columnOptions } from "./components/SelectionBuilder";
import { syntheticMeta } from "./model/layers";
import { fieldKitCsv, fieldKitGeoJson, parseNumberList } from "./model/library";
import { rateLimit } from "./model/timing";

export const PLAN_PREVIEW_MS = 150;

/** Decode the base64 Float32 dose of a plan preview. */
export function decodeDose(b64: string): Float32Array {
  const bytes = base64ToBytes(b64);
  return viewOf(bytes.buffer as ArrayBuffer, "float32", bytes.byteOffset, bytes.length / 4) as Float32Array;
}

/** Budget slider range: up to every cell at the lever's largest dose (pp·cells for canopy). */
export function budgetRange(lever: Lever | undefined, n: number): [number, number] {
  const top = Math.max(...(lever?.doses.length ? lever.doses : [lever?.design_dose ?? 10]).map((d) => Math.abs(d)), 1);
  return [Math.max(1, top), Math.max(top * 10, top * n)];
}

export function defaultParams(lever: Lever, n: number): PlanParams {
  const [lo, hi] = budgetRange(lever, n);
  return {
    lever: lever.var,
    budget: Math.min(hi, Math.max(lo, 20000)),
    cost: { scalar: lever.cost_per_unit || 1 },
    cap: { plantable: lever.role === "canopy" && lever.headroom_available },
    objective: "cooling",
    multipliers: [0.25, 0.5, 1, 2, 4],
  };
}

/**
 * Live planned-mode preview: every params change goes through a 150 ms rate limiter (at most
 * one request per window while a slider moves, and always one with the final value); a
 * newer request aborts the older one.
 */
export function usePlanPreview(rid: string, params: PlanParams | null) {
  const [state, setState] = useState<{ preview: PlanPreview | null; error: string | null; loading: boolean }>({ preview: null, error: null, loading: false });
  const ctrl = useRef<AbortController | null>(null);
  const send = useMemo(
    () =>
      rateLimit((p: PlanParams) => {
        ctrl.current?.abort();
        const c = new AbortController();
        ctrl.current = c;
        previewPlan(rid, p, c.signal).then(
          (preview) => !c.signal.aborted && setState({ preview, error: null, loading: false }),
          (e: unknown) => !c.signal.aborted && setState((s) => ({ ...s, error: errorMessage(e), loading: false })),
        );
      }, PLAN_PREVIEW_MS),
    [rid],
  );
  const key = params ? JSON.stringify(params) : null;
  useEffect(() => {
    if (!params) return;
    setState((s) => ({ ...s, loading: true }));
    send(params);
    // `key` stands for `params`
  }, [key, send]);
  useEffect(
    () => () => {
      send.cancel();
      ctrl.current?.abort();
    },
    [send],
  );
  return state;
}

function DoseMap({ grid, dose, extra, title }: { grid: GridData; dose: Float32Array | null; extra?: { meta: LayerMeta; load: () => Promise<Float32Array> }[]; title: string }) {
  const unitless = useMemo<LayerGroup[]>(() => {
    const layers = [syntheticMeta({ key: "plan:dose", label: "Planned dose", unit: "", scale: "seq", zero_blank: true, desc: "Dose per treated cell; untreated cells are blank." }, dose)];
    return [{ id: "plan", label: "Plan", layers: [...layers, ...(extra ?? []).map((e) => e.meta)] }];
  }, [dose, extra]);
  const load = useCallback(
    async (meta: LayerMeta): Promise<LayerValues> => {
      if (meta.key === "plan:dose") return dose ?? new Float32Array(grid.n);
      const e = extra?.find((x) => x.meta.key === meta.key);
      return e ? e.load() : new Float32Array(grid.n).fill(NaN);
    },
    [dose, extra, grid.n],
  );
  return <MapView grid={grid} groups={unitless} loadLayer={load} height={420} title={title} />;
}

export function PlanForm({ rid, grid, levers, onSaved }: { rid: string; grid: GridData; levers: Lever[]; onSaved: (p: Plan) => void }) {
  const regions = useRegions(rid);
  const layers = useRunLayers(rid);
  const usable = levers.filter((l) => l.direction === "increase" || l.direction === "decrease");
  const [params, setParams] = useState<PlanParams | null>(() => (usable[0] ? defaultParams(usable[0], grid.n) : null));
  const [name, setName] = useState("");
  const [verify, setVerify] = useState(true);
  const [mult, setMult] = useState("0.25, 0.5, 1, 2, 4");
  const [saving, setSaving] = useState(false);
  const live = usePlanPreview(rid, params);
  const dose = useMemo(() => (live.preview ? decodeDose(live.preview.dose) : null), [live.preview]);
  const columns = useMemo(() => columnOptions(layers.data ?? []), [layers.data]);
  if (!params) return <EmptyState title="No levers" body="This run has no actionable levers with responses (S4), so there is nothing to plan." />;
  const lever = levers.find((l) => l.var === params.lever);
  const [bLo, bHi] = budgetRange(lever, grid.n);
  const set = (patch: Partial<PlanParams>) => setParams((p) => (p ? { ...p, ...patch } : p));
  const u = unitLabel(lever?.unit ?? "");
  const pv = live.preview;
  const save = async () => {
    setSaving(true);
    try {
      const r = await createPlan(rid, params, name.trim() || `${lever?.label ?? params.lever} budget ${fmtInt(params.budget)}`, verify);
      if (r.job) useJobs.getState().upsert(r.job);
      invalidate(`run:${rid}:lab`);
      onSaved(r.plan);
    } catch (e) {
      toast("error", "Could not save the plan", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };
  const costIsColumn = "column" in params.cost;
  return (
    <div className="lab-split">
      <section className="card stack" aria-label="Plan parameters">
        <h3>Plan</h3>
        <datalist id="plan-columns">
          {columns.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </datalist>
        <label className="field">
          <span className="field-label">Lever</span>
          <select value={params.lever} onChange={(e) => setParams(defaultParams(levers.find((l) => l.var === e.target.value) ?? levers[0], grid.n))}>
            {usable.map((l) => (
              <option key={l.var} value={l.var}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        <Slider label={`Budget (${u ? u + "·" : ""}cells)`} value={params.budget} min={bLo} max={bHi} log onChange={(v) => set({ budget: v })} format={(v) => fmtInt(Math.round(v))} />
        <div className="field">
          <span className="field-label">Cost</span>
          <Seg<"scalar" | "column">
            label="Cost source"
            size="small"
            value={costIsColumn ? "column" : "scalar"}
            onChange={(v) => set({ cost: v === "scalar" ? { scalar: lever?.cost_per_unit || 1 } : { column: columns[0]?.value ?? "" } })}
            options={[
              { value: "scalar", label: "Per unit" },
              { value: "column", label: "Cost layer" },
            ]}
          />
          {"column" in params.cost ? (
            <input list="plan-columns" aria-label="Cost column" placeholder="layer:… or csv:<path>:<column>" value={params.cost.column} onChange={(e) => set({ cost: { column: e.target.value } })} />
          ) : (
            <NumberField label="Cost per unit" value={params.cost.scalar} min={0} onChange={(v) => set({ cost: { scalar: v ?? 1 } })} />
          )}
        </div>
        <fieldset className="stack" style={{ gap: 4, border: 0, padding: 0 }}>
          <legend className="field-label">Cap</legend>
          <label className="row">
            <input type="checkbox" checked={params.cap.plantable} disabled={!(lever?.role === "canopy" && lever.headroom_available)} onChange={(e) => set({ cap: { ...params.cap, plantable: e.target.checked } })} />
            Plantable headroom{lever?.role === "canopy" && lever.headroom_available ? "" : " (canopy with planner layers only)"}
          </label>
          {params.cap.plantable ? (
            <NumberField label="Paved share" value={params.cap.paved_share ?? null} min={0} max={1} step={0.05} nullable placeholder="config default" onChange={(v) => set({ cap: { ...params.cap, paved_share: v ?? undefined } })} />
          ) : null}
          <select
            aria-label="Region cap"
            value={params.cap.region && "kind" in params.cap.region && params.cap.region.kind === "region" ? params.cap.region.id : ""}
            onChange={(e) => set({ cap: { ...params.cap, region: e.target.value ? { kind: "region", id: e.target.value } : undefined } })}
          >
            <option value="">Whole city</option>
            {(regions.data ?? []).map((r) => (
              <option key={r.id} value={r.id}>
                Only in {r.name}
              </option>
            ))}
          </select>
        </fieldset>
        <NumberField label="Minimum dose per treated cell" unit={lever?.unit} value={params.min_dose ?? null} min={0} nullable placeholder="none" onChange={(v) => set({ min_dose: v ?? undefined })} />
        <div className="field">
          <span className="field-label">Objective</span>
          <Seg<"cooling" | "people">
            label="Objective"
            size="small"
            value={params.objective}
            onChange={(v) => set({ objective: v })}
            options={[
              { value: "cooling", label: "Most cooling" },
              { value: "people", label: "Most people cooled" },
            ]}
          />
        </div>
        <div className="field">
          <span className="field-label">Equity</span>
          <select
            aria-label="Equity source"
            value={params.equity?.source ?? ""}
            onChange={(e) => set({ equity: e.target.value ? { source: e.target.value as EquitySource, focus: params.equity?.focus ?? 0.3, column: e.target.value === "column" ? columns[0]?.value : undefined } : undefined })}
          >
            <option value="">No equity weighting</option>
            <option value="share_60_plus">Share aged 60+</option>
            <option value="share_under_5">Share under 5</option>
            <option value="density">Population density rank</option>
            <option value="column">A column…</option>
          </select>
          {params.equity?.source === "column" ? (
            <input list="plan-columns" aria-label="Equity column" placeholder="layer:… or csv:<path>:<column>" value={params.equity.column ?? ""} onChange={(e) => set({ equity: { ...params.equity!, column: e.target.value } })} />
          ) : null}
          {params.equity ? <Slider label="Equity focus" value={params.equity.focus} min={0} max={1} step={0.05} onChange={(v) => set({ equity: { ...params.equity!, focus: v } })} format={(v) => fmtNum(v, 2)} /> : null}
        </div>
        <label className="field">
          <span className="field-label">Pareto multipliers of the budget</span>
          <input
            value={mult}
            onChange={(e) => setMult(e.target.value)}
            onBlur={() => {
              const r = parseNumberList(mult);
              if (!r.bad.length && r.values.length) set({ multipliers: r.values.filter((v) => v > 0) });
            }}
          />
        </label>
        <hr />
        <label className="field">
          <span className="field-label">Name</span>
          <input value={name} placeholder={`${lever?.label ?? params.lever} budget ${fmtInt(params.budget)}`} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="row">
          <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} /> Verify exactly after saving (closed loop, needs the engine)
        </label>
        <Button variant="primary" busy={saving} onClick={() => void save()} disabled={!pv}>
          Save plan
        </Button>
      </section>
      <section className="stack" aria-label="Planned allocation">
        {live.error ? <p className="callout" data-tone="crit">{live.error}</p> : null}
        <KpiRow label="Planned allocation">
          <Kpi label="Planned benefit" value={pv ? fmtNum(pv.planned_total, 1) : "—"} unit={`${unitLabel(grid.meta.units.target)}·cells`} note="open loop (sum of per-cell responses)" />
          <Kpi label="Cells treated" value={pv ? fmtInt(pv.n_cells_treated) : "—"} note={pv ? `mean dose ${fmtNum(pv.mean_dose_treated, 2)} ${u}` : undefined} />
          <Kpi label="Cost" value={pv ? fmtNum(pv.total_cost, 0) : "—"} note={pv && pv.min_dose_dropped_cost > 0 ? `${fmtNum(pv.min_dose_dropped_cost, 0)} freed by the minimum dose (not re-spent)` : undefined} />
          <Kpi label="Gini of doses" value={pv ? fmtNum(pv.gini, 2) : "—"} />
        </KpiRow>
        {pv ? <p className="cap">{pv.caption}</p> : <p className="cap">{live.loading ? "Planning…" : "Adjust the parameters to plan."}</p>}
        {pv ? (
          <Pareto
            title="Benefit vs budget"
            points={pv.pareto.map((r) => ({ x: r.budget, y: r.benefit, label: `${fmtInt(r.n_cells)} cells` }))}
            selected={pv.pareto.reduce((best, r, i) => (Math.abs(r.budget - params.budget) < Math.abs(pv.pareto[best].budget - params.budget) ? i : best), 0)}
            xLabel="Budget"
            yLabel="Planned benefit"
            yUnit={`${unitLabel(grid.meta.units.target)}·cells`}
            curveLabel="Planned (open loop)"
            onPointClick={(i) => set({ budget: pv.pareto[i].budget })}
            caption={`${pv.objective}; ${pv.constraint}. Select a point to use its budget.`}
          />
        ) : null}
        <DoseMap grid={grid} dose={dose} title="Planned dose" />
      </section>
    </div>
  );
}

function FieldKitBlock({ plid, name }: { plid: string; name: string }) {
  const [kit, setKit] = useState<FieldKit | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async () => {
    setBusy(true);
    try {
      setKit(await getFieldKit(plid));
    } catch (e) {
      toast("error", "Could not build the field kit", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || plid;
  if (!kit) {
    return (
      <section className="card">
        <h3>Field kit</h3>
        <p className="cap">A ranked list of cells to treat with lon/lat, logger sites and before/after pairs for evaluation.</p>
        <Button busy={busy} onClick={() => void load()}>
          Build field kit
        </Button>
      </section>
    );
  }
  const geo = () => {
    const g = fieldKitGeoJson(kit);
    downloadText(g.geojson, `${slug}-field-kit.geojson`, "application/geo+json");
    if (g.skipped) toast("warning", `${g.skipped} rows have no lon/lat and were left out of the GeoJSON`);
  };
  return (
    <section className="stack" aria-label="Field kit">
      <div className="row">
        <h3>Field kit</h3>
        <span className="spacer" />
        <Button size="small" icon="download" onClick={() => downloadText(fieldKitCsv(kit, "cells"), `${slug}-field-list.csv`, "text/csv;charset=utf-8")}>
          Cells CSV
        </Button>
        <Button size="small" icon="download" onClick={() => downloadText(fieldKitCsv(kit, "sites"), `${slug}-logger-sites.csv`, "text/csv;charset=utf-8")}>
          Sites CSV
        </Button>
        <Button size="small" icon="download" onClick={() => downloadText(fieldKitCsv(kit, "pairs"), `${slug}-before-after-pairs.csv`, "text/csv;charset=utf-8")}>
          Pairs CSV
        </Button>
        <Button size="small" icon="download" onClick={geo}>
          GeoJSON
        </Button>
      </div>
      <Table
        caption="Ranked cells"
        csvName={null}
        rowKey={(c) => String(c.id)}
        columns={[
          { key: "rank", label: "Rank", align: "right", value: (c) => c.rank },
          { key: "id", label: "Cell id", value: (c) => c.id },
          { key: "lon", label: "Lon", align: "right", value: (c) => c.lon, render: (c) => fmtNum(c.lon, 5) },
          { key: "lat", label: "Lat", align: "right", value: (c) => c.lat, render: (c) => fmtNum(c.lat, 5) },
          { key: "zone", label: "Zone", value: (c) => c.zone },
          { key: "dose", label: "Dose", align: "right", value: (c) => c.dose, render: (c) => fmtNum(c.dose, 2) },
          { key: "pb", label: "Planned benefit", align: "right", value: (c) => c.planned_benefit, render: (c) => fmtNum(c.planned_benefit, 3) },
          { key: "cl", label: "Closed-loop ΔT", align: "right", value: (c) => c.closed_loop_delta, render: (c) => fmtNum(c.closed_loop_delta, 3) },
          { key: "people", label: "People", align: "right", value: (c) => c.people, render: (c) => fmtInt(c.people) },
          { key: "pp", label: "Plantable headroom", align: "right", value: (c) => c.plantable_pp, render: (c) => fmtNum(c.plantable_pp, 1) },
        ]}
        rows={kit.cells}
      />
      <p className="cap">
        {fmtInt(kit.sites.length)} logger sites and {fmtInt(kit.pairs.length)} before/after pairs (treated = dose at or above the median positive dose).
      </p>
    </section>
  );
}

function PlanDetail({ rid, plid, grid, pid }: { rid: string; plid: string; grid: GridData; pid: string | null }) {
  const plan = usePlan(plid, rid);
  const [busy, setBusy] = useState<string | null>(null);
  const [job, setJob] = useState<string | null>(null);
  const extra = useMemo(() => {
    const unit = grid.meta.units.target;
    const list = [
      { meta: syntheticMeta({ key: "plan:planned_benefit", label: "Planned benefit", unit, scale: "seq" }, null), load: () => getPlanLayer(plid, "planned_benefit") },
    ];
    if (plan.data?.realised) list.push({ meta: syntheticMeta({ key: "plan:closed_loop_delta", label: "Closed-loop ΔT (exact)", unit, sign_note: "negative = cooler" }, null), load: () => getPlanLayer(plid, "closed_loop_delta") });
    return list;
  }, [plid, plan.data?.realised, grid.meta.units.target]);
  const [dose, setDose] = useState<Float32Array | null>(null);
  useEffect(() => {
    let live = true;
    getPlanLayer(plid, "dose").then((d) => live && setDose(d), () => {});
    return () => {
      live = false;
    };
  }, [plid]);
  if (plan.error && !plan.data) return <EmptyState error={plan.error} />;
  const p = plan.data;
  if (!p) return <p className="cap">Loading the plan…</p>;
  const u = unitLabel(grid.meta.units.target);
  const act = async (key: string, f: () => Promise<void>) => {
    setBusy(key);
    try {
      await f();
    } catch (e) {
      toast("error", "That did not work", { body: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };
  const verifyNow = (frontier: boolean) =>
    act(frontier ? "frontier" : "verify", async () => {
      const j = await verifyPlan(plid, frontier);
      useJobs.getState().upsert(j);
      setJob(j.id);
    });
  const gap = p.realised ? p.realised.total / (p.planned.planned_total || NaN) : null;
  return (
    <div className="stack">
      <div className="row">
        <Link to={labHref(rid, "plans")}>All plans</Link>
        <span className="spacer" />
        <Button size="small" busy={busy === "verify"} onClick={() => void verifyNow(false)}>
          {p.realised ? "Verify again" : "Verify exactly"}
        </Button>
        <Button size="small" busy={busy === "frontier"} onClick={() => void verifyNow(true)}>
          Verify frontier
        </Button>
        <Button
          size="small"
          busy={busy === "scenario"}
          onClick={() =>
            void act("scenario", async () => {
              const s = await planToScenario(plid);
              navigate(labHref(rid, `s/${encodeURIComponent(s.id)}`));
            })
          }
        >
          Plan → scenario
        </Button>
        <Button
          size="small"
          icon="download"
          busy={busy === "pack"}
          disabled={!pid}
          onClick={() =>
            void act("pack", async () => {
              const r = await exportPack(pid!, "plan_pack", { plan_id: plid });
              useJobs.getState().upsert(r.job);
              toast("info", "Building the plan pack", { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
            })
          }
        >
          Plan pack
        </Button>
        <Button
          size="small"
          variant="danger"
          busy={busy === "delete"}
          onClick={() =>
            void act("delete", async () => {
              await deletePlan(plid);
              invalidate(`run:${rid}:lab`);
              navigate(labHref(rid, "plans"));
            })
          }
        >
          Delete
        </Button>
      </div>
      {job ? <JobStrip jobId={job} /> : null}
      <KpiRow label="Plan">
        <Kpi label="Planned benefit" value={fmtNum(p.planned.planned_total, 1)} unit={`${u}·cells`} note="open loop" />
        <Kpi
          label="Realised (exact)"
          value={p.realised ? fmtNum(p.realised.total, 1) : "—"}
          unit={p.realised ? `${u}·cells` : undefined}
          note={p.realised && gap !== null ? `${fmtPct(gap)} of planned · spillover non-additivity` : "not verified yet"}
          tone={p.realised ? "good" : undefined}
        />
        <Kpi label="Cells treated" value={fmtInt(p.planned.n_cells_treated)} note={`mean dose ${fmtNum(p.planned.mean_dose_treated, 2)}`} />
        <Kpi label="Cost" value={fmtNum(p.planned.total_cost, 0)} note={p.planned.min_dose_dropped_cost > 0 ? `${fmtNum(p.planned.min_dose_dropped_cost, 0)} freed by the minimum dose` : undefined} />
      </KpiRow>
      {p.realised ? (
        <p className="cap" data-spillover="true">
          Planned {fmtNum(p.planned.planned_total, 0)} vs realised {fmtNum(p.realised.total, 0)} {u}·cells: the plan adds up per-cell responses, while the exact closed loop lets neighbouring treatments overlap (spillover non-additivity).
        </p>
      ) : null}
      <div className="grid2">
        <Pareto
          title="Benefit vs budget"
          points={p.planned.pareto.map((r) => ({ x: r.budget, y: r.benefit, label: `${fmtInt(r.n_cells)} cells` }))}
          realised={(p.frontier ?? []).map((f) => ({ x: f.budget, y: Math.abs(f.realised), label: "closed loop" }))}
          selected={p.planned.pareto.findIndex((r) => r.budget === p.params.budget)}
          xLabel="Budget"
          yLabel="Benefit"
          yUnit={`${u}·cells`}
          curveLabel="Planned (open loop)"
          realisedLabel="Realised (exact closed loop)"
          caption={p.frontier?.length ? "The closed-loop frontier comes from exact runs at each multiplier." : "Verify the frontier to add the exact closed-loop points."}
        />
        <section className="card">
          <h3>Parameters</h3>
          <dl className="kv">
            <dt>Lever</dt>
            <dd>{p.params.lever}</dd>
            <dt>Budget</dt>
            <dd className="num">{fmtInt(p.params.budget)}</dd>
            <dt>Cost</dt>
            <dd>{"scalar" in p.params.cost ? `${fmtNum(p.params.cost.scalar, 2)} per unit` : p.params.cost.column}</dd>
            <dt>Cap</dt>
            <dd>{[p.params.cap.plantable ? "plantable headroom" : null, p.params.cap.region ? "region" : null].filter(Boolean).join(" + ") || "none"}</dd>
            <dt>Objective</dt>
            <dd>{p.params.objective}</dd>
            <dt>Equity</dt>
            <dd>{p.params.equity ? `${p.params.equity.source}${p.params.equity.column ? ` (${p.params.equity.column})` : ""}, focus ${fmtNum(p.params.equity.focus, 2)}` : "none"}</dd>
            <dt>Created</dt>
            <dd>{fmtDateTime(p.created_utc)}</dd>
          </dl>
          {p.realised ? (
            <p className="cap">
              Exact result <span className="mono">{p.realised.result_id}</span> · mean ΔT in treated cells {fmtNum(p.realised.mean_treated, 3)} {u}
            </p>
          ) : null}
        </section>
      </div>
      <DoseMap grid={grid} dose={dose} extra={extra} title={p.name} />
      <FieldKitBlock plid={plid} name={p.name} />
    </div>
  );
}

function PlanList({ rid }: { rid: string }) {
  const plans = usePlans(rid);
  if (plans.error && !plans.data) return <EmptyState error={plans.error} />;
  return (
    <Table<Plan>
      caption="Saved plans"
      csvName="plans"
      rowKey={(p) => p.id}
      columns={[
        { key: "name", label: "Plan", value: (p) => p.name, render: (p) => <Link to={labHref(rid, `plans/${encodeURIComponent(p.id)}`)}>{p.name}</Link> },
        { key: "lever", label: "Lever", value: (p) => p.params.lever },
        { key: "budget", label: "Budget", align: "right", value: (p) => p.params.budget, render: (p) => fmtInt(p.params.budget) },
        { key: "planned", label: "Planned", align: "right", value: (p) => p.planned.planned_total, render: (p) => fmtNum(p.planned.planned_total, 1) },
        { key: "realised", label: "Realised", align: "right", value: (p) => p.realised?.total ?? null, render: (p) => (p.realised ? fmtNum(p.realised.total, 1) : <Badge>not verified</Badge>) },
        { key: "created", label: "Created", value: (p) => p.created_utc, render: (p) => fmtDateTime(p.created_utc) },
      ]}
      rows={plans.data ?? []}
      empty={<span className="cap">{plans.data ? "No saved plans yet." : "Loading…"}</span>}
    />
  );
}

export default function Plans() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
  const plid = params.plid ?? null;
  const detail = useRunDetail(rid);
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = detail.data?.run.project_id ?? ctxPid;
  const grid = useRunGrid(rid);
  const levers = useLevers(rid);
  if (grid.error) return <EmptyState error={grid.error} />;
  if (levers.error && !levers.data) return <EmptyState error={levers.error} />;
  if (!grid.data || !levers.data) return <p className="cap">Loading…</p>;
  return (
    <LabFrame rid={rid} title={plid ? "Budget plan" : "Budget plans"}>
      {plid ? (
        <PlanDetail rid={rid} plid={plid} grid={grid.data} pid={pid} />
      ) : (
        <>
          <PlanForm rid={rid} grid={grid.data} levers={levers.data} onSaved={(p) => navigate(labHref(rid, `plans/${encodeURIComponent(p.id)}`))} />
          <h3>Saved plans</h3>
          <PlanList rid={rid} />
        </>
      )}
    </LabFrame>
  );
}
