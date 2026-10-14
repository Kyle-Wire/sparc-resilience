// Budget plans (`/r/:rid/lab/plans`, `/r/:rid/lab/plans/:plid`; SPEC §7.10, §2 J7).
// - The first entry, "Plan from this run" (read-only; phase 2, the canonical home of the run's own
//   S7 budget plan, formerly the Budget run tab): ../runhub/Budget's sections (key numbers,
//   Pareto, dose and closed-loop maps, top cells). New plans start from the form under it.
// - The params form (lever, budget on a log slider, scalar or column cost, plantable/region cap,
//   minimum dose, objective, equity source and focus, Pareto multipliers) previews the planned
//   allocation inline (< 1 s), at most once per 150 ms while the slider moves.
// - A saved plan shows its Pareto curve (with the closed-loop frontier once verified), the
//   dose map, Verify / Verify frontier (exact engine), planned vs realised with the spillover
//   label, Plan → scenario, the field kit (ranked cells, logger sites, before/after pairs;
//   CSV and GeoJSON rendered in the browser) and the plan pack export.
// Plain words (phase 4): places, amounts, "planned benefit" (open loop) and "checked benefit"
// (closed loop, the full model) from the glossary, each technical term with an ⓘ; no native titles.
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
import { isActiveStatus, type LayerGroup, type LayerMeta } from "../../api/types";
import { Pareto } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { InfoTip, type InfoTipProps } from "../../components/ui/InfoTip";
import { JobStrip } from "../../components/ui/JobStrip";
import { Kpi, KpiRow } from "../../components/ui/Kpi";
import { NumberField } from "../../components/ui/NumberField";
import { Seg } from "../../components/ui/Seg";
import { Slider } from "../../components/ui/Slider";
import { Table } from "../../components/ui/Table";
import { downloadText } from "../../components/ui/download";
import { term } from "../../content/glossary";
import { useRunDetail } from "../../layouts/resources";
import { MapView } from "../../map/MapView";
import { useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import type { GridData } from "../../map/grid";
import { Link, navigate, useRoute } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { RunBudgetPlan } from "../runhub/Budget";
import { ConfirmDialog } from "./components/Dialogs";
import { LabFrame, labHref } from "./components/LabFrame";
import { Tip } from "./components/Tip";
import { columnOptions } from "./components/SelectionBuilder";
import { syntheticMeta } from "./model/layers";
import { fieldKitCsv, fieldKitGeoJson, parseNumberList } from "./model/library";
import { rateLimit } from "./model/timing";
import { useTray } from "./model/tray";
import { leverUnitWord, leverWords } from "./model/words";

/** Plain names of the plan's two benefits: the quick sum (open loop) and the full model's check (closed loop). */
const PLANNED = term("open_loop").label;
const CHECKED = term("closed_loop").label;
/** "°F·places": cooling added up over places (glossary total_cooling). */
const placeDegrees = (u: string) => `${u}·places`;
/** The unit of a cooling total on a tile, with the glossary's total_cooling ⓘ. */
function PlaceDegrees({ u }: { u: string }) {
  return (
    <>
      {placeDegrees(u)}
      <InfoTip term="total_cooling" label={placeDegrees(u)} />
    </>
  );
}
/** The unit of a cost on a tile, with its ⓘ (glossary cost_units). */
const COST_UNITS = term("cost_units").label.toLowerCase();

/** Why the checked cooling is above or below the plan, in a few words (the checked/planned ratio; "" within 5%). */
function gapWhy(ratio: number): string {
  return ratio > 1.05 ? "cooling spreads to nearby places" : ratio < 0.95 ? "neighbouring changes overlap" : "";
}

/**
 * The plan's check in a sentence, the right way round: the full model can find more cooling than the
 * plan (it also counts cooling that spreads to nearby places) or less (neighbouring changes overlap).
 */
export function checkedVsPlanned(planned: number, checked: number): string {
  const engine = term("exact_engine").label.toLowerCase();
  const ratio = checked / (planned || NaN);
  if (ratio > 1.05) return `Higher than planned: the plan adds up each place's cooling on its own, while the ${engine} also counts cooling that spreads to nearby places.`;
  if (ratio < 0.95) return `Lower than planned: the plan adds up each place's cooling on its own, while in the ${engine} neighbouring changes overlap, so their cooling does not simply add up.`;
  return `About as planned: the ${engine} counts cooling that spreads to and overlaps with neighbouring places, and the two roughly cancel out.`;
}
function CostUnits() {
  return (
    <>
      {COST_UNITS}
      <InfoTip term="cost_units" />
    </>
  );
}
/** The scale of the unevenness tiles. */
const UNEVEN_SCALE = "0 = shared evenly, 1 = one place gets it all";

/**
 * How unevenly the plan spreads its amounts. The server's Gini is of the doses (the amounts placed),
 * not of the cooling, so the label says amounts; the ⓘ keeps the glossary's technical name.
 */
const UNEVEN = "Unevenness of the amounts";
const UNEVEN_TIP: InfoTipProps = {
  term: "gini",
  label: UNEVEN.toLowerCase(),
  text: "How unevenly the plan spreads its amounts over the places it treats. 0 means every treated place gets the same; near 1 means a few places get almost all of it.",
  technical: `${term("gini").technical} of the doses`,
};

/** A KPI label with its ⓘ. */
function TipLabel({ label, tip }: { label: string; tip: InfoTipProps }) {
  return (
    <>
      {label}
      <InfoTip {...tip} label={tip.label ?? label} />
    </>
  );
}

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

function DoseMap({ grid, dose, extra, title, unit }: { grid: GridData; dose: Float32Array | null; extra?: { meta: LayerMeta; load: () => Promise<Float32Array> }[]; title: string; unit: string }) {
  const unitless = useMemo<LayerGroup[]>(() => {
    // the amount in the lever's plain unit word ("pts" for percentage points), as the rest of the Lab writes it
    const layers = [syntheticMeta({ key: "plan:dose", label: "Planned amount", unit, scale: "seq", zero_blank: true, desc: "Amount of change per treated place; untreated places are blank." }, dose)];
    return [{ id: "plan", label: "Plan", layers: [...layers, ...(extra ?? []).map((e) => e.meta)] }];
  }, [dose, extra, unit]);
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
  if (!params) return <EmptyState title="No levers" body="This run has no levers with a measured cooling response, so there is nothing to plan." />;
  const lever = levers.find((l) => l.var === params.lever);
  const [bLo, bHi] = budgetRange(lever, grid.n);
  const set = (patch: Partial<PlanParams>) => setParams((p) => (p ? { ...p, ...patch } : p));
  // the lever's plain unit word ("pts" for percentage points), as the Lab, What cools and the Library write it
  const u = leverUnitWord(lever);
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
        <h3 className="lab-tip-head">
          Plan
          <InfoTip term="budget_plan" />
        </h3>
        <datalist id="plan-columns">
          {columns.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </datalist>
        <label className="field">
          <span className="field-label">What to change</span>
          <select value={params.lever} onChange={(e) => setParams(defaultParams(levers.find((l) => l.var === e.target.value) ?? levers[0], grid.n))}>
            {usable.map((l) => (
              <option key={l.var} value={l.var}>
                {leverWords(l).name}
              </option>
            ))}
          </select>
        </label>
        <Slider label={`Budget (${COST_UNITS})`} value={params.budget} min={bLo} max={bHi} log onChange={(v) => set({ budget: v })} format={(v) => fmtInt(Math.round(v))} />
        <div className="field">
          <span className="field-label">
            Cost
            <InfoTip term="cost_units" />
          </span>
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
            <label className="field">
              <span className="field-label">Cost per unit</span>
              <NumberField label="Cost per unit" value={params.cost.scalar} min={0} onChange={(v) => set({ cost: { scalar: v ?? 1 } })} />
            </label>
          )}
        </div>
        <fieldset className="stack" style={{ gap: 4, border: 0, padding: 0 }}>
          <legend className="field-label">Limits</legend>
          <label className="row">
            <input type="checkbox" checked={params.cap.plantable} disabled={!(lever?.role === "canopy" && lever.headroom_available)} onChange={(e) => set({ cap: { ...params.cap, plantable: e.target.checked } })} />
            Only {term("headroom").label.toLowerCase()}
            {lever?.role === "canopy" && lever.headroom_available ? "" : " (trees & canopy only, with the planting-space layers)"}
            <InfoTip term="headroom" />
          </label>
          {params.cap.plantable ? (
            <label className="field">
              <span className="field-label">
                Plantable share of paved ground
                <InfoTip label="plantable share of paved ground" text="The share of paved and built ground counted as space that could take trees (0 to 1). Empty uses the project's setting." technical="cap.paved_share" />
              </span>
              <NumberField
                label="Plantable share of paved ground"
                value={params.cap.paved_share ?? null}
                min={0}
                max={1}
                step={0.05}
                nullable
                placeholder="project default"
                style={{ width: "11em" }}
                onChange={(v) => set({ cap: { ...params.cap, paved_share: v ?? undefined } })}
              />
            </label>
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
        <label className="field">
          <span className="field-label">Smallest amount per treated place</span>
          <NumberField label="Smallest amount per treated place" unit={u || undefined} value={params.min_dose ?? null} min={0} nullable placeholder="none" onChange={(v) => set({ min_dose: v ?? undefined })} />
        </label>
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
          <span className="field-label">
            Who to favour
            <InfoTip label="who to favour" text="Gives extra weight to places where more vulnerable people live, so the plan leans towards them. How strongly sets the weight (0 = not at all)." technical="Equity weighting (source, focus)" />
          </span>
          <select
            aria-label="Who to favour"
            value={params.equity?.source ?? ""}
            onChange={(e) => set({ equity: e.target.value ? { source: e.target.value as EquitySource, focus: params.equity?.focus ?? 0.3, column: e.target.value === "column" ? columns[0]?.value : undefined } : undefined })}
          >
            <option value="">No one in particular</option>
            <option value="share_60_plus">Places with more people aged 60+</option>
            <option value="share_under_5">Places with more children under 5</option>
            <option value="density">The most densely populated places</option>
            <option value="column">By a map layer…</option>
          </select>
          {params.equity?.source === "column" ? (
            <input list="plan-columns" aria-label="Map layer to favour by" placeholder="layer:… or csv:<path>:<column>" value={params.equity.column ?? ""} onChange={(e) => set({ equity: { ...params.equity!, column: e.target.value } })} />
          ) : null}
          {params.equity ? <Slider label="How strongly" value={params.equity.focus} min={0} max={1} step={0.05} onChange={(v) => set({ equity: { ...params.equity!, focus: v } })} format={(v) => fmtNum(v, 2)} /> : null}
        </div>
        <label className="field">
          <span className="field-label">
            Other budgets to compare (multiples of this one)
            <InfoTip term="pareto" />
          </span>
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
          <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} /> Check the plan with the {term("exact_engine").label.toLowerCase()} after saving
          <InfoTip term="closed_loop" />
        </label>
        <Button variant="primary" busy={saving} onClick={() => void save()} disabled={!pv}>
          Save plan
        </Button>
      </section>
      <section className="stack" aria-label="Planned allocation">
        {live.error ? <p className="callout" data-tone="crit">{live.error}</p> : null}
        <KpiRow label="Planned allocation">
          <Kpi label={<TipLabel label={PLANNED} tip={{ term: "open_loop" }} />} value={pv ? fmtNum(pv.planned_total, 1) : "—"} unit={<PlaceDegrees u={unitLabel(grid.meta.units.target)} />} note="each place's cooling added up, not yet checked" />
          <Kpi label="Places treated" value={pv ? fmtInt(pv.n_cells_treated) : "—"} note={pv ? `average amount ${fmtNum(pv.mean_dose_treated, 1)}${u ? ` ${u}` : ""}` : undefined} />
          <Kpi label="Cost" value={pv ? fmtNum(pv.total_cost, 0) : "—"} unit={pv ? <CostUnits /> : undefined} note={pv && pv.min_dose_dropped_cost > 0 ? `${fmtNum(pv.min_dose_dropped_cost, 0)} left unspent by the smallest-amount rule` : undefined} />
          <Kpi label={<TipLabel label={UNEVEN} tip={UNEVEN_TIP} />} value={pv ? fmtNum(pv.gini, 2) : "—"} note={UNEVEN_SCALE} />
        </KpiRow>
        {pv ? <p className="cap">{pv.caption}</p> : <p className="cap">{live.loading ? "Planning…" : "Adjust the parameters to plan."}</p>}
        {pv ? (
          <Pareto
            title="Cooling for each budget"
            points={pv.pareto.map((r) => ({ x: r.budget, y: r.benefit, label: `${fmtInt(r.n_cells)} places` }))}
            selected={pv.pareto.reduce((best, r, i) => (Math.abs(r.budget - params.budget) < Math.abs(pv.pareto[best].budget - params.budget) ? i : best), 0)}
            xLabel="Budget"
            xUnit={COST_UNITS}
            yLabel={PLANNED}
            yUnit={placeDegrees(unitLabel(grid.meta.units.target))}
            curveLabel={PLANNED}
            onPointClick={(i) => set({ budget: pv.pareto[i].budget })}
            caption={
              <>
                Goal: {pv.objective}. Limit: {pv.constraint}. Select a point to use its budget.
                <InfoTip term="pareto" />
              </>
            }
          />
        ) : null}
        <DoseMap grid={grid} dose={dose} title="Planned amount" unit={u} />
      </section>
    </div>
  );
}

function FieldKitBlock({ plid, name, unit, amountUnit }: { plid: string; name: string; unit: string; amountUnit: string }) {
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
        <p className="cap">A ranked list of places to treat, with their coordinates, plus sites for temperature loggers and before/after pairs to measure the effect.</p>
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
          Places CSV
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
          { key: "id", label: "Place id", value: (c) => c.id },
          { key: "lon", label: "Lon", align: "right", value: (c) => c.lon, render: (c) => fmtNum(c.lon, 5) },
          { key: "lat", label: "Lat", align: "right", value: (c) => c.lat, render: (c) => fmtNum(c.lat, 5) },
          { key: "zone", label: "Zone", value: (c) => c.zone },
          { key: "dose", label: "Amount", unit: amountUnit || undefined, align: "right", value: (c) => c.dose, render: (c) => fmtNum(c.dose, 2) },
          // the cell's footprint cooling: what its dose cools around it, summed (target·cells), not ΔT at the cell
          { key: "pb", label: `${PLANNED} (here and nearby)`, unit: placeDegrees(unitLabel(unit)), align: "right", value: (c) => c.planned_benefit, render: (c) => fmtNum(c.planned_benefit, 3) },
          { key: "cl", label: `Change here (${CHECKED.toLowerCase()})`, unit: unitLabel(unit), align: "right", value: (c) => c.closed_loop_delta, render: (c) => fmtNum(c.closed_loop_delta, 3) },
          { key: "people", label: "People", align: "right", value: (c) => c.people, render: (c) => fmtInt(c.people) },
          { key: "pp", label: term("headroom").label, align: "right", value: (c) => c.plantable_pp, render: (c) => fmtNum(c.plantable_pp, 1) },
        ]}
        rows={kit.cells}
      />
      <p className="cap">
        {fmtInt(kit.sites.length)} logger sites and {fmtInt(kit.pairs.length)} before/after pairs (a treated place gets at least the median amount of the places the plan treats).
      </p>
    </section>
  );
}

function PlanDetail({ rid, plid, grid, pid }: { rid: string; plid: string; grid: GridData; pid: string | null }) {
  const plan = usePlan(plid, rid);
  const levers = useLevers(rid);
  // the plan's amounts in its lever's plain unit word ("pts" for percentage points)
  const amountUnit = leverUnitWord(levers.data?.find((l) => l.var === plan.data?.params.lever));
  const [busy, setBusy] = useState<string | null>(null);
  const [job, setJob] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const pin = useTray((s) => s.pin);
  // A verify or frontier job of this plan, including the one "Save plan" started.
  const running = useJobs((s) => Object.values(s.jobs).find((j) => (j.kind === "engine.plan_verify" || j.kind === "engine.plan_frontier") && j.params?.plan_id === plid && isActiveStatus(j.status))?.id ?? null);
  const shownJob = job ?? running;
  const extra = useMemo(() => {
    const unit = grid.meta.units.target;
    const list = [
      // a footprint total on each treated cell (target·cells: the cooling its dose brings to its neighbourhood)
      { meta: syntheticMeta({ key: "plan:planned_benefit", label: `${PLANNED} (here and nearby, on the treated place)`, unit: placeDegrees(unit), scale: "seq" }, null), load: () => getPlanLayer(plid, "planned_benefit") },
    ];
    if (plan.data?.realised) list.push({ meta: syntheticMeta({ key: "plan:closed_loop_delta", label: `Temperature change (${CHECKED.toLowerCase()})`, unit, sign_note: "negative = cooler" }, null), load: () => getPlanLayer(plid, "closed_loop_delta") });
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
          {p.realised ? "Check again" : `Check with the ${term("exact_engine").label.toLowerCase()}`}
        </Button>
        <Tip tip={`Check every budget on the curve with the ${term("exact_engine").label.toLowerCase()}, not only this one.`}>
          <Button size="small" busy={busy === "frontier"} onClick={() => void verifyNow(true)}>
            Check every budget
          </Button>
        </Tip>
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
          Make a scenario from it
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
        <Button size="small" variant="ghost" icon="pin" onClick={() => (pin(rid, { kind: "plan", id: plid }, p.name) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
          Pin to compare
        </Button>
        <Button size="small" variant="danger" onClick={() => setConfirmDelete(true)}>
          Delete
        </Button>
      </div>
      {confirmDelete ? (
        <ConfirmDialog
          title="Delete this plan?"
          confirmLabel="Delete plan"
          onClose={() => setConfirmDelete(false)}
          onConfirm={async () => {
            await deletePlan(plid);
            invalidate(`run:${rid}:lab`);
            navigate(labHref(rid, "plans"));
          }}
        >
          <p>
            <strong>{p.name}</strong> and its stored allocation will be deleted. Scenarios made from it lose the place-by-place amounts they came from.
          </p>
        </ConfirmDialog>
      ) : null}
      {shownJob ? <JobStrip jobId={shownJob} /> : null}
      <KpiRow label="Plan">
        <Kpi label={<TipLabel label={PLANNED} tip={{ term: "open_loop" }} />} value={fmtNum(p.planned.planned_total, 1)} unit={<PlaceDegrees u={u} />} note="each place's cooling added up" />
        <Kpi
          label={<TipLabel label={CHECKED} tip={{ term: "closed_loop" }} />}
          value={p.realised ? fmtNum(p.realised.total, 1) : "—"}
          unit={p.realised ? <PlaceDegrees u={u} /> : undefined}
          note={p.realised && gap !== null ? `${fmtPct(gap)} of planned${gapWhy(gap) ? ` · ${gapWhy(gap)}` : ""}` : "not checked yet"}
          tone={p.realised ? "good" : undefined}
        />
        <Kpi label="Places treated" value={fmtInt(p.planned.n_cells_treated)} note={`average amount ${fmtNum(p.planned.mean_dose_treated, 1)}${amountUnit ? ` ${amountUnit}` : ""}`} />
        <Kpi label="Cost" value={fmtNum(p.planned.total_cost, 0)} unit={<CostUnits />} note={p.planned.min_dose_dropped_cost > 0 ? `${fmtNum(p.planned.min_dose_dropped_cost, 0)} left unspent by the smallest-amount rule` : undefined} />
      </KpiRow>
      {p.realised ? (
        <p className="cap" data-spillover="true">
          Planned {fmtNum(p.planned.planned_total, 0)} vs checked {fmtNum(p.realised.total, 0)} {placeDegrees(u)}. {checkedVsPlanned(p.planned.planned_total, p.realised.total)}
          <InfoTip term="closed_loop" />
        </p>
      ) : null}
      <div className="grid2">
        <Pareto
          title="Cooling for each budget"
          points={p.planned.pareto.map((r) => ({ x: r.budget, y: r.benefit, label: `${fmtInt(r.n_cells)} places` }))}
          realised={(p.frontier ?? []).map((f) => ({ x: f.budget, y: Math.abs(f.realised), label: CHECKED.toLowerCase() }))}
          selected={p.planned.pareto.findIndex((r) => r.budget === p.params.budget)}
          xLabel="Budget"
          xUnit={COST_UNITS}
          yLabel={term("total_cooling").label}
          yUnit={placeDegrees(u)}
          curveLabel={PLANNED}
          realisedLabel={CHECKED}
          caption={
            <>
              {p.frontier?.length
                ? `The ${CHECKED.toLowerCase()} points come from ${term("exact_engine").label.toLowerCase()} runs at each budget.`
                : `Check every budget to add the ${term("exact_engine").label.toLowerCase()}'s points.`}
              <InfoTip term="pareto" />
            </>
          }
        />
        <section className="card">
          <h3>Parameters</h3>
          <dl className="kv">
            <dt>What to change</dt>
            <dd>{p.params.lever}</dd>
            <dt>Budget</dt>
            <dd className="num">{fmtInt(p.params.budget)}</dd>
            <dt>Cost</dt>
            <dd>{"scalar" in p.params.cost ? `${fmtNum(p.params.cost.scalar, 2)} per unit` : p.params.cost.column}</dd>
            <dt>Limits</dt>
            <dd>{[p.params.cap.plantable ? term("headroom").label.toLowerCase() : null, p.params.cap.region ? "one area" : null].filter(Boolean).join(" + ") || "none"}</dd>
            <dt>Objective</dt>
            <dd>{p.params.objective}</dd>
            <dt>Who to favour</dt>
            <dd>{p.params.equity ? `${p.params.equity.source}${p.params.equity.column ? ` (${p.params.equity.column})` : ""}, focus ${fmtNum(p.params.equity.focus, 2)}` : "none"}</dd>
            <dt>Created</dt>
            <dd>{fmtDateTime(p.created_utc)}</dd>
          </dl>
          {p.realised ? (
            <p className="cap">
              {term("exact_engine").label} result <span className="mono">{p.realised.result_id}</span> · average change in the treated places {fmtNum(p.realised.mean_treated, 3)} {u}
            </p>
          ) : null}
        </section>
      </div>
      <DoseMap grid={grid} dose={dose} extra={extra} title={p.name} unit={amountUnit} />
      <FieldKitBlock plid={plid} name={p.name} unit={grid.meta.units.target} amountUnit={amountUnit} />
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
        { key: "lever", label: "What to change", value: (p) => p.params.lever },
        { key: "budget", label: "Budget", align: "right", value: (p) => p.params.budget, render: (p) => fmtInt(p.params.budget) },
        { key: "planned", label: PLANNED, align: "right", value: (p) => p.planned.planned_total, render: (p) => fmtNum(p.planned.planned_total, 1) },
        { key: "realised", label: CHECKED, align: "right", value: (p) => p.realised?.total ?? null, render: (p) => (p.realised ? fmtNum(p.realised.total, 1) : <Badge>not checked</Badge>) },
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
        <div className="stack composer" data-composer="plans">
          <section className="composer-section" data-plan="run">
            <RunBudgetPlan
              title="Plan from this run"
              intro={`The run's own ${term("budget_plan").label.toLowerCase()}: ${term("budget_plan").tip.charAt(0).toLowerCase()}${term("budget_plan").tip.slice(1)} Read-only: make a new plan below to change it.`}
              actions={<Badge>read-only</Badge>}
            />
          </section>
          <section className="composer-section stack" aria-labelledby="plans-new">
            <h2 id="plans-new" className="composer-heading">
              New plan
            </h2>
            <PlanForm rid={rid} grid={grid.data} levers={levers.data} onSaved={(p) => navigate(labHref(rid, `plans/${encodeURIComponent(p.id)}`))} />
          </section>
          <section className="composer-section stack" aria-labelledby="plans-saved">
            <h2 id="plans-saved" className="composer-heading">
              Saved plans
            </h2>
            <PlanList rid={rid} />
          </section>
        </div>
      )}
    </LabFrame>
  );
}
