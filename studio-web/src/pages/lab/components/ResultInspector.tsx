// Exact result inspector (SPEC §7.7): the plain-language card (headline, likely range,
// confidence, qualifiers, what it buys) with an expert toggle (SE, jackknife, n, method);
// region table; ring-profile spill chart; realised vs requested and mediator moves; cost;
// causal check; the uncertainty IntervalStack; impacts (exposure today and in the futures,
// equity quintiles, hot days with the fetch action, climate offset); preview vs exact;
// result layers on the map; DEMO badge and the stale banner. Exports the decision pack.
import { useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { mutate } from "../../../api/resource";
import { exportPack, postImpacts, useResult, type Impacts, type Result } from "../../../api/lab";
import { Bars, DotRange, Gauge, IntervalStack, RingProfile, type IntervalLayer, type Ring } from "../../../charts";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { ActionButton, EmptyState } from "../../../components/ui/EmptyState";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { PlainResult } from "../../../components/ui/PlainResult";
import { Table } from "../../../components/ui/Table";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtSigned, fmtSignedValue, unitLabel } from "../../../theme/format";
import { buysLines, cityLine, plainWording, withRange } from "../model/plain";
import { useTray } from "../model/tray";
import { AcrossRunsPanel } from "./AcrossRuns";

/** Spill rings for the chart: ring 0 (r_m = 0) is the edited set; ring k spans (r_{k−1}, r_k]. */
export function spillRings(rings: Result["spill"]["rings"]): Ring[] {
  const sorted = [...rings].sort((a, b) => a.r_m - b.r_m);
  let prev = 0;
  return sorted.map((r) => {
    const ring: Ring = r.r_m === 0 ? { r0_m: 0, r1_m: 0, mean: r.mean, se: r.se, n: r.n } : { r0_m: prev, r1_m: r.r_m, mean: r.mean, se: r.se, n: r.n };
    prev = r.r_m;
    return ring;
  });
}

/** The uncertainty layers of a result (narrow to wide) for the IntervalStack. */
export function uncertaintyLayers(u: NonNullable<Result["uncertainty"]>): IntervalLayer[] {
  const out: IntervalLayer[] = [];
  const add = (id: string, label: string, v: [number, number] | null) => v && out.push({ id, label, lo: v[0], hi: v[1] });
  add("estimation", "Estimation (95%, fold jackknife)", u.estimation_95);
  add("specification", "Specification (across runs / variants)", u.specification);
  add("attribution", "Attribution (simulation check)", u.attribution);
  add("causal", "Independent causal band", u.causal_band);
  add("envelope", "Envelope (all sources)", u.envelope);
  return out;
}

function ImpactsBlock({ resId, impacts, unit, onImpacts }: { resId: string; impacts: Impacts | null; unit: string; onImpacts: (i: Impacts) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const u = unitLabel(unit);
  const compute = async () => {
    setBusy(true);
    try {
      onImpacts(await postImpacts(resId, {}));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };
  if (!impacts) {
    return (
      <section className="card">
        <h3>Impacts</h3>
        {error ? <EmptyState error={error} /> : <p className="cap">Residents at or above each threshold today and in the futures, equity by group, hot days avoided and the climate offset.</p>}
        <Button busy={busy} onClick={() => void compute()}>
          Compute impacts
        </Button>
      </section>
    );
  }
  const thresholds = impacts.thresholds;
  const tKey = (rec: Record<string, number>, t: number) => Object.keys(rec).find((k) => Number(k) === t) ?? String(t);
  const equityGroups = Object.entries(impacts.equity);
  return (
    <section className="stack" aria-label="Impacts">
      <h3>Impacts</h3>
      <Table
        caption="Residents at or above each threshold, without and with the scenario"
        csvName="exposure"
        rowKey={(r, i) => `${r.case}-${r.adapted}-${i}`}
        columns={[
          { key: "case", label: "Case", value: (r) => r.case },
          { key: "adapted", label: "Scenario", value: (r) => (r.adapted ? "with" : "without") },
          { key: "mean", label: "Person-mean temperature", unit: u, align: "right", value: (r) => r.person_mean_temp, render: (r) => fmtNum(r.person_mean_temp, 2) },
          ...thresholds.map((t) => ({
            key: `ge_${t}`,
            label: `≥ ${fmtNum(t, 1)} ${u}`,
            align: "right" as const,
            value: (r: Impacts["exposure"][number]) => r.people_ge[tKey(r.people_ge, t)] ?? null,
            render: (r: Impacts["exposure"][number]) => `${fmtInt(Math.round(r.people_ge[tKey(r.people_ge, t)] ?? NaN))} (${fmtPct(r.share_people_ge[tKey(r.share_people_ge, t)] ?? null)})`,
          })),
        ]}
        rows={impacts.exposure}
      />
      {equityGroups.length ? (
        <div className="grid2">
          {equityGroups.map(([g, e]) => (
            <Bars
              key={g}
              title={`Cooling by ${g.replace(/_/g, " ")} quintile`}
              units={`${u} (negative = cooler)`}
              categories={e.quintiles.map((q) => `Q${q.quintile}`)}
              series={[{ id: "cool", label: "Mean change", values: e.quintiles.map((q) => q.mean_cooling) }]}
              valueLabel="Mean ΔT"
              unit={u}
              categoryLabel="Quintile (1 = lowest)"
              caption={`Concentration index ${fmtSigned(e.concentration_index, 3)} (negative: the cooling favours the lower quintiles of ${g.replace(/_/g, " ")}).`}
            />
          ))}
        </div>
      ) : null}
      <section className="card">
        <h4>Hot days</h4>
        {impacts.hot_days ? (
          <Table
            caption={`Person-days avoided (station ${impacts.hot_days.station})`}
            csvName="hot-days"
            columns={Object.keys(impacts.hot_days.cases[0] ?? {}).map((k) => ({ key: k, label: k.replace(/_/g, " "), value: (r: Record<string, unknown>) => (typeof r[k] === "number" || typeof r[k] === "string" ? (r[k] as number | string) : null) }))}
            rows={impacts.hot_days.cases}
          />
        ) : impacts.hot_days_action ? (
          <div className="row">
            <span className="cap">Hot days need the station's daily history.</span>
            <ActionButton action={impacts.hot_days_action} size="small" />
          </div>
        ) : (
          <p className="cap">Not available for this run.</p>
        )}
      </section>
      {impacts.climate_offset.length ? (
        <div className="grid3">
          {impacts.climate_offset.map((o) => (
            <Gauge
              key={`${o.experiment}-${o.period}`}
              title={`${o.experiment.toUpperCase()} ${o.period}`}
              label="of median warming offset"
              value={o.offset_share}
              min={0}
              max={Math.max(1, o.offset_share)}
              ticks={[1]}
              format={(v) => fmtPct(v)}
              caption={`Cancels ${fmtPct(o.offset_share)} of ${o.experiment.toUpperCase()} ${o.period} median warming.`}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
}

export type ResultInspectorProps = {
  rid: string;
  resId: string;
  pid: string | null;
  unit: string;
  nFolds: number | null;
  onShowLayer?: (key: string) => void;
};

export function ResultInspector({ rid, resId, pid, unit, nFolds, onShowLayer }: ResultInspectorProps) {
  const res = useResult(resId, rid);
  const pin = useTray((s) => s.pin);
  const [packBusy, setPackBusy] = useState(false);
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  const r = res.data;
  if (!r) return <p className="cap">Loading the result…</p>;
  const u = unitLabel(unit);
  const s = r.summary;
  // The card computes its confidence from lo/hi; fill them from the SE when absent (SPEC §7.7).
  const main = withRange(s.edited ?? r.city);
  const causalDisagrees = r.causal_check ? !r.causal_check.model_within : false;
  const wording = plainWording({ edited: s.edited, city: r.city, unit, fracExtrapolatedEdited: r.extrapolated_edited ?? s.frac_extrapolated_edited, causalDisagrees, buys: buysLines(r, unit) });
  const known = new Set(wording.qualifiers.map((q) => q.toLowerCase()));
  const extraQualifiers = r.plain.qualifiers.filter((q) => !known.has(q.toLowerCase()) && !/outside observed conditions|causal check|preview only/i.test(q));
  const buys = wording.buys.length ? wording.buys : r.plain.buys;
  const edited = r.regions.find((g) => g.auto && /edited/i.test(g.name) && !/ring/i.test(g.name));
  const realized = Object.entries(r.realized);
  const mediators = Object.entries(r.mediators);
  const uLayers = r.uncertainty ? uncertaintyLayers(r.uncertainty) : [];

  const exportDecision = async () => {
    if (!pid) return;
    setPackBusy(true);
    try {
      const { job } = await exportPack(pid, "decision_pack", { result_id: resId });
      useJobs.getState().upsert(job);
      toast("info", "Building the decision pack", { href: `/jobs/${job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", "Could not start the decision pack", { body: errorMessage(e) });
    } finally {
      setPackBusy(false);
    }
  };

  return (
    <section className="result-inspector stack" aria-label="Exact result">
      <header className="row">
        <h2>Exact result{r.scenario ? `: ${r.scenario.name}` : ""}</h2>
        {r.demo ? <Badge tone="demo">DEMO</Badge> : null}
        {r.scenario ? <span className="cap">revision {r.scenario.revision}</span> : null}
        <span className="cap">{fmtDateTime(s.created_utc)}</span>
        <span className="spacer" />
        <Button size="small" icon="pin" onClick={() => (pin(rid, { kind: "result", id: resId }, r.scenario?.name ?? resId) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
          Pin to compare
        </Button>
        <Button size="small" icon="download" busy={packBusy} disabled={!pid} onClick={() => void exportDecision()} title="Self-contained brief, maps, tables and GIS files (exact results only)">
          Decision pack
        </Button>
      </header>
      {r.stale || s.stale ? (
        <div className="callout" role="status" data-stale="true">
          <Badge tone="warn">stale</Badge> The run's checkpoint or the core code changed since this result was computed. It stays viewable; run exact again to refresh it.
        </div>
      ) : null}
      {onShowLayer ? (
        <div className="row" role="group" aria-label="Show result layers on the map">
          <span className="cap">On the map:</span>
          <Button size="small" variant="ghost" onClick={() => onShowLayer(`res:${resId}:delta`)}>
            ΔT
          </Button>
          <Button size="small" variant="ghost" onClick={() => onShowLayer(`res:${resId}:delta_sd`)}>
            Spread
          </Button>
          <Button size="small" variant="ghost" onClick={() => onShowLayer(`res:${resId}:extrapolation`)}>
            Extrapolation
          </Button>
          {realized.map(([v]) => (
            <Button key={v} size="small" variant="ghost" onClick={() => onShowLayer(`res:${resId}:realized_${v}`)}>
              Realised {v}
            </Button>
          ))}
        </div>
      ) : null}
      <div className="grid2">
        <PlainResult
          title="In plain words"
          likely={main}
          unit={unit}
          subject={s.edited ? "the edited area" : "the city"}
          qualifiers={[...(wording.city ? [wording.city] : []), ...wording.qualifiers, ...extraQualifiers, ...buys.map((b) => `What it buys: ${b}`)]}
          expert={{
            n: edited?.n_cells ?? null,
            folds: nFolds,
            method: "Exact engine on the run's fold models. SE = std over folds of the fold means × √(K−1) (jackknife); likely range = estimate ± 1.96·SE.",
            extra: [
              ["City mean", `${fmtSignedValue(r.city.estimate, unit, 4)} ± ${fmtNum(r.city.se, 4)}`],
              ["Cell spread (p10–p90)", `${fmtSigned(r.p10, 3)} to ${fmtSigned(r.p90, 3)} ${u}`],
              ["Mean per-cell fold SD", fmtNum(r.mean_delta_sd, 4)],
              ["Edited cells extrapolated", fmtPct(r.extrapolated_edited)],
            ],
          }}
        />
        <div className="stack">
          <KpiRow label="Result summary">
            <Kpi label="Edited area" value={s.edited ? fmtSigned(s.edited.estimate, 3) : "—"} unit={u} note={s.edited?.phrase} />
            <Kpi label="City-wide" value={fmtSigned(r.city.estimate, 4)} unit={u} note={cityLine(r.city, unit)} />
            <Kpi label="Cooling outside the edits" value={fmtPct(r.spill.outside_share)} note="spill beyond the edited cells" />
            <Kpi label="Cost" value={fmtNum(r.cost.total, 0)} note={r.cost.cooling_per_cost !== null ? `${fmtNum(Math.abs(r.cost.cooling_per_cost) * 1000, 3)} ${u}·cells per 1k` : undefined} />
          </KpiRow>
          {r.preview_vs_exact ? (
            <p className="cap" data-preview-vs-exact="true">
              Preview vs exact: mean absolute error {fmtNum(r.preview_vs_exact.mean_abs_err, 4)} {u} ({fmtPct(r.preview_vs_exact.rel_err)} relative).
            </p>
          ) : null}
          {r.warnings.length ? (
            <ul className="edit-issues">
              {r.warnings.map((w, i) => (
                <li key={i} data-level="warn">
                  {w.message}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>

      <div className="grid2">
        <RingProfile
          title="How far the cooling reaches"
          units={`${u} (negative = cooler)`}
          rings={spillRings(r.spill.rings)}
          unit={u}
          valueLabel="Mean ΔT"
          caption={`${r.spill.outside_share !== null ? `${fmtPct(r.spill.outside_share)} of the total cooling lands outside the edited cells.` : "No cooling to split between inside and outside."}${Object.keys(r.spill.lever_ranges).length ? ` Influence ranges: ${Object.entries(r.spill.lever_ranges).map(([v, m]) => `${v} ${fmtInt(m)} m`).join(", ")}.` : ""}`}
        />
        <Table
          caption="Regions"
          csvName="regions"
          rowKey={(g) => g.name}
          columns={[
            { key: "name", label: "Region", value: (g) => g.name, render: (g) => (g.auto ? <em>{g.name}</em> : g.name) },
            { key: "n", label: "Cells", align: "right", value: (g) => g.n_cells, render: (g) => fmtInt(g.n_cells) },
            { key: "mean", label: "Mean ΔT", unit: u, align: "right", value: (g) => g.mean?.estimate ?? null, render: (g) => (g.mean ? `${fmtSigned(g.mean.estimate, 3)}${g.mean.se !== null ? ` ± ${fmtNum(1.96 * g.mean.se, 3)}` : ""}` : "—") },
            { key: "pw", label: "People-weighted", unit: u, align: "right", value: (g) => g.people_weighted, render: (g) => fmtSigned(g.people_weighted, 3) },
            { key: "total", label: "Total", unit: `${u}·cells`, align: "right", value: (g) => g.total, render: (g) => fmtSigned(g.total, 1) },
            { key: "c01", label: "Cooled ≥ 0.1", align: "right", value: (g) => g.frac_cooled_01, render: (g) => fmtPct(g.frac_cooled_01) },
            { key: "c05", label: "Cooled ≥ 0.5", align: "right", value: (g) => g.frac_cooled_05, render: (g) => fmtPct(g.frac_cooled_05) },
          ]}
          rows={r.regions}
        />
      </div>

      <div className="grid2">
        <section className="card">
          <h3>Realised vs requested</h3>
          {realized.length ? (
            <Table
              csvName="realised"
              rowKey={([v]) => v}
              columns={[
                { key: "v", label: "Lever", value: ([v]) => v },
                { key: "rq", label: "Requested mean", align: "right", value: ([, x]) => x.requested_mean, render: ([, x]) => fmtNum(x.requested_mean, 3) },
                { key: "rl", label: "Realised mean", align: "right", value: ([, x]) => x.realized_mean, render: ([, x]) => fmtNum(x.realized_mean, 3) },
                { key: "rqt", label: "Requested total", align: "right", value: ([, x]) => x.requested_total, render: ([, x]) => fmtNum(x.requested_total, 1) },
                { key: "rlt", label: "Realised total", align: "right", value: ([, x]) => x.realized_total, render: ([, x]) => fmtNum(x.realized_total, 1) },
                { key: "clip", label: "Clipped", align: "right", value: ([, x]) => x.clipped_share, render: ([, x]) => fmtPct(x.clipped_share) },
              ]}
              rows={realized}
            />
          ) : (
            <p className="cap">No lever changes recorded.</p>
          )}
          {mediators.length ? (
            <p className="cap">
              Mediators: {mediators.map(([v, m]) => `${v} follows ${m.mean_change >= 0 ? "+" : "−"}${fmtNum(Math.abs(m.mean_change), 3)}`).join(" · ")}.
            </p>
          ) : null}
          <p className="cap">
            Cost {fmtNum(r.cost.total, 0)}
            {Object.keys(r.cost.per_lever).length ? ` (${Object.entries(r.cost.per_lever).map(([v, c]) => `${v} ${fmtNum(c, 0)}`).join(", ")})` : ""}.
          </p>
        </section>
        {r.causal_check ? (
          <DotRange
            title="Causal check"
            units={`${u} (negative = cooler)`}
            rows={[
              {
                id: "model",
                label: "City mean (model, exact)",
                est: r.city.estimate,
                lo: r.city.lo,
                hi: r.city.hi,
                check: { est: r.causal_check.delta, lo: r.causal_check.lo, hi: r.causal_check.hi, label: "independent causal band" },
              },
            ]}
            valueLabel="City-mean ΔT"
            unit={u}
            signed
            rangeLabel="Likely range"
            caption={r.causal_check.model_within ? "The model's estimate lies inside the independent causal band." : "The model's estimate lies OUTSIDE the independent causal band: treat the size of the effect with caution."}
          />
        ) : (
          <section className="card">
            <h3>Causal check</h3>
            <p className="cap">No causal audit for these levers in this run.</p>
          </section>
        )}
      </div>

      {uLayers.length ? (
        <IntervalStack
          title="Uncertainty"
          units={`${u} (negative = cooler)`}
          rows={[{ id: "city", label: "City mean", estimate: r.city.estimate, layers: uLayers }]}
          valueLabel="City-mean ΔT"
          unit={u}
          caption={`${r.uncertainty?.envelope_excludes_zero ? "The full envelope excludes zero." : "The full envelope includes zero."} Sources: ${r.uncertainty?.sources.join(", ") || "fold jackknife only"}.`}
        />
      ) : null}

      {r.scenario ? <AcrossRunsPanel pid={pid} sid={r.scenario.id} unit={unit} /> : null}

      <ImpactsBlock resId={resId} impacts={r.impacts} unit={unit} onImpacts={(i) => mutate(`result:${resId}`, (prev: Result | undefined) => (prev ? { ...prev, impacts: i } : prev) as Result)} />
    </section>
  );
}
