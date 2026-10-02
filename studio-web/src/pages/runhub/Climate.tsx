// Climate (SPEC §6.4): warming per SSP × period (median, p10–p90, min–max, per-model strip),
// the model × projection table, exposure bars (share of cells ≥ T today vs futures and
// adaptation variants, with a threshold toggle), the "% of median warming offset" gauges, a
// future-temperature map computed client-side (observed + warming) and a link to Lab › Climate.
import { useMemo, useState } from "react";
import type { ClimateSections, ClimateProjection } from "../../api/runs";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { Bars, DotRange, Gauge, SmallMultiples } from "../../charts";
import { EmptyState } from "../../components/ui/EmptyState";
import { Seg } from "../../components/ui/Seg";
import { MapView } from "../../map/MapView";
import { runLayerLoader, useRunGrid, useRunLayers } from "../../map/data";
import { findLayer } from "../../map/LayerPicker";
import { Link } from "../../router";
import { fmtNum, fmtPct, fmtSignedValue, unitLabel } from "../../theme/format";
import { Block, GenericTableView, Section, ViewPage, useRid, useUnits } from "./common";

const projLabel = (p: Pick<ClimateProjection, "label" | "period">) => `${p.label} ${p.period.replace("-", "–")}`;

function Warming({ w }: { w: ClimateProjection[] }) {
  const u = unitLabel(useUnits().target);
  const mid = w.find((p) => /2041/.test(p.period) && /245/.test(p.experiment)) ?? w[0];
  return (
    <DotRange
      title="Warming by scenario and period"
      units={`${u} above the baseline`}
      rows={w.map((p) => ({
        id: p.id,
        label: projLabel(p),
        est: p.median,
        lo: p.p10,
        hi: p.p90,
        lo2: p.min,
        hi2: p.max,
        strip: Object.values(p.by_model).filter((v): v is number => v !== null),
      }))}
      valueLabel="Warming"
      unit={u}
      rangeLabel="10th–90th percentile of models"
      outerLabel="model range"
      caption={
        mid
          ? `${projLabel(mid)}: median warming ${fmtSignedValue(mid.median, u, 2)} across ${mid.n_models ?? "?"} models (10th–90th percentile ${fmtNum(mid.p10, 2)} to ${fmtNum(mid.p90, 2)} ${u}). Grey dots are individual models.`
          : undefined
      }
    />
  );
}

function Exposure({ e }: { e: NonNullable<ClimateSections["exposure"]> }) {
  const u = unitLabel(useUnits().target);
  const ths = e.thresholds.map(String);
  const [th, setTh] = useState(ths[0] ?? "");
  const t = ths.includes(th) ? th : ths[0];
  if (!t) return <p className="cap">No exposure thresholds configured.</p>;
  const keyOf = (rec: Record<string, unknown>) => Object.keys(rec).find((k) => Number(k) === Number(t)) ?? t;
  const today = e.present[keyOf(e.present)] ?? null;
  const cats = ["Today", ...e.groups.map((g) => `${g.label}${g.variant && g.variant !== "no adaptation" ? ` · ${g.variant}` : ""}`)];
  const vals = [today, ...e.groups.map((g) => g.share[keyOf(g.share)]?.median ?? null)];
  const lo = [null, ...e.groups.map((g) => g.share[keyOf(g.share)]?.p10 ?? null)];
  const hi = [null, ...e.groups.map((g) => g.share[keyOf(g.share)]?.p90 ?? null)];
  const pct = (a: (number | null)[]) => a.map((v) => (v === null ? null : v * 100));
  const worst = e.groups.reduce<{ label: string; v: number } | null>((m, g) => {
    const v = g.share[keyOf(g.share)]?.median;
    return v !== null && v !== undefined && (!m || v > m.v) ? { label: g.label, v } : m;
  }, null);
  return (
    <Bars
      title={`Share of cells at or above ${fmtNum(Number(t), 0)} ${u}`}
      units="% of cells"
      categories={cats}
      orientation="h"
      series={[{ id: "share", label: "Share of cells", values: pct(vals), lo: pct(lo), hi: pct(hi) }]}
      valueLabel="Cells at or above the threshold"
      unit="%"
      decimals={1}
      domain={[0, 100]}
      actions={
        ths.length > 1 ? (
          <Seg<string> label="Threshold" size="small" value={t} onChange={setTh} options={ths.map((x) => ({ value: x, label: `≥ ${fmtNum(Number(x), 0)} ${u}` }))} />
        ) : undefined
      }
      caption={`Today ${fmtPct(today, 1)} of cells reach ${fmtNum(Number(t), 0)} ${u}${worst ? `; ${worst.label} raises this to ${fmtPct(worst.v, 1)} (median model; whiskers are the 10th–90th percentile of models)` : ""}.`}
    />
  );
}

/** Observed temperature + the projection's warming (SPEC §6.3 Climate layers, client-side). */
function FutureMap({ warming }: { warming: ClimateProjection[] }) {
  const rid = useRid();
  const units = useUnits();
  const u = unitLabel(units.target);
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const [pid, setPid] = useState(warming[0]?.id ?? "");
  const [stat, setStat] = useState<"median" | "p10" | "p90">("median");
  const proj = warming.find((p) => p.id === pid) ?? warming[0];
  const obs = useMemo(() => (layers.data ? findLayer(layers.data, "obs") : null), [layers.data]);
  const base = useMemo(() => (grid.data ? runLayerLoader(rid, grid.data.meta.etag) : null), [rid, grid.data]);
  const delta = proj ? (proj[stat] ?? 0) : 0;
  const groups = useMemo<LayerGroup[]>(() => {
    if (!obs || !proj) return [];
    const meta: LayerMeta = {
      ...obs,
      key: `future:${proj.id}:${stat}`,
      group: "climate",
      label: `${projLabel(proj)} (${stat === "median" ? "median" : stat === "p10" ? "10th percentile" : "90th percentile"} warming)`,
      desc: `Observed temperature plus ${fmtSignedValue(delta, u, 2)} of warming.`,
      stats: {
        ...obs.stats,
        lo: shift(obs.stats.lo, delta),
        hi: shift(obs.stats.hi, delta),
        mean: shift(obs.stats.mean, delta),
        p1: shift(obs.stats.p1, delta),
        p2: shift(obs.stats.p2, delta),
        p50: shift(obs.stats.p50, delta),
        p98: shift(obs.stats.p98, delta),
        p99: shift(obs.stats.p99, delta),
      },
      center: obs.center !== null ? obs.center : null,
    };
    return [{ id: "climate", label: "Future temperature", layers: [meta] }];
  }, [obs, proj, stat, delta, u]);
  const load = useMemo(() => (base && obs ? async () => addConst(await base(obs), delta) : null), [base, obs, delta]);
  if (grid.error) return <EmptyState error={grid.error} />;
  if (layers.data && !obs) return <p className="cap">This run has no observed-temperature layer to project.</p>;
  if (!grid.data || !load || !groups.length) return <p className="cap">Loading the map…</p>;
  return (
    <div className="stack">
      <div className="row">
        <label className="row cap">
          Projection
          <select value={proj?.id ?? ""} onChange={(e) => setPid(e.target.value)} aria-label="Projection">
            {warming.map((p) => (
              <option key={p.id} value={p.id}>
                {projLabel(p)}
              </option>
            ))}
          </select>
        </label>
        <Seg<"median" | "p10" | "p90">
          label="Warming statistic"
          size="small"
          value={stat}
          onChange={setStat}
          options={[
            { value: "p10", label: "p10" },
            { value: "median", label: "median" },
            { value: "p90", label: "p90" },
          ]}
        />
      </div>
      <MapView grid={grid.data} groups={groups} loadLayer={load} layerKey={groups[0].layers[0].key} height={420} title="Future temperature" />
    </div>
  );
}

const shift = (v: number | null, d: number) => (v === null ? null : v + d);

function addConst(values: ArrayLike<number>, d: number): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i] + d;
  return out;
}

export default function Climate() {
  const rid = useRid();
  return (
    <ViewPage
      view="climate"
      title="Climate"
      intro="Downscaled CMIP6 change factors added to the campaign temperatures: how hot the city gets under each scenario, and how much configured adaptation offsets."
      actions={
        <Link to={`/r/${encodeURIComponent(rid)}/lab/climate`} className="btn small">
          Climate × adaptation in the Lab
        </Link>
      }
    >
      {(s) => (
        <>
          <Section title="Warming" data={s.warming}>
            {(w) => <Warming w={w} />}
          </Section>
          <div className="grid2">
            <Section title="Exposure" data={s.exposure}>
              {(e) => <Exposure e={e} />}
            </Section>
            <Section title="Warming offset" data={s.offset}>
              {(o) =>
                o.length ? (
                  <SmallMultiples
                    title="Share of median warming offset"
                    items={o}
                    minPanelWidth={200}
                    panelTitle={(x) => `${x.label} · ${x.variant}`}
                    renderPanel={(x) => (
                      <Gauge
                        bare
                        title={x.label}
                        value={x.share}
                        min={0}
                        max={Math.max(1, x.share ?? 0)}
                        ticks={[1]}
                        label="of median warming offset"
                        format={(v) => fmtPct(v, 0)}
                        tone={(x.share ?? 0) >= 1 ? "good" : "accent"}
                        size={180}
                      />
                    )}
                    table={{
                      columns: [
                        { key: "p", label: "Projection" },
                        { key: "v", label: "Variant" },
                        { key: "s", label: "Share offset", unit: "%" },
                      ],
                      rows: o.map((x) => [x.label, x.variant, x.share === null ? null : x.share * 100]),
                    }}
                    caption="1 (the tick) means the adaptation fully cancels the median warming of that projection."
                  />
                ) : (
                  <Block title="Warming offset">
                    <p className="cap">
                      No adaptation variants were configured for this run. Try scenarios against these futures in{" "}
                      <Link to={`/r/${encodeURIComponent(rid)}/lab/climate`}>Lab › Climate</Link>.
                    </p>
                  </Block>
                )
              }
            </Section>
          </div>
          <Section title="Model × projection table" data={s.models_table}>
            {(t) => (
              <Block title="Warming by model">
                <GenericTableView table={t} caption="Warming by model and projection" csvName="warming-by-model" />
              </Block>
            )}
          </Section>
          <Section title="Future-temperature map" data={s.warming}>
            {(w) => (
              <Block title="Future temperature">
                <FutureMap warming={w} />
              </Block>
            )}
          </Section>
        </>
      )}
    </ViewPage>
  );
}
