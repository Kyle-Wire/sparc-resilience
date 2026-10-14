// Climate (SPEC §6.4), the first part of Heat risk › Future climate (phase 2, the canonical home
// of the climate numbers): warming per SSP × period (median, p10–p90, min–max, per-model strip,
// with the model × projection table folded under it), exposure bars (share of cells ≥ T today vs
// futures and adaptation variants, with a threshold toggle), the "% of median warming offset"
// gauges (only for a run with adaptation variants, and never when Heat risk embeds the page:
// there "Can adaptation keep up?", the Lab's Climate × adaptation matrix that follows on the same
// page (#adapt), is the one home of the offset) and a future-temperature map computed
// client-side (observed + warming).
// Plain words (phase 4): futures by emissions and period ("Moderate emissions, mid-century
// (2041–2060)"; "Moderate, mid-century" on chart rows), the pathway code only in a tooltip's
// technical line; p10/median/p90 as the low end / middle / high end of the climate models; cells
// as places; the offset as the share of the expected warming a scenario cancels out.
import { useMemo, useState } from "react";
import type { ClimateSections, ClimateProjection } from "../../api/runs";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { Bars, DotRange, Gauge, SmallMultiples } from "../../charts";
import { EmptyState } from "../../components/ui/EmptyState";
import { InfoTip } from "../../components/ui/InfoTip";
import { Seg } from "../../components/ui/Seg";
import { Term } from "../../components/ui/Term";
import { term } from "../../content/glossary";
import { MapView } from "../../map/MapView";
import { runLayerLoader, useRunGrid, useRunLayers } from "../../map/data";
import { findLayer } from "../../map/LayerPicker";
import { Link } from "../../router";
import { fmtNum, fmtPct, fmtSignedValue, unitLabel } from "../../theme/format";
import { futureName, futureShort, plainCase } from "./climateWords";
import { Block, GenericTableView, Section, ViewPage, useRid, useUnits } from "./common";
import { heatHref, useDewpoint } from "./HeatRisk";

/** A projection in full ("Moderate emissions, mid-century (2041–2060)") and on a chart row ("Moderate, mid-century"). */
const projLabel = (p: Pick<ClimateProjection, "experiment" | "label" | "period">) => futureName(p.experiment, p.period, p.label);
const projShort = (p: Pick<ClimateProjection, "experiment" | "label" | "period">) => futureShort(p.experiment, p.period, p.label);
/** The climate models' low end, middle and high end (10th percentile, median, 90th percentile). */
const STAT_WORD = { p10: "low end", median: "middle estimate", p90: "high end" } as const;

/** The ⓘ of the warming figures: what the emissions futures and the climate-model range are. */
function WarmingNote() {
  return (
    <>
      <Term id="ssp" /> · <Term id="climate_models" />
    </>
  );
}

function Warming({ w }: { w: ClimateProjection[] }) {
  const u = unitLabel(useUnits().target);
  const mid = w.find((p) => /2041/.test(p.period) && /245/.test(p.experiment)) ?? w[0];
  return (
    <DotRange
      title="Warming by emissions future and period"
      note={<WarmingNote />}
      units={`${u} warmer than 1995–2014`}
      rows={w.map((p) => ({
        id: p.id,
        label: projShort(p),
        est: p.median,
        lo: p.p10,
        hi: p.p90,
        lo2: p.min,
        hi2: p.max,
        strip: Object.values(p.by_model).filter((v): v is number => v !== null),
      }))}
      valueLabel="Warming"
      unit={u}
      rangeLabel="Most models"
      outerLabel="All models"
      caption={
        mid
          ? `${projLabel(mid)}: ${fmtSignedValue(mid.median, u, 2)}, the middle estimate of ${mid.n_models ?? "?"} climate models (most models: ${fmtNum(mid.p10, 2)} to ${fmtNum(mid.p90, 2)} ${u}). Grey dots are individual models.`
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
  if (!t) return <p className="cap">No temperatures to count are set in this run's settings.</p>;
  const keyOf = (rec: Record<string, unknown>) => Object.keys(rec).find((k) => Number(k) === Number(t)) ?? t;
  const today = e.present[keyOf(e.present)] ?? null;
  const cats = ["Today", ...e.groups.map((g) => `${futureShort(g.experiment, g.period, g.label)}${g.variant && g.variant !== "no adaptation" ? ` · ${g.variant}` : ""}`)];
  const vals = [today, ...e.groups.map((g) => g.share[keyOf(g.share)]?.median ?? null)];
  const lo = [null, ...e.groups.map((g) => g.share[keyOf(g.share)]?.p10 ?? null)];
  const hi = [null, ...e.groups.map((g) => g.share[keyOf(g.share)]?.p90 ?? null)];
  const pct = (a: (number | null)[]) => a.map((v) => (v === null ? null : v * 100));
  const worst = e.groups.reduce<{ label: string; v: number } | null>((m, g) => {
    const v = g.share[keyOf(g.share)]?.median;
    // ">=": on a tie (several futures at 100%) the later group, listed hotter, wins
    return v !== null && v !== undefined && (!m || v >= m.v) ? { label: futureName(g.experiment, g.period, g.label), v } : m;
  }, null);
  return (
    <Bars
      title={`Share of places with an air temperature at or above ${fmtNum(Number(t), 0)} ${u}`}
      note={
        <>
          Each <Term id="cell">place</Term> is one square of the map
        </>
      }
      labelGutter={Math.max(...cats.map((c) => c.length)) * 6.6 + 16}
      units="% of places"
      categories={cats}
      orientation="h"
      series={[{ id: "share", label: "Share of places", values: pct(vals), lo: pct(lo), hi: pct(hi) }]}
      valueLabel="Places at or above the air temperature"
      unit="%"
      decimals={1}
      domain={[0, 100]}
      actions={
        ths.length > 1 ? (
          <Seg<string> label="Air temperature" size="small" value={t} onChange={setTh} options={ths.map((x) => ({ value: x, label: `≥ ${fmtNum(Number(x), 0)} ${u}` }))} />
        ) : undefined
      }
      caption={`Today ${fmtPct(today, 1)} of places reach an air temperature of ${fmtNum(Number(t), 0)} ${u}.${
        worst
          ? worst.v >= 0.995
            ? ` Every place reaches ${fmtNum(Number(t), 0)} ${u} in the hottest futures (e.g. ${worst.label}).`
            : ` In the hottest future, ${worst.label}, ${fmtPct(worst.v, 1)} of places do (the climate models' middle estimate; whiskers show the range of most models).`
          : ""
      }`}
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
      label: `${projLabel(proj)} (${STAT_WORD[stat]} of warming)`,
      desc: `Measured temperature plus ${fmtSignedValue(delta, u, 2)} of warming.`,
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
  if (layers.data && !obs) return <p className="cap">This run has no measured-temperature layer to project.</p>;
  if (!grid.data || !load || !groups.length) return <p className="cap">Loading the map…</p>;
  return (
    <div className="stack">
      <div className="row">
        <label className="row cap">
          Future climate
          <select value={proj?.id ?? ""} onChange={(e) => setPid(e.target.value)} aria-label="Future climate">
            {warming.map((p) => (
              <option key={p.id} value={p.id}>
                {projLabel(p)}
              </option>
            ))}
          </select>
        </label>
        <Seg<"median" | "p10" | "p90">
          label="Climate-model estimate"
          size="small"
          value={stat}
          onChange={setStat}
          options={[
            { value: "p10", label: "Low end" },
            { value: "median", label: "Middle" },
            { value: "p90", label: "High end" },
          ]}
        />
        <InfoTip term="climate_models" technical="Low end = 10th percentile, middle = median, high end = 90th percentile of the CMIP6 models" />
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

function Offset({ o }: { o: NonNullable<ClimateSections["offset"]> }) {
  return (
    <SmallMultiples
      title={term("climate_offset").label}
      note={
        <>
          Cancels out this share of the expected warming
          <InfoTip term="climate_offset" />
        </>
      }
      items={o}
      minPanelWidth={200}
      panelTitle={(x) => `${plainCase(x.label)} · ${x.variant}`}
      renderPanel={(x) => (
        <Gauge
          bare
          title={x.label}
          value={x.share}
          min={0}
          max={Math.max(1, x.share ?? 0)}
          ticks={[1]}
          label="of the expected warming cancelled"
          format={(v) => fmtPct(v, 0)}
          tone={(x.share ?? 0) >= 1 ? "good" : "accent"}
          size={180}
        />
      )}
      table={{
        columns: [
          { key: "p", label: "Future climate" },
          { key: "v", label: "Adaptation" },
          { key: "s", label: "Warming cancelled", unit: "%" },
        ],
        rows: o.map((x) => [plainCase(x.label), x.variant, x.share === null ? null : x.share * 100]),
      }}
      caption="The tick marks 100%: there the adaptation cancels out all of the expected warming in that future."
    />
  );
}

/**
 * Future climate. `embedded` (Heat risk › Future climate, where "Can adaptation keep up?" follows
 * on the same page): no offset gauges and no link down to that section.
 */
export default function Climate({ embedded }: { embedded?: boolean } = {}) {
  const rid = useRid();
  const [dewpoint] = useDewpoint();
  const adaptHref = `${heatHref(rid, "future", dewpoint)}#adapt`;
  return (
    <ViewPage
      view="climate"
      title="Future climate"
      intro={
        <>
          How much hotter the city gets in each <Term id="ssp">emissions future</Term> and time period: the{" "}
          <Term id="climate_models" technical="Downscaled CMIP6 change factors">climate models&apos;</Term> warming added to the measured temperatures
          {embedded ? "." : ", and how much of it the configured adaptation cancels out."}
        </>
      }
      actions={
        embedded ? undefined : (
          <Link to={adaptHref} className="cap">
            Can adaptation keep up? ↓
          </Link>
        )
      }
    >
      {(s) => (
        <>
          <Section title="Warming" data={s.warming}>
            {(w) => <Warming w={w} />}
          </Section>
          <Section title="Warming by climate model" data={s.models_table}>
            {(t) => (
              <details className="card" data-models-table="true">
                <summary>Warming by climate model</summary>
                <GenericTableView table={{ ...t, columns: t.columns.map((c) => ({ ...c, label: plainCase(c.label) })) }} caption="Warming by climate model and future" csvName="warming-by-model" />
              </details>
            )}
          </Section>
          {!embedded && s.offset?.length ? (
            <div className="grid2">
              <Section title="Exposure" data={s.exposure}>
                {(e) => <Exposure e={e} />}
              </Section>
              <Section title={term("climate_offset").label} data={s.offset}>
                {(o) => <Offset o={o} />}
              </Section>
            </div>
          ) : (
            // No adaptation variants (or Heat risk's matrix below owns the offset): the exposure spans the width.
            <Section title="Exposure" data={s.exposure}>
              {(e) => <Exposure e={e} />}
            </Section>
          )}
          <Section title="Future temperature map" data={s.warming}>
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
