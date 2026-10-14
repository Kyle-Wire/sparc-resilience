// Climate × adaptation (SPEC §7.11): pick adaptations (exact results and configured scenarios),
// free thresholds, the warming statistic (median / p10 / p90 / one model) and the SSP × period
// filters. Charts: warming dot-range with the per-model strip, exposure bars (p10–p90 across
// models), offset gauges. Maps of future temperature and of ≥ T exceedance are computed in the
// browser (observed + warming statistic + adaptation ΔT).
// Phase 2: the tool lives under Heat risk › Future climate as "Can adaptation keep up?"
// (`ClimateAdaptation embedded`), below the run's own warming chart and future-temperature map,
// so the embedded form leaves those out (each climate number once) and shows the scenario ×
// projection matrix once at least one adaptation is picked. The default export is the complete
// tool in the Lab's frame (the old `/r/:rid/lab/climate` URL redirects to Heat risk › Future).
// Plain words (phase 4): emissions futures and periods by their glossary names ("Moderate
// emissions", "Mid-century (2041–2060)"), the pathway codes only in an ⓘ's technical line; the
// statistic as the climate models' low end / middle / high end; cells as places; the offset
// gauges as "Cancels out N% of the expected warming".
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { errorMessage } from "../../api/client";
import { exploreClimate, getResultLayer, useClimateFactors, useRunScenarios, useScenarios, type ClimateExplore, type ItemRef, type PeopleExposureRow, type WarmingRow } from "../../api/lab";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { Bars, DotRange, Gauge } from "../../charts";
import { EmptyState } from "../../components/ui/EmptyState";
import { InfoTip } from "../../components/ui/InfoTip";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { periodLabel, sspLabel, term } from "../../content/glossary";
import { useRunDetail } from "../../layouts/resources";
import { MapView } from "../../map/MapView";
import { runLayerLoader, useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import { quantiles } from "../../map/domain";
import { codecs, useRoute, useUrlState } from "../../router";
import { emissionsName, futureName, futureShort, pathwayCodes, periodName } from "../runhub/climateWords";
import { useUi } from "../../stores/ui";
import { fmtInt, fmtNum, fmtPct, fmtSignedValue, fmtTempChange, unitLabel } from "../../theme/format";
import { LabFrame } from "./components/LabFrame";
import { Tip } from "./components/Tip";
import { decodeStatistic, encodeStatistic, exceedance, futureTemperature, presentToday, shareAt, shareAtOrAbove, warmingValue } from "./model/climate";
import { decodeItemRef, encodeItemRef, itemLayerKey } from "./model/doc";
import { syntheticMeta } from "./model/layers";
import { parseNumberList } from "./model/library";
import { debounce } from "./model/timing";

/** A projection in full ("Moderate emissions, mid-century (2041–2060)") and on a chart row ("Moderate, mid-century"). */
const projLabel = (r: { experiment: string; label: string; period: string }) => futureName(r.experiment, r.period, r.label);
const projShort = (r: { experiment: string; label: string; period: string }) => futureShort(r.experiment, r.period, r.label);

/** The warming statistic in plain words: the climate models' middle estimate, low end, high end, or one model. */
function statWord(stat: ReturnType<typeof decodeStatistic>): string {
  if (typeof stat === "object") return `climate model ${stat.model}`;
  return stat === "median" ? "middle climate-model estimate" : stat === "p10" ? "low end of the climate models" : "high end of the climate models";
}

/** Room for chart-row labels (Bars cuts them to the gutter): about 6.6 px a character. */
const gutterFor = (labels: string[]) => Math.max(0, ...labels.map((l) => l.length)) * 6.6 + 16;

function ClimateMaps({
  rid,
  warming,
  adaptations,
  threshold,
  stat,
  unit,
  embedded = false,
}: {
  rid: string;
  warming: WarmingRow[];
  adaptations: { ref: ItemRef; label: string }[];
  threshold: number | null;
  stat: ReturnType<typeof decodeStatistic>;
  unit: string;
  /** Adapted futures only (the unadapted future map and the warming figures are shown above it). */
  embedded?: boolean;
}) {
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const [proj, setProj] = useState(0);
  const [chosen, setAdapt] = useState<string>("");
  // Embedded, the map always shows one of the picked adaptations (no "No adaptation" choice).
  const keys = adaptations.map((a) => encodeItemRef(a.ref));
  const adapt = !embedded || keys.includes(chosen) ? chosen : keys[0] ?? "";
  const [obs, setObs] = useState<Float32Array | null>(null);
  const [delta, setDelta] = useState<{ key: string; values: Float32Array | null }>({ key: "", values: null });
  const load = useMemo(() => (grid.data ? runLayerLoader(rid, grid.data.meta.etag) : null), [rid, grid.data]);
  const all = useMemo(() => (layers.data ?? []).flatMap((g) => g.layers), [layers.data]);
  useEffect(() => {
    const m = all.find((l) => l.key === "obs");
    if (!m || !load) return;
    let live = true;
    load(m).then((v) => live && setObs(v as Float32Array), () => {});
    return () => {
      live = false;
    };
  }, [all, load]);
  useEffect(() => {
    let live = true;
    const ref = adapt ? decodeItemRef(adapt) : null;
    if (!ref || !load) {
      setDelta({ key: adapt, values: null });
      return;
    }
    const p: Promise<LayerValues> =
      ref.kind === "result"
        ? getResultLayer(ref.id, "delta")
        : (() => {
            const key = itemLayerKey(ref);
            const m = all.find((l) => l.key === key);
            return m ? load(m) : Promise.reject(new Error(`layer ${key} is not in this run`));
          })();
    p.then((v) => live && setDelta({ key: adapt, values: v as Float32Array }), () => live && setDelta({ key: adapt, values: null }));
    return () => {
      live = false;
    };
  }, [adapt, all, load]);

  const row = warming[Math.min(proj, warming.length - 1)];
  const w = row ? warmingValue(row, stat) : null;
  const future = useMemo(() => (obs && w !== null ? futureTemperature(obs, w, delta.key === adapt ? delta.values : null) : null), [obs, w, delta, adapt]);
  const exceed = useMemo(() => (future && threshold !== null ? exceedance(future, threshold) : null), [future, threshold]);
  const center = useMemo(() => (obs ? quantiles(obs, [0.5])[0] : null), [obs]);
  const groups = useMemo<LayerGroup[]>(() => {
    const ls: LayerMeta[] = [
      syntheticMeta(
        { key: "climate:future", label: "Future temperature", unit, scale: "div", center: center ?? 0, decimals: 1, desc: "Measured temperature + warming + the adaptation's temperature change (worked out in your browser)." },
        future,
      ),
    ];
    if (threshold !== null) ls.push(syntheticMeta({ key: "climate:exceed", label: `At or above ${fmtNum(threshold, 1)} ${unitLabel(unit)}`, unit: "", scale: "cat", labels: ["below", "at or above"], dtype: "uint8" }, exceed));
    return [{ id: "climate", label: "Climate", layers: ls }];
  }, [future, exceed, threshold, unit, center]);
  const loader = useCallback(async (m: LayerMeta): Promise<LayerValues> => (m.key === "climate:exceed" ? exceed ?? new Uint8Array(0) : future ?? new Float32Array(0)), [future, exceed]);

  if (!grid.data) return <p className="cap">Loading the map…</p>;
  if (!all.some((l) => l.key === "obs")) return <p className="cap">The measured temperature layer is not in this run.</p>;
  const share = future && threshold !== null ? shareAtOrAbove(future, threshold) : null;
  return (
    <section className="stack" aria-label="Future maps">
      <div className="row">
        <select aria-label="Future climate for the map" value={proj} onChange={(e) => setProj(Number(e.target.value))}>
          {warming.map((r, i) => (
            <option key={`${r.experiment}-${r.period}`} value={i}>
              {projLabel(r)}
            </option>
          ))}
        </select>
        <select aria-label="Adaptation for the map" value={adapt} onChange={(e) => setAdapt(e.target.value)}>
          {embedded ? null : <option value="">No adaptation</option>}
          {adaptations.map((a) => (
            <option key={encodeItemRef(a.ref)} value={encodeItemRef(a.ref)}>
              {a.label}
            </option>
          ))}
        </select>
        <span className="cap">
          {embedded
            ? share !== null
              ? `${fmtPct(share)} of places at or above an air temperature of ${fmtNum(threshold, Number.isInteger(threshold) ? 0 : 1)} ${unitLabel(unit)} (warming: ${statWord(stat)})`
              : null
            : `Warming ${w !== null ? fmtSignedValue(w, unit, 2) : "—"} (${statWord(stat)})${share !== null ? ` · ${fmtPct(share)} of places at or above an air temperature of ${fmtNum(threshold, Number.isInteger(threshold) ? 0 : 1)} ${unitLabel(unit)}` : ""}`}
        </span>
      </div>
      {future ? <MapView grid={grid.data} groups={groups} loadLayer={loader} height={460} title="Future temperature" /> : <p className="cap">Loading measured temperatures…</p>}
    </section>
  );
}

/**
 * Climate × adaptation for run `rid`. `embedded` (Heat risk › Future climate, "Can adaptation
 * keep up?"): no Lab frame, no warming chart (the page's Warming chart is the one), and the
 * exposure matrix, offset gauges, residents table and adapted-future map appear once an
 * adaptation is picked (nothing is asked of the server before that).
 */
export function ClimateAdaptation({ rid, embedded = false }: { rid: string; embedded?: boolean }) {
  const detail = useRunDetail(rid);
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = detail.data?.run.project_id ?? ctxPid;
  const grid = useRunGrid(rid);
  const factors = useClimateFactors(rid);
  const runSc = useRunScenarios(rid);
  const scenarios = useScenarios(pid, { run: rid }, rid);
  const [adaptRaw, setAdaptRaw] = useUrlState("adapt", codecs.list());
  const [statRaw, setStatRaw] = useUrlState("stat", codecs.string("median"));
  const [tText, setTText] = useUrlState("t", codecs.string(""));
  const [exps, setExps] = useUrlState("exp", codecs.list());
  const [pers, setPers] = useUrlState("period", codecs.list());
  const [thresholdIdx, setThresholdIdx] = useState(0);
  const [out, setOut] = useState<{ data: ClimateExplore | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: false });
  const stat = decodeStatistic(statRaw);
  const unit = grid.data?.meta.units.target ?? "";
  const u = unitLabel(unit);

  const names = useMemo(() => new Map((scenarios.data ?? []).flatMap((s) => (s.latest ? [[s.latest.id, s.name] as const] : []))), [scenarios.data]);
  const options = useMemo(() => {
    const opts: { ref: ItemRef; label: string; configName?: string }[] = [];
    for (const r of runSc.data?.results ?? []) if (r.kind === "exact") opts.push({ ref: { kind: "result", id: r.id }, label: names.get(r.id) ?? `Result ${r.id}` });
    for (const c of runSc.data?.configured ?? []) opts.push({ ref: { kind: "configured", slug: c.slug }, label: c.label || c.name, ...(c.label && c.label !== c.name ? { configName: c.name } : {}) });
    return opts;
  }, [runSc.data, names]);
  const adaptations = useMemo(() => adaptRaw.map(decodeItemRef).filter((r): r is ItemRef => r !== null), [adaptRaw]);
  // Toggle one adaptation chip; entries that are not item refs (an old `?adapt=1`) are dropped.
  const toggleAdapt = (k: string) => {
    const cur = adaptRaw.filter((x) => decodeItemRef(x) !== null);
    setAdaptRaw(cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]);
  };
  const picked = !embedded || adaptations.length > 0;
  const frame = (children: ReactNode) =>
    embedded ? (
      <div className="stack" data-climate-adaptation="embedded">
        {children}
      </div>
    ) : (
      <LabFrame rid={rid} title="Climate × adaptation">
        {children}
      </LabFrame>
    );
  const thresholds = useMemo(() => parseNumberList(tText).values, [tText]);

  const explore = useMemo(
    () =>
      debounce((body: Parameters<typeof exploreClimate>[1]) => {
        setOut((o) => ({ ...o, loading: true }));
        exploreClimate(rid, body).then(
          (data) => setOut({ data, error: null, loading: false }),
          (e: unknown) => setOut((o) => ({ ...o, error: errorMessage(e), loading: false })),
        );
      }, 200),
    [rid],
  );
  const bodyKey = JSON.stringify([adaptRaw, statRaw, thresholds, exps, pers]);
  useEffect(() => {
    if (!factors.data?.present || !picked) return;
    explore({
      adaptations,
      ...(thresholds.length ? { thresholds } : {}),
      ...(exps.length ? { experiments: exps } : {}),
      ...(pers.length ? { periods: pers } : {}),
      statistic: stat,
    });
    return () => explore.cancel();
    // bodyKey stands for the request body
  }, [bodyKey, factors.data?.present, picked, explore]);

  if (factors.error && !factors.data) return <EmptyState error={factors.error} />;
  if (!factors.data || !grid.data) return <p className="cap">Loading…</p>;
  const f = factors.data;
  if (!f.present) {
    return frame(
      <EmptyState
        title="No climate projections yet"
        body="Fetch the climate models' projections (CMIP6 change factors) for this city to see future warming and how much of it each adaptation cancels out."
        action={f.action}
      />,
    );
  }
  const warming = f.warming.filter((w) => (!exps.length || exps.includes(w.experiment)) && (!pers.length || pers.includes(w.period)));
  const data = out.data;
  const ths = data?.thresholds ?? thresholds;
  const t = ths[Math.min(thresholdIdx, Math.max(0, ths.length - 1))] ?? null;
  const tKey = (rec: Record<string, unknown>) => (t === null ? "" : Object.keys(rec).find((k) => Number(k) === t) ?? String(t));
  const projections = data?.projections ?? [];
  const variants = projections[0]?.variants.map((v) => v.name) ?? [];
  const present = presentToday(data);
  const gaugeProj = projections.find((p) => /245/.test(p.experiment) && /2041/.test(p.period)) ?? projections[0];

  return frame(
    <>
      <div className="lab-form">
        <div className="field">
          <span className="field-label">
            Climate-model estimate
            <InfoTip term="climate_models" technical="Middle = median, low end = 10th percentile, high end = 90th percentile of the CMIP6 models" />
          </span>
          <Seg<string>
            label="Climate-model estimate"
            size="small"
            value={typeof stat === "object" ? "model" : stat}
            onChange={(v) => setStatRaw(v === "model" ? encodeStatistic({ model: f.models[0] ?? "" }) : v)}
            options={[
              { value: "median", label: "Middle" },
              { value: "p10", label: "Low end" },
              { value: "p90", label: "High end" },
              { value: "model", label: "One model", disabled: !f.models.length },
            ]}
          />
          {typeof stat === "object" ? (
            <select aria-label="Model" value={stat.model} onChange={(e) => setStatRaw(encodeStatistic({ model: e.target.value }))}>
              {f.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : null}
        </div>
        <label className="field">
          <span className="field-label">Thresholds ({u})</span>
          <input value={tText} placeholder={data ? data.thresholds.join(", ") : "from the run's settings"} onChange={(e) => setTText(e.target.value)} />
        </label>
        <div className="field">
          <span className="field-label">
            Emissions futures
            <InfoTip term="ssp" technical={pathwayCodes(f.experiments)} />
          </span>
          <div className="chips" role="group" aria-label="Emissions futures">
            {f.experiments.map((e) => (
              <button key={e} type="button" className="chip" aria-pressed={!exps.length || exps.includes(e)} onClick={() => setExps(exps.includes(e) ? exps.filter((x) => x !== e) : [...exps, e])}>
                {emissionsName(e)}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <span className="field-label">
            Time periods
            <InfoTip term="climate_period" />
          </span>
          <div className="chips" role="group" aria-label="Time periods">
            {f.periods.map((p) => (
              <button key={p} type="button" className="chip" aria-pressed={!pers.length || pers.includes(p)} onClick={() => setPers(pers.includes(p) ? pers.filter((x) => x !== p) : [...pers, p])}>
                {periodName(p)}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="field">
        <span className="field-label">Adaptations</span>
        <div className="chips" role="group" aria-label="Adaptations">
          {options.map((o) => {
            const k = encodeItemRef(o.ref);
            return (
              <Tip key={k} tip={o.configName ? `Its name in the run's settings: ${o.configName}` : null}>
                <button type="button" className="chip" aria-pressed={adaptRaw.includes(k)} onClick={() => toggleAdapt(k)}>
                  {o.label}
                </button>
              </Tip>
            );
          })}
          {!options.length ? <span className="cap">No exact results or configured scenarios on this run yet.</span> : null}
        </div>
      </div>
      {out.error && picked ? <p className="callout" data-tone="crit">{out.error}</p> : null}
      {embedded ? null : (
        <DotRange
          title="Warming by emissions future and period"
          units={`${u} warmer than 1995–2014`}
          rows={warming.map((w) => ({
            id: `${w.experiment}-${w.period}`,
            label: projShort(w),
            est: warmingValue(w, stat),
            lo: w.p10,
            hi: w.p90,
            lo2: w.min,
            hi2: w.max,
            strip: Object.values(w.by_model).filter((v) => Number.isFinite(v)),
          }))}
          valueLabel="Warming"
          unit={u}
          rangeLabel="Most models"
          outerLabel="All models"
          caption={`Dots: the ${statWord(stat)}; bars: the range of most climate models (10th to 90th percentile); grey points: individual models.`}
        />
      )}
      {!picked ? (
        <p className="cap">
          Pick one or more adaptations above to see, for each future climate, how much of the expected warming they cancel out and how many places and residents stay at or
          above each temperature.
        </p>
      ) : data && t !== null ? (
        <>
          <div className="row">
            <span className="cap">Air temperature</span>
            <Seg<number> label="Air temperature" size="small" value={Math.min(thresholdIdx, ths.length - 1)} onChange={setThresholdIdx} options={ths.map((x, i) => ({ value: i, label: `${fmtNum(x, 1)} ${u}` }))} />
            {out.loading ? <span className="spinner" aria-label="Updating" /> : null}
          </div>
          <Bars
            title={`Share of places with an air temperature at or above ${fmtNum(t, Number.isInteger(t) ? 0 : 1)} ${u}`}
            note={
              <>
                Each place is one square of the map
                <InfoTip term="cell" />
              </>
            }
            units="share of places"
            labelGutter={gutterFor(projections.map(projShort))}
            categories={["Today", ...projections.map(projShort)]}
            series={variants.map((name, k) => ({
              id: name,
              label: name,
              values: [k === 0 && present && t !== null ? shareAt(present.share_at_or_above, t) : null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.median ?? null)],
              lo: [null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.p10 ?? null)],
              hi: [null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.p90 ?? null)],
              muted: k === 0 && variants.length > 1,
            }))}
            valueLabel="Share of places"
            unit="share"
            categoryLabel="Future climate"
            domain={[0, 1]}
            caption="Bars: the climate models' middle estimate; whiskers: the range of most models (10th to 90th percentile)."
          />
          {gaugeProj && gaugeProj.variants.some((v) => v.offset_share_of_median_warming !== null) ? (
            <div className="grid3">
              {gaugeProj.variants
                .filter((v) => v.offset_share_of_median_warming !== null)
                .map((v) => (
                  <Gauge
                    key={v.name}
                    title={v.name}
                    note={
                      <>
                        {term("climate_offset").label}
                        <InfoTip term="climate_offset" />
                      </>
                    }
                    label="of the expected warming cancelled"
                    value={v.offset_share_of_median_warming}
                    min={0}
                    max={Math.max(1, v.offset_share_of_median_warming ?? 0)}
                    ticks={[1]}
                    format={(x) => fmtPct(x)}
                    caption={`Cancels out ${fmtPct(v.offset_share_of_median_warming)} of the warming expected by ${periodLabel(gaugeProj.period)}, with ${sspLabel(gaugeProj.experiment)}. Citywide: ${fmtTempChange(v.adaptation_mean_delta, unit, 3)}.`}
                  />
                ))}
            </div>
          ) : null}
          {data.people_exposure?.length ? (
            <Table<PeopleExposureRow>
              caption="Residents at or above each air temperature"
              csvName="people-exposure"
              rowKey={(r, i) => `${r.experiment ?? "today"}-${r.period ?? ""}-${r.variant}-${i}`}
              columns={[
                { key: "case", label: "Climate", value: (r) => (r.period ? (r.experiment ? futureName(r.experiment, r.period, r.label) : `${r.label} ${periodLabel(r.period)}`) : r.label) },
                { key: "variant", label: "Adaptation", value: (r) => r.variant },
                { key: "warming", label: "Warming", unit: u, align: "right", value: (r) => r.warming, render: (r) => (r.warming !== null ? fmtNum(r.warming, 2) : "—") },
                ...data.thresholds.map((th) => ({
                  key: `ge_${th}`,
                  label: `≥ ${fmtNum(th, 1)} ${u}`,
                  align: "right" as const,
                  value: (r: PeopleExposureRow) => shareAt(r.people_ge, th),
                  render: (r: PeopleExposureRow) => {
                    const n = shareAt(r.people_ge, th);
                    return n === null ? "—" : `${fmtInt(Math.round(n))} (${fmtPct(shareAt(r.share_people_ge, th))})`;
                  },
                })),
              ]}
              rows={data.people_exposure}
            />
          ) : null}
        </>
      ) : out.loading ? (
        <p className="cap">Computing exposure…</p>
      ) : null}
      {picked ? (
        <ClimateMaps
          rid={rid}
          warming={warming}
          adaptations={options.filter((o) => adaptRaw.includes(encodeItemRef(o.ref)))}
          threshold={t}
          stat={stat}
          unit={unit}
          embedded={embedded}
        />
      ) : null}
    </>,
  );
}

/** The Climate × adaptation tool in the Lab's frame (the complete page, for run `:rid`). */
export default function Climate() {
  const { params } = useRoute();
  return <ClimateAdaptation rid={params.rid ?? ""} />;
}
