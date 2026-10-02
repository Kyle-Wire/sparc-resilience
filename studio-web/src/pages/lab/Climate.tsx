// Climate × adaptation (`/r/:rid/lab/climate`, SPEC §7.11): pick adaptations (exact results and
// configured scenarios), free thresholds, the warming statistic (median / p10 / p90 / one
// model) and the SSP × period filters. Charts: warming dot-range with the per-model strip,
// exposure bars (p10–p90 across models), offset gauges. Maps of future temperature and of
// ≥ T exceedance are computed in the browser (observed + warming statistic + adaptation ΔT).
import { useCallback, useEffect, useMemo, useState } from "react";
import { errorMessage } from "../../api/client";
import { exploreClimate, getResultLayer, useClimateFactors, useRunScenarios, useScenarios, type ClimateExplore, type ItemRef, type WarmingRow } from "../../api/lab";
import type { LayerGroup, LayerMeta } from "../../api/types";
import { Bars, DotRange, Gauge } from "../../charts";
import { EmptyState } from "../../components/ui/EmptyState";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { useRunDetail } from "../../layouts/resources";
import { MapView } from "../../map/MapView";
import { runLayerLoader, useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import { quantiles } from "../../map/domain";
import { codecs, useRoute, useUrlState } from "../../router";
import { useUi } from "../../stores/ui";
import { fmtNum, fmtPct, fmtSignedValue, unitLabel } from "../../theme/format";
import { LabFrame } from "./components/LabFrame";
import { decodeStatistic, encodeStatistic, exceedance, futureTemperature, presentToday, shareAt, shareAtOrAbove, statisticLabel, warmingValue } from "./model/climate";
import { decodeItemRef, encodeItemRef, itemLayerKey } from "./model/doc";
import { syntheticMeta } from "./model/layers";
import { parseNumberList } from "./model/library";
import { debounce } from "./model/timing";

const projLabel = (r: { label: string; period: string }) => `${r.label} ${r.period.replace("-", "–")}`;

function ClimateMaps({
  rid,
  warming,
  adaptations,
  threshold,
  stat,
  unit,
}: {
  rid: string;
  warming: WarmingRow[];
  adaptations: { ref: ItemRef; label: string }[];
  threshold: number | null;
  stat: ReturnType<typeof decodeStatistic>;
  unit: string;
}) {
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const [proj, setProj] = useState(0);
  const [adapt, setAdapt] = useState<string>("");
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
      syntheticMeta({ key: "climate:future", label: "Future temperature", unit, scale: "div", center: center ?? 0, decimals: 1, desc: "Observed + warming + adaptation ΔT (computed in the browser)." }, future),
    ];
    if (threshold !== null) ls.push(syntheticMeta({ key: "climate:exceed", label: `At or above ${fmtNum(threshold, 1)} ${unitLabel(unit)}`, unit: "", scale: "cat", labels: ["below", "at or above"], dtype: "uint8" }, exceed));
    return [{ id: "climate", label: "Climate", layers: ls }];
  }, [future, exceed, threshold, unit, center]);
  const loader = useCallback(async (m: LayerMeta): Promise<LayerValues> => (m.key === "climate:exceed" ? exceed ?? new Uint8Array(0) : future ?? new Float32Array(0)), [future, exceed]);

  if (!grid.data) return <p className="cap">Loading the map…</p>;
  if (!all.some((l) => l.key === "obs")) return <p className="cap">The observed temperature layer is not in this run.</p>;
  const share = future && threshold !== null ? shareAtOrAbove(future, threshold) : null;
  return (
    <section className="stack" aria-label="Future maps">
      <div className="row">
        <select aria-label="Projection for the map" value={proj} onChange={(e) => setProj(Number(e.target.value))}>
          {warming.map((r, i) => (
            <option key={`${r.experiment}-${r.period}`} value={i}>
              {projLabel(r)}
            </option>
          ))}
        </select>
        <select aria-label="Adaptation for the map" value={adapt} onChange={(e) => setAdapt(e.target.value)}>
          <option value="">No adaptation</option>
          {adaptations.map((a) => (
            <option key={encodeItemRef(a.ref)} value={encodeItemRef(a.ref)}>
              {a.label}
            </option>
          ))}
        </select>
        <span className="cap">
          Warming {w !== null ? fmtSignedValue(w, unit, 2) : "—"} ({statisticLabel(stat)}){share !== null ? ` · ${fmtPct(share)} of cells at or above ${fmtNum(threshold, 1)} ${unitLabel(unit)}` : ""}
        </span>
      </div>
      {future ? <MapView grid={grid.data} groups={groups} loadLayer={loader} height={460} title="Future temperature" /> : <p className="cap">Loading observed temperatures…</p>}
    </section>
  );
}

export default function Climate() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
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
    const opts: { ref: ItemRef; label: string }[] = [];
    for (const r of runSc.data?.results ?? []) if (r.kind === "exact") opts.push({ ref: { kind: "result", id: r.id }, label: names.get(r.id) ?? `Result ${r.id}` });
    for (const c of runSc.data?.configured ?? []) opts.push({ ref: { kind: "configured", slug: c.slug }, label: c.name });
    return opts;
  }, [runSc.data, names]);
  const adaptations = useMemo(() => adaptRaw.map(decodeItemRef).filter((r): r is ItemRef => r !== null), [adaptRaw]);
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
    if (!factors.data?.present) return;
    explore({
      adaptations,
      ...(thresholds.length ? { thresholds } : {}),
      ...(exps.length ? { experiments: exps } : {}),
      ...(pers.length ? { periods: pers } : {}),
      statistic: stat,
    });
    return () => explore.cancel();
    // bodyKey stands for the request body
  }, [bodyKey, factors.data?.present, explore]);

  if (factors.error && !factors.data) return <EmptyState error={factors.error} />;
  if (!factors.data || !grid.data) return <p className="cap">Loading…</p>;
  const f = factors.data;
  if (!f.present) {
    return (
      <LabFrame rid={rid} title="Climate × adaptation">
        <EmptyState title="No climate change factors yet" body="Fetch CMIP6 change factors for this city to see future warming and how much each adaptation offsets." action={f.action} />
      </LabFrame>
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

  return (
    <LabFrame rid={rid} title="Climate × adaptation">
      <div className="lab-form">
        <div className="field">
          <span className="field-label">Statistic across models</span>
          <Seg<string>
            label="Statistic"
            size="small"
            value={typeof stat === "object" ? "model" : stat}
            onChange={(v) => setStatRaw(v === "model" ? encodeStatistic({ model: f.models[0] ?? "" }) : v)}
            options={[
              { value: "median", label: "Median" },
              { value: "p10", label: "p10" },
              { value: "p90", label: "p90" },
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
          <input value={tText} placeholder={data ? data.thresholds.join(", ") : "from the config"} onChange={(e) => setTText(e.target.value)} />
        </label>
        <div className="field">
          <span className="field-label">Scenarios (SSP)</span>
          <div className="chips" role="group" aria-label="Experiments">
            {f.experiments.map((e) => (
              <button key={e} type="button" className="chip" aria-pressed={!exps.length || exps.includes(e)} onClick={() => setExps(exps.includes(e) ? exps.filter((x) => x !== e) : [...exps, e])}>
                {e.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <span className="field-label">Periods</span>
          <div className="chips" role="group" aria-label="Periods">
            {f.periods.map((p) => (
              <button key={p} type="button" className="chip" aria-pressed={!pers.length || pers.includes(p)} onClick={() => setPers(pers.includes(p) ? pers.filter((x) => x !== p) : [...pers, p])}>
                {p}
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
              <button key={k} type="button" className="chip" aria-pressed={adaptRaw.includes(k)} onClick={() => setAdaptRaw(adaptRaw.includes(k) ? adaptRaw.filter((x) => x !== k) : [...adaptRaw, k])}>
                {o.label}
              </button>
            );
          })}
          {!options.length ? <span className="cap">No exact results or configured scenarios on this run yet.</span> : null}
        </div>
      </div>
      {out.error ? <p className="callout" data-tone="crit">{out.error}</p> : null}
      <DotRange
        title="Warming by scenario and period"
        units={`${u} above the baseline`}
        rows={warming.map((w) => ({
          id: `${w.experiment}-${w.period}`,
          label: projLabel(w),
          est: warmingValue(w, stat),
          lo: w.p10,
          hi: w.p90,
          lo2: w.min,
          hi2: w.max,
          strip: Object.values(w.by_model).filter((v) => Number.isFinite(v)),
        }))}
        valueLabel="Warming"
        unit={u}
        rangeLabel="10th–90th percentile of models"
        outerLabel="model range"
        caption={`Dots: ${statisticLabel(stat)}; grey points: individual models.`}
      />
      {data && t !== null ? (
        <>
          <div className="row">
            <span className="cap">Threshold</span>
            <Seg<number> label="Threshold" size="small" value={Math.min(thresholdIdx, ths.length - 1)} onChange={setThresholdIdx} options={ths.map((x, i) => ({ value: i, label: `${fmtNum(x, 1)} ${u}` }))} />
            {out.loading ? <span className="spinner" aria-label="Updating" /> : null}
          </div>
          <Bars
            title={`Share of cells at or above ${fmtNum(t, 1)} ${u}`}
            units="share of cells"
            categories={["Today", ...projections.map(projLabel)]}
            series={variants.map((name, k) => ({
              id: name,
              label: name,
              values: [k === 0 && present && t !== null ? shareAt(present.share_at_or_above, t) : null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.median ?? null)],
              lo: [null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.p10 ?? null)],
              hi: [null, ...projections.map((p) => p.variants[k]?.share_at_or_above[tKey(p.variants[k].share_at_or_above)]?.p90 ?? null)],
              muted: k === 0 && variants.length > 1,
            }))}
            valueLabel="Share of cells"
            unit="share"
            categoryLabel="Period"
            domain={[0, 1]}
            caption="Bars: median across models; whiskers: 10th–90th percentile of models."
          />
          {gaugeProj && gaugeProj.variants.some((v) => v.offset_share_of_median_warming !== null) ? (
            <div className="grid3">
              {gaugeProj.variants
                .filter((v) => v.offset_share_of_median_warming !== null)
                .map((v) => (
                  <Gauge
                    key={v.name}
                    title={v.name}
                    label={`of ${projLabel(gaugeProj)} median warming offset`}
                    value={v.offset_share_of_median_warming}
                    min={0}
                    max={Math.max(1, v.offset_share_of_median_warming ?? 0)}
                    ticks={[1]}
                    format={(x) => fmtPct(x)}
                    caption={`Cancels ${fmtPct(v.offset_share_of_median_warming)} of ${projLabel(gaugeProj)} median warming (mean ΔT ${fmtSignedValue(v.adaptation_mean_delta, unit, 3)}).`}
                  />
                ))}
            </div>
          ) : null}
          {data.people_exposure?.length ? (
            <Table
              caption="Residents at or above each threshold"
              csvName="people-exposure"
              columns={Object.keys(data.people_exposure[0]).map((k) => ({
                key: k,
                label: k.replace(/_/g, " "),
                value: (r: Record<string, unknown>) => (typeof r[k] === "number" || typeof r[k] === "string" ? (r[k] as number | string) : r[k] === null || r[k] === undefined ? null : JSON.stringify(r[k])),
              }))}
              rows={data.people_exposure}
            />
          ) : null}
        </>
      ) : out.loading ? (
        <p className="cap">Computing exposure…</p>
      ) : null}
      <ClimateMaps rid={rid} warming={warming} adaptations={options.filter((o) => adaptRaw.includes(encodeItemRef(o.ref)))} threshold={t} stat={stat} unit={unit} />
    </LabFrame>
  );
}
