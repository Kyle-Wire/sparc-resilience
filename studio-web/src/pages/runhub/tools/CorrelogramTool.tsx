// Correlogram (SPEC §6.5): FFT autocorrelation of any layer (influence.fft_acf) with a
// 19-permutation band (POST /api/runs/{rid}/stats/acf).
import { useMemo, useState } from "react";
import { acf, useAnalysis, type AcfRequest } from "../../../api/analysis";
import { LineBand } from "../../../charts";
import { EmptyState } from "../../../components/ui/EmptyState";
import { NumberField } from "../../../components/ui/NumberField";
import { fmtDistance } from "../format";
import { allLayers, LayerSelect, type ToolProps } from "./shared";

export function CorrelogramTool({ rid, groups, layerKey }: ToolProps) {
  const layers = useMemo(() => allLayers(groups).filter((l) => l.scale !== "cat"), [groups]);
  const [layer, setLayer] = useState<string | null>(null);
  const [maxLag, setMaxLag] = useState<number | null>(null);
  const key = layer ?? (layerKey && layers.some((l) => l.key === layerKey) ? layerKey : (layers[0]?.key ?? null));
  const meta = layers.find((l) => l.key === key);
  const body: AcfRequest | null = key ? { layer: key, n_perm: 19, ...(maxLag ? { max_lag_m: maxLag } : {}) } : null;
  const res = useAnalysis(rid, "acf", body, acf);
  const d = res.data;
  // lags without pairs come back as null: plot only the lags with an autocorrelation and a band
  const pts = useMemo(() => {
    const out = { x: [] as number[], y: [] as number[], mean: [] as number[], lo: [] as number[], hi: [] as number[] };
    if (!d) return out;
    d.lags_m.forEach((lag, i) => {
      const a = d.acf[i];
      const m = d.band_mean[i];
      const sd = d.band_sd[i];
      if (lag == null || a == null || m == null || sd == null) return;
      out.x.push(lag);
      out.y.push(a);
      out.mean.push(m);
      out.lo.push(m - 1.96 * sd);
      out.hi.push(m + 1.96 * sd);
    });
    return out;
  }, [d]);
  const reach = pts.x.filter((_lag, i) => pts.y[i] > pts.hi[i]).reduce((m, v) => Math.max(m, v), -1);
  return (
    <div className="stack" data-tool="correlogram">
      <div className="row">
        <LayerSelect groups={groups} value={key} onChange={setLayer} label="Layer" filter={(l) => l.scale !== "cat"} />
        <NumberField label="Maximum lag (m)" value={maxLag} nullable min={0} placeholder="auto" onChange={(v) => setMaxLag(v)} unit="m" />
      </div>
      {res.error ? <EmptyState error={res.error} /> : null}
      {!d && !res.error && body ? <p className="cap">Computing…</p> : null}
      {d && meta ? (
        <LineBand
          title={`Correlogram of ${meta.label}`}
          units="autocorrelation"
          series={[
            { id: "band", label: "19-permutation band (95%)", x: pts.x, y: pts.mean, lo: pts.lo, hi: pts.hi, muted: true, dashed: true, points: false },
            { id: "acf", label: meta.label, x: pts.x, y: pts.y, emphasis: true },
          ]}
          xLabel="Lag"
          xUnit="m"
          xDecimals={0}
          yLabel="Autocorrelation"
          refLines={[{ axis: "y", value: 0 }]}
          decimals={2}
          caption={reach > 0 ? `${meta.label} stays more alike than chance out to ${fmtDistance(reach)}.` : `${meta.label} is no more alike than chance at any lag.`}
        />
      ) : null}
    </div>
  );
}
