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
  const lo = d ? d.band_mean.map((m, i) => m - 1.96 * d.band_sd[i]) : [];
  const hi = d ? d.band_mean.map((m, i) => m + 1.96 * d.band_sd[i]) : [];
  const reach = d ? d.lags_m.filter((_lag, i) => d.acf[i] > hi[i]).reduce((m, v) => Math.max(m, v), -1) : -1;
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
            { id: "band", label: "19-permutation band (95%)", x: d.lags_m, y: d.band_mean, lo, hi, muted: true, dashed: true, points: false },
            { id: "acf", label: meta.label, x: d.lags_m, y: d.acf, emphasis: true },
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
