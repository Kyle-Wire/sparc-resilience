// Influence (SPEC §6.4): influence range per predictor (click → its influence circle on the
// map), the target correlogram with its permutation band and a directional toggle, ring-
// kernel β by distance, the anisotropy roses, and the priors (L, block size, residual range).
import { useState } from "react";
import type { InfluenceSections } from "../../api/runs";
import { Bars, LineBand, Rose, SmallMultiples, type LineSeries } from "../../charts";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { navigate } from "../../router";
import { fmtNum } from "../../theme/format";
import { Block, Section, ViewPage, useRid } from "./common";
import { fmtDistance } from "./format";

const DIRS = ["all", "0", "45", "90", "135"] as const;
type Dir = (typeof DIRS)[number];

function Correlogram({ c }: { c: NonNullable<InfluenceSections["correlogram"]> }) {
  const [dir, setDir] = useState<Dir>("all");
  const hasDir = !!c.directional && Object.keys(c.directional).length > 0;
  const band = c.band_mean.map((m, i) => (m === null || c.band_sd[i] === null ? null : m));
  const lo = c.band_mean.map((m, i) => (m === null || c.band_sd[i] === null ? null : m - 1.96 * (c.band_sd[i] as number)));
  const hi = c.band_mean.map((m, i) => (m === null || c.band_sd[i] === null ? null : m + 1.96 * (c.band_sd[i] as number)));
  const series: LineSeries[] = [{ id: "perm", label: "Permutation band (95%)", x: c.lags_m, y: band, lo, hi, muted: true, points: false, dashed: true }];
  if (dir === "all" || !hasDir) series.push({ id: "acf", label: "All directions", x: c.lags_m, y: c.acf, emphasis: true });
  else series.push({ id: `acf${dir}`, label: `${dir}° from north`, x: c.lags_m, y: c.directional?.[dir] ?? [], emphasis: true });
  const above = c.acf.map((v, i) => (v !== null && hi[i] !== null && v > (hi[i] as number) ? c.lags_m[i] : null)).filter((v): v is number => v !== null);
  const reach = above.length ? Math.max(...above) : null;
  return (
    <LineBand
      title="Target correlogram"
      units="autocorrelation"
      series={series}
      xLabel="Lag"
      xUnit="m"
      xDecimals={0}
      yLabel="Autocorrelation"
      refLines={[{ axis: "y", value: 0 }]}
      decimals={2}
      actions={
        hasDir ? (
          <Seg<Dir> label="Direction" size="small" value={dir} onChange={setDir} options={DIRS.map((d) => ({ value: d, label: d === "all" ? "All" : `${d}°` }))} />
        ) : undefined
      }
      caption={reach !== null ? `The residual temperature pattern stays above the permutation band out to ${fmtDistance(reach)}.` : "No lag rises above the permutation band."}
    />
  );
}

function longestReach(r: NonNullable<InfluenceSections["ranges"]>): string | undefined {
  const top = r.reduce<(typeof r)[number] | null>((m, x) => (x.range_m !== null && (!m || (m.range_m ?? 0) < x.range_m) ? x : m), null);
  return top ? `Longest reach: ${top.label} (${fmtDistance(top.range_m)}). Select a bar to draw that predictor's influence circle on the map.` : undefined;
}

export default function Influence() {
  const rid = useRid();
  const showCircle = (p: string) => navigate(`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(p)}&ov=influence&infl=${encodeURIComponent(p)}`);
  return (
    <ViewPage view="influence" title="Influence" intro="How far each predictor reaches: the range over which a cell's surroundings still change its temperature.">
      {(s) => (
        <>
          <Section title="Influence ranges" data={s.ranges}>
            {(r) => (
              <Bars
                title="Influence range per predictor"
                units="m"
                categories={r.map((x) => x.label)}
                orientation="h"
                series={[{ id: "range", label: "Influence range", values: r.map((x) => x.range_m) }]}
                valueLabel="Range"
                unit="m"
                decimals={0}
                onBarClick={(label) => {
                  const p = r.find((x) => x.label === label);
                  if (p) showCircle(p.predictor);
                }}
                caption={longestReach(r)}
              />
            )}
          </Section>
          <Section title="Correlogram" data={s.correlogram}>
            {(c) => <Correlogram c={c} />}
          </Section>
          <Section title="Ring kernels" data={s.rings}>
            {(rings) => (
              <SmallMultiples
                title="Ring-kernel β by distance"
                items={rings}
                panelTitle={(r) => `${r.label}${r.family ? ` · ${r.family}` : ""}${r.significant === false ? " · not significant" : ""}`}
                renderPanel={(r) => {
                  const mids = r.betas.map((_, i) => ((r.edges_m[i] ?? 0) + (r.edges_m[i + 1] ?? r.edges_m[i] ?? 0)) / 2);
                  return (
                    <LineBand
                      bare
                      title={r.label}
                      series={[{ id: r.predictor, label: r.label, x: mids, y: r.betas, muted: r.significant === false }]}
                      xLabel="Distance"
                      xUnit="m"
                      xDecimals={0}
                      yLabel="β"
                      refLines={[{ axis: "y", value: 0 }]}
                      height={170}
                      decimals={3}
                    />
                  );
                }}
                table={{
                  columns: [
                    { key: "p", label: "Predictor" },
                    { key: "r0", label: "Ring from", unit: "m" },
                    { key: "r1", label: "Ring to", unit: "m" },
                    { key: "b", label: "β" },
                  ],
                  rows: rings.flatMap((r) => r.betas.map((b, i) => [r.label, r.edges_m[i] ?? null, r.edges_m[i + 1] ?? null, b])),
                }}
                caption={`β per distance ring of the fitted kernel; ${rings.filter((r) => r.significant).length} of ${rings.length} kernels are significant.`}
              />
            )}
          </Section>
          <Section title="Anisotropy" data={s.anisotropy}>
            {(an) => (
              <div className="stack">
                <SmallMultiples
                  title="Anisotropy roses"
                  units="range by direction (m)"
                  items={an}
                  minPanelWidth={200}
                  panelTitle={(a) => `${a.label}${a.reliable ? "" : " (not reliable)"}`}
                  renderPanel={(a) => (
                    <Rose
                      bare
                      title={a.label}
                      petals={Object.entries(a.ranges_m).map(([deg, v]) => ({ angle_deg: Number(deg), value: v, reliable: a.reliable }))}
                      unit="m"
                      valueLabel="Range"
                      decimals={0}
                      size={190}
                    />
                  )}
                  table={{
                    columns: [
                      { key: "p", label: "Predictor" },
                      { key: "d", label: "Direction", unit: "°" },
                      { key: "r", label: "Range", unit: "m" },
                    ],
                    rows: an.flatMap((a) => Object.entries(a.ranges_m).map(([deg, v]) => [a.label, Number(deg), v])),
                  }}
                  caption={`${an.filter((a) => a.reliable).length} of ${an.length} anisotropies are confirmed by the bootstrap; dashed petals are not.`}
                />
                <Table
                  caption="Anisotropy"
                  csvName="anisotropy"
                  rowKey={(r) => r.predictor}
                  columns={[
                    { key: "p", label: "Predictor", value: (r) => r.label },
                    { key: "ratio", label: "Minor / major range", align: "right", value: (r) => r.ratio, render: (r) => fmtNum(r.ratio, 2) },
                    { key: "theta", label: "Major axis", unit: "° from north", align: "right", value: (r) => r.theta_deg, render: (r) => fmtNum(r.theta_deg, 0) },
                    { key: "rel", label: "Reliable", value: (r) => (r.reliable ? "yes" : "no") },
                  ]}
                  rows={an}
                />
              </div>
            )}
          </Section>
          <Section title="Priors" data={s.priors}>
            {(p) => (
              <Block title="Length scales">
                <dl className="kv" aria-label="Length scales">
                  <dt>Relaxation length prior L</dt>
                  <dd>{fmtDistance(p.L_prior_m)}</dd>
                  <dt>CV block size</dt>
                  <dd>{fmtDistance(p.block_size_m)}</dd>
                  <dt>Residual range</dt>
                  <dd>{fmtDistance(p.resid_range_m)}</dd>
                </dl>
              </Block>
            )}
          </Section>
        </>
      )}
    </ViewPage>
  );
}
