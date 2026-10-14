// Influence (SPEC §6.4): influence range per predictor (click → its influence circle on the
// map), the target correlogram with its permutation band and a directional toggle, ring-
// kernel β by distance, the anisotropy roses, and the priors (L, block size, residual range).
// What cools reuses one lever's reach (LeverReach), so under Model details (`detailsOnly`) the
// ring kernels leave out the levers' panels (they are on What cools) and keep the other predictors.
import { useState } from "react";
import type { InfluenceSections } from "../../api/runs";
import { Bars, LineBand, Rose, SmallMultiples, type LineSeries } from "../../charts";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { Link, navigate, runHref } from "../../router";
import { fmtNum } from "../../theme/format";
import { Block, OlderCode, Section, ViewPage, useRid, type DetailPageProps } from "./common";
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

type RingKernel = InfluenceSections["rings"][number];
type InfluenceRange = InfluenceSections["ranges"][number];

/** The map with a predictor's influence circle drawn (the Influence ranges bars link here). */
export function influenceCircleHref(rid: string, predictor: string): string {
  return `/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(predictor)}&ov=influence&infl=${encodeURIComponent(predictor)}`;
}

/** Midpoint of each distance ring of a kernel. */
function ringMids(r: RingKernel): number[] {
  return r.betas.map((_, i) => ((r.edges_m[i] ?? 0) + (r.edges_m[i + 1] ?? r.edges_m[i] ?? 0)) / 2);
}

/** One predictor's ring-kernel β by distance (a panel of "Ring-kernel β by distance"). */
function RingKernelPanel({ ring }: { ring: RingKernel }) {
  return (
    <LineBand
      bare
      title={ring.label}
      series={[{ id: ring.predictor, label: ring.label, x: ringMids(ring), y: ring.betas, muted: ring.significant === false }]}
      xLabel="Distance"
      xUnit="m"
      xDecimals={0}
      yLabel="β"
      refLines={[{ axis: "y", value: 0 }]}
      height={170}
      decimals={3}
    />
  );
}

/** The entry of `list` for a lever: same key as the predictor, else the same label. */
function forLever<T extends { predictor: string; label: string }>(list: T[] | null | undefined, key: string, label: string): T | null {
  if (!list) return null;
  const l = label.trim().toLowerCase();
  return list.find((x) => x.predictor === key) ?? list.find((x) => x.label.trim().toLowerCase() === l) ?? null;
}

/**
 * How far one lever's cooling reaches (What cools): its ring-kernel β by distance with the
 * influence range marked, else the influence range alone, with a link to its influence circle
 * on the map. `ranges` / `rings` are the Influence view's sections (null: not in this run).
 */
export function LeverReach({
  lever,
  label,
  ranges,
  rings,
}: {
  lever: string;
  label: string;
  ranges: InfluenceSections["ranges"] | null | undefined;
  rings: InfluenceSections["rings"] | null | undefined;
}) {
  const rid = useRid();
  const title = `${label}: how far cooling reaches`;
  if (!ranges && !rings) return <OlderCode title={title} />;
  const ring = forLever(rings, lever, label);
  const range: InfluenceRange | null = forLever(ranges, lever, label);
  const predictor = ring?.predictor ?? range?.predictor ?? lever;
  const r = range?.range_m ?? null;
  const reach = r !== null && Number.isFinite(r) ? r : null;
  const actions = (
    <Link className="btn small ghost" to={influenceCircleHref(rid, predictor)}>
      Influence circle on the map
    </Link>
  );
  const rangeText = reach !== null ? `Influence range ${fmtDistance(reach)}.` : "";
  if (ring && ring.betas.length) {
    const mids = ringMids(ring);
    const finite = mids.filter((m) => Number.isFinite(m));
    // Keep the influence-range marker inside the plot when it falls outside the rings.
    const xDomain: [number, number] | undefined = finite.length ? [Math.min(...finite, reach ?? Infinity), Math.max(...finite, reach ?? -Infinity)] : undefined;
    return (
      <LineBand
        title={title}
        units="ring-kernel β by distance"
        series={[{ id: ring.predictor, label: ring.label, x: mids, y: ring.betas, muted: ring.significant === false }]}
        xLabel="Distance"
        xUnit="m"
        xDecimals={0}
        xDomain={xDomain && xDomain[1] > xDomain[0] ? xDomain : undefined}
        yLabel="β"
        refLines={[{ axis: "y", value: 0 }, ...(reach !== null ? [{ axis: "x" as const, value: reach, label: "influence range" }] : [])]}
        decimals={3}
        actions={actions}
        caption={`${rangeText}${ring.significant === false ? `${rangeText ? " " : ""}This kernel is not significant.` : ""}` || undefined}
      />
    );
  }
  if (range) {
    return (
      <Bars
        title={title}
        units="m"
        categories={[range.label]}
        orientation="h"
        series={[{ id: "range", label: "Influence range", values: [range.range_m] }]}
        valueLabel="Range"
        unit="m"
        decimals={0}
        height={110}
        actions={actions}
        caption={rangeText || undefined}
      />
    );
  }
  return <p className="cap">No influence range or ring kernel for {label} in this run.</p>;
}

function longestReach(r: NonNullable<InfluenceSections["ranges"]>): string | undefined {
  const top = r.reduce<(typeof r)[number] | null>((m, x) => (x.range_m !== null && (!m || (m.range_m ?? 0) < x.range_m) ? x : m), null);
  return top ? `Longest reach: ${top.label} (${fmtDistance(top.range_m)}). Select a bar to draw that predictor's influence circle on the map.` : undefined;
}

/** Whether a ring kernel is one of the run's levers (by key, else by label), whose reach What cools shows. */
function isLeverRing(r: RingKernel, levers: readonly string[]): boolean {
  const l = r.label.trim().toLowerCase();
  return levers.includes(r.predictor) || levers.some((k) => k.toLowerCase() === l);
}

export default function Influence({
  title = "Influence",
  intro = "How far each predictor reaches: the range over which a cell's surroundings still change its temperature.",
  level,
  detailsOnly,
}: DetailPageProps = {}) {
  const rid = useRid();
  const showCircle = (p: string) => navigate(influenceCircleHref(rid, p));
  const onWhatCools = (
    <>
      Each lever&apos;s kernel (how far its cooling reaches) is on <Link to={runHref(rid, "effects")}>What cools</Link>.
    </>
  );
  return (
    <ViewPage view="influence" title={title} intro={intro} level={level}>
      {(s, vm) => (
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
            {(all) => {
              const levers = Object.keys(vm.units.levers ?? {});
              const rings = detailsOnly ? all.filter((r) => !isLeverRing(r, levers)) : all;
              if (!rings.length)
                return (
                  <p className="cap" data-lever-kernels="effects">
                    {onWhatCools}
                  </p>
                );
              const shown = `β per distance ring of the fitted kernel; ${rings.filter((r) => r.significant).length} of ${rings.length} kernels are significant.`;
              return (
              <SmallMultiples
                title="Ring-kernel β by distance"
                items={rings}
                panelTitle={(r) => `${r.label}${r.family ? ` · ${r.family}` : ""}${r.significant === false ? " · not significant" : ""}`}
                renderPanel={(r) => <RingKernelPanel ring={r} />}
                table={{
                  columns: [
                    { key: "p", label: "Predictor" },
                    { key: "r0", label: "Ring from", unit: "m" },
                    { key: "r1", label: "Ring to", unit: "m" },
                    { key: "b", label: "β" },
                  ],
                  rows: rings.flatMap((r) => r.betas.map((b, i) => [r.label, r.edges_m[i] ?? null, r.edges_m[i + 1] ?? null, b])),
                }}
                caption={
                  rings.length < all.length ? (
                    <>
                      {shown} {onWhatCools}
                    </>
                  ) : (
                    shown
                  )
                }
              />
              );
            }}
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
