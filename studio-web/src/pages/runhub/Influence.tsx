// Influence (SPEC §6.4): influence range per predictor (click → its influence circle on the
// map), the target correlogram with its permutation band and a directional toggle, ring-
// kernel β by distance, the anisotropy roses, and the priors (L, block size, residual range).
// What cools reuses one lever's reach (LeverReach), so under Model details (`detailsOnly`) the
// ring kernels leave out the levers' panels (they are on What cools) and keep the other predictors.
// Plain words (phase 4): LeverReach is What cools' "How far the cooling spreads" in planner words;
// this page keeps its technical names, each with an ⓘ (InfoTip) saying what it means.
import { useState } from "react";
import type { InfluenceSections } from "../../api/runs";
import { Bars, LineBand, Rose, SmallMultiples, type LineSeries } from "../../charts";
import { InfoTip } from "../../components/ui/InfoTip";
import { lever as leverWords } from "../../content/glossary";
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
      note={
        <InfoTip
          label="Target correlogram"
          text="How alike the temperatures of two places are, by the distance between them. Above the grey band the likeness is more than chance; where the line drops into it, the pattern has faded."
          technical="Autocorrelation of the target by lag, with a 95% permutation band"
        />
      }
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

/** Title of a lever's reach chart on What cools. */
export const REACH_TITLE = "How far the cooling spreads";

/**
 * A ring profile on one scale for every lever: each ring's β as a percentage of the strongest
 * ring's size (sign kept: negative = cooler), so the charts read alike whatever the lever's units.
 */
export function ringShares(betas: readonly (number | null)[]): (number | null)[] {
  const top = Math.max(0, ...betas.filter((b): b is number => b !== null && Number.isFinite(b)).map(Math.abs));
  return betas.map((b) => (b === null || !Number.isFinite(b) || top === 0 ? null : (100 * b) / top));
}

/**
 * The reach caption's reading of the zero line, for the side of it most rings are on. The chart is of
 * the measure itself (more paving nearby), so for a lever that cools by going down (`decrease`, "Less
 * paving") the sentence also says what removing it does.
 */
export function zeroLineSentence(betas: readonly (number | null)[], measure: string, decrease = false): string {
  const pos = betas.filter((b) => b !== null && b > 0).length;
  const neg = betas.filter((b) => b !== null && b < 0).length;
  if (decrease)
    return pos > neg
      ? `Above the zero line: ${measure} nearby makes a place warmer, so removing it cools.`
      : `Below the zero line: ${measure} nearby makes a place cooler, so removing it warms.`;
  return pos > neg ? `Above the zero line, more ${measure} nearby makes a place warmer.` : `Below the zero line, more ${measure} nearby makes a place cooler.`;
}

/**
 * How far one lever's cooling spreads (What cools), in plain words: its ring-kernel β by distance
 * with the influence range marked, else the influence range alone, with a link to its influence
 * circle on the map. `ranges` / `rings` are the Influence view's sections (null: not in this run);
 * `label` (the catalogue label) finds the lever's entries, `plain` (default `label`) names it;
 * `decrease`: the lever cools by going down (its caption says what removing the measure does).
 */
export function LeverReach({
  lever,
  label,
  plain = label,
  ranges,
  rings,
  decrease = false,
}: {
  lever: string;
  label: string;
  plain?: string;
  ranges: InfluenceSections["ranges"] | null | undefined;
  rings: InfluenceSections["rings"] | null | undefined;
  decrease?: boolean;
}) {
  const rid = useRid();
  const title = REACH_TITLE;
  if (!ranges && !rings) return <OlderCode title={`${plain}: ${title.toLowerCase()}`} />;
  const ring = forLever(rings, lever, label);
  const range: InfluenceRange | null = forLever(ranges, lever, label);
  const predictor = ring?.predictor ?? range?.predictor ?? lever;
  const r = range?.range_m ?? null;
  const reach = r !== null && Number.isFinite(r) ? r : null;
  const actions = (
    <Link className="btn small ghost" to={influenceCircleHref(rid, predictor)}>
      Show the reach on the map
    </Link>
  );
  const rangeText = reach !== null ? `${plain}: most of its effect on temperature (90%) is felt within ${fmtDistance(reach)}.` : "";
  // what the lever changes, as a noun ("paving", "surface reflectivity"), from the glossary
  const measure = leverWords(lever)?.measure ?? label.toLowerCase();
  if (ring && ring.betas.length) {
    const mids = ringMids(ring);
    const finite = mids.filter((m) => Number.isFinite(m));
    // Keep the influence-range marker inside the plot when it falls outside the rings.
    const xDomain: [number, number] | undefined = finite.length ? [Math.min(...finite, reach ?? Infinity), Math.max(...finite, reach ?? -Infinity)] : undefined;
    const caption = [
      rangeText,
      zeroLineSentence(ring.betas, measure, decrease),
      ring.significant === false ? "The data cannot confirm this pattern (not statistically significant), so treat it as a rough guide." : "",
    ]
      .filter(Boolean)
      .join(" ");
    return (
      <LineBand
        title={title}
        exportName={`${plain} ${title}`}
        note={
          <>
            How strongly {measure} at each distance changes a place&apos;s temperature, fading with distance.
            <InfoTip term="ring_kernel" label={title} />
          </>
        }
        units={`% of the strongest effect of ${measure} nearby (negative = cooler)`}
        series={[{ id: ring.predictor, label: plain, x: mids, y: ringShares(ring.betas), muted: ring.significant === false }]}
        xLabel="Distance"
        xUnit="m"
        xDecimals={0}
        xDomain={xDomain && xDomain[1] > xDomain[0] ? xDomain : undefined}
        yLabel="Strength (% of the strongest)"
        refLines={[{ axis: "y", value: 0 }, ...(reach !== null ? [{ axis: "x" as const, value: reach, label: "90% of the effect" }] : [])]}
        decimals={0}
        actions={actions}
        caption={caption}
      />
    );
  }
  if (range) {
    return (
      <Bars
        title={title}
        exportName={`${plain} ${title}`}
        note={
          <>
            How far {label} still changes the temperature around it.
            <InfoTip term="length_scale" label={title} />
          </>
        }
        units="m"
        categories={[plain]}
        orientation="h"
        series={[{ id: "range", label: "Reach (90% of the effect)", values: [range.range_m] }]}
        valueLabel="Reach"
        unit="m"
        decimals={0}
        height={110}
        actions={actions}
        caption={rangeText || undefined}
      />
    );
  }
  return <p className="cap">This run does not measure how far the cooling from {label} spreads.</p>;
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
                note={<InfoTip term="length_scale" label="Influence range" />}
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
                note={<InfoTip term="ring_kernel" label="Ring kernel" />}
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
                  note={<InfoTip term="anisotropy" label="Anisotropy" />}
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
              <Block title="Length scales" tip={<InfoTip term="priors" label="Length scales" />}>
                <dl className="kv" aria-label="Length scales">
                  <dt>
                    Relaxation length prior L
                    <InfoTip
                      term="length_scale"
                      label="Relaxation length prior L"
                      text="The distance over which the heat-physics model starts out assuming heat spreads, before the data adjusts it."
                      technical="Relaxation length prior L (physics base model)"
                    />
                  </dt>
                  <dd>{fmtDistance(p.L_prior_m)}</dd>
                  <dt>
                    CV block size
                    <InfoTip term="cv_design" label="CV block size" text="The size of the square areas the city is split into for testing: the model is trained without one area and tested on it." />
                  </dt>
                  <dd>{fmtDistance(p.block_size_m)}</dd>
                  <dt>
                    Residual range
                    <InfoTip
                      label="Residual range"
                      text="How far apart two places must be before what the model gets wrong in one says nothing about the other."
                      technical="Range of the residual correlogram"
                    />
                  </dt>
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
