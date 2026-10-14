// Data & QA (SPEC §6.4): QA flags beside the forcing card (radiation, wind and station: the
// run's input weather, read from the accuracy view; phase 2 moved it here from Accuracy, since
// Data & checks is the home of input QA), tiles (input, dropped, clipped, grid fill with the
// cell size, collisions, background, noise floor), the fractional-part histogram (evidence of
// classed targets; full width, with a one-line note, when the run has no zone column), the
// dose-scale table, predictor histograms, the correlation matrix, per-zone counts, coarse
// aggregation stats and join provenance.
// Plain words (phase 4): the technical names stay (this is Model details), but every jargon tile,
// block and chart has an ⓘ (InfoTip) or a glossary <Term> saying what it means.
import type { AccuracySections, DataSections, KeyValue, ViewKpi } from "../../api/runs";
import { useView } from "../../api/runs";
import { Bars, Heatmap, Histogram, SmallMultiples } from "../../charts";
import { InfoTip } from "../../components/ui/InfoTip";
import { Table } from "../../components/ui/Table";
import { Term } from "../../components/ui/Term";
import { term } from "../../content/glossary";
import { fmtNum, fmtPct, fmtValue, unitLabel } from "../../theme/format";
import { Block, FlagList, KpiTiles, Section, ViewPage, useRid, useUnits, type DetailPageProps, type KpiTip } from "./common";
import { cellText } from "./format";

type FracHist = NonNullable<DataSections["frac_hist"]>;

/** What each QA tile means (by KPI id; "noise" is the fixture's id of the noise floor). */
export const QA_TILE_TIPS: Partial<Record<string, KpiTip>> = {
  n_input: { label: "Input points", text: "The temperature readings in the input file, before any were left out." },
  n_points: { term: "cell", label: "Cells modelled" },
  dropped: { label: "Dropped (non-finite)", text: "Readings left out because a value was blank, infinite or not a number.", technical: "Non-finite values (NaN, ±inf)" },
  clipped: {
    label: "Clipped values",
    text: "Map-layer values outside their allowed range (set in the project's checks) that were pulled back to its edge, e.g. a tree cover above 100%. The note lists them by layer.",
    technical: "qa.clip (values clipped to the valid range)",
  },
  fill: { term: "grid_fill", label: "Grid fill" },
  collisions: { label: "Cell collisions", text: "The share of filled squares that received more than one reading. Their readings are averaged into one.", technical: "Cell collision fraction" },
  background: {
    label: "Background",
    text: "The reference temperature taken away from every reading, so the model predicts how much warmer or cooler each place is than it. By default, the median reading.",
    technical: "Background (subtracted from the target)",
  },
  noise_floor: { term: "noise_floor", label: "Rounding-noise floor" },
  noise: { term: "noise_floor", label: "Rounding-noise floor" },
};

function FractionalParts({ h }: { h: FracHist }) {
  const units = useUnits();
  const u = unitLabel(units.target);
  const cats = h.shares.map((_, i) => `${fmtNum(h.edges[i] ?? i / 10, 1)}–${fmtNum(h.edges[i + 1] ?? (i + 1) / 10, 1)}`);
  const peak = h.shares.reduce((m, v, i) => (v > (h.shares[m] ?? -1) ? i : m), 0);
  const even = h.shares.length ? 1 / h.shares.length : 0;
  const parts = [
    h.integer_share !== null ? `${fmtPct(h.integer_share, 1)} of readings are whole numbers` : null,
    h.half_share !== null ? `${fmtPct(h.half_share, 1)} end in .5` : null,
    h.shares.length ? `the fullest bar (${cats[peak]}) holds ${fmtPct(h.shares[peak], 0)}, where evenly spread decimals would put ${fmtPct(even, 0)}` : null,
    h.noise_sd !== null ? `${term("noise_floor").label.toLowerCase()} (rounding-noise floor) ${fmtValue(h.noise_sd, units.target, 3)}` : null,
  ].filter(Boolean);
  return (
    <Bars
      title="Fractional part of the target"
      note={
        <>
          Were the readings rounded?
          <InfoTip term="fractional_part" label="Fractional part of the target" />
        </>
      }
      units="share of points"
      categories={cats}
      categoryLabel={`Fractional part (${u || "units"})`}
      series={[{ id: "share", label: "Share of points", values: h.shares.map((v) => v * 100) }]}
      valueLabel="Share of points"
      unit="%"
      decimals={1}
      caption={parts.length ? parts.join("; ") + "." : undefined}
    />
  );
}

/**
 * The QA tiles with the cell size folded into the grid fill ("63.5%", "of 30 m cells"): one
 * tile fewer, so the row breaks evenly. Unchanged when either tile is absent.
 */
export function foldCellSize(tiles: readonly ViewKpi[]): ViewKpi[] {
  const fill = tiles.find((t) => t.id === "fill");
  const cell = tiles.find((t) => t.id === "cell_m");
  if (!fill || !cell || typeof cell.value !== "number" || !Number.isFinite(cell.value)) return [...tiles];
  const of = `of ${fmtNum(cell.value, cell.value % 1 ? 1 : 0)} m cells`;
  return tiles.filter((t) => t !== cell).map((t) => (t === fill ? { ...t, note: t.note ? `${of}; ${t.note}` : of } : t));
}

type Forcing = NonNullable<AccuracySections["forcing"]>;

function WindArrow({ deg, speed }: { deg: number | null; speed: number | null }) {
  if (deg === null) return null;
  // The arrow points where the wind blows to (from `deg`, clockwise from north).
  return (
    <svg width="40" height="40" viewBox="-20 -20 40 40" role="img" aria-label={`Wind from ${fmtNum(deg, 0)}° at ${fmtNum(speed, 1)} m/s`}>
      <circle r="18" fill="none" style={{ stroke: "var(--line-strong)" }} />
      <g transform={`rotate(${deg + 180})`}>
        <path d="M0,-14 L5,-4 L1.5,-4 L1.5,12 L-1.5,12 L-1.5,-4 L-5,-4 Z" style={{ fill: "var(--ink-2)" }} />
      </g>
    </svg>
  );
}


/** The run's input weather (from the accuracy view's forcing card): date and hours, radiation, wind and station, with its checks. */
export function ForcingCard({ f }: { f: Forcing }) {
  return (
    <Block title="Forcing" tip={<InfoTip term="forcing" label="Forcing" />}>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <WindArrow deg={f.wind_dir_deg} speed={f.wind_speed} />
        <dl className="kv" aria-label="Forcing">
          <dt>Date</dt>
          <dd>{f.date ?? "—"}</dd>
          <dt>Hours</dt>
          <dd>{f.hours ?? "—"}</dd>
          <dt>
            Shortwave down
            <InfoTip label="Shortwave down" text="Sunlight reaching the ground, in watts per square metre." technical="Downwelling shortwave radiation (SW↓)" />
          </dt>
          <dd>{fmtValue(f.sw_down, "W/m²", 0)}</dd>
          <dt>
            Net longwave
            <InfoTip
              label="Net longwave"
              text="Heat radiation the ground gains from the sky minus what it gives off, in watts per square metre. Negative means the ground loses heat this way."
              technical="Net longwave radiation (LW↓ − LW↑)"
            />
          </dt>
          <dd>{fmtValue(f.lw_net, "W/m²", 0)}</dd>
          <dt>Wind</dt>
          <dd>{f.wind_speed !== null ? `${fmtValue(f.wind_speed, "m/s", 1)}${f.wind_dir_deg !== null ? ` from ${fmtNum(f.wind_dir_deg, 0)}°` : ""}` : "—"}</dd>
          <dt>Station</dt>
          <dd>{f.station ?? "—"}</dd>
        </dl>
      </div>
      {f.checks.length ? (
        <ul className="cap" style={{ margin: 0, paddingLeft: 18 }}>
          {f.checks.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
      ) : null}
    </Block>
  );
}

/** QA flags beside the forcing card when the run has one (the accuracy view carries it), else full width. */
function FlagsAndForcing({ flags }: { flags: DataSections["flags"] | null | undefined }) {
  const rid = useRid();
  const acc = useView(rid || null, "accuracy");
  const forcing = acc.data && acc.data.availability !== "missing" ? acc.data.sections.forcing : null;
  const flagBlock = (
    <Section title="QA flags" data={flags}>
      {(f) => (
        <Block
          title="QA flags"
          tip={
            <InfoTip
              label="QA flags"
              text="Notes and warnings raised while the input data were checked. A warning can change how the results should be read; an info note needs no action."
              technical="Data QA flags"
            />
          }
        >
          <FlagList flags={f} />
        </Block>
      )}
    </Section>
  );
  if (!forcing) return flagBlock;
  return (
    <div className="grid2" data-forcing="true">
      {flagBlock}
      <ForcingCard f={forcing} />
    </div>
  );
}

type DoseRow = { key: string; lever: string; unit: string; dose: number; inSd: number | null; pct: number | null; sd: number | null };

function DoseScale({ rows }: { rows: NonNullable<DataSections["dose_scale"]> }) {
  const flat: DoseRow[] = rows.flatMap((r) =>
    r.doses.map((d, i) => ({ key: `${r.lever}-${i}`, lever: r.lever, unit: r.unit, dose: d, inSd: r.doses_in_sd[i] ?? null, pct: r.percentile[i] ?? null, sd: r.sd })),
  );
  return (
    <Table<DoseRow>
      caption="Dose scale"
      csvName="dose-scale"
      rowKey={(r) => r.key}
      highlight={(r) => r.inSd !== null && r.inSd > 1}
      columns={[
        { key: "lever", label: "Lever", value: (r) => r.lever },
        { key: "dose", label: "Dose", align: "right", value: (r) => r.dose, render: (r) => fmtValue(r.dose, r.unit, 2) },
        { key: "sd", label: "Lever sd", align: "right", value: (r) => r.sd, render: (r) => fmtValue(r.sd, r.unit, 2) },
        {
          key: "insd",
          label: "Dose / sd",
          align: "right",
          value: (r) => r.inSd,
          render: (r) => (r.inSd === null ? "—" : `${fmtNum(r.inSd, 2)}${r.inSd > 1 ? " (beyond 1 sd)" : ""}`),
        },
        { key: "pct", label: "Median cell moves to percentile", align: "right", value: (r) => r.pct, render: (r) => (r.pct === null ? "—" : `${fmtNum(r.pct, 0)}th`) },
      ]}
      rows={flat}
    />
  );
}

function KeyValues({ items, label }: { items: KeyValue[]; label: string }) {
  if (!items.length) return <p className="cap">Nothing to report.</p>;
  return (
    <dl className="kv" aria-label={label}>
      {items.map((k, i) => (
        <div key={i} className="kv-row" style={{ display: "contents" }}>
          <dt>{k.label}</dt>
          <dd>{typeof k.value === "number" ? fmtValue(k.value, k.unit ?? null, k.decimals ?? 2) : cellText(k.value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function DataQa({
  title = "Data & QA",
  intro = "What went into the model, and the checks on it: the readings, the map grid, the size of the tested changes and the map layers.",
}: DetailPageProps = {}) {
  return (
    <ViewPage view="data" title={title} intro={intro}>
      {(s) => (
        <>
          <Section title="QA tiles" data={s.qa_tiles}>
            {(t) => <KpiTiles kpis={foldCellSize(t)} label="Data QA" tips={QA_TILE_TIPS} />}
          </Section>
          <FlagsAndForcing flags={s.flags} />
          {s.zone_counts && !s.zone_counts.length ? (
            // No zone column: the histogram takes the full row, the note one line under it.
            <div className="stack" style={{ gap: 6 }} data-no-zones="true">
              <Section title="Fractional-part histogram" data={s.frac_hist}>
                {(h) => <FractionalParts h={h} />}
              </Section>
              <p className="cap" style={{ margin: 0 }}>
                This run has no zone column.
              </p>
            </div>
          ) : (
            <div className="grid2">
              <Section title="Fractional-part histogram" data={s.frac_hist}>
                {(h) => <FractionalParts h={h} />}
              </Section>
              <Section title="Per-zone counts" data={s.zone_counts}>
                {(z) => (
                  <Bars
                    title="Points per zone"
                    note="Readings in each zone or neighbourhood of the input (used for reporting only)."
                    categories={z.map((r) => r.zone)}
                    categoryLabel="Zone"
                    series={[{ id: "n", label: "Points", values: z.map((r) => r.n) }]}
                    valueLabel="Points"
                    decimals={0}
                    caption={`${z.length} zones, ${fmtNum(
                      z.reduce((a, r) => a + r.n, 0),
                      0,
                    )} points.`}
                  />
                )}
              </Section>
            </div>
          )}
          <Section title="Dose scale" data={s.dose_scale}>
            {(d) => (
              <Block title="Dose scale" tip={<InfoTip term="dose_scale" label="Dose scale" />}>
                <p className="cap">
                  Each tested amount (dose) next to how much the lever already varies across the city: its standard deviation (sd). Amounts beyond 1 sd are highlighted: few
                  places today are that far from the typical value, so the model has little to go on there. The last column shows where a typical place would rank in
                  today&apos;s city after the change.
                </p>
                <DoseScale rows={d} />
              </Block>
            )}
          </Section>
          <Section title="Predictor histograms" data={s.predictor_hists}>
            {(p) => (
              <SmallMultiples
                title="Predictor distributions"
                note={
                  <>
                    How each <Term id="predictor">map layer</Term> is spread across the cells.
                  </>
                }
                items={p}
                panelTitle={(h) => `${h.label}${h.unit ? ` (${unitLabel(h.unit)})` : ""}`}
                renderPanel={(h) => <Histogram bare title={h.label} bins={{ edges: h.edges, counts: h.counts }} xLabel={h.label} unit={unitLabel(h.unit)} height={150} />}
                table={{
                  columns: [
                    { key: "p", label: "Predictor" },
                    { key: "lo", label: "From" },
                    { key: "hi", label: "To" },
                    { key: "n", label: "Cells" },
                  ],
                  rows: p.flatMap((h) => h.counts.map((c, i) => [h.label, h.edges[i], h.edges[i + 1], c])),
                }}
                caption={`${p.length} predictors.`}
              />
            )}
          </Section>
          <Section title="Correlation matrix" data={s.corr_matrix}>
            {(c) => {
              let best: [string, string, number] | null = null;
              c.names.forEach((a, i) =>
                c.names.forEach((b, j) => {
                  const v = c.values[i]?.[j];
                  if (j > i && v !== null && v !== undefined && (!best || Math.abs(v) > Math.abs(best[2]))) best = [a, b, v];
                }),
              );
              const b = best as [string, string, number] | null;
              return (
                <Heatmap
                  title="Predictor correlations"
                  rows={c.names}
                  cols={c.names}
                  values={c.values}
                  scale="div"
                  center={0}
                  domain={[-1, 1]}
                  valueLabel="Pearson r"
                  decimals={2}
                  labels={c.names.length <= 10}
                  note={
                    <>
                      Which map layers rise and fall together
                      <InfoTip
                        label="Pearson r"
                        text="How closely two layers move together across the grid: 1 always together, −1 always opposite, 0 unrelated. The model finds it hard to tell the effects of strongly linked layers apart."
                        technical="Pearson correlation coefficient"
                      />
                    </>
                  }
                  caption={b ? `Strongest pair: ${b[0]} and ${b[1]} (r = ${fmtNum(b[2], 2)}).` : undefined}
                />
              );
            }}
          </Section>
          <div className="grid2">
            <Section title="Coarse aggregation" data={s.coarse}>
              {(c) => (
                <Block title="Coarse aggregation" tip={<InfoTip term="coarse" label="Coarse aggregation" />}>
                  <KeyValues items={c} label="Coarse aggregation" />
                </Block>
              )}
            </Section>
            <Section title="Joins" data={s.joins}>
              {(j) => (
                <Block
                  title="Join provenance"
                  tip={
                    <InfoTip
                      label="Join provenance"
                      text="Extra data files joined onto the readings by a shared id column, with a fingerprint of each file so a later run can tell whether it changed."
                      technical="Joins (key, matched / unmatched rows, SHA-256)"
                    />
                  }
                >
                  {j.length ? (
                    <Table
                      caption="Joins"
                      csvName="joins"
                      rowKey={(r) => r.path}
                      columns={[
                        { key: "path", label: "File", value: (r) => r.path },
                        { key: "key", label: "Key", value: (r) => (r.key && r.right_key ? `${r.key} = ${r.right_key}` : (r.key ?? "—")) },
                        { key: "m", label: "Matched", align: "right", value: (r) => r.n_matched },
                        { key: "u", label: "Unmatched", align: "right", value: (r) => r.n_unmatched },
                        { key: "h", label: "SHA-256", value: (r) => r.sha256, render: (r) => <span className="mono cap">{r.sha256 ? r.sha256.slice(0, 12) + "…" : "—"}</span> },
                      ]}
                      rows={j}
                    />
                  ) : (
                    <p className="cap">No joined files.</p>
                  )}
                </Block>
              )}
            </Section>
          </div>
        </>
      )}
    </ViewPage>
  );
}
