// Data & QA (SPEC §6.4): QA flags, tiles (input, dropped, clipped, grid fill, collisions,
// background, noise floor), the fractional-part histogram (evidence of classed targets), the
// dose-scale table, predictor histograms, the correlation matrix, per-zone counts, coarse
// aggregation stats and join provenance.
import type { DataSections, KeyValue } from "../../api/runs";
import { Bars, Heatmap, Histogram, SmallMultiples } from "../../charts";
import { Table } from "../../components/ui/Table";
import { fmtNum, fmtPct, fmtValue, unitLabel } from "../../theme/format";
import { Block, FlagList, KpiTiles, Section, ViewPage, useUnits } from "./common";
import { cellText } from "./format";

type FracHist = NonNullable<DataSections["frac_hist"]>;

function FractionalParts({ h }: { h: FracHist }) {
  const units = useUnits();
  const u = unitLabel(units.target);
  const cats = h.shares.map((_, i) => `${fmtNum(h.edges[i] ?? i / 10, 1)}–${fmtNum(h.edges[i + 1] ?? (i + 1) / 10, 1)}`);
  const peak = h.shares.reduce((m, v, i) => (v > (h.shares[m] ?? -1) ? i : m), 0);
  const even = h.shares.length ? 1 / h.shares.length : 0;
  const parts = [
    h.integer_share !== null ? `${fmtPct(h.integer_share, 1)} of targets are whole numbers` : null,
    h.half_share !== null ? `${fmtPct(h.half_share, 1)} end in .5` : null,
    h.shares.length ? `the fullest bin (${cats[peak]}) holds ${fmtPct(h.shares[peak], 0)} against ${fmtPct(even, 0)} for evenly spread decimals` : null,
    h.noise_sd !== null ? `rounding noise floor ${fmtValue(h.noise_sd, units.target, 3)}` : null,
  ].filter(Boolean);
  return (
    <Bars
      title="Fractional part of the target"
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

export default function DataQa() {
  return (
    <ViewPage view="data" title="Data & QA" intro="What went into the model: input checks, the grid, the lever doses and the predictors.">
      {(s) => (
        <>
          <Section title="QA tiles" data={s.qa_tiles}>
            {(t) => <KpiTiles kpis={t} label="Data QA" />}
          </Section>
          <Section title="QA flags" data={s.flags}>
            {(f) => (
              <Block title="QA flags">
                <FlagList flags={f} />
              </Block>
            )}
          </Section>
          <div className="grid2">
            <Section title="Fractional-part histogram" data={s.frac_hist}>
              {(h) => <FractionalParts h={h} />}
            </Section>
            <Section title="Per-zone counts" data={s.zone_counts}>
              {(z) =>
                z.length ? (
                  <Bars
                    title="Points per zone"
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
                ) : (
                  <p className="cap">This run has no zone column.</p>
                )
              }
            </Section>
          </div>
          <Section title="Dose scale" data={s.dose_scale}>
            {(d) => (
              <Block title="Dose scale">
                <p className="cap">Doses beyond one standard deviation of the lever are highlighted: the model sees few cells that far from today.</p>
                <DoseScale rows={d} />
              </Block>
            )}
          </Section>
          <Section title="Predictor histograms" data={s.predictor_hists}>
            {(p) => (
              <SmallMultiples
                title="Predictor distributions"
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
                  caption={b ? `Strongest pair: ${b[0]} and ${b[1]} (r = ${fmtNum(b[2], 2)}).` : undefined}
                />
              );
            }}
          </Section>
          <div className="grid2">
            <Section title="Coarse aggregation" data={s.coarse}>
              {(c) => (
                <Block title="Coarse aggregation">
                  <KeyValues items={c} label="Coarse aggregation" />
                </Block>
              )}
            </Section>
            <Section title="Joins" data={s.joins}>
              {(j) => (
                <Block title="Join provenance">
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
