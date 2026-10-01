// Response (SPEC §6.4), per lever: the dose–response line with its ±1.96 SE ribbon (hollow
// points where more than 20% of cells are extrapolated) and the realised dose, curve shapes,
// own vs footprint effects with fold means, links to the lever's maps, and the literature
// panel (published ranges, SPARC ± SE, the causal point and a factor-of-2 band).
import type { ResponseLever, ResponseSections } from "../../api/runs";
import { Bars, DotRange, LineBand, SmallMultiples, type DotRangeRow } from "../../charts";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { Link, codecs, useUrlState } from "../../router";
import { fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../theme/format";
import { Block, OlderCode, Section, ViewPage, useRid, useUnits } from "./common";

function DoseResponse({ name, lever }: { name: string; lever: ResponseLever }) {
  const u = unitLabel(useUnits().target);
  const lu = unitLabel(lever.unit);
  const c = lever.curve;
  if (!c) return <OlderCode title="Dose–response curve" />;
  const lo = c.benefit.map((b, i) => (b === null || c.se[i] === null ? null : b - 1.96 * (c.se[i] as number)));
  const hi = c.benefit.map((b, i) => (b === null || c.se[i] === null ? null : b + 1.96 * (c.se[i] as number)));
  const hollow = c.frac_extrapolated.map((f) => f !== null && f > 0.2);
  const last = c.dose.length - 1;
  const caption =
    last > 0 && c.benefit[last] !== null
      ? `${lever.direction === "decrease" ? "Lowering" : "Raising"} ${lever.label} by ${fmtValue(c.dose[last], lu, 2)} cools the city by ${fmtValue(c.benefit[last], u, 2)} on average (±${fmtValue(1.96 * (c.se[last] ?? 0), u, 2)}). ${hollow.some(Boolean) ? `Hollow points: more than 20% of cells pushed beyond the observed range (${fmtPct(c.frac_extrapolated[last], 0)} at the top dose).` : ""}`
      : undefined;
  const shortfall = c.realized_dose.map((r, i) => (r === null ? null : c.dose[i] - r));
  const maxShort = Math.max(0, ...shortfall.filter((v): v is number => v !== null));
  return (
    <div className="grid2">
      <LineBand
        title={`${lever.label}: dose–response`}
        units={`cooling, ${u} (positive = cooler)`}
        series={[{ id: name, label: "City-mean cooling", x: c.dose, y: c.benefit, lo, hi, hollow }]}
        xLabel={`${lever.label} ${lever.direction === "decrease" ? "decrease" : "increase"}`}
        xUnit={lu}
        yLabel="Cooling"
        yUnit={u}
        yInclude={[0]}
        refLines={[{ axis: "y", value: 0 }]}
        decimals={2}
        caption={caption}
      />
      <LineBand
        title={`${lever.label}: requested vs realised dose`}
        units={lu}
        series={[
          { id: "realised", label: "Realised dose", x: c.dose, y: c.realized_dose },
          { id: "requested", label: "Requested dose", x: c.dose, y: c.dose, dashed: true, muted: true, points: false },
        ]}
        xLabel="Requested dose"
        xUnit={lu}
        yLabel="Realised dose"
        yUnit={lu}
        decimals={2}
        caption={maxShort > 0 ? `Clipping to the lever bounds loses up to ${fmtValue(maxShort, lu, 2)} of the requested dose.` : "Every requested dose is realised in full."}
      />
    </div>
  );
}

const SHAPES: { key: keyof NonNullable<ResponseLever["shapes"]>; label: string }[] = [
  { key: "saturating", label: "saturating" },
  { key: "linear", label: "linear" },
  { key: "sigmoid", label: "S-shaped" },
  { key: "censored", label: "knee beyond tested doses" },
  { key: "insufficient", label: "too few doses" },
];

function Effects({ lever }: { lever: ResponseLever }) {
  const u = unitLabel(useUnits().target);
  const lu = unitLabel(lever.unit);
  const e = lever.effects;
  const f = lever.fold_means;
  const rows: { id: string; label: string; v: number | null; unit: string; folds: number[] }[] = [
    { id: "own", label: "Own cell", v: e?.own ?? null, unit: `${u} per ${lu}`, folds: f?.own ?? [] },
    { id: "footprint", label: "Footprint (all cells)", v: e?.footprint ?? null, unit: `${u}·cells per ${lu}`, folds: f?.footprint ?? [] },
  ];
  return (
    <div className="stack">
      {e ? (
        <Table
          caption="Own vs footprint effect"
          csvName="own-vs-footprint"
          rowKey={(r) => r.id}
          columns={[
            { key: "label", label: "Effect of +1 unit", value: (r) => r.label },
            { key: "v", label: "Mean ΔT", align: "right", value: (r) => r.v, render: (r) => `${fmtSigned(r.v, 4)} ${r.unit}` },
            { key: "folds", label: "Fold means", value: (r) => r.folds.map((x) => fmtSigned(x, 4)).join(", ") },
          ]}
          rows={rows}
        />
      ) : (
        <OlderCode title="Own vs footprint effects" />
      )}
      {e ? (
        <p className="cap">
          Footprint / own ratio {fmtNum(e.ratio, 1)}: most of the cooling from a change lands in neighbouring cells.
          {e.median_d90 !== null ? ` Half the cells reach 90% of their maximum cooling by ${fmtValue(e.median_d90, lu, 1)}.` : ""}
          {e.median_max_cooling !== null ? ` Median maximum cooling ${fmtValue(e.median_max_cooling, u, 2)}.` : ""}
        </p>
      ) : null}
      {f ? (
        <SmallMultiples
          title="Fold means"
          items={rows}
          panelTitle={(r) => `${r.label} (${r.unit})`}
          renderPanel={(r) => <DotRange bare title={r.label} rows={[{ id: r.id, label: r.label, est: r.v, strip: r.folds }]} valueLabel="ΔT per unit" decimals={4} signed />}
          table={{
            columns: [
              { key: "e", label: "Effect" },
              { key: "f", label: "Fold" },
              { key: "v", label: "Mean ΔT per unit" },
            ],
            rows: rows.flatMap((r) => r.folds.map((v, i) => [r.label, i + 1, v])),
          }}
          caption="Grey dots: the effect estimated in each CV fold. Agreement across folds means the estimate does not hinge on one area."
        />
      ) : null}
    </div>
  );
}

function Literature({ lit, lever }: { lit: NonNullable<ResponseSections["literature"]>; lever: string }) {
  const rows = lit.rows.filter((r) => r.quantity === lever);
  const mine = lit.sparc.filter((r) => r.quantity === lever);
  if (!rows.length && !mine.length) return <p className="cap">No published values for this lever.</p>;
  const unit = unitLabel(rows[0]?.unit ?? mine[0]?.unit ?? "");
  const dot: DotRangeRow[] = [
    ...rows.map((r) => ({
      id: r.key,
      label: r.citation ?? r.key,
      est: r.low !== null && r.high !== null ? (r.low + r.high) / 2 : (r.low ?? r.high),
      lo: r.low,
      hi: r.high,
      muted: true,
    })),
    ...mine.map((m) => ({
      id: `sparc-${m.scenario}`,
      label: `SPARC: ${m.scenario}`,
      est: m.cooling,
      lo: m.cooling !== null && m.se !== null ? m.cooling - 1.96 * m.se : null,
      hi: m.cooling !== null && m.se !== null ? m.cooling + 1.96 * m.se : null,
      lo2: m.cooling !== null ? m.cooling / 2 : null,
      hi2: m.cooling !== null ? m.cooling * 2 : null,
      check: m.causal !== null ? { est: m.causal, label: "causal estimate" } : null,
    })),
  ];
  return (
    <div className="stack">
      <DotRange
        title="Literature check"
        units={`cooling, ${unit}`}
        rows={dot}
        valueLabel="Cooling"
        unit={unit}
        rangeLabel="published range / SPARC 95%"
        outerLabel="factor-of-2 band"
        decimals={2}
        caption={
          mine.length && mine[0].cooling !== null
            ? `SPARC's ${mine[0].scenario} cools by ${fmtValue(mine[0].cooling, unit, 2)}; published values ${rows.some((r) => r.low !== null && mine[0].cooling !== null && r.low <= mine[0].cooling * 2 && (r.high ?? r.low) >= mine[0].cooling / 2) ? "overlap its factor-of-2 band" : "fall outside its factor-of-2 band"}.`
            : undefined
        }
      />
      <Table
        caption="Published values"
        csvName="literature"
        rowKey={(r) => r.key}
        columns={[
          { key: "c", label: "Source", value: (r) => r.citation ?? r.key },
          { key: "per", label: "Per", value: (r) => r.per },
          { key: "v", label: "Value", value: (r) => r.value },
          { key: "s", label: "Status", value: (r) => r.status ?? "—" },
        ]}
        rows={rows}
      />
    </div>
  );
}

export default function Response() {
  const rid = useRid();
  const [lever, setLever] = useUrlState("lever", codecs.optString());
  return (
    <ViewPage
      view="response"
      title="Response"
      intro="What each lever does at increasing doses: city-mean cooling, curve shapes per cell, and how much of the effect spills into neighbouring cells."
    >
      {(s) => (
        <>
          {s.levers === null ? (
            <Section title="Literature" data={s.literature}>
              {() => null}
            </Section>
          ) : null}
          <Section title="Levers" data={s.levers}>
            {(levers) => {
              const names = Object.keys(levers);
              if (!names.length) return <p className="cap">No actionable levers were swept in this run.</p>;
              const cur = lever && levers[lever] ? lever : names[0];
              const L = levers[cur];
              const base = `/r/${encodeURIComponent(rid)}/map?layer=`;
              return (
                <div className="stack">
                  <Seg<string> label="Lever" value={cur} onChange={(v) => setLever(v)} options={names.map((n) => ({ value: n, label: levers[n].label || n }))} />
                  <DoseResponse name={cur} lever={L} />
                  <div className="grid2">
                    {L.shapes ? (
                      <Bars
                        title="Curve shapes across cells"
                        categories={[L.label]}
                        orientation="h"
                        mode="percent"
                        series={SHAPES.filter((x) => L.shapes?.[x.key] !== undefined && L.shapes?.[x.key] !== null).map((x) => ({
                          id: x.key,
                          label: x.label,
                          values: [L.shapes?.[x.key] ?? null],
                          muted: x.key === "insufficient",
                        }))}
                        valueLabel="cells"
                        decimals={0}
                        height={110}
                        caption={`Saturating ${fmtPct(L.shapes.saturating, 0)}, linear ${fmtPct(L.shapes.linear, 0)}, S-shaped ${fmtPct(L.shapes.sigmoid, 0)} of cells.`}
                      />
                    ) : (
                      <OlderCode title="Curve shapes" />
                    )}
                    <Block title="Maps for this lever">
                      <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 4 }}>
                        <li>
                          <Link to={`${base}fp_${encodeURIComponent(cur)}`}>Footprint effect per unit</Link>
                        </li>
                        <li>
                          <Link to={`${base}own_${encodeURIComponent(cur)}`}>Own-cell effect per unit</Link>
                        </li>
                        <li>
                          <Link to={`${base}cls_${encodeURIComponent(cur)}`}>Curve shape per cell</Link>
                        </li>
                        <li>
                          <Link to={`${base}d90_${encodeURIComponent(cur)}`}>Dose reaching 90% of the maximum</Link>
                        </li>
                        <li>
                          <Link to={`${base}A_${encodeURIComponent(cur)}`}>Maximum cooling</Link>
                        </li>
                      </ul>
                      <p className="cap">Pin a cell on the map to see its own fitted curve.</p>
                    </Block>
                  </div>
                  <Effects lever={L} />
                  <Section title="Literature" data={s.literature}>
                    {(lit) => <Literature lit={lit} lever={cur} />}
                  </Section>
                </div>
              );
            }}
          </Section>
        </>
      )}
    </ViewPage>
  );
}
