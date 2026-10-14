// Response (SPEC §6.4), per lever: the dose–response line with its ±1.96 SE ribbon (hollow
// points where more than 20% of cells are extrapolated) and the realised dose, curve shapes,
// own vs footprint effects with fold means, links to the lever's maps, and the literature
// panel (published ranges, SPARC ± SE, the causal point and a factor-of-2 band). What cools
// reuses the curve, the curve shapes and the saturation sentence (DoseResponseCurve,
// CurveShapes, SaturationSummary), so under Model details (`detailsOnly`) this page leaves them
// out and keeps the technical extras: the realised dose, the lever's maps, own vs footprint with
// fold means, and the literature check.
// Plain words (phase 4): the shared parts are What cools' and speak to a planner (glossary lever
// names and shapes, likely ranges, a "most of the benefit" marker at the median d90); the Model
// details parts keep their technical names, each with an ⓘ (InfoTip) or a glossary <Term>.
import type { ResponseLever, ResponseSections } from "../../api/runs";
import { Bars, DotRange, LineBand, SmallMultiples, type DotRangeRow, type RefLine } from "../../charts";
import { InfoTip } from "../../components/ui/InfoTip";
import { Seg } from "../../components/ui/Seg";
import { Table } from "../../components/ui/Table";
import { Term } from "../../components/ui/Term";
import { lever as leverWords, shape, term } from "../../content/glossary";
import { Link, codecs, useUrlState } from "../../router";
import { fmtNum, fmtPct, fmtRange, fmtSig, fmtSigned, fmtTempChange, fmtValue, unitLabel } from "../../theme/format";
import { Block, OlderCode, Section, ViewPage, useRid, useUnits, type DetailPageProps } from "./common";

/** A lever's plain name: the glossary's ("Trees & canopy") for a known lever, else its catalogue label. */
export function leverName(key: string, L: Pick<ResponseLever, "label">): string {
  return leverWords(key)?.name ?? (L.label || key);
}

/**
 * A lever's unit as a planner reads it, the glossary's unit word: "pts" for percentage points, "0–1"
 * for a reflectivity (albedo) on its 0–1 scale; else its display unit.
 */
export function leverUnit(key: string, L: Pick<ResponseLever, "unit">): string {
  const u = unitLabel(L.unit);
  const w = leverWords(key)?.unitWord;
  return w && (u === "pp" || u === "albedo" || u === "" || u === "1") ? w : u;
}

/** A unit that is a scale ("0–1"), not a quantity: amounts read "0.2 on the 0–1 scale". */
const isScale = (unit: string) => /^[\d.]+–[\d.]+$/.test(unit);

/** What the lever changes, as a noun for sentences: the glossary's measure ("paving", "surface reflectivity"), else the catalogue label. */
export function leverMeasure(key: string, L: Pick<ResponseLever, "label">): string {
  return leverWords(key)?.measure ?? inSentence(L.label || key);
}

/** An amount of a lever with its unit: "40 pts", "0.2 on the 0–1 scale", "12%". */
function amount(v: number, unit: string): string {
  const n = fmtSig(Math.abs(v), 2);
  return !unit ? n : unit === "%" ? `${n}%` : isScale(unit) ? `${n} on the ${unit} scale` : `${n} ${unit}`;
}

/** A change of a lever in words: "+32 pts" when it is added, "a cut of 20 pts" when it is removed. */
function change(v: number, unit: string, decrease: boolean): string {
  return decrease ? `a cut of ${amount(v, unit)}` : `+${amount(v, unit)}`;
}

/** The lever's catalogue label as a word in a sentence ("Canopy" → "canopy"; "NDVI" stays). */
function inSentence(label: string): string {
  return /^[A-Z][a-z]/.test(label) ? label.charAt(0).toLowerCase() + label.slice(1) : label;
}

/** The marker on the curve: the median dose by which a cell reaches 90% of its maximum cooling, when it lies on the curve. */
function d90Marker(L: ResponseLever): number | null {
  const d = L.effects?.median_d90 ?? null;
  const xs = L.curve?.dose.filter((x) => Number.isFinite(x)) ?? [];
  if (d === null || !Number.isFinite(d) || !xs.length) return null;
  return d >= Math.min(...xs) && d <= Math.max(...xs) ? d : null;
}

/** Label of the "most of the benefit" marker on the curve. */
export const MOST_OF_THE_BENEFIT = "most of the benefit";

/**
 * One lever's dose–response line, in plain words for What cools: the city-mean cooling with its
 * likely range (±1.96 SE ribbon), hollow points where more than 20% of cells are extrapolated, and a
 * "most of the benefit" marker at the median d90. Shared by this page and What cools.
 */
export function DoseResponseCurve({ name, lever }: { name: string; lever: ResponseLever }) {
  const u = unitLabel(useUnits().target);
  const lu = leverUnit(name, lever);
  const plain = leverName(name, lever);
  const word = leverMeasure(name, lever);
  const c = lever.curve;
  if (!c) return <OlderCode title={`${plain}: ${term("dose_response").label.toLowerCase()}`} />;
  const lo = c.benefit.map((b, i) => (b === null || c.se[i] === null ? null : b - 1.96 * (c.se[i] as number)));
  const hi = c.benefit.map((b, i) => (b === null || c.se[i] === null ? null : b + 1.96 * (c.se[i] as number)));
  const hollow = c.frac_extrapolated.map((f) => f !== null && f > 0.2);
  const last = c.dose.length - 1;
  const top = last > 0 ? c.benefit[last] : null;
  const se = last > 0 ? c.se[last] : null;
  const decrease = lever.direction === "decrease";
  const d90 = d90Marker(lever);
  // Benefit is cooling (positive = cooler); a negative benefit warms the city.
  const likely = term("interval_95").label.toLowerCase();
  // the likely range of the cooling at the largest amount; when it crosses zero the sentence says so,
  // and words each end as cooler or warmer (a negative cooling is a warming)
  const rLo = top === null || se === null ? null : top - 1.96 * se;
  const rHi = top === null || se === null ? null : top + 1.96 * se;
  const crosses = rLo !== null && rHi !== null && rLo < 0 && rHi > 0;
  const range =
    rLo === null || rHi === null || top === null
      ? ""
      : crosses
        ? ` (${likely}: ${fmtTempChange(-rLo, u, 2)} to ${fmtTempChange(-rHi, u, 2)})`
        : top >= 0
          ? ` (${likely} ${fmtRange(rLo, rHi, u, 2)})`
          : ` (${likely} ${fmtRange(-rHi, -rLo, u, 2)})`;
  const effect =
    top === null
      ? ""
      : top >= 0
        ? `the city cools by ${fmtValue(top, u, 2)} on average${crosses ? ", but it might not cool at all" : ""}${range}`
        : `the city warms by ${fmtValue(-top, u, 2)} on average${crosses ? ", but it might not warm at all" : ""}${range}`;
  const sentences =
    top !== null
      ? [
          isScale(lu)
            ? `If every place ${decrease ? "lowers" : "raises"} ${word} by ${fmtSig(Math.abs(c.dose[last]), 2)} (on a ${lu} scale, where there is room), ${effect}.`
            : `If every place ${decrease ? "removes" : "adds"} ${amount(c.dose[last], lu)} of ${word} (where there is room), ${effect}.`,
          d90 !== null ? `The “${MOST_OF_THE_BENEFIT}” line: by ${change(d90, lu, decrease)}, a typical place gets about 90% of the cooling it can get.` : "",
          hollow.some(Boolean)
            ? `Hollow points: more than 1 in 5 places go beyond anything measured in the city today (${fmtPct(c.frac_extrapolated[last], 0)} at the largest amount), so the model is less sure there.`
            : "",
        ]
      : [];
  const refLines: RefLine[] = [{ axis: "y", value: 0 }, ...(d90 !== null ? [{ axis: "x" as const, value: d90, label: MOST_OF_THE_BENEFIT }] : [])];
  return (
    <LineBand
      title={term("dose_response").label}
      exportName={`${plain} ${term("dose_response").label}`}
      units={`${u ? `${u} of ` : ""}cooling, citywide average (higher = cooler)`}
      note={
        <>
          How much the city cools as every place gets more. The shaded band is the <Term id="interval_95">{likely}</Term>.
          <InfoTip term="dose_response" />
        </>
      }
      series={[{ id: name, label: "Citywide average cooling", x: c.dose, y: c.benefit, lo, hi, hollow }]}
      xLabel={`${word} ${decrease ? "removed" : "added"}`}
      xUnit={lu}
      yLabel="Cooling"
      yUnit={u}
      yInclude={[0]}
      refLines={refLines}
      decimals={2}
      caption={sentences.filter(Boolean).join(" ") || undefined}
    />
  );
}

/** Requested vs realised dose: how much of each requested dose survives clipping to the lever bounds. */
function RealisedDose({ lever }: { lever: ResponseLever }) {
  const lu = unitLabel(lever.unit);
  const c = lever.curve;
  if (!c) return null;
  const shortfall = c.realized_dose.map((r, i) => (r === null ? null : c.dose[i] - r));
  const maxShort = Math.max(0, ...shortfall.filter((v): v is number => v !== null));
  return (
    <LineBand
      title={`${lever.label}: requested vs realised dose`}
      units={lu}
      note={<InfoTip term="realised" label="Realised dose" />}
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
  );
}

function DoseResponse({ name, lever }: { name: string; lever: ResponseLever }) {
  if (!lever.curve) return <OlderCode title="Dose–response curve" />;
  return (
    <div className="grid2">
      <DoseResponseCurve name={name} lever={lever} />
      <RealisedDose lever={lever} />
    </div>
  );
}

type ShapeId = "saturating" | "linear" | "sigmoid" | "censored" | "insufficient";

/** The bar's order: the four glossary shapes, then the cells with too few doses to fit. */
const SHAPE_ORDER: readonly ShapeId[] = ["saturating", "linear", "sigmoid", "censored", "insufficient"];

/** A shape's plain label: the glossary's ("Diminishing", "Hit the limit"), with "returns" after the one-word ones. */
export function shapeLabel(k: ShapeId): string {
  if (k === "insufficient") return "Too few amounts to tell";
  const label = shape(k).label;
  return k === "saturating" || k === "linear" ? `${label} returns` : label;
}

/**
 * Each cell in exactly one shape, as the "Curve shape per cell" map layer counts them. Core counts
 * a censored cell (no knee within the tested doses or its headroom) among the saturating or S-shaped
 * ones as well, so here it is taken out of those (saturating first, as nearly all are) and shown
 * once, as Hit the limit; the shares then add up to 1 with the cells that had too few doses.
 */
export function shapeShares(s: NonNullable<ResponseLever["shapes"]>): Record<ShapeId, number | null> {
  const cens = s.censored ?? 0;
  const fromSat = s.saturating !== null ? Math.min(cens, s.saturating) : 0;
  const fromSig = s.sigmoid !== null ? Math.min(cens - fromSat, s.sigmoid) : 0;
  return {
    saturating: s.saturating !== null ? Math.max(0, s.saturating - fromSat) : null,
    linear: s.linear,
    sigmoid: s.sigmoid !== null ? Math.max(0, s.sigmoid - fromSig) : null,
    censored: s.censored,
    insufficient: s.insufficient ?? null,
  };
}

/** Share of cells per fitted curve shape for one lever, in the glossary's plain words. Shared by this page and What cools. */
export function CurveShapes({ name, lever }: { name: string; lever: ResponseLever }) {
  const plain = leverName(name, lever);
  const title = "How the cooling builds up, place by place";
  if (!lever.shapes) return <OlderCode title={`${plain}: how the cooling builds up`} />;
  const shares = shapeShares(lever.shapes);
  const shown = SHAPE_ORDER.filter((k) => shares[k] !== null && (k !== "insufficient" || (shares[k] as number) > 0));
  return (
    <Bars
      title={title}
      exportName={`${plain} ${title}`}
      note={
        <>
          Each place&apos;s cooling, sorted by how it builds up as more is added.
          <InfoTip term="saturation" />
        </>
      }
      categories={[plain]}
      orientation="h"
      mode="percent"
      series={shown.map((k) => ({
        id: k,
        label: shapeLabel(k),
        values: [shares[k]],
        // one look per category: s1/s2/s3 for the fitted shapes, hatched grey for Hit the limit
        // (the default cycle would reuse s1), solid grey for too few doses
        muted: k === "insufficient" || k === "censored",
        hatch: k === "censored",
      }))}
      valueLabel="places"
      decimals={0}
      height={110}
      labelGutter={200}
      caption={`${plain}, share of places: ${shown.map((k) => `${shapeLabel(k).toLowerCase()} ${fmtPct(shares[k], 0)}`).join("; ")}.`}
    />
  );
}

/** "In 72% of places", "In every place" (the share of cells a shape covers, in words). */
function inPlaces(share: number, first: boolean): string {
  const of = first ? " of places" : "";
  if (share >= 1) return first ? "In every place" : "In all the rest";
  const pct = fmtPct(share, 0);
  if (pct === "100%") return `In nearly every place`;
  if (pct === "0%") return `In under 1%${of}`;
  return `In ${pct}${of}`;
}

/**
 * Diminishing returns in plain words, per lever: the share of places where each extra step cools
 * less than the last (exclusive shapes, as the bar shows), the biggest other shape when it covers
 * more places, and the median dose by which a place gets 90% of its possible cooling (left out when
 * the run has none). Null when the run has no curve shapes.
 */
export function saturationText(name: string, L: ResponseLever): { shapes: string; d90: string | null } | null {
  if (!L.shapes) return null;
  const s = shapeShares(L.shapes);
  const word = leverMeasure(name, L);
  const decrease = L.direction === "decrease";
  const step = decrease ? `each extra cut in ${word}` : `each extra step of ${word}`;
  const short = decrease ? "each extra cut" : "each extra step";
  const say: Record<"saturating" | "linear" | "sigmoid" | "censored", (stepWords: string) => string> = {
    saturating: (w) => `${w} cools less than the last (${shapeLabel("saturating").toLowerCase()})`,
    linear: (w) => `${w} cools about as much as the last (${shapeLabel("linear").toLowerCase()})`,
    sigmoid: () => `the cooling starts slowly, then builds once enough is ${decrease ? "removed" : "added"} (${shapeLabel("sigmoid").toLowerCase()})`,
    censored: () => `it is still cooling at the most the space allows (${shapeLabel("censored").toLowerCase()})`,
  };
  const fitted = (["saturating", "linear", "sigmoid", "censored"] as const).filter((k) => s[k] !== null);
  if (!fitted.length) return null;
  const sat = s.saturating ?? 0;
  const biggest = fitted.reduce((m, k) => ((s[k] as number) > (s[m] as number) ? k : m), fitted[0]);
  const parts: string[] = [];
  if (sat > 0) parts.push(`${inPlaces(sat, true)}, ${say.saturating(step)}.`);
  if (biggest !== "saturating" && (s[biggest] as number) > sat) parts.push(`${inPlaces(s[biggest] as number, sat <= 0)}, ${say[biggest](sat > 0 ? short : step)}.`);
  const d = L.effects?.median_d90 ?? null;
  const lu = leverUnit(name, L);
  const d90 = d !== null && Number.isFinite(d) && d > 0 ? `About 90% of the possible cooling comes by ${change(d, lu, decrease)}.` : null;
  if (!parts.length && !d90) return null;
  return { shapes: parts.join(" "), d90 };
}

/** The plain sentence on diminishing returns at the top of a lever's section on What cools, each part with its ⓘ. */
export function SaturationSummary({ name, lever }: { name: string; lever: ResponseLever }) {
  const t = saturationText(name, lever);
  if (!t) return null;
  return (
    <p className="prose" data-saturation={name}>
      {t.shapes ? (
        <>
          {t.shapes}
          <InfoTip term="saturation" />
        </>
      ) : null}
      {t.shapes && t.d90 ? " " : null}
      {t.d90 ? (
        <>
          {t.d90}
          <InfoTip term="d90" />
        </>
      ) : null}
    </p>
  );
}

/**
 * What the footprint / own-cell ratio r says. The footprint sums the change over every cell
 * (the changed cell included), so the share landing in neighbouring cells is 1 − 1/r.
 */
export function spillText(ratio: number | null): string {
  if (ratio === null || !Number.isFinite(ratio) || ratio === 0) return "No footprint / own-cell ratio for this lever.";
  const r = `Footprint / own-cell ratio ${fmtNum(ratio, 1)}`;
  if (ratio < 0) return `${r}: the neighbouring cells change in the opposite direction and outweigh the own-cell effect.`;
  if (ratio < 1) return `${r}: the neighbouring cells change in the opposite direction, offsetting part of the own-cell effect.`;
  const share = 1 - 1 / ratio;
  return `${r}: ${fmtPct(share, 0)} of the effect of a change lands in neighbouring cells${share > 0.5 ? ", more than in the changed cell itself" : ""}.`;
}

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
          title="Own vs footprint effect"
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
        <p className="cap" data-effect-terms="true">
          <Term id="own_cell">Own cell</Term>: the changed cell only. <Term id="footprint">Footprint</Term>: summed over every cell, the changed one included.{" "}
          {spillText(e.ratio)}
          {e.median_d90 !== null ? (
            <>
              {" "}
              Half the cells reach <Term id="d90">90% of their maximum cooling</Term> by {fmtValue(e.median_d90, lu, 1)}.
            </>
          ) : null}
          {e.median_max_cooling !== null ? ` Median maximum cooling ${fmtValue(e.median_max_cooling, u, 2)}.` : ""}
        </p>
      ) : null}
      {f ? (
        <SmallMultiples
          title="Fold means"
          note={
            <InfoTip
              term="held_out_fold"
              label="Fold means"
              text="The effect estimated again on each test area (cross-validation fold). When they agree, the estimate does not hinge on one part of the city."
              technical="Per-fold means of the own-cell and footprint effects"
            />
          }
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

/**
 * The literature panel's caption. SPARC's number is a rate: core `literature.sparc_effects` takes
 * the uniform scenario whose realised dose is nearest +0.10 (10 pp of canopy) and rescales its
 * city-mean cooling to "per +0.10", so it is not that scenario's own cooling unless the dose is
 * exactly the step. A scenario with more than 20% of cells extrapolated is flagged.
 */
export function literatureCaption(
  m: { scenario: string; dose: number | null; cooling: number | null; frac_extrapolated?: number | null },
  rows: { low: number | null; high: number | null }[],
  per: string,
  unit: string,
  leverUnit: string,
): string | undefined {
  const c = m.cooling;
  if (c === null) return undefined;
  const lu = unitLabel(leverUnit);
  const dose = m.dose !== null ? ` (realised dose ${fmtSig(m.dose, 3)}${lu ? (lu === "%" ? "%" : ` ${lu}`) : ""})` : "";
  const overlap = rows.some((r) => r.low !== null && r.low <= c * 2 && (r.high ?? r.low) >= c / 2);
  const extrap = m.frac_extrapolated !== undefined && m.frac_extrapolated !== null && m.frac_extrapolated > 0.2 ? ` That scenario pushes ${fmtPct(m.frac_extrapolated, 0)} of cells beyond the observed range, so treat the rate with caution.` : "";
  return `SPARC: ${c >= 0 ? `${fmtValue(c, unit, 2)} of cooling` : `${fmtValue(-c, unit, 2)} of warming`} per ${per}, scaled from ${m.scenario}${dose}; published values ${overlap ? "overlap its factor-of-2 band" : "fall outside its factor-of-2 band"}.${extrap}`;
}

function Literature({ lit, lever, leverUnit }: { lit: NonNullable<ResponseSections["literature"]>; lever: string; leverUnit: string }) {
  const rows = lit.rows.filter((r) => r.quantity === lever);
  const mine = lit.sparc.filter((r) => r.quantity === lever);
  if (!rows.length && !mine.length) return <p className="cap">No published values for this lever.</p>;
  const unit = unitLabel(rows[0]?.unit ?? mine[0]?.unit ?? "");
  // Published values and SPARC's are all rates per the same step ("+0.10 canopy cover").
  const per = rows[0]?.per ?? "+0.10";
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
      label: `SPARC per ${per} · from ${m.scenario}`,
      est: m.cooling,
      lo: m.cooling !== null && m.se !== null ? m.cooling - 1.96 * m.se : null,
      hi: m.cooling !== null && m.se !== null ? m.cooling + 1.96 * m.se : null,
      lo2: m.cooling !== null ? m.cooling / 2 : null,
      hi2: m.cooling !== null ? m.cooling * 2 : null,
      hollow: m.frac_extrapolated !== undefined && m.frac_extrapolated !== null && m.frac_extrapolated > 0.2,
      check: m.causal !== null ? { est: m.causal, label: "causal estimate" } : null,
    })),
  ];
  return (
    <div className="stack">
      <DotRange
        title="Literature check"
        note={
          <InfoTip
            label="Literature check"
            text="SPARC's cooling per step next to values published by other studies. Published values inside the factor-of-2 band (half to double SPARC's number) agree with it."
            technical="Published ranges vs SPARC ± 1.96 SE, with the causal estimate and a factor-of-2 band"
          />
        }
        units={`cooling per ${per}, ${unit}`}
        rows={dot}
        valueLabel={`Cooling per ${per}`}
        unit={unit}
        rangeLabel="published range / SPARC 95%"
        outerLabel="factor-of-2 band"
        decimals={2}
        caption={mine.length ? literatureCaption(mine[0], rows, per, unit, leverUnit) : undefined}
      />
      <Table
        title="Published values"
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

/** Links to one lever's maps (footprint, own-cell, curve shape, d90, maximum cooling). */
function LeverMaps({ rid, lever }: { rid: string; lever: string }) {
  const base = `/r/${encodeURIComponent(rid)}/map?layer=`;
  return (
    <Block title="Maps for this lever">
      <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 4 }}>
        <li>
          <Link to={`${base}fp_${encodeURIComponent(lever)}`}>Footprint effect per unit</Link>
        </li>
        <li>
          <Link to={`${base}own_${encodeURIComponent(lever)}`}>Own-cell effect per unit</Link>
        </li>
        <li>
          <Link to={`${base}cls_${encodeURIComponent(lever)}`}>Curve shape per cell</Link>
        </li>
        <li>
          <Link to={`${base}d90_${encodeURIComponent(lever)}`}>Dose reaching 90% of the maximum</Link>
        </li>
        <li>
          <Link to={`${base}A_${encodeURIComponent(lever)}`}>Maximum cooling</Link>
        </li>
      </ul>
      <p className="cap">Pin a cell on the map to see its own fitted curve.</p>
    </Block>
  );
}

export default function Response({
  title = "Response",
  intro = "What each lever does at increasing doses: city-mean cooling, curve shapes per cell, and how much of the effect spills into neighbouring cells.",
  level,
  detailsOnly,
}: DetailPageProps = {}) {
  const rid = useRid();
  const [lever, setLever] = useUrlState("lever", codecs.optString());
  return (
    <ViewPage view="response" title={title} intro={intro} level={level}>
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
              return (
                <div className="stack">
                  <Seg<string> label="Lever" value={cur} onChange={(v) => setLever(v)} options={names.map((n) => ({ value: n, label: levers[n].label || n }))} />
                  {detailsOnly ? (
                    // The curve and the curve shapes are on What cools (linked from the section intro).
                    <div className="grid2">
                      {L.curve ? <RealisedDose lever={L} /> : <OlderCode title="Requested vs realised dose" />}
                      <LeverMaps rid={rid} lever={cur} />
                    </div>
                  ) : (
                    <>
                      <DoseResponse name={cur} lever={L} />
                      <div className="grid2">
                        <CurveShapes name={cur} lever={L} />
                        <LeverMaps rid={rid} lever={cur} />
                      </div>
                    </>
                  )}
                  <Effects lever={L} />
                  <Section title="Literature" data={s.literature}>
                    {(lit) => <Literature lit={lit} lever={cur} leverUnit={L.unit} />}
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
