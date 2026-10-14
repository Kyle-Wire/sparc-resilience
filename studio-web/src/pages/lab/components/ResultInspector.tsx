// Pieces of the full-model result (phase 3). The old one-page inspector became result/ResultView
// (the accordion) + result/AnswerCard; this module keeps the pure helpers (spillRings,
// uncertaintyLayers, tested in library.test.ts), the props type ResultView still takes, and the
// impacts pieces the accordion lays out: heat risk by NWS category, residents at or above each
// temperature, who benefits (equity quintiles), hot days, and the share of future warming offset.
// Rows are split by case so that today's lines show in "Heat risk & people" and the futures in
// "Against future warming", each once.
import type { Impacts, Result } from "../../../api/lab";
import { Bars, Gauge, type IntervalLayer, type Ring } from "../../../charts";
import { ActionButton } from "../../../components/ui/EmptyState";
import { Table } from "../../../components/ui/Table";
import { InfoTip } from "../../../components/ui/InfoTip";
import { Term } from "../../../components/ui/Term";
import { ssp, term } from "../../../content/glossary";
import { fmtInt, fmtNum, fmtPct, fmtSigned, unitLabel } from "../../../theme/format";
import { concentrationCaption } from "../model/plain";
import { casePathway, emissionsWord, futureCaseLabel, futureLabel, humiditySourceWord, periodWord } from "../model/words";

/** Spill rings for the chart: ring 0 (r_m = 0) is the edited set; ring k spans (r_{k−1}, r_k]. */
export function spillRings(rings: Result["spill"]["rings"]): Ring[] {
  const sorted = [...rings].sort((a, b) => a.r_m - b.r_m);
  let prev = 0;
  return sorted.map((r) => {
    const ring: Ring = r.r_m === 0 ? { r0_m: 0, r1_m: 0, mean: r.mean, se: r.se, n: r.n } : { r0_m: prev, r1_m: r.r_m, mean: r.mean, se: r.se, n: r.n };
    prev = r.r_m;
    return ring;
  });
}

/** The uncertainty layers of a result (narrow to wide) for the IntervalStack. */
export function uncertaintyLayers(u: NonNullable<Result["uncertainty"]>): IntervalLayer[] {
  const out: IntervalLayer[] = [];
  const add = (id: string, label: string, v: [number, number] | null) => v && out.push({ id, label, lo: v[0], hi: v[1] });
  add("estimation", "Estimation (95%, fold jackknife)", u.estimation_95);
  add("specification", "Specification (across runs / variants)", u.specification);
  add("attribution", "Attribution (simulation check)", u.attribution);
  add("causal", "Independent causal band", u.causal_band);
  add("envelope", "Envelope (all sources)", u.envelope);
  return out;
}

/** Plain labels of the uncertainty layers (by `IntervalLayer.id`). */
export const UNCERTAINTY_PLAIN: Record<string, string> = {
  estimation: "Model fit (the likely range)",
  specification: "Across model versions and runs",
  attribution: "Simulation check",
  causal: "Independent cause-and-effect check",
  envelope: "All sources together",
};

/** Props of the result view (formerly the inspector). */
export type ResultInspectorProps = {
  rid: string;
  resId: string;
  pid: string | null;
  unit: string;
  nFolds: number | null;
  onShowLayer?: (key: string) => void;
};

/** A future-climate case ("SSP2-4.5 2041-2060", "ssp245 2041-2060"); anything else is today. */
export function isFutureCase(c: string): boolean {
  return !/^(today|present|obs)/i.test(c) && (/ssp|rcp/i.test(c) || /\b(20[3-9]\d|2100)\b/.test(c));
}

/** The humidity cases of the heat-risk rows, as the Heat risk page words them (sparc/studio/runs/heat.py assumptions). */
const HUMIDITY_LABEL: Record<string, string> = { observed: "as measured", constant_dewpoint: "today's moisture (low end)", constant_rh: "as muggy as today (high end)" };

/** The futures' humidity range in one sentence, as the Heat risk page says it (only when both cases are listed). */
const FUTURE_HUMIDITY_RANGE =
  "For future years we show a range: at the low end the air holds the same amount of water as today, so it feels a little less muggy as it warms; at the high end it takes up more water as it warms and stays as muggy as today, so the heat feels worse.";

/**
 * A case of the impacts tables in plain words: today's ("Measured hot afternoon" for heat risk,
 * "Today" for residents by temperature) or a future ("Moderate emissions, mid-century (2041–2060)").
 * The pathway code (SSP2-4.5…) shows only in the cell's tooltip.
 */
export function caseWords(c: string, today = "Today"): string {
  if (!isFutureCase(c)) return /^(today|present|obs)/i.test(c) ? today : c;
  return futureCaseLabel(c) ?? c;
}

/** A case cell: a future explains its emissions words on hover, focus or tap, with the code as the technical line. */
const caseCell = (c: string, today?: string) => {
  const words = caseWords(c, today);
  if (words === c || !isFutureCase(c)) return words;
  const pathway = casePathway(c);
  return (
    <Term id="ssp" text={(pathway ? ssp(pathway)?.tip : null) ?? term("ssp").tip} technical={c}>
      {words}
    </Term>
  );
};

/** A dew point (°C, as the impacts carry it) in the page's unit: "64 °F" on a °F page. */
export function dewPointText(dewpointC: number, unit: string): string {
  if (!Number.isFinite(dewpointC)) return "";
  // the unit as the run states it ("degF", "°F", "F") or its display form
  const u = unitLabel(unit).replace(/^deg\s*/i, "°").replace(/^(F|C)$/i, "°$1").toUpperCase();
  if (u === "°F") return `${fmtNum((dewpointC * 9) / 5 + 32, 0)} °F`;
  if (u === "K") return `${fmtNum(dewpointC + 273.15, 0)} K`;
  return `${fmtNum(dewpointC, 0)} °C`;
}

/** "Humidity: as measured in the field survey (dew point 64 °F)." */
export function humidityNote(heat: Pick<NonNullable<Impacts["heat"]>, "dewpoint_C" | "source">, unit: string): string {
  const where = humiditySourceWord(heat.source);
  const dp = dewPointText(heat.dewpoint_C, unit);
  return `Humidity: as measured${where ? ` ${where}` : ""}${dp ? ` (dew point ${dp})` : ""}.`;
}

type HeatRow = NonNullable<Impacts["heat"]>["rows"][number];

/**
 * Heat risk (NWS heat index) before → after, one row per case and humidity assumption. Today's
 * table leaves out "moved out" (the answer card's fact); the futures' table keeps it.
 */
export function HeatRiskTable({ heat, rows, future, unit = "degF" }: { heat: NonNullable<Impacts["heat"]>; rows: HeatRow[]; future: boolean; unit?: string }) {
  const who = heat.measure === "people" ? "Residents" : "Places";
  const n = (v: number) => fmtInt(Math.round(v));
  const bothHumidities = rows.some((r) => r.humidity === "constant_dewpoint") && rows.some((r) => r.humidity === "constant_rh");
  return (
    <section className="stack" aria-label={future ? "Heat risk in future climates" : "Heat risk"} style={{ gap: 8 }}>
      <Table
        title={future ? `${who} at heat risk as the climate warms, before → after` : `${who} at heat risk today, before → after`}
        csvName={future ? "heat-risk-future" : "heat-risk"}
        rowKey={(r, i) => `${r.case}-${r.humidity}-${i}`}
        columns={[
          { key: "case", label: future ? "Future" : "Case", value: (r) => caseWords(r.case, "Measured hot afternoon"), render: (r) => caseCell(r.case, "Measured hot afternoon") },
          { key: "hum", label: "Humidity", value: (r) => HUMIDITY_LABEL[r.humidity] ?? r.humidity },
          {
            key: "ec",
            label: "Extreme caution or worse",
            align: "right",
            value: (r) => r.after.ec_or_worse,
            render: (r) => `${n(r.before.ec_or_worse)} → ${n(r.after.ec_or_worse)}`,
          },
          ...(future ? [{ key: "eca", label: "Moved out", align: "right" as const, value: (r: HeatRow) => r.ec_avoided, render: (r: HeatRow) => n(r.ec_avoided) }] : []),
          {
            key: "dg",
            label: "Danger or worse",
            align: "right",
            value: (r) => r.after.danger_or_worse,
            render: (r) => `${n(r.before.danger_or_worse)} → ${n(r.after.danger_or_worse)}`,
          },
          { key: "dhi", label: "Change in feels-like temperature", unit: "°F", align: "right", value: (r) => r.mean_hi_change, render: (r) => fmtSigned(r.mean_hi_change, 2) },
        ]}
        rows={rows}
      />
      <p className="cap">
        {future
          ? bothHumidities
            ? `Each future adds the warming the climate models expect (their middle estimate). ${FUTURE_HUMIDITY_RANGE}`
            : "Each future adds the warming the climate models expect (their middle estimate), with today's measured humidity."
          : `Extreme caution means a feels-like temperature of 90 °F or more (heat cramps and exhaustion possible); Danger, 103 °F or more. ${humidityNote(heat, unit)}`}
      </p>
    </section>
  );
}

type ExposureRow = Impacts["exposure"][number];

/** Residents at or above each air-temperature threshold (not the feels-like levels of heat risk), without and with the changes. */
export function ExposureTable({ impacts, rows, unit, future }: { impacts: Impacts; rows: ExposureRow[]; unit: string; future: boolean }) {
  const u = unitLabel(unit);
  const tKey = (rec: Record<string, number>, t: number) => Object.keys(rec).find((k) => Number(k) === t) ?? String(t);
  return (
    <Table
      title={future ? "Residents at or above each air temperature in future climates" : "Residents at or above each air temperature today"}
      csvName={future ? "exposure-future" : "exposure"}
      rowKey={(r, i) => `${r.case}-${r.adapted}-${i}`}
      columns={[
        { key: "case", label: future ? "Future" : "Case", value: (r) => caseWords(r.case), render: (r) => caseCell(r.case) },
        { key: "adapted", label: "Your changes", value: (r) => (r.adapted ? "with" : "without") },
        { key: "mean", label: term("person_mean").label, tip: <InfoTip term="person_mean" />, unit: u, align: "right", value: (r) => r.person_mean_temp, render: (r) => fmtNum(r.person_mean_temp, 2) },
        ...impacts.thresholds.map((t) => ({
          key: `ge_${t}`,
          label: `Air temperature ≥ ${fmtNum(t, Number.isInteger(t) ? 0 : 1)} ${u}`,
          align: "right" as const,
          value: (r: ExposureRow) => r.people_ge[tKey(r.people_ge, t)] ?? null,
          render: (r: ExposureRow) => `${fmtInt(Math.round(r.people_ge[tKey(r.people_ge, t)] ?? NaN))} (${fmtPct(r.share_people_ge[tKey(r.share_people_ge, t)] ?? null)})`,
        })),
      ]}
      rows={rows}
    />
  );
}

/** Who benefits: mean cooling by quintile of each equity variable. */
export function EquityBars({ impacts, unit }: { impacts: Impacts; unit: string }) {
  const u = unitLabel(unit);
  const groups = Object.entries(impacts.equity);
  if (!groups.length) return null;
  return (
    <div className="grid2">
      {groups.map(([g, e]) => (
        <Bars
          key={g}
          title={`Who benefits: cooling by ${g.replace(/_/g, " ")} (fifths)`}
          units={`${u} (positive = cooler)`}
          categories={e.quintiles.map((q) => `Q${q.quintile}`)}
          series={[{ id: "cool", label: "Mean cooling", values: e.quintiles.map((q) => q.mean_cooling) }]}
          valueLabel="Mean cooling"
          unit={u}
          categoryLabel="Fifth of places (1 = lowest)"
          caption={concentrationCaption(e.concentration_index, e.resident_mean_cooling, g)}
        />
      ))}
    </div>
  );
}

/** Hot days avoided (person-days), or the action that fetches the station's daily history. */
export function HotDays({ impacts }: { impacts: Impacts }) {
  if (impacts.hot_days) {
    return (
      <Table
        title={`Hot days avoided, in person-days (weather station ${impacts.hot_days.station})`}
        csvName="hot-days"
        columns={Object.keys(impacts.hot_days.cases[0] ?? {}).map((k) => ({
          key: k,
          label: k.replace(/_/g, " "),
          value: (r: Record<string, unknown>) => (typeof r[k] === "number" || typeof r[k] === "string" ? (r[k] as number | string) : null),
        }))}
        rows={impacts.hot_days.cases}
      />
    );
  }
  if (impacts.hot_days_action) {
    return (
      <div className="row">
        <span className="cap">Hot days avoided need the weather station&apos;s daily history.</span>
        <ActionButton action={impacts.hot_days_action} size="small" />
      </div>
    );
  }
  return null;
}

/** Share of each future's median warming the citywide cooling cancels out (impacts.climate_offset). */
export function ClimateOffsets({ impacts }: { impacts: Impacts }) {
  const offsets = impacts.climate_offset.filter((o) => typeof o.offset_share === "number" && Number.isFinite(o.offset_share));
  if (!offsets.length) return null;
  return (
    <div className="grid3">
      {offsets.map((o) => (
        <Gauge
          key={`${o.experiment}-${o.period}`}
          title={futureLabel(o.experiment, o.period)}
          label="of the expected warming cancelled out"
          value={o.offset_share}
          min={0}
          max={Math.max(1, o.offset_share)}
          ticks={[1]}
          format={(v) => fmtPct(v)}
          caption={`The citywide cooling as a share of the warming the climate models expect by ${periodWord(o.period)} (${emissionsWord(o.experiment) ?? "this pathway"}, their middle estimate).`}
        />
      ))}
    </div>
  );
}
