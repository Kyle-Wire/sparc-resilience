// Future climates and heat-risk levels in plain words (phase 4), shared by Heat risk (Today, Who is
// exposed, Future climate) and the Lab's Climate × adaptation:
// - an emissions future by its glossary label, the period by its name and years: ("ssp245",
//   "2041-2060") → "Moderate emissions, mid-century (2041–2060)". The pathway code (SSP2-4.5) only
//   ever appears in a tooltip's technical line.
// - a compact form for chart rows, whose labels are cut at about 26 characters: "Moderate, mid-century".
// - server strings that name a case by code ("SSP2-4.5 2041–2060 (with the cooling package)")
//   rewritten in the same words (plainCase), the rest of the string kept.
// - the heat-risk levels with each level's feels-like range, for an ⓘ (heatLevelsTip).
import { InfoTip, type InfoTipProps } from "../../components/ui/InfoTip";
import { Term } from "../../components/ui/Term";
import { heatCategory, periodLabel, ssp, term, type HeatKey } from "../../content/glossary";

const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/** "SSP2-4.5", "ssp245", "RCP 4.5" inside a longer string. */
const PATHWAY = /\b(ssp\s*\d[\d.-]*\d|rcp\s*\d[\d.]*)/i;
/** "2041-2060" / "2041–2060" inside a longer string. */
const YEARS = /(\d{4})\s*[-–]\s*(\d{4})/;

/** "2041-2060" → "2041–2060". */
const years = (period: string) => period.trim().replace(/^(\d{4})\s*[-–]\s*(\d{4})$/, "$1–$2");

/** An emissions future, capitalised: "ssp245" → "Moderate emissions"; an unknown pathway keeps `fallback` (its label) or its code. */
export function emissionsName(experiment: string, fallback?: string): string {
  const e = ssp(experiment);
  return e ? cap(e.label) : fallback || experiment.toUpperCase();
}

/** A period at the start of a sentence or a chip: "2041-2060" → "Mid-century (2041–2060)". */
export function periodName(period: string): string {
  return cap(periodLabel(period));
}

/** A future climate in full: ("ssp245", "2041-2060") → "Moderate emissions, mid-century (2041–2060)". */
export function futureName(experiment: string, period: string, fallback?: string): string {
  return `${emissionsName(experiment, fallback)}, ${periodLabel(period)}`;
}

/**
 * A future climate for a chart row (labels there are cut at about 26 characters): ("ssp245",
 * "2041-2060") → "Moderate, mid-century"; a period without a name keeps its years ("Moderate, 2061–2080").
 */
export function futureShort(experiment: string, period: string, fallback?: string): string {
  const e = ssp(experiment);
  const em = e ? cap(e.label.replace(/ emissions$/, "")) : fallback || experiment.toUpperCase();
  return `${em}, ${periodLabel(period).replace(/\s*\(\d{4}–\d{4}\)$/, "")}`;
}

/** The technical line of a future climate: "SSP2-4.5, 2041–2060". */
export function futureTechnical(experiment: string, period: string): string {
  return `${ssp(experiment)?.technical ?? experiment}, ${years(period)}`;
}

/**
 * A server string that names a future by code ("SSP2-4.5 2041–2060", "ssp245:2041-2060 (with the
 * cooling package)") with the code and years replaced by plain words (`short`: the chart-row form);
 * a string without both (Today, a zone) comes back as given.
 */
export function plainCase(text: string, short = false): string {
  const p = PATHWAY.exec(text);
  const y = YEARS.exec(text);
  if (!p || !y) return text;
  const period = `${y[1]}-${y[2]}`;
  const name = short ? futureShort(p[1], period, p[1]) : futureName(p[1], period, p[1]);
  const start = Math.min(p.index, y.index);
  const end = Math.max(p.index + p[0].length, y.index + y[0].length);
  return `${text.slice(0, start)}${name}${text.slice(end)}`;
}

/**
 * A future climate in running text or a table cell: the emissions words explain themselves on
 * hover, focus or tap (what the pathway means, its code in the technical line), then the period.
 * Not for use inside a button or link (the term is a button): use futureName() there.
 */
export function FutureName({ experiment, period, fallback }: { experiment: string; period: string; fallback?: string }) {
  const e = ssp(experiment);
  if (!e) return <>{futureName(experiment, period, fallback)}</>;
  return (
    <>
      <Term id="ssp" text={e.tip ?? term("ssp").tip} technical={futureTechnical(experiment, period)}>
        {cap(e.label)}
      </Term>
      , {periodLabel(period)}
    </>
  );
}

/** The emissions futures behind a set of plain labels, for an ⓘ's technical line: "SSP2-4.5 = moderate emissions · SSP5-8.5 = very high emissions". */
export function pathwayCodes(experiments: readonly string[]): string {
  return experiments.map((x) => `${ssp(x)?.technical ?? x} = ${ssp(x)?.label ?? x}`).join(" · ");
}

const LEVELS: readonly HeatKey[] = ["caution", "extreme_caution", "danger", "extreme_danger"];

/** The heat-risk levels, one line each with its feels-like range (from the glossary). */
function HeatLevelLines() {
  return (
    <>
      {LEVELS.map((k) => {
        const h = heatCategory(k);
        return (
          <span key={k} style={{ display: "block", marginTop: 4 }}>
            <strong>{h.label}</strong>: {h.tip}
          </span>
        );
      })}
    </>
  );
}

/** InfoTip props for the heat-risk levels: what they are, then each level's feels-like range. */
export function heatLevelsTip(label?: string): InfoTipProps {
  return {
    term: "heat_risk_level",
    label,
    text: (
      <>
        {term("heat_risk_level").tip}
        <HeatLevelLines />
      </>
    ),
  };
}

/** ⓘ explaining the heat-risk levels and the feels-like temperature at which each starts. */
export function HeatLevelsTip({ label }: { label?: string }) {
  return <InfoTip {...heatLevelsTip(label)} />;
}
