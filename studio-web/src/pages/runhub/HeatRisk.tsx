// Heat risk run tab (`/r/:rid/heat/:section?`, phase 2), one URL per section (default Today):
// - Today: the Heat view's baseline (NWS categories, residents by category, the heat-index
//   histogram and map, the humidity assumption). The canonical home of the heat-risk baseline.
// - Who is exposed: the Planner pack's exposure, person-mean temperature, hot days and equity.
// - Future climate: the Climate view (warming by SSP × period, exposure, future map; embedded, so
//   without its offset gauges), the Heat view's residents per category in each future, then "Can
//   adaptation keep up?" (the Lab's Climate × adaptation matrix, embedded: the one home of the
//   warming offset). Each climate number is shown once.
// The what-if dewpoint is page state (`?dewpoint=`, °C) so Today and Future climate agree; the
// section links carry it. How sure each scenario is lives on Trust; the planner pack's "where to
// act" parts on What cools; its GIS files on Reports.
import { lazy, Suspense } from "react";
import { Term } from "../../components/ui/Term";
import { SectionLoading, SectionNav } from "../../layouts/SectionNav";
import { codecs, HEAT_SECTIONS, runHref, useRoute, useUrlState, type HeatSection } from "../../router";

const Heat = lazy(() => import("./Heat"));
const HeatFutures = lazy(() => import("./Heat").then((m) => ({ default: m.HeatFutures })));
const Planner = lazy(() => import("./Planner"));
const Climate = lazy(() => import("./Climate"));
const ClimateAdaptation = lazy(() => import("../lab/Climate").then((m) => ({ default: m.ClimateAdaptation })));

export const HEAT_SECTION_LABELS: Record<HeatSection, string> = { today: "Today", people: "Who is exposed", future: "Future climate" };

/** The Heat risk section of a `:section` param (unknown or absent → "today"). */
export function heatSectionOf(raw: string | null | undefined): HeatSection {
  return (HEAT_SECTIONS as readonly string[]).includes(raw ?? "") ? (raw as HeatSection) : "today";
}

/** Query key of the what-if dewpoint (°C) shared by Today and Future climate; absent = the campaign's. */
export const DEWPOINT_KEY = "dewpoint";

/** The page's what-if dewpoint (°C, null = the campaign's) and its setter (pushes a history entry). */
export function useDewpoint(): [number | null, (v: number | null) => void] {
  return useUrlState(DEWPOINT_KEY, codecs.float(null), { push: true });
}

/** URL of a Heat risk section, carrying a what-if dewpoint to the sections that use it. */
export function heatHref(rid: string, section: HeatSection, dewpoint?: number | null): string {
  const base = runHref(rid, "heat", section);
  return dewpoint !== null && dewpoint !== undefined && section !== "people" ? `${base}?${DEWPOINT_KEY}=${encodeURIComponent(String(dewpoint))}` : base;
}

/** Future climate: the warming and its exposure, heat risk in each future, then whether adaptation keeps up. */
function FutureClimate({ rid }: { rid: string }) {
  return (
    <>
      <section id="climate" className="composer-section">
        <Climate embedded />
      </section>
      <section id="futures" className="composer-section">
        <HeatFutures />
      </section>
      <section id="adapt" className="composer-section" aria-labelledby="heat-adapt">
        <h2 id="heat-adapt" className="composer-heading">
          Can adaptation keep up?
        </h2>
        <p className="cap prose">
          Try scenarios against each future climate: how much of the expected warming each one{" "}
          <Term id="climate_offset">cancels out</Term>
          , and how many places and residents stay at or above each temperature.
        </p>
        <Suspense fallback={<SectionLoading />}>
          <ClimateAdaptation rid={rid} embedded />
        </Suspense>
      </section>
    </>
  );
}

export default function HeatRisk() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
  const section = heatSectionOf(params.section);
  const [dewpoint] = useDewpoint();
  return (
    <div className="stack composer" data-composer="heat">
      <SectionNav label="Heat risk" current={section} items={HEAT_SECTIONS.map((s) => ({ id: s, label: HEAT_SECTION_LABELS[s], to: heatHref(rid, s, dewpoint) }))} />
      <Suspense fallback={<SectionLoading />}>{section === "today" ? <Heat /> : section === "people" ? <Planner /> : <FutureClimate rid={rid} />}</Suspense>
    </div>
  );
}
