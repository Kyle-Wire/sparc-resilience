// Model details run tab (`/r/:rid/details/:section?`, phase 2): every technical view, complete,
// behind a sticky left section rail (kicker "Model details", ten sections; below 900px it turns
// into a horizontal scroll strip, ./details.css). Each section renders the original page(s) by
// their default exports, the first titled with the rail's label and given the section's one
// muted line (what it is for and, where a planner tab has the plain version, a link to it) as its
// `intro`, so the line sits under the title (DetailPageProps). Response, Influence and Distance
// get `detailsOnly`, which leaves out the charts and notices a planner tab already shows.
// Documents also take a document (`/r/:rid/details/docs/:doc`, read by ./Docs from the route
// params). The default section is Data & checks (the only home of the run's data QA and input
// forcing). The Run log is the compact run record (./Overview RunRecord with `detailsOnly`: no
// outputs grid, commit, versions or run directory, which Files and Provenance hold, and stage
// timings only for a run without a job log) over Mission Control of the run's latest job
// (../tracking/RunTrack, `embedded`: its Outputs and Config & provenance tabs become links to
// Files and Provenance); `/jobs/:jid` is unchanged.
import { lazy, Suspense, useEffect, useRef, type ComponentType, type LazyExoticComponent } from "react";
import { SectionLoading, SectionNav } from "../../layouts/SectionNav";
import { DETAIL_SECTIONS, Link, runHref, useRoute, type DetailSection, type RunSection } from "../../router";
import { SectionBoundary, type DetailPageProps } from "./common";
import "./details.css";

/** What a section passes its pages: DetailPageProps, and `embedded` for the job log (../tracking/RunTrack). */
type PageProps = DetailPageProps & { embedded?: boolean };

type Page = LazyExoticComponent<ComponentType<PageProps>>;

/** One page of a section and what the section passes it (`title` and `intro` default to the rail label and the section's line on the first page). */
type SectionPage = { page: Page; props?: PageProps };

export const DETAIL_SECTION_LABELS: Record<DetailSection, string> = {
  data: "Data & checks",
  accuracy: "Accuracy",
  reach: "Reach & spatial structure",
  response: "Response curves",
  causal: "Cause-and-effect audit",
  uncertainty: "Uncertainty sources",
  runlog: "Run log",
  provenance: "Provenance",
  docs: "Documents",
  files: "Files",
};

/** The one-line intro at the top of each section: what the section is for. */
export const DETAIL_SECTION_INTROS: Record<DetailSection, string> = {
  data: "Every check on the input data before training: the grid, the lever doses, the predictors and the weather forcing.",
  accuracy: "The complete accuracy record behind the trust grade: the cross-validation design, every model and the blend.",
  reach: "How far the model's skill holds up away from measurements, and how far each predictor's influence carries.",
  response: "The full dose-response record of every lever, city-wide and cell by cell.",
  causal: "The independent cause-and-effect check of every lever, with all its diagnostics.",
  uncertainty: "Every source of uncertainty in the scenario estimates and how they add up.",
  runlog: "How the run went: its record and the log of its latest job.",
  provenance: "What it takes to audit or repeat this run exactly.",
  docs: "The run's written reports: methods, model card and the study write-ups.",
  files: "Every file the run wrote, to inspect or download.",
};

type PlainHome = { section: RunSection; label: string };
const TRUST: PlainHome = { section: "trust", label: "How much to trust this" };
const WHAT_COOLS: PlainHome = { section: "effects", label: "What cools" };

/** Sections whose numbers or charts a planner tab also shows in plain language: those tabs, linked from the intro. */
const PLAIN_HOME: Partial<Record<DetailSection, PlainHome[]>> = {
  accuracy: [TRUST],
  reach: [TRUST, WHAT_COOLS],
  response: [WHAT_COOLS],
  causal: [TRUST],
  uncertainty: [TRUST],
};

/** Sections whose page has no heading of its own: the composer titles them. */
const NEEDS_HEADING: ReadonlySet<DetailSection> = new Set<DetailSection>(["runlog"]);

/** The page(s) each section renders, top to bottom. */
const PAGES: Record<DetailSection, SectionPage[]> = {
  data: [{ page: lazy(() => import("./DataQa")) }],
  accuracy: [{ page: lazy(() => import("./Accuracy")) }],
  reach: [
    { page: lazy(() => import("./Distance")), props: { detailsOnly: true } },
    { page: lazy(() => import("./Influence")), props: { title: "How far each predictor reaches", level: 3, detailsOnly: true } },
  ],
  response: [{ page: lazy(() => import("./Response")), props: { detailsOnly: true } }],
  causal: [{ page: lazy(() => import("./Causal")) }],
  uncertainty: [{ page: lazy(() => import("./Uncertainty")) }],
  runlog: [
    { page: lazy(() => import("./Overview").then((m) => ({ default: m.RunRecord }))), props: { detailsOnly: true } },
    { page: lazy(() => import("../tracking/RunTrack")), props: { embedded: true } },
  ],
  provenance: [{ page: lazy(() => import("./Provenance")) }],
  docs: [{ page: lazy(() => import("./Docs")) }],
  files: [{ page: lazy(() => import("./Files")) }],
};

/** The section of a Model details URL: the docs route, else a known `:section`, else "data". */
export function detailSectionOf(routePath: string | null | undefined, raw: string | null | undefined): DetailSection {
  if (routePath && /\/details\/docs(\/|$)/.test(routePath)) return "docs";
  return (DETAIL_SECTIONS as readonly string[]).includes(raw ?? "") ? (raw as DetailSection) : "data";
}

/** In the narrow (strip) layout, scroll the current section's link into view inside the strip only. */
function useCurrentInStrip(section: DetailSection) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const ul = ref.current?.querySelector("ul");
    const a = ref.current?.querySelector<HTMLElement>('a[aria-current="page"]');
    if (!ul || !a || ul.scrollWidth <= ul.clientWidth) return;
    const box = ul.getBoundingClientRect();
    const at = a.getBoundingClientRect();
    if (at.left < box.left) ul.scrollLeft -= box.left - at.left + 12;
    else if (at.right > box.right) ul.scrollLeft += at.right - box.right + 12;
  }, [section]);
  return ref;
}

/** The section's one line (and its plain-summary links): the first page renders it under its title. */
function SectionIntro({ rid, section }: { rid: string; section: DetailSection }) {
  const plain = PLAIN_HOME[section];
  return (
    <span className="detail-intro" data-intro={section}>
      {DETAIL_SECTION_INTROS[section]}
      {plain?.length ? (
        <>
          {" "}
          <span className="detail-intro-plain">
            Plain summary:{" "}
            {plain.map((h, i) => (
              <span key={h.section}>
                {i ? " · " : ""}
                <Link to={runHref(rid, h.section)} data-plain-link={h.section}>
                  {h.label}
                </Link>
              </span>
            ))}
          </span>
        </>
      ) : null}
    </span>
  );
}

export default function Details() {
  const { params, route } = useRoute();
  const rid = params.rid ?? "";
  const section = detailSectionOf(route?.path, params.section);
  const railRef = useCurrentInStrip(section);
  const label = DETAIL_SECTION_LABELS[section];
  const headed = NEEDS_HEADING.has(section);
  const intro = <SectionIntro rid={rid} section={section} />;
  return (
    <div className="detail-layout" data-composer="details">
      <div className="detail-rail" ref={railRef}>
        <p className="kicker detail-rail-kicker" aria-hidden="true">
          Model details
        </p>
        <SectionNav
          variant="rail"
          label="Model details"
          current={section}
          items={DETAIL_SECTIONS.map((s) => ({ id: s, label: DETAIL_SECTION_LABELS[s], to: runHref(rid, "details", s) }))}
        />
      </div>
      <div className="stack detail-body" data-section={section}>
        {headed ? (
          // A section whose page has no heading: the composer titles it, its line under the title.
          <header className="stack detail-header">
            <h2 className="detail-heading">{label}</h2>
            <p className="cap prose">{intro}</p>
          </header>
        ) : null}
        <SectionBoundary key={section} title={label}>
          <Suspense fallback={<SectionLoading />}>
            {PAGES[section].map(({ page: P, props }, i) => (
              // Each page drops its own intro; the first takes the rail label as its title and the section's line as its intro.
              <P key={`${section}:${i}`} {...(i === 0 && !headed ? { title: label, intro } : { intro: null })} {...props} />
            ))}
          </Suspense>
        </SectionBoundary>
      </div>
    </div>
  );
}
