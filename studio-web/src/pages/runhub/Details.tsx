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
// Plain words (phase 4): the sections keep their technical names, but each intro says in plain
// words what the section is for, with its jargon as glossary <Term>s, and a section with a
// technical title gets an ⓘ beside it (TitleTipProvider: the page's ViewPage shows it).
import { lazy, Suspense, useEffect, useRef, type ComponentType, type LazyExoticComponent, type ReactNode } from "react";
import { InfoTip, type InfoTipProps } from "../../components/ui/InfoTip";
import { Term } from "../../components/ui/Term";
import { SectionLoading, SectionNav } from "../../layouts/SectionNav";
import { DETAIL_SECTIONS, Link, runHref, useRoute, type DetailSection, type RunSection } from "../../router";
import { SectionBoundary, TitleTipProvider, type DetailPageProps } from "./common";
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

/** The one-line intro at the top of each section: what the section is for, in plain words (its jargon as glossary terms). */
export const DETAIL_SECTION_INTROS: Record<DetailSection, ReactNode> = {
  data: (
    <>
      Checks on everything that went into the model before training: the temperature readings, the map grid, the <Term id="dose_scale">size of the tested changes</Term>, the{" "}
      <Term id="predictor">map layers</Term> and the <Term id="forcing">weather inputs</Term>.
    </>
  ),
  accuracy: (
    <>
      How close the model comes to measured temperatures in places it was not trained on, in full: <Term id="cv_design">how it was tested</Term>, every model, and{" "}
      <Term id="stacker">how they are blended</Term>. The <Term id="trust_grade">trust grade</Term> comes from these numbers.
    </>
  ),
  reach: (
    <>
      Whether the model stays accurate farther from the measurements, and <Term id="length_scale">how far each map layer&apos;s influence carries</Term>.
    </>
  ),
  response: (
    <>
      How much each lever cools <Term id="dose_response">at each amount of change</Term>, from small to large, citywide and place by place, with the checks behind each curve.
    </>
  ),
  causal: (
    <>
      A <Term id="dag_audit">separate cause-and-effect analysis</Term> of every lever, to rule out cooling that is only coincidence, with all its diagnostics.
    </>
  ),
  uncertainty: (
    <>
      Every source of doubt in the scenario estimates, and how they add up to <Term id="envelope">the widest likely range</Term>.
    </>
  ),
  runlog: "How the run went: its record and the log of its latest job, stage by stage.",
  provenance: (
    <>
      What it takes to check or repeat this run exactly: <Term id="provenance">fingerprints of its inputs, settings and code</Term>, and the software it ran on.
    </>
  ),
  docs: "The run's written reports: how it was done (methods), what the model is for and where it falls short (model card), and the study write-ups.",
  files: "Every file the run wrote, to preview, convert or download.",
};

/** The ⓘ beside a section's technical title: what the name means, in plain words. Plain titles have none. */
export const DETAIL_SECTION_TIPS: Partial<Record<DetailSection, InfoTipProps>> = {
  accuracy: {
    term: "r2",
    label: "Accuracy",
    text: "How close the model's predictions come to measured temperatures, always scored on test areas it was not trained on.",
    technical: "Spatial-block cross-validation: held-out R², RMSE and interval coverage",
  },
  reach: {
    label: "Reach & spatial structure",
    text: "Two questions: does the model stay accurate farther from the measurements, and how far does each map layer, such as tree cover, affect the temperature around it?",
    technical: "Skill vs CV block size; influence length scales and ring kernels",
  },
  response: { term: "partial_dependence", label: "Response curves" },
  causal: { term: "dag_audit", label: "Cause-and-effect audit" },
  uncertainty: {
    term: "envelope",
    label: "Uncertainty sources",
    text: "Where the doubt in a scenario's answer comes from: the model fit, how the analysis was set up, a simulation check and a separate cause-and-effect check. Added up, they give the widest range.",
    technical: "Estimation, specification, attribution and causal bands, and their envelope",
  },
  provenance: { term: "provenance", label: "Provenance" },
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
  const tipProps = DETAIL_SECTION_TIPS[section];
  const tip = tipProps ? <InfoTip {...tipProps} /> : null;
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
            {/* a composer-titled section (the Run log) has a plain title: no ⓘ */}
            <h2 className="detail-heading">{label}</h2>
            <p className="cap prose">{intro}</p>
          </header>
        ) : null}
        <SectionBoundary key={section} title={label}>
          <TitleTipProvider tip={headed ? null : tip}>
            <Suspense fallback={<SectionLoading />}>
              {PAGES[section].map(({ page: P, props }, i) => (
                // Each page drops its own intro; the first takes the rail label as its title and the section's line as its intro.
                <P key={`${section}:${i}`} {...(i === 0 && !headed ? { title: label, intro } : { intro: null })} {...props} />
              ))}
            </Suspense>
          </TitleTipProvider>
        </SectionBoundary>
      </div>
    </div>
  );
}
