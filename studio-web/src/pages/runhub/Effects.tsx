// What cools run tab (`/r/:rid/effects`, phase 2): the planner's page on what each change does.
// One section per lever: its dose–response curve with the band, the curve shapes across cells,
// and how far its cooling reaches (ring kernel and influence range, from the Influence view).
// Then Where to act (the Planner pack's part, PlannerWhereToAct; without a planner pack its
// one-sentence empty state stands alone, under no second heading) and a link to try other
// amounts in the Scenario Lab. The technical parts (realised dose, own vs footprint and fold
// means, the literature check, correlogram, anisotropy, priors, the causal audit) live under
// Model details, where the complete Response and Influence pages render.
// `?lever=<key>` (kept by the legacy /response redirect) scrolls to that lever's section.
import { useEffect } from "react";
import type { InfluenceSections, ResponseLever } from "../../api/runs";
import { useView } from "../../api/runs";
import { Link, codecs, runHref, useUrlState } from "../../router";
import { Block, Section, ViewPage, useRid } from "./common";
import { LeverReach } from "./Influence";
import { PlannerWhereToAct } from "./Planner";
import { CurveShapes, DoseResponseCurve } from "./Response";

/** Anchor id of a lever's section on What cools. */
export function leverAnchor(lever: string): string {
  return `lever-${lever.replace(/[^A-Za-z0-9_-]/g, "_")}`;
}

/** The Influence view's reach sections as What cools reads them (undefined while loading). */
type Reach = { ranges: InfluenceSections["ranges"] | null; rings: InfluenceSections["rings"] | null } | undefined;

/** Stand-in for the reach chart while the Influence view loads, or when it failed. */
function ReachPending({ label, error }: { label: string; error: boolean }) {
  return error ? (
    <p className="cap">{label}: how far cooling reaches could not be loaded.</p>
  ) : (
    <p className="cap" role="status">
      <span className="spinner" aria-hidden="true" /> Loading how far cooling reaches…
    </p>
  );
}

function LeverSection({ name, lever, reach, reachError }: { name: string; lever: ResponseLever; reach: Reach; reachError: boolean }) {
  const label = lever.label || name;
  return (
    <Block id={leverAnchor(name)} title={label}>
      <div className="grid2">
        <DoseResponseCurve name={name} lever={lever} />
        {reach ? <LeverReach lever={name} label={label} ranges={reach.ranges} rings={reach.rings} /> : <ReachPending label={label} error={reachError} />}
      </div>
      <CurveShapes lever={lever} />
    </Block>
  );
}

function Levers({ levers, reach, reachError }: { levers: Record<string, ResponseLever>; reach: Reach; reachError: boolean }) {
  const [focus] = useUrlState("lever", codecs.optString());
  const names = Object.keys(levers);
  useEffect(() => {
    if (!focus || !levers[focus]) return;
    document.getElementById(leverAnchor(focus))?.scrollIntoView?.({ block: "start" });
  }, [focus, levers]);
  if (!names.length) return <p className="cap">No actionable levers were swept in this run.</p>;
  return (
    <div className="stack">
      {names.map((n) => (
        <LeverSection key={n} name={n} lever={levers[n]} reach={reach} reachError={reachError} />
      ))}
    </div>
  );
}

export default function Effects() {
  const rid = useRid();
  const infl = useView(rid, "influence");
  const planner = useView(rid, "planner");
  // Without a planner pack, Where to act is its titled empty state alone (no heading above it).
  const whereEmpty = planner.data?.availability === "missing";
  const vm = infl.data;
  // A view that is missing entirely reads as "not in this run" per lever (null sections).
  const reach: Reach = vm ? (vm.availability === "missing" ? { ranges: null, rings: null } : { ranges: vm.sections.ranges, rings: vm.sections.rings }) : undefined;
  return (
    <div className="stack composer" data-composer="effects">
      <section id="levers" className="composer-section" aria-label="What each change does">
        <ViewPage view="response" title="What cools" intro="What each lever does at increasing doses, and how far its cooling reaches.">
          {(s) => (
            <Section title="Levers" data={s.levers}>
              {(levers) => <Levers levers={levers} reach={reach} reachError={!!infl.error} />}
            </Section>
          )}
        </ViewPage>
      </section>
      <section id="where" className="composer-section" aria-labelledby={whereEmpty ? undefined : "effects-where"} aria-label={whereEmpty ? "Where to act" : undefined}>
        {whereEmpty ? null : (
          <h2 id="effects-where" className="composer-heading">
            Where to act
          </h2>
        )}
        <PlannerWhereToAct />
      </section>
      <p data-lab-link="true">
        <Link to={runHref(rid, "lab", "sweeps")}>Try other amounts in the Scenario Lab →</Link>
      </p>
    </div>
  );
}
