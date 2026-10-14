// Configured scenarios (SPEC §6.4): dot-and-whisker of each scenario's city-mean ΔT with its
// likely range and p10–p90 whisker (hollow when > 20% extrapolated, orange diamond for the
// causal band, flagged when the model falls outside it), ladders per lever, and the table
// with tier, folds and realised doses plus "Open in Lab" and "Clone to edit".
// Phase 2: their home is Scenario Lab › Library, whose first group "Set up with this run" is
// ConfiguredScenarios (the same sections, with the Lab's row actions and the
// `?configured=<slug>` focus), without what other tabs own: the ladders repeat What cools'
// dose-response curves (a one-line link instead), and the causal band is How much to trust
// this's (a small badge per table row instead of the chart's diamonds), and the chart has no
// table view (the table lists each scenario's numbers once, with its actions). The default export
// (the former run tab) renders them in full (`full`). The Verdict column shows only when a row has one.
// Plain words (phase 4): "citywide change", "amount", the glossary's terms with an ⓘ; status chips
// and buttons explain themselves in tooltips, never a native title.
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ScenarioRow, ScenariosSections, Sections } from "../../api/runs";
import { cloneConfiguredScenario } from "../../api/runs";
import { errorMessage } from "../../api/client";
import { DotRange, LineBand, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { InfoTip } from "../../components/ui/InfoTip";
import { Table } from "../../components/ui/Table";
import { term } from "../../content/glossary";
import { TipSpan } from "../../layouts/tips";
import { Link, navigate, runHref } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { fmtPct, fmtSig, unitLabel } from "../../theme/format";
import { Tip } from "../lab/components/Tip";
import { Block, Section, TipChip, VerdictPill, ViewPage, useRid, useUnits } from "./common";
import { confidenceWords, likelyText } from "./format";
import { leverName, leverUnit } from "./Response";

/**
 * A realised change in plain words: the glossary's lever name, a signed amount and its unit word
 * ("Trees & canopy +5 pts", "Less paving −19.3 pts", "Cool roofs & surfaces +0.05 (0–1)").
 */
export function realisedWords(key: string, v: number | null, leverUnits: Record<string, string>): string {
  const name = leverName(key, { label: key });
  if (v === null || !Number.isFinite(v)) return `${name} —`;
  const w = leverUnit(key, { unit: leverUnits[key] ?? "" });
  const unit = /^[\d.]+–[\d.]+$/.test(w) ? `(${w})` : w;
  return `${name} ${v > 0 ? "+" : ""}${fmtSig(v, 3)}${unit ? ` ${unit}` : ""}`;
}

/** The share beyond measured conditions, with a word when it is more than a sliver ("22% (partly beyond)", "73% (mostly beyond)"). */
function beyondText(frac: number | null): string {
  const pct = fmtPct(frac, 0);
  // the word from the share itself, as the Lab answer card words it ("mostly" only above half); the
  // server's tier calls anything over 20% "beyond", which read "22% (mostly beyond)"
  const word = frac === null ? null : frac > 0.5 ? "mostly beyond" : frac > 0.05 ? "partly beyond" : null;
  return word ? `${pct} (${word})` : pct;
}

/** A configured scenario's plain name: the server's label of a ladder scenario ("Trees & canopy +5 pts"), else its name. */
export function scenarioLabel(r: { name: string; label?: string | null }): string {
  return r.label || r.name;
}

/** The plain name, with the name in the run's settings ("Canopy Increase +5") in its tooltip when they differ. */
function ScenarioName({ r }: { r: Pick<ScenarioRow, "slug" | "name" | "label"> }) {
  const label = scenarioLabel(r);
  const text = <span data-configured={r.slug}>{label}</span>;
  return label === r.name ? text : <TipSpan tip="Its name in the run's settings." technical={r.name}>{text}</TipSpan>;
}

/** A change (negative = cooler) as cooling (higher = cooler), the way What cools and the Summary draw it. */
function cooling(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? (v === 0 ? 0 : -v) : null;
}

/** "Set up with this run" (the glossary's name for the run's configured scenarios), and "Scenarios set up with this run". */
const CONFIGURED = term("configured_scenario").label;
const SCENARIOS_CONFIGURED = `Scenarios ${CONFIGURED.charAt(0).toLowerCase()}${CONFIGURED.slice(1)}`;
/** The independent cause-and-effect check, in sentences. */
const CAUSAL = term("causal_band").label.toLowerCase();

/** Clone a configured scenario into the project's Scenario Lab and open the copy in Design. */
export function CloneButton({ rid, slug, pid: pidProp }: { rid: string; slug: string; pid?: string | null }) {
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = pidProp ?? ctxPid;
  const [busy, setBusy] = useState(false);
  return (
    <Tip tip={pid ? "Save an editable copy in the Scenario Lab." : "Open the run from its project to clone this scenario."}>
      <Button
        size="small"
        busy={busy}
        disabled={!pid}
        onClick={async () => {
          if (!pid) return;
          setBusy(true);
          try {
            const sid = await cloneConfiguredScenario(rid, pid, slug);
            navigate(`/r/${encodeURIComponent(rid)}/lab/s/${encodeURIComponent(sid)}`);
          } catch (e) {
            toast("error", "Could not clone the scenario", { body: errorMessage(e) });
          } finally {
            setBusy(false);
          }
        }}
      >
        Clone to edit
      </Button>
    </Tip>
  );
}

/** Row actions of the configured-scenarios table (replaces the default Open in Lab · Clone to edit). */
export type ScenarioActions = (r: ScenarioRow) => ReactNode;

/** The selector of a configured scenario's row (its name cell carries `data-configured`). */
export const configuredSelector = (slug: string) => `[data-configured="${slug.replace(/["\\]/g, "\\$&")}"]`;

/**
 * Every configured scenario with its city-mean change, confidence, verdict, extrapolation, tier,
 * folds and realised doses. `focus` (a slug) highlights its row and scrolls it into view once.
 */
/** A configured scenario's causal check as a small badge (the band itself is on How much to trust this). */
function CausalBadge({ c }: { c: ScenarioRow["causal"] }) {
  if (!c || c.model_within === null) return null;
  return c.model_within ? (
    <TipChip tip={`${term("causal_band").tip} The model's change is inside its range.`}>
      <Badge>cause-and-effect check agrees</Badge>
    </TipChip>
  ) : (
    <TipChip tip={`${term("causal_band").tip} The model's change is outside its range.`}>
      <Badge tone="warn">cause-and-effect check: different size</Badge>
    </TipChip>
  );
}

export function ScenarioTable({
  rows,
  rid,
  focus = null,
  actions,
  causalBadges,
}: {
  rows: ScenarioRow[];
  rid: string;
  focus?: string | null;
  actions?: ScenarioActions;
  /** Mark each row's causal check with a small badge (where the chart does not show the band). */
  causalBadges?: boolean;
}) {
  const units = useUnits();
  const u = unitLabel(units.target);
  const leverUnits = units.levers ?? {};
  const scrolled = useRef<string | null>(null);
  useEffect(() => {
    if (!focus || scrolled.current === focus) return;
    const el = document.querySelector(configuredSelector(focus));
    if (!el) return;
    scrolled.current = focus;
    el.scrollIntoView?.({ block: "center" });
  }, [focus, rows]);
  return (
    <Table<ScenarioRow>
      caption={SCENARIOS_CONFIGURED}
      csvName="configured-scenarios"
      rowKey={(r) => r.slug}
      highlight={focus ? (r) => r.slug === focus : undefined}
      columns={[
        {
          key: "name",
          label: "Scenario",
          value: (r) => scenarioLabel(r),
          render: (r) =>
            causalBadges && r.causal ? (
              <span className="row" style={{ gap: 6 }}>
                <ScenarioName r={r} />
                <CausalBadge c={r.causal} />
              </span>
            ) : (
              <ScenarioName r={r} />
            ),
        },
        { key: "delta", label: "Citywide change", unit: u, align: "right", value: (r) => r.delta.estimate, render: (r) => likelyText(r.delta, u) },
        { key: "conf", label: "Confidence", value: (r) => confidenceWords(r.delta) },
        ...(rows.some((r) => r.verdict?.verdict)
          ? [{ key: "verdict", label: "Verdict", value: (r: ScenarioRow) => r.verdict?.label ?? "—", render: (r: ScenarioRow) => <VerdictPill verdict={r.verdict} /> }]
          : []),
        {
          key: "extrap",
          label: term("extrapolated").label,
          align: "right",
          tip: <InfoTip term="extrapolated" />,
          value: (r) => r.frac_extrapolated,
          render: (r) => beyondText(r.frac_extrapolated),
        },
        { key: "folds", label: "Can be compared", tip: <InfoTip term="paired_se" />, value: (r) => (r.has_folds ? "yes" : "no") },
        {
          key: "realized",
          label: term("realised").label,
          value: (r) =>
            Object.entries(r.realized)
              .map(([k, v]) => realisedWords(k, v, leverUnits))
              .join("; "),
          render: (r) =>
            Object.entries(r.realized)
              .map(([k, v]) => realisedWords(k, v, leverUnits))
              .join(" · ") || "—",
        },
        {
          key: "actions",
          label: "Actions",
          sortable: false,
          value: () => null,
          render: (r) =>
            actions ? (
              actions(r)
            ) : (
              <span className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
                <Link to={`/r/${encodeURIComponent(rid)}/lab/library?configured=${encodeURIComponent(r.slug)}`} className="btn small">
                  Open in Lab
                </Link>
                <CloneButton rid={rid} slug={r.slug} />
              </span>
            ),
        },
      ]}
      rows={rows}
    />
  );
}

/** The run tab's ladders: each lever at increasing doses (Lab Library links to What cools instead). */
function Ladders({ ladders: data }: { ladders: Sections<ScenariosSections>["ladders"] }) {
  const u = unitLabel(useUnits().target);
  return (
    <Section title={`${term("ladder").label}s`} data={data}>
      {(ladders) =>
        ladders.length ? (
          <SmallMultiples
            title={`${term("ladder").label}s`}
            units={`${u} of cooling, citywide (higher = cooler)`}
            items={ladders}
            panelTitle={(l) => l.label}
            renderPanel={(l) => (
              <LineBand
                bare
                title={l.label}
                series={[
                  {
                    id: l.lever,
                    label: l.label,
                    x: l.points.map((p) => p.dose),
                    y: l.points.map((p) => cooling(p.estimate)),
                    lo: l.points.map((p) => cooling(p.hi)),
                    hi: l.points.map((p) => cooling(p.lo)),
                    hollow: l.points.map((p) => p.hollow),
                  },
                ]}
                xLabel={term("dose").label}
                xUnit={unitLabel(l.unit)}
                yLabel="Citywide cooling"
                yUnit={u}
                yInclude={[0]}
                refLines={[{ axis: "y", value: 0 }]}
                height={190}
              />
            )}
            table={{
              columns: [
                { key: "l", label: "What changes" },
                { key: "d", label: term("dose").label },
                { key: "e", label: "Citywide cooling", unit: u },
                { key: "lo", label: "Low", unit: u },
                { key: "hi", label: "High", unit: u },
              ],
              rows: ladders.flatMap((l) => l.points.map((p) => [l.label, p.dose, cooling(p.estimate), cooling(p.hi), cooling(p.lo)])),
            }}
            caption={
              <>
                Each ladder runs one lever at growing amounts; hollow points lie mostly {term("extrapolated").label.toLowerCase()}.
                <InfoTip term="ladder" />
              </>
            }
          />
        ) : (
          <p className="cap">No amount ladders are set up for this run.</p>
        )
      }
    </Section>
  );
}

/**
 * The sections of a scenarios ViewModel (inside its ViewPage, for the units): the city-mean
 * change per scenario, the fold-detail note, then the table. `full` (the former run tab) adds the
 * causal band to the chart and the ladders; without it the ladders are a link to What cools and
 * the causal check a badge per row.
 */
export function ScenarioSections({
  s,
  rid,
  focus,
  actions,
  full,
}: {
  s: Sections<ScenariosSections>;
  rid: string;
  focus?: string | null;
  actions?: ScenarioActions;
  full?: boolean;
}) {
  const u = unitLabel(useUnits().target);
  return (
    <>
      <Section title="Scenario effects" data={s.rows}>
        {(rows) => {
          const flagged = full ? rows.filter((r) => r.causal && r.causal.model_within === false) : [];
          const best = rows.reduce<ScenarioRow | null>((m, r) => (!m || r.delta.estimate < m.delta.estimate ? r : m), null);
          return (
            // drawn as cooling (higher = cooler), the same way round as What cools and the Summary
            <DotRange
              title="Citywide cooling per scenario"
              units={`${u} of cooling (higher = cooler)`}
              rows={rows.map((r) => ({
                id: r.slug,
                label: scenarioLabel(r),
                est: cooling(r.delta.estimate),
                lo: cooling(r.delta.hi),
                hi: cooling(r.delta.lo),
                lo2: cooling(r.p90),
                hi2: cooling(r.p10),
                hollow: (r.frac_extrapolated ?? 0) > 0.2,
                check:
                  full && r.causal
                    ? {
                        est: -r.causal.delta,
                        lo: cooling(r.causal.hi),
                        hi: cooling(r.causal.lo),
                        label: r.causal.model_within === false ? `${CAUSAL} (model outside)` : CAUSAL,
                      }
                    : null,
              }))}
              valueLabel="Citywide cooling"
              unit={u}
              signed
              rangeLabel="Likely range"
              outerLabel="Middle 80% of places"
              onRowClick={(slug) => navigate(`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(`sc:${slug}`)}`)}
              // In the Lab the table below lists each scenario's numbers with its actions: no second listing here.
              tableView={full ? undefined : false}
              caption={`${best ? `Strongest: ${scenarioLabel(best)}, ${likelyText(best.delta, u)}. ` : ""}${flagged.length ? `The model's change is outside the ${CAUSAL}'s range for ${flagged.map((r) => scenarioLabel(r)).join(", ")}. ` : ""}Select a dot to map that scenario's change.`}
            />
          );
        }}
      </Section>
      {s.has_detail === false ? (
        <p className="cap">
          <Badge>no result by test area</Badge> This run kept no result for each test area, so comparing these scenarios with the {term("paired_se").label.toLowerCase()} needs a{" "}
          {term("exact_engine").label.toLowerCase()} run again in the Lab.
          <InfoTip term="paired_se" />
        </p>
      ) : null}
      {full ? (
        <Ladders ladders={s.ladders} />
      ) : s.ladders?.length ? (
        <p className="cap" data-ladders-link="true" style={{ margin: 0 }}>
          How each lever&apos;s cooling grows with the amount: <Link to={runHref(rid, "effects")}>What cools →</Link>
        </p>
      ) : null}
      <Section title="Scenario table" data={s.rows}>
        {(rows) => (
          <Block title={`All ${SCENARIOS_CONFIGURED.toLowerCase()}`}>
            <ScenarioTable rows={rows} rid={rid} focus={focus} actions={actions} causalBadges={!full} />
          </Block>
        )}
      </Section>
    </>
  );
}

export default function Scenarios() {
  const rid = useRid();
  return (
    <ViewPage view="scenarios" title={SCENARIOS_CONFIGURED} intro={`${term("configured_scenario").tip} The charts show cooling: higher is cooler.`}>
      {(s) => <ScenarioSections s={s} rid={rid} full />}
    </ViewPage>
  );
}

/**
 * Scenario Lab › Library's first group (phase 2, the canonical home of the run's configured
 * scenarios): the run tab's sections under the group's title (a kicker under the Library's own
 * title) and intro, with the Lab's row actions, and a note when `focus` names no configured
 * scenario of this run. The chart has no table view: the table lists the same numbers once.
 */
export function ConfiguredScenarios({ title, intro, focus = null, actions }: { title: string; intro?: ReactNode; focus?: string | null; actions?: ScenarioActions }) {
  const rid = useRid();
  return (
    <ViewPage view="scenarios" title={title} intro={intro} level={3} kicker>
      {(s) => (
        <>
          {focus && s.rows && !s.rows.some((r) => r.slug === focus) ? <p className="callout">This run has no configured scenario “{focus}”.</p> : null}
          <ScenarioSections s={s} rid={rid} focus={focus} actions={actions} />
        </>
      )}
    </ViewPage>
  );
}
