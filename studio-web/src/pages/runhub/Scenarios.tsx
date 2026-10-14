// Configured scenarios (SPEC §6.4): dot-and-whisker of each scenario's city-mean ΔT with its
// likely range and p10–p90 whisker (hollow when > 20% extrapolated, orange diamond for the
// causal band, flagged when the model falls outside it), ladders per lever, and the table
// with tier, folds and realised doses plus "Open in Lab" and "Clone to edit".
// Phase 2: their home is Scenario Lab › Library, whose first group "Set up with this run" is
// ConfiguredScenarios (the same sections, with the Lab's row actions and the
// `?configured=<slug>` focus), without what other tabs own: the ladders repeat What cools'
// dose-response curves (a one-line link instead), and the causal band is How much to trust
// this's (a small badge per table row instead of the chart's diamonds). The default export (the
// former run tab) renders them in full (`full`). The Verdict column shows only when a row has one.
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ScenarioRow, ScenariosSections, Sections } from "../../api/runs";
import { cloneConfiguredScenario } from "../../api/runs";
import { errorMessage } from "../../api/client";
import { DotRange, LineBand, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Table } from "../../components/ui/Table";
import { Link, navigate, runHref } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { fmtPct, fmtValue, unitLabel } from "../../theme/format";
import { Block, Section, VerdictPill, ViewPage, useRid, useUnits } from "./common";
import { confidenceWords, likelyText } from "./format";

/** Clone a configured scenario into the project's Scenario Lab and open the copy in Design. */
export function CloneButton({ rid, slug, pid: pidProp }: { rid: string; slug: string; pid?: string | null }) {
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = pidProp ?? ctxPid;
  const [busy, setBusy] = useState(false);
  return (
    <Button
      size="small"
      busy={busy}
      disabled={!pid}
      title={pid ? "Save an editable copy in the Scenario Lab" : "Open the run from its project to clone"}
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
    <Badge title="The model's change is inside the band of the independent causal check">causal agrees</Badge>
  ) : (
    <Badge tone="warn" title="The model's change is outside the band of the independent causal check">
      causal differs
    </Badge>
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
  const u = unitLabel(useUnits().target);
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
      caption="Configured scenarios"
      csvName="configured-scenarios"
      rowKey={(r) => r.slug}
      highlight={focus ? (r) => r.slug === focus : undefined}
      columns={[
        {
          key: "name",
          label: "Scenario",
          value: (r) => r.name,
          render: (r) =>
            causalBadges && r.causal ? (
              <span className="row" style={{ gap: 6 }}>
                <span data-configured={r.slug}>{r.name}</span>
                <CausalBadge c={r.causal} />
              </span>
            ) : (
              <span data-configured={r.slug}>{r.name}</span>
            ),
        },
        { key: "delta", label: "City-mean change", unit: u, align: "right", value: (r) => r.delta.estimate, render: (r) => likelyText(r.delta, u) },
        { key: "conf", label: "Confidence", value: (r) => confidenceWords(r.delta) },
        ...(rows.some((r) => r.verdict?.verdict)
          ? [{ key: "verdict", label: "Verdict", value: (r: ScenarioRow) => r.verdict?.label ?? "—", render: (r: ScenarioRow) => <VerdictPill verdict={r.verdict} /> }]
          : []),
        { key: "extrap", label: "Extrapolated", align: "right", value: (r) => r.frac_extrapolated, render: (r) => fmtPct(r.frac_extrapolated, 0) },
        { key: "tier", label: "Tier", value: (r) => r.tier ?? "—" },
        { key: "folds", label: "Has folds", value: (r) => (r.has_folds ? "yes" : "no") },
        {
          key: "realized",
          label: "Realised doses",
          value: (r) =>
            Object.entries(r.realized)
              .map(([k, v]) => `${k} ${v ?? "—"}`)
              .join("; "),
          render: (r) =>
            Object.entries(r.realized)
              .map(([k, v]) => `${k} ${fmtValue(v, "", 2)}`)
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
    <Section title="Ladders" data={data}>
      {(ladders) =>
        ladders.length ? (
          <SmallMultiples
            title="Scenario ladders"
            units={`${u} (negative = cooler)`}
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
                    y: l.points.map((p) => p.estimate),
                    lo: l.points.map((p) => p.lo),
                    hi: l.points.map((p) => p.hi),
                    hollow: l.points.map((p) => p.hollow),
                  },
                ]}
                xLabel="Dose"
                xUnit={unitLabel(l.unit)}
                yLabel="City-mean ΔT"
                yUnit={u}
                yInclude={[0]}
                refLines={[{ axis: "y", value: 0 }]}
                height={190}
              />
            )}
            table={{
              columns: [
                { key: "l", label: "Lever" },
                { key: "d", label: "Dose" },
                { key: "e", label: "City-mean ΔT", unit: u },
                { key: "lo", label: "Low", unit: u },
                { key: "hi", label: "High", unit: u },
              ],
              rows: ladders.flatMap((l) => l.points.map((p) => [l.label, p.dose, p.estimate, p.lo, p.hi])),
            }}
            caption="Each ladder repeats one lever at increasing doses; hollow points are mostly extrapolated."
          />
        ) : (
          <p className="cap">No single-lever ladders are configured.</p>
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
            <DotRange
              title="City-mean change per scenario"
              units={`${u} (negative = cooler)`}
              rows={rows.map((r) => ({
                id: r.slug,
                label: r.name,
                est: r.delta.estimate,
                lo: r.delta.lo,
                hi: r.delta.hi,
                lo2: r.p10,
                hi2: r.p90,
                hollow: (r.frac_extrapolated ?? 0) > 0.2,
                check:
                  full && r.causal
                    ? { est: r.causal.delta, lo: r.causal.lo, hi: r.causal.hi, label: r.causal.model_within === false ? "causal band (model outside)" : "causal band" }
                    : null,
              }))}
              valueLabel="City-mean ΔT"
              unit={u}
              signed
              rangeLabel="Likely range"
              outerLabel="10th–90th percentile of cells"
              onRowClick={(slug) => navigate(`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(`sc:${slug}`)}`)}
              caption={`${best ? `Strongest: ${best.name}, ${likelyText(best.delta, u)}. ` : ""}${flagged.length ? `The model is outside the causal band for ${flagged.map((r) => r.name).join(", ")}. ` : ""}Select a dot to map that scenario's change.`}
            />
          );
        }}
      </Section>
      {s.has_detail === false ? (
        <p className="cap">
          <Badge>no fold detail</Badge> This run has no scenario_detail.npz, so paired comparisons need an exact re-run in the Lab.
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
          <Block title="All configured scenarios">
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
    <ViewPage view="scenarios" title="Configured scenarios" intro="The scenarios written in the project config, evaluated at the end of the run (S5). Negative changes are cooler.">
      {(s) => <ScenarioSections s={s} rid={rid} full />}
    </ViewPage>
  );
}

/**
 * Scenario Lab › Library's first group (phase 2, the canonical home of the run's configured
 * scenarios): the run tab's sections under the Library's own title and intro, with the Lab's row
 * actions, and a note when `focus` names no configured scenario of this run.
 */
export function ConfiguredScenarios({ title, intro, focus = null, actions }: { title: string; intro?: ReactNode; focus?: string | null; actions?: ScenarioActions }) {
  const rid = useRid();
  return (
    <ViewPage view="scenarios" title={title} intro={intro}>
      {(s) => (
        <>
          {focus && s.rows && !s.rows.some((r) => r.slug === focus) ? <p className="callout">This run has no configured scenario “{focus}”.</p> : null}
          <ScenarioSections s={s} rid={rid} focus={focus} actions={actions} />
        </>
      )}
    </ViewPage>
  );
}
