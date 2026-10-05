// Configured scenarios (SPEC §6.4): dot-and-whisker of each scenario's city-mean ΔT with its
// likely range and p10–p90 whisker (hollow when > 20% extrapolated, orange diamond for the
// causal band, flagged when the model falls outside it), ladders per lever, and the table
// with tier, folds and realised doses plus "Open in Lab" and "Clone to edit".
import { useState } from "react";
import type { ScenarioRow } from "../../api/runs";
import { cloneConfiguredScenario } from "../../api/runs";
import { errorMessage } from "../../api/client";
import { DotRange, LineBand, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Table } from "../../components/ui/Table";
import { Link, navigate } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { fmtPct, fmtValue, unitLabel } from "../../theme/format";
import { Block, Section, VerdictPill, ViewPage, useRid, useUnits } from "./common";
import { confidenceWords, likelyText } from "./format";

function CloneButton({ rid, slug }: { rid: string; slug: string }) {
  const pid = useUi((s) => s.context.projectId);
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

function ScenarioTable({ rows, rid }: { rows: ScenarioRow[]; rid: string }) {
  const u = unitLabel(useUnits().target);
  return (
    <Table<ScenarioRow>
      caption="Configured scenarios"
      csvName="configured-scenarios"
      rowKey={(r) => r.slug}
      columns={[
        { key: "name", label: "Scenario", value: (r) => r.name },
        { key: "delta", label: "City-mean change", unit: u, align: "right", value: (r) => r.delta.estimate, render: (r) => likelyText(r.delta, u) },
        { key: "conf", label: "Confidence", value: (r) => confidenceWords(r.delta) },
        { key: "verdict", label: "Verdict", value: (r) => r.verdict?.label ?? "—", render: (r) => <VerdictPill verdict={r.verdict} /> },
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
          render: (r) => (
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

export default function Scenarios() {
  const rid = useRid();
  return (
    <ViewPage view="scenarios" title="Configured scenarios" intro="The scenarios written in the project config, evaluated at the end of the run (S5). Negative changes are cooler.">
      {(s, vm) => {
        const u = unitLabel(vm.units.target);
        return (
          <>
            <Section title="Scenario effects" data={s.rows}>
              {(rows) => {
                const flagged = rows.filter((r) => r.causal && r.causal.model_within === false);
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
                      check: r.causal
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
            <Section title="Ladders" data={s.ladders}>
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
            <Section title="Scenario table" data={s.rows}>
              {(rows) => (
                <Block title="All configured scenarios">
                  <ScenarioTable rows={rows} rid={rid} />
                </Block>
              )}
            </Section>
          </>
        );
      }}
    </ViewPage>
  );
}
