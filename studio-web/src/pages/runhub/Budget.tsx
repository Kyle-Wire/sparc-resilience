// Budget (S7) (SPEC §6.4): KPIs (cells treated, mean dose, planned vs realised labelled
// spillover non-additivity, cost, Gini, constraint, objective), the open-loop Pareto chart
// with segments and the server-computed caption, dose and closed-loop maps, the top-cells
// table and "Re-plan in Lab". `status: "no positive-benefit segments"` is an explained
// empty state, not a missing output.
// Phase 2: its home is Scenario Lab › Plans, whose first entry "Plan from this run" (read-only)
// is RunBudgetPlan. The default export (the former run tab) renders the same sections.
import type { ReactNode } from "react";
import type { BudgetSections, Sections } from "../../api/runs";
import { Pareto } from "../../charts";
import { Table } from "../../components/ui/Table";
import { Link } from "../../router";
import { fmtInt, fmtNum, fmtValue, unitLabel } from "../../theme/format";
import { fmtLonLat } from "../../map/grid";
import { Block, KpiTiles, Section, ViewPage, useRid, useUnits } from "./common";
import { RunLayerMap } from "./RunLayerMap";

type TopCell = NonNullable<BudgetSections["top_cells"]>[number];

/**
 * The sections of a budget ViewModel (inside its ViewPage, for the units): the explained "no
 * allocation" state (with `replan`, its way out), or the key numbers, the Pareto curve, the dose
 * and closed-loop maps and the top cells.
 */
export function BudgetPlanSections({ s, rid, replan }: { s: Sections<BudgetSections>; rid: string; replan?: ReactNode }) {
  const u = unitLabel(useUnits().target);
  if (s.status && s.status !== "ok") {
    return (
      <div className="empty" role="status" data-status={s.status}>
        <div className="empty-title">No allocation: {s.status}</div>
        <div className="empty-body">
          No cell has a positive cooling footprint for this lever, so the optimiser had nothing to buy. Try another lever or objective in the Lab.
        </div>
        {replan}
      </div>
    );
  }
  return (
    <>
      <Section title="Budget key numbers" data={s.kpis}>
        {(k) => <KpiTiles kpis={k} label="Budget" />}
      </Section>
      <Section title="Pareto" data={s.pareto}>
        {(p) => {
          const sel = p.budget !== null ? p.points.findIndex((q) => q.budget === p.budget) : -1;
          return (
            <Pareto
              title="Cooling bought per budget (open-loop)"
              units={unitLabel(p.unit)}
              points={p.points.map((q) => ({ x: q.budget, y: q.total_benefit, label: q.n_segments !== null ? `${fmtInt(q.n_segments)} segments` : undefined }))}
              realised={p.realised?.map((q) => ({ x: q.budget, y: q.total_benefit }))}
              selected={sel >= 0 ? sel : null}
              xLabel="Budget"
              xUnit={unitLabel(p.budget_unit)}
              yLabel="Total cooling"
              yUnit={unitLabel(p.unit)}
              curveLabel="planned (open-loop)"
              realisedLabel="realised (closed-loop, spillover non-additivity)"
              caption={<>{s.caption ?? ""} Each point is a budget; "segments" are the dose steps bought, which can share a cell. The ring marks this run's budget.</>}
            />
          );
        }}
      </Section>
      <Block title="Dose and closed-loop maps">
        <RunLayerMap rid={rid} keys={["alloc_dose", "alloc_delta"]} title="Budget allocation" />
      </Block>
      <Section title="Top cells" data={s.top_cells}>
        {(cells) => (
          <Block title="Top cells">
            <Table<TopCell>
              caption="Top cells by cooling"
              csvName="budget-top-cells"
              rowKey={(r) => r.row}
              columns={[
                { key: "rank", label: "Rank", align: "right", value: (r) => r.rank },
                { key: "id", label: "Cell id", value: (r) => String(r.id) },
                {
                  key: "loc",
                  label: "Location",
                  value: (r) => (r.lon !== null && r.lat !== null ? `${r.lat},${r.lon}` : null),
                  render: (r) => (r.lon !== null && r.lat !== null ? fmtLonLat([r.lon, r.lat]) : "—"),
                },
                { key: "zone", label: "Zone", value: (r) => (r.zone === null ? null : String(r.zone)) },
                { key: "dose", label: "Dose", align: "right", value: (r) => r.dose, render: (r) => fmtNum(r.dose, 2) },
                { key: "benefit", label: "Cooling", unit: `${u}·cells`, align: "right", value: (r) => r.benefit, render: (r) => fmtValue(r.benefit, "", 2) },
              ]}
              rows={cells}
            />
          </Block>
        )}
      </Section>
    </>
  );
}

export default function Budget() {
  const rid = useRid();
  const lab = `/r/${encodeURIComponent(rid)}/lab/plans`;
  return (
    <ViewPage
      view="budget"
      title="Budget"
      intro="The run's own budget allocation (S7): where a fixed amount of the lever buys the most cooling."
      actions={
        <Link to={lab} className="btn small">
          Re-plan in Lab
        </Link>
      }
    >
      {(s) => (
        <BudgetPlanSections
          s={s}
          rid={rid}
          replan={
            <Link to={lab} className="btn">
              Re-plan in Lab
            </Link>
          }
        />
      )}
    </ViewPage>
  );
}

/**
 * Scenario Lab › Plans' first entry (phase 2, the canonical home of the run's budget plan): the
 * run tab's sections under the Lab's own title, intro and header actions.
 */
export function RunBudgetPlan({ title, intro, actions, replan }: { title: string; intro?: ReactNode; actions?: ReactNode; replan?: ReactNode }) {
  const rid = useRid();
  return (
    <ViewPage view="budget" title={title} intro={intro} actions={actions}>
      {(s) => <BudgetPlanSections s={s} rid={rid} replan={replan} />}
    </ViewPage>
  );
}
