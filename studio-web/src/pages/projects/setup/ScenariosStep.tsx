// Setup step "scenarios" (SPEC §9.3): configured ladders (one scenario per increment) and
// joint packages, each showing the exact names core will generate ("Impervious Decrease −10"
// with U+2212). Richer scenarios (selections, brushes, budgets) live in the Scenario Lab.
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { CfgNumber, CfgNumberList, CfgSeg, CfgSelect, CfgText, IssueLines, useCfg, type Opt } from "../components/cfg";
import { StepIssues } from "../components/StepIssues";
import { getList, getRecord } from "../model/dotted";
import { isDecrease, ladderNames, scenarioSlugs, configuredScenarioNames } from "../model/scenarioNames";
import { stepIssues } from "../model/steps";

const DIRECTIONS = [
  { value: "increase", label: "increase" },
  { value: "decrease", label: "decrease" },
] as const;

const dir = (v: unknown) => (v === "decrease" ? "decrease" : "increase");

function Names({ names }: { names: string[] }) {
  if (!names.length) return <p className="cap">No scenarios yet: add increments.</p>;
  return (
    <ul className="scenario-names" aria-label="Generated scenario names">
      {names.map((n, i) => (
        <li key={`${i}:${n}`} className="mono">
          {n}
        </li>
      ))}
    </ul>
  );
}

export function ScenariosStep() {
  const c = useCfg();
  const raw = c.draft.raw;
  const levers = Object.keys(getRecord(raw, "actionable"));
  const leverOpts: Opt[] = levers.map((l) => ({ value: l, label: l }));
  const ladders = getList<Record<string, unknown>>(raw, "scenarios");
  const packages = getList<Record<string, unknown>>(raw, "joint_scenarios");
  const all = scenarioSlugs(configuredScenarioNames(raw));
  const unit = (v: unknown) => {
    const u = getRecord<unknown>(raw, ["actionable", String(v ?? "")]).unit;
    return typeof u === "string" ? u : null;
  };
  return (
    <div className="stack setup-step" data-step="scenarios">
      <StepIssues issues={stepIssues("scenarios", c.draft.issues)} />
      <div className="callout" data-tone="info">
        These are the configured scenarios S5 runs on every launch. Richer scenarios (selections, brushed edits, budgets) live in the Scenario Lab once a run has a checkpoint.
      </div>
      <Card title="Ladders" eyebrow="One scenario per increment">
        {levers.length === 0 ? <p className="cap">Mark some predictors as levers first (Levers step).</p> : null}
        {ladders.map((l, i) => (
          <div key={i} className="ladder" data-cfg-path={`scenarios.${i}`}>
            <div className="grid3">
              <CfgText path={`scenarios.${i}.name`} label={`Ladder ${i + 1} name`} placeholder="Canopy Increase" />
              <CfgSelect path={`scenarios.${i}.variable`} label="Lever" options={leverOpts} />
              <CfgSeg path={`scenarios.${i}.direction`} label="Direction" options={DIRECTIONS} fromValue={dir} />
            </div>
            <CfgNumberList path={`scenarios.${i}.increments`} label="Increments" unit={unit(l.variable)} signed={isDecrease(l.direction) ? "−" : "+"} />
            <Names names={ladderNames(l)} />
            <div className="row">
              <Button size="small" variant="ghost" icon="x" onClick={() => c.set("scenarios", ladders.filter((_, k) => k !== i))}>
                Remove ladder
              </Button>
            </div>
          </div>
        ))}
        <div className="row">
          <Button
            size="small"
            icon="plus"
            disabled={!levers.length}
            onClick={() => {
              const v = levers[0];
              const down = getRecord<unknown>(raw, ["actionable", v]).direction === "decrease";
              c.set(["scenarios", ladders.length], { name: `${v} ${down ? "Decrease" : "Increase"}`, variable: v, direction: down ? "decrease" : "increase", increments: [5, 10, 20] });
            }}
          >
            Add a ladder
          </Button>
        </div>
        <IssueLines issues={c.issues("scenarios")} />
      </Card>
      <Card title="Packages" eyebrow="Several edits applied together">
        {packages.map((p, i) => {
          const ivs = getList<Record<string, unknown>>(p, "interventions");
          return (
            <div key={i} className="package" data-cfg-path={`joint_scenarios.${i}`}>
              <CfgText path={`joint_scenarios.${i}.name`} label={`Package ${i + 1} name`} placeholder="Green Infrastructure Package" />
              {ivs.map((iv, k) => (
                <div key={k} className="grid3 intervention">
                  <CfgSelect path={`joint_scenarios.${i}.interventions.${k}.variable`} label={`Edit ${k + 1}: lever`} options={leverOpts} />
                  <CfgSeg path={`joint_scenarios.${i}.interventions.${k}.direction`} label="Direction" options={DIRECTIONS} fromValue={dir} />
                  <div className="row" style={{ alignItems: "end" }}>
                    <CfgNumber path={`joint_scenarios.${i}.interventions.${k}.increment`} label="Increment" unit={unit(iv.variable)} />
                    <Button size="small" variant="ghost" icon="x" onClick={() => c.set(["joint_scenarios", i, "interventions"], ivs.filter((_, j) => j !== k))}>
                      Remove
                    </Button>
                  </div>
                </div>
              ))}
              <div className="row">
                <Button size="small" icon="plus" disabled={!levers.length} onClick={() => c.set(["joint_scenarios", i, "interventions", ivs.length], { variable: levers[0], direction: "increase", increment: 10 })}>
                  Add an edit
                </Button>
                <Button size="small" variant="ghost" icon="x" onClick={() => c.set("joint_scenarios", packages.filter((_, j) => j !== i))}>
                  Remove package
                </Button>
              </div>
            </div>
          );
        })}
        <div className="row">
          <Button size="small" icon="plus" disabled={!levers.length} onClick={() => c.set(["joint_scenarios", packages.length], { name: "Cooling package", interventions: [] })}>
            Add a package
          </Button>
        </div>
        <IssueLines issues={c.issues("joint_scenarios")} />
      </Card>
      <Card title="Every configured scenario" eyebrow={`${all.length} scenarios`}>
        <table className="tbl" aria-label="Configured scenarios">
          <thead>
            <tr>
              <th scope="col">Name (as generated)</th>
              <th scope="col">Id (slug, used by layers and the headline scenario)</th>
            </tr>
          </thead>
          <tbody>
            {all.map((s, i) => (
              <tr key={`${i}:${s.slug}`}>
                <td>{s.name}</td>
                <td className="mono">{s.slug}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
