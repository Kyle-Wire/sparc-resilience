// Setup step "levers" (SPEC §9.3): predictors (numeric columns), encodings (categorical /
// circular), qa.clip, the actionable editor (toggle, min/max, unit, dose chips, direction,
// cost) with each lever's dose-scale table from the data check (> 1 sd amber), coupling sums
// and the mediators mini-editor (parents, context, monotone signs).
import { useFileInspect } from "../../../api/projects";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg } from "../../../components/ui/Seg";
import { fmtSig } from "../../../theme/format";
import { CfgChips, CfgNumber, CfgNumberList, CfgSeg, CfgText, IssueLines, useCfg, type Opt } from "../components/cfg";
import { DoseScaleTable } from "../components/DoseScaleTable";
import { StepIssues } from "../components/StepIssues";
import { predictorCandidates } from "../model/columns";
import { getList, getPath, getRecord, getString } from "../model/dotted";
import { stepIssues } from "../model/steps";
import { runDataCheck } from "./actions";
import { rawKey } from "./store";

type Lever = { min?: number; max?: number; unit?: string; doses?: number[]; direction?: string; cost_per_unit?: number };

const DIRECTIONS = [
  { value: "increase", label: "increase" },
  { value: "decrease", label: "decrease" },
] as const;

function ClipRow({ name, range }: { name: string; range: { min: number | null; max: number | null } }) {
  const c = useCfg();
  const path = ["qa", "clip", name];
  const v = getPath(c.draft.raw, path);
  const pair = Array.isArray(v) ? (v as unknown[]) : [];
  const lo = typeof pair[0] === "number" ? (pair[0] as number) : null;
  const hi = typeof pair[1] === "number" ? (pair[1] as number) : null;
  // qa.clip takes both bounds: a missing one falls back to the data's own min or max.
  const put = (a: number | null, b: number | null) =>
    c.set(path, a === null && b === null ? undefined : [a ?? range.min ?? b ?? 0, b ?? range.max ?? a ?? 0]);
  return (
    <tr data-cfg-path={`qa.clip.${name}`}>
      <th scope="row" className="mono">
        {name}
      </th>
      <td>
        <NumberField label={`${name} lower clip`} value={lo} onChange={(n) => put(n, hi)} nullable />
      </td>
      <td>
        <NumberField label={`${name} upper clip`} value={hi} onChange={(n) => put(lo, n)} nullable />
      </td>
      <td>
        <IssueLines issues={c.issues(`qa.clip.${name}`, true)} />
      </td>
    </tr>
  );
}

function LeverCard({ name, range }: { name: string; range: { min: number | null; max: number | null } }) {
  const c = useCfg();
  const base = `actionable.${name}`;
  const lever = getRecord<unknown>(c.draft.raw, ["actionable", name]) as Lever;
  const on = getPath(c.draft.raw, ["actionable", name]) !== undefined;
  const check = c.draft.check;
  const stale = !!check && c.draft.checkKey !== rawKey(c.draft.raw);
  const sign = lever.direction === "decrease" ? "−" : "+";
  return (
    <div className="lever-card" data-lever={name} data-cfg-path={base}>
      <label className="row lever-toggle">
        <input
          type="checkbox"
          checked={on}
          onChange={(e) =>
            c.set(
              ["actionable", name],
              e.target.checked
                ? { min: range.min ?? 0, max: range.max ?? 100, unit: "", doses: [0, 5, 10, 20, 30], direction: "increase", cost_per_unit: 1 }
                : undefined,
            )
          }
        />
        <span className="mono">{name}</span>
        <span className="cap">
          {on ? "actionable lever" : "not a lever"}
          {range.min !== null && range.max !== null ? ` · data ${fmtSig(range.min, 4)} to ${fmtSig(range.max, 4)}` : ""}
        </span>
      </label>
      {on ? (
        <div className="stack" style={{ gap: 8 }}>
          <div className="grid3">
            <CfgNumber path={`${base}.min`} label="Lowest value" hint="An edit never goes below this." />
            <CfgNumber path={`${base}.max`} label="Highest value" hint="An edit never goes above this." />
            <CfgText path={`${base}.unit`} label="Unit" placeholder="pp, reflectance…" />
            <CfgSeg path={`${base}.direction`} label="Direction" options={DIRECTIONS} fromValue={(v) => (v === "decrease" ? "decrease" : "increase")} />
            <CfgNumber path={`${base}.cost_per_unit`} label="Cost per unit" hint="Cost of one unit of dose on one cell (default 1)." min={0} />
          </div>
          <CfgNumberList path={`${base}.doses`} label="Dose ladder (S4 sweep)" unit={lever.unit ?? null} signed={sign} hint="Magnitudes, including 0; the direction sets the sign." />
          <DoseScaleTable lever={name} entry={check?.dose_scale?.[name]} unit={lever.unit ?? null} stale={stale} />
          <IssueLines issues={c.issues(base)} />
        </div>
      ) : null}
    </div>
  );
}

function Coupling({ predictors }: { predictors: Opt[] }) {
  const c = useCfg();
  const list = getList<Record<string, unknown>>(c.draft.raw, "coupling");
  return (
    <div className="stack" style={{ gap: 8 }}>
      {list.length === 0 ? <p className="cap">No coupling. Add one when edits of several columns must not push their sum past a cap (e.g. canopy + impervious ≤ 100%).</p> : null}
      {list.map((_, i) => (
        <div key={i} className="coupling-row">
          <CfgChips path={`coupling.${i}.sum`} label={`Coupling ${i + 1}: columns summed`} options={predictors} />
          <CfgNumber path={`coupling.${i}.max`} label="Cap on the sum" />
          <Button size="small" variant="ghost" icon="x" onClick={() => c.set("coupling", list.filter((__, k) => k !== i))}>
            Remove
          </Button>
        </div>
      ))}
      <div className="row">
        <Button size="small" icon="plus" onClick={() => c.set(["coupling", list.length], { sum: [], max: 100 })}>
          Add a coupling sum
        </Button>
      </div>
      <IssueLines issues={c.issues("coupling")} />
    </div>
  );
}

const MONO = [
  { value: "1", label: "+1" },
  { value: "0", label: "free" },
  { value: "-1", label: "−1" },
] as const;

function Mediators({ predictors, levers }: { predictors: string[]; levers: string[] }) {
  const c = useCfg();
  const meds = getRecord<Record<string, unknown>>(c.draft.raw, "mediators");
  const names = Object.keys(meds);
  const candidates = predictors.filter((p) => !levers.includes(p) && !names.includes(p));
  return (
    <div className="stack" style={{ gap: 10 }}>
      {names.length === 0 ? <p className="cap">No mediators. A mediator (e.g. NDVI) is a predictor the levers move; scenarios then update it through a small fitted model.</p> : null}
      {names.map((m) => {
        const parents = getList<string>(meds[m], "parents").map(String);
        const mono = getRecord<number>(meds[m], "monotone");
        return (
          <div key={m} className="mediator" data-cfg-path={`mediators.${m}`}>
            <div className="row">
              <strong className="mono">{m}</strong>
              <span className="spacer" />
              <Button size="small" variant="ghost" icon="x" onClick={() => c.set(["mediators", m], undefined)}>
                Remove
              </Button>
            </div>
            <CfgChips path={["mediators", m, "parents"]} label="Parents (levers that move it)" options={levers.map((l) => ({ value: l, label: l }))} />
            <CfgChips path={["mediators", m, "context"]} label="Context (other inputs of its model)" options={predictors.filter((p) => p !== m && !parents.includes(p)).map((p) => ({ value: p, label: p }))} />
            {parents.length ? (
              <div className="field">
                <span className="field-label">Monotone signs</span>
                <div className="row">
                  {parents.map((p) => (
                    <span key={p} className="row" style={{ gap: 4 }}>
                      <span className="mono cap">{p}</span>
                      <Seg
                        size="small"
                        label={`Sign of ${m} in ${p}`}
                        options={MONO}
                        value={mono[p] === 1 ? "1" : mono[p] === -1 ? "-1" : "0"}
                        onChange={(v) => c.set(["mediators", m, "monotone", p], v === "0" ? undefined : Number(v))}
                      />
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            <IssueLines issues={c.issues(`mediators.${m}`, true)} />
          </div>
        );
      })}
      {candidates.length ? (
        <label className="row">
          <span className="cap">Add a mediator</span>
          <select aria-label="Add a mediator" value="" onChange={(e) => e.target.value && c.set(["mediators", e.target.value], { parents: [], context: [], monotone: {} })}>
            <option value="">Choose a predictor…</option>
            {candidates.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
      ) : null}
    </div>
  );
}

export function LeversStep() {
  const c = useCfg();
  const d = c.draft;
  const dataPath = getString(d.raw, "data.path") ?? getString(d.effective, "data.path");
  const inspect = useFileInspect(c.pid, dataPath);
  const candidates = predictorCandidates(inspect.data, d.raw, d.effective);
  const pv = c.value("predictors");
  const predictors = Array.isArray(pv) ? pv.map(String) : [];
  const predOpts: Opt[] = predictors.map((p) => ({ value: p, label: p }));
  const levers = Object.keys(getRecord(d.raw, "actionable"));
  const range = (name: string) => {
    const col = inspect.data?.columns.find((x) => x.name === name);
    return { min: col?.min ?? null, max: col?.max ?? null };
  };
  return (
    <div className="stack setup-step" data-step="levers">
      <StepIssues issues={stepIssues("levers", d.issues)} />
      <Card title="Predictors" eyebrow="Model inputs">
        <CfgChips path="predictors" label="Predictors" options={[...candidates, ...predictors.filter((p) => !candidates.includes(p))].map((p) => ({ value: p, label: p }))} hint="Numeric columns the models learn from (target, coordinates, id and zone excluded)." />
        <div className="grid2">
          <CfgChips path="encodings.categorical" label="Categorical (one-hot)" options={predOpts} />
          <CfgChips path="encodings.circular_degrees" label="Circular, in degrees (sin/cos)" options={predOpts} />
        </div>
        <details>
          <summary>QA clip ranges</summary>
          <table className="tbl clip-table" aria-label="QA clip ranges">
            <thead>
              <tr>
                <th scope="col">Predictor</th>
                <th scope="col">Lower</th>
                <th scope="col">Upper</th>
                <th scope="col">Issues</th>
              </tr>
            </thead>
            <tbody>
              {predictors.map((p) => (
                <ClipRow key={p} name={p} range={range(p)} />
              ))}
            </tbody>
          </table>
          <IssueLines issues={c.issues("qa.clip")} />
        </details>
      </Card>
      <Card
        title="Levers"
        eyebrow="What a plan can change"
        actions={
          <Button size="small" icon="refresh" busy={d.checking} onClick={() => void runDataCheck(c.pid)}>
            {d.check ? "Re-run the data check" : "Run the data check"}
          </Button>
        }
      >
        <p className="cap">Each actionable predictor gets a dose sweep in S4 and can be edited in scenarios. The dose-scale table shows how far each dose reaches in the data.</p>
        {predictors.length === 0 ? <p className="cap">Choose predictors first.</p> : null}
        <div className="stack" style={{ gap: 10 }}>
          {predictors.map((p) => (
            <LeverCard key={p} name={p} range={range(p)} />
          ))}
        </div>
        <IssueLines issues={c.issues("actionable")} />
      </Card>
      <Card title="Coupling" eyebrow="Sums with a cap">
        <Coupling predictors={predOpts} />
      </Card>
      <Card title="Mediators" eyebrow="Predictors the levers move">
        <Mediators predictors={predictors} levers={levers} />
      </Card>
    </div>
  );
}
