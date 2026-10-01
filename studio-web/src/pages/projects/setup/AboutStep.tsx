// Setup step "about" (SPEC §9.3): report fields of the config (title, place, area,
// limitations, caveats, run name), and the project's headline scenario (by slug) and cost
// model defaults, which are project fields saved with `PATCH /api/projects/{pid}`.
import { useEffect, useState } from "react";
import { errorMessage } from "../../../api/client";
import { patchProject } from "../../../api/projects";
import { invalidate } from "../../../api/resource";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { NumberField } from "../../../components/ui/NumberField";
import { useProject } from "../../../layouts/resources";
import { toast } from "../../../stores/ui";
import { CfgLines, CfgText, useCfg } from "../components/cfg";
import { StepIssues } from "../components/StepIssues";
import { getRecord } from "../model/dotted";
import { configuredScenarioNames, scenarioSlugs } from "../model/scenarioNames";
import { stepIssues } from "../model/steps";

type CostModel = Record<string, { per_unit: number }>;

function ProjectDetails() {
  const c = useCfg();
  const project = useProject(c.pid);
  const p = project.data?.project;
  const slugs = scenarioSlugs(configuredScenarioNames(c.draft.raw));
  const levers = getRecord<{ cost_per_unit?: number; unit?: string }>(c.draft.raw, "actionable");
  const [headline, setHeadline] = useState<string>("");
  const [cost, setCost] = useState<CostModel>({});
  const [busy, setBusy] = useState(false);
  // Follow the server's values when they change (not on every refetch of the project).
  const serverKey = p ? JSON.stringify([p.headline_scenario ?? "", p.cost_model ?? {}]) : null;
  useEffect(() => {
    if (serverKey === null) return;
    const [h, cm] = JSON.parse(serverKey) as [string, CostModel];
    setHeadline(h);
    setCost(cm);
  }, [serverKey]);
  if (!p) return <p className="cap">Loading the project…</p>;
  const dirty = (p.headline_scenario ?? "") !== headline || JSON.stringify(p.cost_model ?? {}) !== JSON.stringify(cost);
  const save = async () => {
    setBusy(true);
    try {
      await patchProject(c.pid, { headline_scenario: headline || null, cost_model: cost });
      invalidate(`project:${c.pid}`);
      invalidate("projects");
      toast("success", "Project details saved");
    } catch (e) {
      toast("error", "Could not save the project details", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const known = slugs.some((s) => s.slug === headline);
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="field" data-project-field="headline_scenario">
        <label htmlFor="headline-scenario">Headline scenario</label>
        <select id="headline-scenario" value={headline} onChange={(e) => setHeadline(e.target.value)}>
          <option value="">none</option>
          {headline && !known ? <option value={headline}>{headline} (not configured)</option> : null}
          {slugs.map((s) => (
            <option key={s.slug} value={s.slug}>
              {s.name}
            </option>
          ))}
        </select>
        <span className="hint">Shown first on run overviews and in reports (stored by slug, e.g. {slugs[0]?.slug ?? "canopy-increase-plus-10"}).</span>
      </div>
      <div className="field">
        <span className="field-label">Cost model (per unit of dose on one cell)</span>
        {Object.keys(levers).length === 0 ? <span className="hint">No levers yet.</span> : null}
        <div className="grid3">
          {Object.entries(levers).map(([name, l]) => (
            <label key={name} className="field">
              <span className="mono">{name}</span>
              <NumberField
                label={`Cost per unit of ${name}`}
                value={cost[name]?.per_unit ?? (typeof l.cost_per_unit === "number" ? l.cost_per_unit : null)}
                onChange={(v) => {
                  const next = { ...cost };
                  if (v === null) delete next[name];
                  else next[name] = { per_unit: v };
                  setCost(next);
                }}
                nullable
                min={0}
                unit={l.unit ? `per ${l.unit}·cell` : null}
              />
            </label>
          ))}
        </div>
      </div>
      <div className="row">
        <Button variant="primary" busy={busy} disabled={!dirty} onClick={() => void save()}>
          Save project details
        </Button>
        {dirty ? <span className="cap">Unsaved project details</span> : null}
      </div>
    </div>
  );
}

export function AboutStep() {
  const c = useCfg();
  return (
    <div className="stack setup-step" data-step="about">
      <StepIssues issues={stepIssues("about", c.draft.issues)} />
      <Card title="Report" eyebrow="Pages, model card, reports">
        <div className="grid3">
          <CfgText path="report.title" label="Report title" placeholder="Providence Heat Model" />
          <CfgText path="report.place" label="Place" placeholder="Providence, RI" />
          <CfgText path="report.area" label="Study area" placeholder="the Brown University / Providence study area" />
          <CfgText path="name" label="Run name" placeholder="core_run" hint="Names the runs' manifest and report." />
        </div>
        <CfgLines path="report.limitations" label="Limitations (model card)" />
        <CfgLines path="report.caveats" label="Caveats (results page)" />
      </Card>
      <Card title="Project" eyebrow="Headline and costs">
        <ProjectDetails />
      </Card>
    </div>
  );
}
