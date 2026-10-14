// Setup step "physics" (SPEC §9.3, J2 step 5): role dropdowns for the six physical roles, the
// missing-role warning (canopy and impervious are what placebo, simcheck, planner and the
// emulator edit), forcing values (window, SW↓, LW net, wind), the campaign forcing file with
// its checks, and the advanced drawer.
import { useState } from "react";
import { useInputs } from "../../../api/inputs";
import { useProjectFiles } from "../../../api/projects";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { InfoTip } from "../../../components/ui/InfoTip";
import { term } from "../../../content/glossary";
import { Drawer } from "../../../components/ui/Drawer";
import { NumberField } from "../../../components/ui/NumberField";
import { Pill } from "../../../components/ui/Pill";
import { Link } from "../../../router";
import { fmtSig } from "../../../theme/format";
import { CfgJson, CfgNumber, CfgSeg, CfgSelect, CfgToggle, IssueLines, useCfg, type Opt } from "../components/cfg";
import { StepIssues } from "../components/StepIssues";
import { getPath, getRecord } from "../model/dotted";
import { stepIssues } from "../model/steps";

export const ROLES: readonly { role: string; label: string; hint: string }[] = [
  { role: "canopy", label: "Tree canopy (%)", hint: "Shade, and cooling as leaves give off water" },
  { role: "impervious", label: "Paved and built surface (%)", hint: "Stores heat; rain runs off (impervious surface)" },
  { role: "albedo", label: "Surface reflectivity (albedo)", hint: "How much sunlight surfaces reflect, 0 to 1" },
  { role: "ndvi", label: "Greenness (NDVI)", hint: "How green the vegetation is, from satellite images" },
  { role: "elevation", label: "Elevation", hint: "Higher ground is cooler (m)" },
  { role: "water_distance", label: "Distance to water", hint: "Cooling near rivers, lakes and the sea (m)" },
];

/** The roles the placebo, simcheck, planner and emulator need. */
export const KEY_ROLES = ["canopy", "impervious"] as const;

export const MISSING_ROLES_TEXT = "fake-change tests, simulation checks, the planner pack and quick estimates need these";

/** Key roles (canopy, impervious) without a predictor in the draft. */
export function missingKeyRoles(roles: Record<string, unknown>): string[] {
  return KEY_ROLES.filter((r) => typeof roles[r] !== "string" || roles[r] === "");
}

function Wind() {
  const c = useCfg();
  const v = c.value("physics.wind");
  const pair = Array.isArray(v) ? (v as unknown[]) : null;
  const u = pair && typeof pair[0] === "number" ? (pair[0] as number) : null;
  const w = pair && typeof pair[1] === "number" ? (pair[1] as number) : null;
  const put = (a: number | null, b: number | null) => c.set("physics.wind", a === null && b === null ? undefined : [a ?? 0, b ?? 0]);
  return (
    <div className="field" data-cfg-path="physics.wind">
      <span className="field-label">Wind [u, v] (m/s, blowing towards east / north)</span>
      <div className="row">
        <NumberField label="Wind u (m/s)" value={u} onChange={(n) => put(n, w)} nullable unit="m/s" />
        <NumberField label="Wind v (m/s)" value={w} onChange={(n) => put(u, n)} nullable unit="m/s" />
      </div>
      <span className="hint">Empty: taken from the weather file when it has one; otherwise the wind does not carry heat between places (no advection).</span>
      <IssueLines issues={c.issues("physics.wind", true)} />
    </div>
  );
}

function ForcingLink() {
  const c = useCfg();
  const files = useProjectFiles(c.pid);
  const inputs = useInputs(c.pid);
  const opts: Opt[] = (files.data ?? []).filter((f) => f.kind === "forcing" || f.path.endsWith(".json")).map((f) => ({ value: f.path, label: f.path }));
  const f = inputs.data?.forcing;
  const current = c.value("physics.forcing");
  const values = f && f.physics ? Object.entries(f.physics).filter(([, v]) => typeof v === "number" || typeof v === "string") : [];
  return (
    <div className="stack" style={{ gap: 8 }}>
      <CfgSelect path="physics.forcing" label="Weather file for the measurement day (forcing)" options={opts} none="none (use the values above)" missingNote="not among the project files" hint="A weather file (forcing JSON) sets the time of day, sunshine, heat from the sky and wind for the measurement day, from ERA5 and a weather station." />
      {f && current && f.path === current ? (
        <div className="forcing-summary">
          <p className="cap">
            {f.date ? `Measurement day ${f.date}. ` : ""}
            {f.linked ? "Linked into the config." : "Not linked yet."}
          </p>
          {values.length ? (
            <dl className="kv">
              {values.map(([k, v]) => (
                <div className="kv-row" key={k}>
                  <dt>{k}</dt>
                  <dd>{typeof v === "number" ? fmtSig(v, 4) : String(v)}</dd>
                </div>
              ))}
            </dl>
          ) : null}
          {f.checks.length ? (
            <div className="row">
              {f.checks.map((ck, i) => (
                <Pill key={i} tone="warn" icon="alert">
                  {ck}
                </Pill>
              ))}
            </div>
          ) : (
            <Pill tone="good" icon="check">
              Forcing checks passed
            </Pill>
          )}
        </div>
      ) : (
        <p className="cap">
          Fetch campaign forcing (ERA5 + a nearby station) on the <Link to={`/p/${encodeURIComponent(c.pid)}/setup/inputs`}>Inputs</Link> step.
        </p>
      )}
    </div>
  );
}

function Advanced() {
  return (
    <div className="stack" style={{ gap: 10 }}>
      <CfgSeg
        path="physics.shade_form"
        label="Canopy shading form"
        options={[
          { value: "saturating", label: "saturating" },
          { value: "sigmoid", label: "S-shaped" },
        ]}
        fromValue={(v) => (v === "sigmoid" ? "sigmoid" : "saturating")}
      />
      <CfgJson path="physics.priors" label="Source-coefficient priors" hint='{name: [mean, sd]}, e.g. {"canopy": [-0.02, 0.01]}' />
      <CfgJson path="physics.albedo_map" label="Albedo rescale" hint='For non-broadband albedo layers, e.g. {"from": "auto", "to": [0.08, 0.25]}' />
      <div className="grid2">
        <CfgNumber path="physics.tau_s" label="Advection time scale" unit="s" min={0} />
        <CfgNumber path="physics.L_max_m" label="Largest diffusion length" unit="m" min={0} />
        <CfgNumber path="physics.v_max_m" label="Largest advection displacement" unit="m" min={0} />
        <CfgNumber path="physics.max_iter" label="L-BFGS iterations" min={1} step={1} />
        <CfgNumber path="physics.num_threads" label="Torch threads (physics fit)" min={1} step={1} />
      </div>
      <CfgSeg
        path="physics.fit_advection"
        label="Fit advection"
        options={[
          { value: "auto", label: "auto (when wind is set)" },
          { value: "on", label: "on" },
          { value: "off", label: "off" },
        ]}
        fromValue={(v) => (v === true ? "on" : v === false ? "off" : "auto")}
        toValue={(v) => (v === "on" ? true : v === "off" ? false : "auto")}
      />
      <CfgToggle def path="physics.select_advection" label="Keep advection only if it beats v = 0 out of fold" />
    </div>
  );
}

export function PhysicsStep() {
  const c = useCfg();
  const [advanced, setAdvanced] = useState(false);
  const pv = c.value("predictors");
  const predictors = Array.isArray(pv) ? pv.map(String) : [];
  const roles = getRecord<unknown>(c.draft.raw, "physics.roles");
  const missing = missingKeyRoles({ ...getRecord<unknown>(c.draft.effective, "physics.roles"), ...roles });
  const opts: Opt[] = predictors.map((p) => ({ value: p, label: p }));
  const mapped = ROLES.filter((r) => typeof (getPath(c.draft.raw, ["physics", "roles", r.role]) ?? getPath(c.draft.effective, ["physics", "roles", r.role])) === "string").length;
  return (
    <div className="stack setup-step" data-step="physics">
      <StepIssues issues={stepIssues("physics", c.draft.issues)} />
      <Card
        title="Which layer is which"
        eyebrow={`Physical roles · ${mapped} of ${ROLES.length} chosen`}
        tip={<InfoTip label="Which layer is which" text="Tell the built-in heat-physics model which of your map layers is tree canopy, paving and so on. It uses them to work out how surfaces heat up and cool down." technical="physics.roles" />}
      >
        {missing.length ? (
          <div className="callout roles-warning" role="alert" data-missing={missing.join(",")}>
            <strong>Missing {missing.join(" and ")} role{missing.length > 1 ? "s" : ""}:</strong> {MISSING_ROLES_TEXT}, and cannot run without {missing.length > 1 ? "them" : "it"}.
          </div>
        ) : null}
        <div className="grid3">
          {ROLES.map((r) => (
            <CfgSelect key={r.role} path={`physics.roles.${r.role}`} label={r.label} hint={r.hint} options={opts} none="not mapped" />
          ))}
        </div>
        <div data-cfg-path="physics.roles">
          <IssueLines issues={c.issues("physics.roles")} />
        </div>
      </Card>
      <Card title={term("forcing").label} eyebrow="Forcing · sunshine and wind" tip={<InfoTip term="forcing" />}>
        <div className="grid3">
          <CfgToggle def path="physics.enabled" label="Use the built-in heat-physics model" />
          <CfgSeg
            path="physics.window"
            label="Measurement window"
            options={[
              { value: "day", label: "day" },
              { value: "night", label: "night" },
            ]}
            fromValue={(v) => (v === "night" ? "night" : "day")}
          />
          <CfgNumber path="physics.sw_down" label="Sunshine (downward shortwave, SW↓)" unit="W/m²" />
          <CfgNumber path="physics.lw_net" label="Heat from the sky, net (net longwave)" unit="W/m²" />
          <Wind />
        </div>
        <ForcingLink />
      </Card>
      <div className="row">
        <Button icon="settings" onClick={() => setAdvanced(true)}>
          Advanced physics settings
        </Button>
      </div>
      <Drawer open={advanced} onClose={() => setAdvanced(false)} title="Advanced physics">
        <Advanced />
      </Drawer>
    </div>
  );
}
