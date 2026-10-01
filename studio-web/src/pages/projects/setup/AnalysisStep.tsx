// Setup step "analysis" (SPEC §9.3): causal (treatments, confounders, exclusions, contrasts,
// DAG audit), climate (enabled, table vs live, site pin, SSP and period chips, months,
// thresholds, adaptation picker from the scenario names), optimize (variable, budget, cost,
// plantable, objective, equity), planner (layers, paved share, GHCN station), and the
// advanced settings: influence, cv, models, stacker, response.
import { useFileInspect, useProjectFiles } from "../../../api/projects";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Chips } from "../../../components/ui/Chips";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg } from "../../../components/ui/Seg";
import { Slider } from "../../../components/ui/Slider";
import { fmtNum } from "../../../theme/format";
import { CfgChips, CfgNumber, CfgNumberList, CfgSeg, CfgSelect, CfgText, CfgToggle, IssueLines, useCfg, type Opt } from "../components/cfg";
import { StepIssues } from "../components/StepIssues";
import { allColumns } from "../model/columns";
import { getRecord, getString } from "../model/dotted";
import { configuredScenarioNames } from "../model/scenarioNames";
import { stepIssues } from "../model/steps";
import { usePreviewGrid } from "./actions";

export const SSPS: readonly Opt[] = [
  { value: "ssp126", label: "SSP1-2.6" },
  { value: "ssp245", label: "SSP2-4.5" },
  { value: "ssp370", label: "SSP3-7.0" },
  { value: "ssp585", label: "SSP5-8.5" },
];

export const DEFAULT_PERIODS: Record<string, [number, number]> = { "2021-2040": [2021, 2040], "2041-2060": [2041, 2060], "2081-2100": [2081, 2100] };

const MONTHS: readonly Opt[] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map((m, i) => ({ value: String(i + 1), label: m }));

const BASELINES: readonly Opt[] = [
  { value: "regression_kriging", label: "regression-kriging" },
  { value: "hgb_xy", label: "boosting + x,y" },
  { value: "hgb", label: "boosting" },
  { value: "idw", label: "IDW" },
  { value: "hgb_focal", label: "boosting + focal" },
];

/** A value that is "auto" or a number (cv.block_m, cv.buffer_m, causal.spatial_basis_scale_m). */
function AutoNumber({ path, label, unit }: { path: string; label: string; unit?: string }) {
  const c = useCfg();
  const v = c.value(path);
  const auto = v === "auto" || v === undefined || v === null;
  return (
    <div className="field" data-cfg-path={path}>
      <span className="field-label">{label}</span>
      <div className="row">
        <Seg
          size="small"
          label={label}
          value={auto ? "auto" : "set"}
          options={[
            { value: "auto", label: "auto" },
            { value: "set", label: "set" },
          ]}
          onChange={(m) => c.set(path, m === "auto" ? "auto" : 1000)}
        />
        {!auto ? <NumberField label={label} value={typeof v === "number" ? v : null} onChange={(n) => c.set(path, n ?? "auto")} unit={unit} min={0} /> : null}
      </div>
      <IssueLines issues={c.issues(path)} />
    </div>
  );
}

function Causal({ levers, predictors }: { levers: string[]; predictors: string[] }) {
  const c = useCfg();
  const tv = c.value("causal.treatments");
  const treatments = Array.isArray(tv) ? tv.map(String) : [];
  const audit = c.value("causal.dag_audit");
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="grid2">
        <CfgToggle def path="causal.enabled" label="Run the causal validation (S6)" />
        <CfgChips path="causal.treatments" label="Treatments (levers validated causally)" options={levers.map((l) => ({ value: l, label: l }))} />
      </div>
      {treatments.map((t) => {
        const others = predictors.filter((p) => p !== t).map((p) => ({ value: p, label: p }));
        return (
          <div key={t} className="treatment" data-cfg-path={`causal.confounders.${t}`}>
            <strong className="mono">{t}</strong>
            <CfgChips path={["causal", "confounders", t]} label="Confounders (adjusted for)" options={others} emptyAs="remove" />
            <CfgChips path={["causal", "exclude_controls", t]} label="Descendants never adjusted for (mediators)" options={others} emptyAs="remove" />
            <CfgNumber path={["causal", "contrast", t].join(".")} label="Contrast (dose compared)" unit={String(getRecord<unknown>(c.draft.raw, ["actionable", t]).unit ?? "") || null} />
          </div>
        );
      })}
      <div className="grid2">
        {audit === true || audit === false || audit === undefined || audit === null ? (
          <CfgToggle path="causal.dag_audit" label="Data-driven DAG audit (slower)" />
        ) : (
          <p className="cap">
            DAG audit with custom settings <span className="mono">{JSON.stringify(audit)}</span> (edit in the config editor).
          </p>
        )}
        <AutoNumber path="causal.spatial_basis_scale_m" label="Spatial confounding basis scale" unit="m" />
        <CfgNumber path="causal.n_boot" label="Bootstrap replicates" min={1} step={10} />
      </div>
    </div>
  );
}

function Periods() {
  const c = useCfg();
  const periods = getRecord<[number, number]>(c.value("climate.periods") ?? DEFAULT_PERIODS, []);
  const names = Object.keys(periods);
  const all = [...new Set([...Object.keys(DEFAULT_PERIODS), ...names])];
  return (
    <div className="field" data-cfg-path="climate.periods">
      <span className="field-label">Future periods</span>
      <Chips
        label="Future periods"
        items={all.map((n) => ({ value: n, label: n }))}
        selected={names}
        onToggle={(n, on) => {
          const next: Record<string, [number, number]> = {};
          for (const k of all) {
            if (k === n ? on : names.includes(k)) {
              const v = periods[k] ?? DEFAULT_PERIODS[k];
              if (v) next[k] = v;
            }
          }
          c.set("climate.periods", next);
        }}
      />
      <span className="hint">The CMIP6 input job fetches change factors for these periods.</span>
      <IssueLines issues={c.issues("climate.periods", true)} />
    </div>
  );
}

function Site({ centroid }: { centroid: { lat: number; lon: number } | null }) {
  const c = useCfg();
  const v = c.value("climate.site");
  const pair = Array.isArray(v) ? (v as unknown[]) : null;
  const lat = pair && typeof pair[0] === "number" ? (pair[0] as number) : null;
  const lon = pair && typeof pair[1] === "number" ? (pair[1] as number) : null;
  const put = (a: number | null, b: number | null) => c.set("climate.site", a === null && b === null ? undefined : [a ?? 0, b ?? 0]);
  return (
    <div className="field" data-cfg-path="climate.site">
      <span className="field-label">Site pin [lat, lon]</span>
      <div className="row">
        <NumberField label="Site latitude" value={lat} onChange={(n) => put(n, lon)} nullable min={-90} max={90} unit="°N" />
        <NumberField label="Site longitude" value={lon} onChange={(n) => put(lat, n)} nullable min={-180} max={180} unit="°E" />
        {centroid ? (
          <Button size="small" variant="ghost" icon="target" onClick={() => c.set("climate.site", [Number(centroid.lat.toFixed(4)), Number(centroid.lon.toFixed(4))])}>
            Use the data centroid
          </Button>
        ) : null}
      </div>
      <span className="hint">Empty: the data centroid (needs data.crs).</span>
      <IssueLines issues={c.issues("climate.site", true)} />
    </div>
  );
}

function Climate({ centroid, scenarioNames }: { centroid: { lat: number; lon: number } | null; scenarioNames: string[] }) {
  const c = useCfg();
  const files = useProjectFiles(c.pid);
  const tables: Opt[] = (files.data ?? []).filter((f) => f.kind === "climate" || f.path.endsWith(".csv")).map((f) => ({ value: f.path, label: f.path }));
  const adaptation = c.value("climate.adaptation");
  const months = c.value("climate.months");
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="grid3">
        <CfgToggle path="climate.enabled" label="CMIP6 projections × adaptation" />
        <CfgSeg
          path="climate.source"
          label="Change factors"
          options={[
            { value: "table", label: "table (CSV)" },
            { value: "cmip6", label: "fetch at run time" },
          ]}
          fromValue={(v) => (v === "cmip6" ? "cmip6" : "table")}
        />
        <CfgSelect path="climate.table" label="Change-factor table" options={tables} none="none" missingNote="not among the project files" hint="From the CMIP6 input job (Inputs step)." />
      </div>
      <Site centroid={centroid} />
      <CfgChips path="climate.experiments" label="SSPs" options={SSPS} />
      <Periods />
      <div className="field" data-cfg-path="climate.months">
        <span className="field-label">Months of the season</span>
        <Chips
          label="Months"
          items={MONTHS}
          selected={Array.isArray(months) ? (months as unknown[]).map(String) : []}
          onToggle={(m, on) => {
            const cur = Array.isArray(months) ? (months as unknown[]).map(Number) : [];
            const next = on ? [...cur, Number(m)] : cur.filter((x) => x !== Number(m));
            c.set("climate.months", [...new Set(next)].sort((a, b) => a - b));
          }}
        />
        <IssueLines issues={c.issues("climate.months", true)} />
      </div>
      <div className="grid2">
        <CfgSeg
          path="climate.variable"
          label="CMIP6 variable"
          options={[
            { value: "tasmax", label: "daily max (tasmax)" },
            { value: "tas", label: "daily mean (tas)" },
          ]}
          fromValue={(v) => (v === "tas" ? "tas" : "tasmax")}
        />
        <CfgNumberList path="climate.thresholds" label="Exposure thresholds" unit={getString(c.draft.raw, "data.target_units") ?? "degF"} />
      </div>
      <div className="field" data-cfg-path="climate.adaptation">
        <span className="field-label">Adaptation scenarios paired with warming</span>
        <div className="row">
          <Seg
            size="small"
            label="Adaptation choice"
            value={Array.isArray(adaptation) ? "pick" : "auto"}
            options={[
              { value: "auto", label: "automatic" },
              { value: "pick", label: "choose" },
            ]}
            onChange={(m) => c.set("climate.adaptation", m === "auto" ? undefined : scenarioNames.slice(0, 3))}
          />
        </div>
        {Array.isArray(adaptation) ? (
          <Chips
            label="Adaptation scenarios"
            items={scenarioNames.map((n) => ({ value: n, label: n }))}
            selected={(adaptation as unknown[]).map(String)}
            onToggle={(n, on) => {
              const cur = (adaptation as unknown[]).map(String);
              c.set("climate.adaptation", on ? scenarioNames.filter((x) => x === n || cur.includes(x)) : cur.filter((x) => x !== n));
            }}
          />
        ) : (
          <span className="hint">Automatic: the configured scenarios with little extrapolation.</span>
        )}
        <IssueLines issues={c.issues("climate.adaptation", true)} />
      </div>
    </div>
  );
}

function Optimize({ levers, columns }: { levers: string[]; columns: string[] }) {
  const c = useCfg();
  const focus = c.value("optimize.equity_focus");
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="grid3">
        <CfgToggle def path="optimize.enabled" label="Budget optimisation (S7)" />
        <CfgSelect path="optimize.variable" label="Lever to allocate" options={levers.map((l) => ({ value: l, label: l }))} none="none" />
        <CfgNumber path="optimize.budget" label="Budget" hint="In dose units × cells (e.g. +10 pp on 2,000 cells = 20,000)." min={0} />
        <CfgNumber path="optimize.cost_per_unit" label="Cost per dose unit" min={0} />
        <CfgToggle def path="optimize.plantable" label="Cap canopy by plantable space (needs planner layers)" />
        <CfgSeg
          path="optimize.objective"
          label="Objective"
          options={[
            { value: "cooling", label: "cooling" },
            { value: "people", label: "cooling × residents" },
          ]}
          fromValue={(v) => (v === "people" ? "people" : "cooling")}
        />
        <CfgSelect path="optimize.equity_column" label="Equity column" options={columns.map((x) => ({ value: x, label: x }))} none="none" />
      </div>
      <div data-cfg-path="optimize.equity_focus">
        <Slider label="Equity focus" value={typeof focus === "number" ? focus : 0} min={0} max={1} step={0.05} onChange={(v) => c.set("optimize.equity_focus", v)} format={(v) => fmtNum(v, 2)} />
        <IssueLines issues={c.issues("optimize.equity_focus")} />
      </div>
    </div>
  );
}

function Planner() {
  const c = useCfg();
  const files = useProjectFiles(c.pid);
  const layers: Opt[] = (files.data ?? []).filter((f) => f.kind === "layers" || f.path.endsWith(".parquet")).map((f) => ({ value: f.path, label: f.path }));
  return (
    <div className="grid3">
      <CfgSelect path="planner.layers" label="People and land-cover layers" options={layers} none="none" missingNote="not among the project files" hint="From the People & land cover input job." />
      <CfgNumber path="planner.paved_plantable_share" label="Paved share counted as plantable" min={0} max={1} step={0.05} />
      <CfgText path="planner.ghcn_station" label="GHCN-Daily station" placeholder="USW00014765" hint="Daily Tmax for hot-day counts." />
    </div>
  );
}

function Advanced() {
  const c = useCfg();
  const baselines = c.value("cv.baselines");
  const blMode = baselines === false ? "off" : Array.isArray(baselines) ? "pick" : "all";
  return (
    <div className="stack" style={{ gap: 14 }}>
      <fieldset className="adv-group">
        <legend>Influence (S1)</legend>
        <div className="grid3">
          <CfgNumber path="influence.max_lag_m" label="Longest correlogram lag" unit="m" min={0} />
          <CfgNumber path="influence.n_rings" label="Ring-profile rings" min={1} step={1} />
          <CfgNumber path="influence.n_perm" label="Permutations" min={1} step={1} />
          <CfgNumber path="influence.mass" label="Kernel mass defining the range" min={0} max={1} step={0.05} />
        </div>
        <CfgNumberList path="influence.scales" label="Focal-feature scales (× range)" />
      </fieldset>
      <fieldset className="adv-group">
        <legend>Cross-validation</legend>
        <div className="grid3">
          <CfgNumber path="cv.n_folds" label="Spatial-block folds" min={2} step={1} />
          <AutoNumber path="cv.block_m" label="Block size" unit="m" />
          <AutoNumber path="cv.buffer_m" label="Buffer" unit="m" />
          <CfgNumber path="cv.seed" label="Fold seed" step={1} />
          <CfgToggle path="cv.distance_curve.enabled" label="Skill-vs-distance curve (re-fits; slow)" />
        </div>
        <CfgNumberList path="cv.distance_curve.block_m" label="Curve block sizes (0 = random points)" unit="m" />
        <div className="field" data-cfg-path="cv.baselines">
          <span className="field-label">Reference baselines</span>
          <Seg
            size="small"
            label="Reference baselines"
            value={blMode}
            options={[
              { value: "all", label: "all" },
              { value: "pick", label: "choose" },
              { value: "off", label: "off" },
            ]}
            onChange={(m) => c.set("cv.baselines", m === "all" ? true : m === "off" ? false : BASELINES.map((b) => b.value))}
          />
          {blMode === "pick" ? (
            <Chips
              label="Baselines"
              items={BASELINES}
              selected={(baselines as unknown[]).map(String)}
              onToggle={(b, on) => {
                const cur = (baselines as unknown[]).map(String);
                c.set("cv.baselines", on ? BASELINES.map((x) => x.value).filter((x) => x === b || cur.includes(x)) : cur.filter((x) => x !== b));
              }}
            />
          ) : null}
          <IssueLines issues={c.issues("cv.baselines", true)} />
        </div>
      </fieldset>
      <fieldset className="adv-group">
        <legend>Base models</legend>
        <div className="row">
          {(["ols", "mgwr", "gwrf", "gam", "physics"] as const).map((m) => (
            <CfgToggle def key={m} path={`models.${m}`} label={m.toUpperCase()} />
          ))}
        </div>
        <CfgChips
          path="models.spatial_plus"
          label="Spatial+ for"
          options={[
            { value: "mgwr", label: "MGWR" },
            { value: "gam", label: "GAM" },
          ]}
        />
      </fieldset>
      <fieldset className="adv-group">
        <legend>Stacker</legend>
        <div className="grid3">
          <CfgNumber path="stacker.epochs" label="Training epochs" min={1} step={50} />
          <CfgNumber path="stacker.coverage" label="Interval coverage target" min={0.5} max={0.99} step={0.01} />
        </div>
        <CfgNumberList path="stacker.tune_lambda" label="PDE weights tried (tune_lambda)" />
      </fieldset>
      <fieldset className="adv-group">
        <legend>Response</legend>
        <CfgToggle def path="response.clip_to_support" label="Clamp edits to the observed range of each lever" />
      </fieldset>
    </div>
  );
}

export function AnalysisStep() {
  const c = useCfg();
  const d = c.draft;
  const levers = Object.keys(getRecord(d.raw, "actionable"));
  const pv = c.value("predictors");
  const predictors = Array.isArray(pv) ? pv.map(String) : [];
  const inspect = useFileInspect(c.pid, getString(d.raw, "data.path") ?? getString(d.effective, "data.path"));
  const grid = usePreviewGrid(c.pid, d.check?.preview_token || null);
  const b = grid.data?.meta.bounds_lonlat;
  const centroid = b ? { lon: (b[0] + b[2]) / 2, lat: (b[1] + b[3]) / 2 } : null;
  const names = configuredScenarioNames(d.raw);
  return (
    <div className="stack setup-step" data-step="analysis">
      <StepIssues issues={stepIssues("analysis", d.issues)} />
      <Card title="Causal validation" eyebrow="S6">
        <Causal levers={levers} predictors={predictors} />
      </Card>
      <Card title="Climate" eyebrow="CMIP6 × adaptation">
        <Climate centroid={centroid} scenarioNames={names} />
      </Card>
      <Card title="Optimisation" eyebrow="S7 budget">
        <Optimize levers={levers} columns={allColumns(inspect.data)} />
      </Card>
      <Card title="Planner pack" eyebrow="Post-run">
        <Planner />
      </Card>
      <details className="card advanced">
        <summary>
          <strong>Advanced</strong> <span className="cap">influence, cross-validation, models, stacker, response</span>
        </summary>
        <Advanced />
      </details>
    </div>
  );
}
