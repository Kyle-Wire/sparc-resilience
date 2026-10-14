// Template gallery (SPEC §7.4): planning actions as parameterised forms that emit a
// ScenarioDoc on the server ("Shade the hottest 10%", "Cool roofs on built-up cells", …).
// Each card renders its params_schema (numbers, choices, flags, a lever, and the list of
// [lon, lat] sites of "Around sites", typed or loaded from a CSV of uploaded points); missing
// required parameters keep Create disabled. Requirements the run lacks come back as
// `422 template_unavailable` with `detail.missing` and are shown on the card, in plain words; a
// requirement the run is known to lack (no lever with that role, no map coordinates, no people
// layers) shows before trying. A run that meets them says nothing about them. Titles and descriptions
// are the server's, reworded for a planner where the client knows the template (PLAIN_TITLE,
// PLAIN_DESC): the server's titles carry placeholders and model terms ("Shade the hottest X%",
// "Prioritise by footprint"). Each card is compact (title and one line) and opens with a quiet "Start
// from this"; three show, the rest behind "More examples", so the examples never outweigh step 01.
import { useId, useRef, useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { scenarioFromTemplate, useTemplates, type Lever, type Scenario, type ScenarioTemplate, type TemplateParamSchema, type TemplateRequirement } from "../../../api/lab";
import { Button } from "../../../components/ui/Button";
import { NumberField } from "../../../components/ui/NumberField";
import { toast } from "../../../stores/ui";
import { fmtInt } from "../../../theme/format";
import { parseSites } from "../model/library";
import { LAB_WORDS } from "../model/words";
import { Tip } from "./Tip";

/** What each requirement means for a planner (the data the example needs). */
export const REQ_TEXT: Record<string, string> = {
  layers: "people and land-cover data",
  canopy_role: "tree-cover data",
  impervious_role: "paving data",
  albedo_role: "roof & surface data",
  crs: "map coordinates",
};

/** The examples' titles in plain words, by template id (the server's label otherwise). */
export const PLAIN_TITLE: Record<string, string> = {
  cool_roofs: "Cool roofs where it's most built up",
  shade_hottest: "Shade the hottest places",
  fill_plantable: "Plant the open ground",
  depave: "Less paving where it's heaviest",
  footprint_priority: "Trees where they cool most",
  around_sites: "Changes around chosen sites",
  configured_package: "Start from this run's planned changes",
};

/** A template's title for a planner. */
export const templateTitle = (t: Pick<ScenarioTemplate, "id" | "label">): string => PLAIN_TITLE[t.id] ?? t.label;

/** The examples' descriptions in plain words, by template id (the server's text otherwise). */
export const PLAIN_DESC: Record<string, string> = {
  cool_roofs: "Make roofs and pavements lighter where buildings and paving cover at least half the ground.",
  shade_hottest: "Add tree canopy in the hottest places, by measured temperature.",
  fill_plantable: "Plant trees on a share of the open ground that can take them, plus a little of the paving.",
  depave: "Take paving out of the most paved places, down to a level you choose.",
  footprint_priority: "Add tree canopy where each new tree cools the surrounding neighbourhood the most.",
  around_sites: "Make a change within a set distance of places you list, such as schools or bus stops.",
  configured_package: "Start from the set of changes saved in this run's settings, as changes you can edit.",
};

/** What this run is known to have (unknown when absent): a requirement shows only when it is known to be missing. */
export type TemplateContext = { hasCrs?: boolean; hasLayers?: boolean };

/** Requirements of a template the run is known to lack (levers by role, map coordinates, people layers). */
export function knownMissing(t: Pick<ScenarioTemplate, "requires">, levers: readonly Pick<Lever, "role">[], ctx: TemplateContext = {}): TemplateRequirement[] {
  return t.requires.filter((r) => {
    if (r === "crs") return ctx.hasCrs === false;
    if (r === "layers") return ctx.hasLayers === false;
    const role = r.replace(/_role$/, "");
    return levers.length > 0 && !levers.some((l) => l.role === role);
  });
}

const reqList = (reqs: readonly string[]) => reqs.map((r) => REQ_TEXT[r] ?? r.replace(/_/g, " ")).join(", ");

/** Default parameter values from a template's schema. */
export function templateDefaults(t: ScenarioTemplate): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, s] of Object.entries(t.params_schema.properties ?? {})) if (s.default !== undefined) out[k] = s.default;
  return out;
}

/** Required parameters with no usable value (an empty list counts as missing). */
export function missingParams(t: ScenarioTemplate, params: Record<string, unknown>): string[] {
  return (t.params_schema.required ?? []).filter((k) => {
    const v = params[k];
    return v === undefined || v === null || v === "" || (Array.isArray(v) && v.length === 0);
  });
}

/** A list of [lon, lat] points: typed one per line, or loaded from a CSV with lon/lat columns. */
function SitesField({ label, value, onChange }: { label: string; value: unknown; onChange: (v: [number, number][] | undefined) => void }) {
  const [text, setText] = useState(() => (Array.isArray(value) ? (value as number[][]).map((p) => p.join(", ")).join("\n") : ""));
  const [bad, setBad] = useState<string[]>([]);
  const file = useRef<HTMLInputElement>(null);
  const apply = (t: string) => {
    const r = parseSites(t);
    setBad(r.bad);
    onChange(r.sites.length ? r.sites : undefined);
  };
  const n = Array.isArray(value) ? value.length : 0;
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <textarea
        rows={3}
        aria-label={label}
        placeholder={"lon, lat (one site per line)\n-71.41, 41.82"}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          apply(e.target.value);
        }}
      />
      <div className="row">
        <input
          ref={file}
          type="file"
          accept=".csv,.txt,text/csv,text/plain"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (!f) return;
            void f.text().then((t) => {
              setText(t);
              apply(t);
            });
          }}
        />
        <Button size="small" icon="file" onClick={() => file.current?.click()}>
          Load points CSV
        </Button>
        <span className="hint">{n ? `${fmtInt(n)} site${n === 1 ? "" : "s"}` : "No sites yet"}</span>
      </div>
      {bad.length ? <span className="hint" data-tone="crit">Not a lon, lat pair: {bad.slice(0, 3).join(" · ")}{bad.length > 3 ? ` (+${bad.length - 3} more)` : ""}</span> : null}
    </div>
  );
}

function ParamField({ name, schema, value, onChange, levers }: { name: string; schema: TemplateParamSchema; value: unknown; onChange: (v: unknown) => void; levers: Lever[] }) {
  const label = schema.title ?? name.replace(/_/g, " ");
  if (schema.type === "array") return <SitesField label={label} value={value} onChange={onChange} />;
  if (schema.type === "string" && name === "lever" && levers.length) {
    return (
      <label className="field">
        <span className="field-label">{label}</span>
        <select value={typeof value === "string" ? value : ""} onChange={(e) => onChange(e.target.value || undefined)}>
          <option value="">The canopy lever (default)</option>
          {levers.map((l) => (
            <option key={l.var} value={l.var}>
              {l.label}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (schema.enum) {
    return (
      <label className="field">
        <span className="field-label">{label}</span>
        <select value={String(value ?? "")} onChange={(e) => onChange(typeof schema.enum![0] === "number" ? Number(e.target.value) : e.target.value)}>
          {schema.enum.map((o) => (
            <option key={String(o)} value={String(o)}>
              {String(o)}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (schema.type === "number" || schema.type === "integer") {
    return (
      <div className="field">
        <span className="field-label">{label}</span>
        <NumberField label={label} value={typeof value === "number" ? value : null} min={schema.minimum} max={schema.maximum} step={schema.type === "integer" ? 1 : "any"} nullable onChange={(v) => onChange(v ?? undefined)} />
        {schema.description ? <span className="hint">{schema.description}</span> : null}
      </div>
    );
  }
  if (schema.type === "boolean") {
    return (
      <label className="row">
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {label}
      </label>
    );
  }
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input value={value === undefined || value === null ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)} />
    </label>
  );
}

function TemplateCard({ t, pid, rid, levers, context, onCreated }: { t: ScenarioTemplate; pid: string | null; rid: string; levers: Lever[]; context?: TemplateContext; onCreated: (s: Scenario) => void }) {
  const [open, setOpen] = useState(false);
  const [params, setParams] = useState<Record<string, unknown>>(() => templateDefaults(t));
  const [busy, setBusy] = useState(false);
  const [missing, setMissing] = useState<string[] | null>(null);
  const needed = missingParams(t, params);
  const lacking = missing ?? knownMissing(t, levers, context);
  const titleOf = (k: string) => t.params_schema.properties?.[k]?.title ?? k.replace(/_/g, " ");
  const apply = async () => {
    if (!pid || needed.length) return;
    setBusy(true);
    try {
      onCreated(await scenarioFromTemplate(pid, t.id, params, rid));
    } catch (e) {
      if (e instanceof ApiError && e.code === "template_unavailable") setMissing(Array.isArray(e.detail?.missing) ? (e.detail!.missing as string[]) : [e.message]);
      else toast("error", `Could not start "${templateTitle(t)}"`, { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const desc = PLAIN_DESC[t.id] ?? t.desc;
  return (
    <article className="template card tight" data-template={t.id} data-open={open || undefined}>
      <header className="template-head">
        <h3>{templateTitle(t)}</h3>
        {/* one line on the compact card (the whole sentence in its tooltip); in full once opened */}
        <Tip tip={open ? null : desc} className="template-desc-tip">
          <p className="cap template-desc">{desc}</p>
        </Tip>
      </header>
      {lacking.length ? (
        <p className={missing ? "callout" : "cap"} data-tone={missing ? "crit" : undefined} data-unavailable="true">
          Not available for this run: it needs {reqList(lacking)}.
        </p>
      ) : null}
      {open ? (
        <div className="stack" style={{ gap: 6 }}>
          {Object.entries(t.params_schema.properties ?? {}).map(([k, s]) => (
            <ParamField key={k} name={k} schema={s} value={params[k]} levers={levers} onChange={(v) => setParams((p) => ({ ...p, [k]: v }))} />
          ))}
          {needed.length ? <p className="cap">Needed: {needed.map(titleOf).join(", ")}.</p> : null}
          <div className="row">
            <Button size="small" variant="primary" busy={busy} disabled={!pid || needed.length > 0} onClick={() => void apply()}>
              Create scenario
            </Button>
            <Button size="small" variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Tip tip={pid ? null : "Open the run from its project to create scenarios"} className="template-start-tip">
          <Button size="small" variant="ghost" className="template-start" onClick={() => setOpen(true)} disabled={!pid} aria-label={`Start from this: ${templateTitle(t)}`}>
            Start from this
          </Button>
        </Tip>
      )}
    </article>
  );
}

/** Examples shown before "More examples". */
export const EXAMPLES_SHOWN = 3;

/**
 * The examples ("Start from an example"): compact cards (title and one line), the first
 * EXAMPLES_SHOWN of them with the ones this run can use first (server order otherwise), the rest
 * behind "More examples (N)". Each card opens its own form with "Start from this".
 */
export function TemplateGallery({ pid, rid, levers = [], context, onCreated }: { pid: string | null; rid: string; levers?: Lever[]; context?: TemplateContext; onCreated: (s: Scenario) => void }) {
  const { data, error } = useTemplates();
  const [all, setAll] = useState(false);
  const listId = useId();
  if (error) return <p className="cap">The examples are unavailable: {errorMessage(error)}</p>;
  if (!data) return <p className="cap">Loading the examples…</p>;
  const usable = (t: ScenarioTemplate) => knownMissing(t, levers, context).length === 0;
  const ordered = [...data.filter(usable), ...data.filter((t) => !usable(t))];
  const shown = all ? ordered : ordered.slice(0, EXAMPLES_SHOWN);
  const more = ordered.length - EXAMPLES_SHOWN;
  return (
    <section className="templates stack" aria-label="Templates" style={{ gap: 6 }}>
      <div className="stack" id={listId} style={{ gap: 6 }}>
        {shown.map((t) => (
          <TemplateCard key={t.id} t={t} pid={pid} rid={rid} levers={levers} context={context} onCreated={onCreated} />
        ))}
      </div>
      {more > 0 ? (
        <Button size="small" variant="ghost" className="templates-more" icon={all ? "chevronUp" : "chevronDown"} aria-expanded={all} aria-controls={listId} onClick={() => setAll((v) => !v)}>
          {all ? LAB_WORDS.fewerExamples : LAB_WORDS.moreExamples(more)}
        </Button>
      ) : null}
    </section>
  );
}
