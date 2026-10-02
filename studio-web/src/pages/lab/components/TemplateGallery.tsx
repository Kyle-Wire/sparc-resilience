// Template gallery (SPEC §7.4): planning actions as parameterised forms that emit a
// ScenarioDoc on the server ("Shade the hottest 10%", "Cool roofs on built-up cells", …).
// Each card renders its params_schema (numbers, choices, flags, a lever, and the list of
// [lon, lat] sites of "Around sites", typed or loaded from a CSV of uploaded points); missing
// required parameters keep Create disabled. Requirements the run lacks come back as
// `422 template_unavailable` with `detail.missing` and are shown on the card.
import { useRef, useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { scenarioFromTemplate, useTemplates, type Lever, type Scenario, type ScenarioTemplate, type TemplateParamSchema } from "../../../api/lab";
import { Button } from "../../../components/ui/Button";
import { NumberField } from "../../../components/ui/NumberField";
import { toast } from "../../../stores/ui";
import { fmtInt } from "../../../theme/format";
import { parseSites } from "../model/library";

const REQ_TEXT: Record<string, string> = {
  layers: "people & land-cover layers",
  canopy_role: "a canopy role",
  impervious_role: "an impervious role",
  albedo_role: "an albedo role",
  crs: "a CRS",
};

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

function TemplateCard({ t, pid, rid, levers, onCreated }: { t: ScenarioTemplate; pid: string | null; rid: string; levers: Lever[]; onCreated: (s: Scenario) => void }) {
  const [open, setOpen] = useState(false);
  const [params, setParams] = useState<Record<string, unknown>>(() => templateDefaults(t));
  const [busy, setBusy] = useState(false);
  const [missing, setMissing] = useState<string[] | null>(null);
  const needed = missingParams(t, params);
  const titleOf = (k: string) => t.params_schema.properties?.[k]?.title ?? k.replace(/_/g, " ");
  const apply = async () => {
    if (!pid || needed.length) return;
    setBusy(true);
    try {
      onCreated(await scenarioFromTemplate(pid, t.id, params, rid));
    } catch (e) {
      if (e instanceof ApiError && e.code === "template_unavailable") setMissing(Array.isArray(e.detail?.missing) ? (e.detail!.missing as string[]) : [e.message]);
      else toast("error", `Could not start "${t.label}"`, { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <article className="template card tight" data-template={t.id}>
      <header>
        <div>
          <h3>{t.label}</h3>
          <p className="cap">{t.desc}</p>
        </div>
      </header>
      {t.requires.length ? <p className="cap">Needs {t.requires.map((r) => REQ_TEXT[r] ?? r).join(", ")}.</p> : null}
      {missing ? (
        <p className="callout" data-tone="crit">
          This run lacks {missing.map((m) => REQ_TEXT[m] ?? m).join(", ")}.
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
        <Button size="small" onClick={() => setOpen(true)} disabled={!pid} title={pid ? undefined : "Open the run from its project to create scenarios"}>
          Use template
        </Button>
      )}
    </article>
  );
}

export function TemplateGallery({ pid, rid, levers = [], onCreated }: { pid: string | null; rid: string; levers?: Lever[]; onCreated: (s: Scenario) => void }) {
  const { data, error } = useTemplates();
  if (error) return <p className="cap">Templates are unavailable: {errorMessage(error)}</p>;
  if (!data) return <p className="cap">Loading templates…</p>;
  return (
    <section className="templates stack" aria-label="Templates" style={{ gap: 8 }}>
      <p className="eyebrow">Start from a planning action</p>
      {data.map((t) => (
        <TemplateCard key={t.id} t={t} pid={pid} rid={rid} levers={levers} onCreated={onCreated} />
      ))}
    </section>
  );
}
