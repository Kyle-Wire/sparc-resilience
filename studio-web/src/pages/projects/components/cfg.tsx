// Config-bound form fields for the setup wizard. Each field reads its value from the draft
// (falling back to the DEFAULTS-merged config), writes to the draft at its dotted path, and
// shows the server's validation Issue messages for that path (SPEC §9.3, api.md §1 Issue)
// under the control. Every field carries `data-cfg-path` so a path can be found in the page.
import { createContext, useContext, useState, type ReactNode } from "react";
import type { Issue } from "../../../api/types";
import { Badge } from "../../../components/ui/Badge";
import { Chips } from "../../../components/ui/Chips";
import { Field } from "../../../components/ui/Field";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg, type SegOption } from "../../../components/ui/Seg";
import { fmtPct, MINUS } from "../../../theme/format";
import { getPath, joinPath, splitPath, type Path } from "../model/dotted";
import { draftValue, useDrafts, type Draft } from "../setup/store";

export type CfgContextValue = {
  pid: string;
  draft: Draft;
  issues: Issue[];
};

const CfgCtx = createContext<CfgContextValue | null>(null);

export function CfgProvider({ value, children }: { value: CfgContextValue; children: ReactNode }) {
  return <CfgCtx.Provider value={value}>{children}</CfgCtx.Provider>;
}

/** Issues at a path; `deep` also takes those under it (list items, nested keys). */
export function issuesFor(issues: readonly Issue[], path: string, deep = false): Issue[] {
  return issues.filter((i) => i.path === path || (deep && (i.path.startsWith(path + ".") || i.path.startsWith(path + "["))));
}

export type Cfg = {
  pid: string;
  draft: Draft;
  /** Display value (draft, else effective). */
  value: (path: Path) => unknown;
  /** The draft's own value (undefined when unset). */
  own: (path: Path) => unknown;
  set: (path: Path, value: unknown) => void;
  issues: (path: Path, deep?: boolean) => Issue[];
  suggested: (path: Path) => boolean;
};

export function useCfg(): Cfg {
  const ctx = useContext(CfgCtx);
  const edit = useDrafts((s) => s.edit);
  if (!ctx) throw new Error("useCfg outside the setup wizard");
  const { pid, draft, issues } = ctx;
  const key = (p: Path) => (typeof p === "string" ? p : joinPath(p));
  return {
    pid,
    draft,
    value: (p) => draftValue(draft, p),
    own: (p) => getPath(draft.raw, p),
    set: (p, v) => edit(pid, p, v),
    issues: (p, deep) => issuesFor(issues, key(p), deep),
    suggested: (p) => draft.suggested.includes(key(p)),
  };
}

function SuggestedBadge({ conf }: { conf?: number | null }) {
  return (
    <Badge tone="accent" title={conf !== null && conf !== undefined ? `Suggested from the data (confidence ${fmtPct(conf)})` : "Suggested from the data"}>
      suggested
    </Badge>
  );
}

type Base = { path: Path; label: ReactNode; hint?: ReactNode; deep?: boolean; className?: string; confidence?: number | null };

function Wrap({ path, children }: { path: Path; children: ReactNode }) {
  return <div data-cfg-path={typeof path === "string" ? path : joinPath(path)}>{children}</div>;
}

function LabelText({ label, suggested, confidence }: { label: ReactNode; suggested: boolean; confidence?: number | null }) {
  return (
    <>
      {label}
      {suggested ? (
        <>
          {" "}
          <SuggestedBadge conf={confidence} />
        </>
      ) : null}
    </>
  );
}

/** Text input; an empty value removes the key (the default applies). */
export function CfgText({ path, label, hint, placeholder, list, deep, className, confidence }: Base & { placeholder?: string; list?: string }) {
  const c = useCfg();
  const v = c.value(path);
  return (
    <Wrap path={path}>
      <Field label={<LabelText label={label} suggested={c.suggested(path)} confidence={confidence} />} hint={hint} issues={c.issues(path, deep)} className={className}>
        {(id, describedBy, invalid) => (
          <input
            id={id}
            type="text"
            list={list}
            placeholder={placeholder}
            aria-describedby={describedBy}
            aria-invalid={invalid || undefined}
            value={v === undefined || v === null ? "" : String(v)}
            onChange={(e) => c.set(path, e.target.value === "" ? undefined : e.target.value)}
          />
        )}
      </Field>
    </Wrap>
  );
}

/** Numeric input with unit; `nullable` lets the user clear it (key removed). */
export function CfgNumber({ path, label, hint, unit, min, max, step, nullable = true, deep, className }: Base & { unit?: string | null; min?: number; max?: number; step?: number; nullable?: boolean }) {
  const c = useCfg();
  const v = c.value(path);
  return (
    <Wrap path={path}>
      <Field label={label} hint={hint} issues={c.issues(path, deep)} className={className}>
        {(id, describedBy, invalid) => (
          <NumberField
            id={id}
            aria-describedby={describedBy}
            aria-invalid={invalid || undefined}
            value={typeof v === "number" && Number.isFinite(v) ? v : null}
            onChange={(n) => c.set(path, n === null ? undefined : n)}
            unit={unit}
            min={min}
            max={max}
            step={step}
            nullable={nullable}
          />
        )}
      </Field>
    </Wrap>
  );
}

export type Opt = { value: string; label: string; group?: string };

/** Select from options; `none` adds an empty choice that removes the key. */
export function CfgSelect({ path, label, hint, options, none, deep, className, confidence, missingNote = "not in the data" }: Base & { options: readonly Opt[]; none?: string; missingNote?: string }) {
  const c = useCfg();
  const v = c.value(path);
  const cur = v === undefined || v === null ? "" : String(v);
  const known = options.some((o) => o.value === cur);
  return (
    <Wrap path={path}>
      <Field label={<LabelText label={label} suggested={c.suggested(path)} confidence={confidence} />} hint={hint} issues={c.issues(path, deep)} className={className}>
        {(id, describedBy, invalid) => (
          <select id={id} aria-describedby={describedBy} aria-invalid={invalid || undefined} value={cur} onChange={(e) => c.set(path, e.target.value === "" ? undefined : e.target.value)}>
            {none !== undefined || cur === "" ? <option value="">{none ?? "Choose…"}</option> : null}
            {cur !== "" && !known ? <option value={cur}>{cur} ({missingNote})</option> : null}
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        )}
      </Field>
    </Wrap>
  );
}

/** Checkbox bound to a boolean; `def` is core's DEFAULTS value shown while the key is unset. */
export function CfgToggle({ path, label, hint, className, def = false }: Base & { def?: boolean }) {
  const c = useCfg();
  const raw = c.value(path);
  const v = raw === undefined || raw === null ? def : raw;
  const issues = c.issues(path);
  return (
    <Wrap path={path}>
      <div className={["field", className ?? ""].filter(Boolean).join(" ")}>
        <label className="row" style={{ gap: 6, fontSize: "0.88rem", color: "var(--ink)" }}>
          <input type="checkbox" checked={v === true} onChange={(e) => c.set(path, e.target.checked)} />
          {label}
        </label>
        {hint ? <span className="hint">{hint}</span> : null}
        <IssueLines issues={issues} />
      </div>
    </Wrap>
  );
}

/** Segmented choice bound to a string (or boolean/number when the options say so). */
export function CfgSeg<T extends string | number>({ path, label, hint, options, deep, className, fromValue, toValue }: Base & { options: readonly SegOption<T>[]; fromValue?: (v: unknown) => T; toValue?: (v: T) => unknown }) {
  const c = useCfg();
  const raw = c.value(path);
  const v = fromValue ? fromValue(raw) : (raw as T);
  return (
    <Wrap path={path}>
      <div className={["field", className ?? ""].filter(Boolean).join(" ")}>
        <span className="field-label">{label}</span>
        <Seg options={options} value={v} onChange={(n) => c.set(path, toValue ? toValue(n) : n)} label={typeof label === "string" ? label : joinPath(splitPath(path))} size="small" />
        {hint ? <span className="hint">{hint}</span> : null}
        <IssueLines issues={c.issues(path, deep)} />
      </div>
    </Wrap>
  );
}

/** Multi-select chips over a string list. */
export function CfgChips({ path, label, hint, options, className, emptyAs }: Base & { options: readonly Opt[]; emptyAs?: "remove" | "empty" }) {
  const c = useCfg();
  const v = c.value(path);
  const list = Array.isArray(v) ? (v as unknown[]).map(String) : [];
  const extra = list.filter((x) => !options.some((o) => o.value === x)).map((x) => ({ value: x, label: `${x} (not in the data)` }));
  return (
    <Wrap path={path}>
      <div className={["field", className ?? ""].filter(Boolean).join(" ")}>
        <span className="field-label">{label}</span>
        <Chips
          label={typeof label === "string" ? label : joinPath(splitPath(path))}
          items={[...options, ...extra].map((o) => ({ value: o.value, label: o.label }))}
          selected={list}
          onToggle={(val, on) => {
            const next = on ? [...list, val] : list.filter((x) => x !== val);
            const order = [...options.map((o) => o.value), ...extra.map((o) => o.value)];
            next.sort((a, b) => order.indexOf(a) - order.indexOf(b));
            c.set(path, next.length || emptyAs !== "remove" ? next : undefined);
          }}
        />
        {hint ? <span className="hint">{hint}</span> : null}
        <IssueLines issues={c.issues(path, true)} />
      </div>
    </Wrap>
  );
}

/** Number list as removable chips plus an add box (doses, increments, thresholds…). */
export function NumberListEditor({
  values,
  onChange,
  label,
  unit,
  signed,
  sort = true,
  format,
}: {
  values: number[];
  onChange: (next: number[]) => void;
  label: string;
  unit?: string | null;
  signed?: "+" | "−";
  sort?: boolean;
  format?: (v: number) => string;
}) {
  const [adding, setAdding] = useState<number | null>(null);
  const show = (v: number) => (format ? format(v) : v === 0 ? "0" : `${signed ?? ""}${String(v).replace("-", MINUS)}`);
  const add = () => {
    if (adding === null) return;
    const next = values.includes(adding) ? values : [...values, adding];
    onChange(sort ? [...next].sort((a, b) => a - b) : next);
    setAdding(null);
  };
  return (
    <div className="row" style={{ gap: 6 }}>
      <Chips
        label={label}
        items={values.map((v, i) => ({ value: `${i}`, label: show(v) }))}
        onRemove={(k) => onChange(values.filter((_, i) => String(i) !== k))}
      />
      <span className="row" style={{ gap: 4 }}>
        <NumberField value={adding} onChange={setAdding} nullable unit={unit} label={`Add to ${label}`} style={{ width: "6em" }} onKeyDown={(e) => e.key === "Enter" && add()} />
        <button type="button" className="btn small" onClick={add} disabled={adding === null}>
          Add
        </button>
      </span>
    </div>
  );
}

/** A number-list config field (chips + add). */
export function CfgNumberList({ path, label, hint, unit, signed, sort, className }: Base & { unit?: string | null; signed?: "+" | "−"; sort?: boolean }) {
  const c = useCfg();
  const v = c.value(path);
  const values = Array.isArray(v) ? (v as unknown[]).filter((x): x is number => typeof x === "number" && Number.isFinite(x)) : [];
  return (
    <Wrap path={path}>
      <div className={["field", className ?? ""].filter(Boolean).join(" ")}>
        <span className="field-label">{label}</span>
        <NumberListEditor values={values} onChange={(n) => c.set(path, n)} label={typeof label === "string" ? label : joinPath(splitPath(path))} unit={unit} signed={signed} sort={sort} />
        {hint ? <span className="hint">{hint}</span> : null}
        <IssueLines issues={c.issues(path, true)} />
      </div>
    </Wrap>
  );
}

/** String list as a textarea, one item per line (limitations, caveats). */
export function CfgLines({ path, label, hint, rows = 4, className }: Base & { rows?: number }) {
  const c = useCfg();
  const v = c.value(path);
  const list = Array.isArray(v) ? (v as unknown[]).map(String) : [];
  const [text, setText] = useState(list.join("\n"));
  const [base, setBase] = useState(list.join("\n"));
  if (list.join("\n") !== base) {
    // The draft changed underneath (discard, reload): follow it.
    setBase(list.join("\n"));
    setText(list.join("\n"));
  }
  return (
    <Wrap path={path}>
      <Field label={label} hint={hint ?? "One per line."} issues={c.issues(path, true)} className={className}>
        {(id, describedBy) => (
          <textarea
            id={id}
            rows={rows}
            aria-describedby={describedBy}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => {
              const next = text
                .split("\n")
                .map((s) => s.trim())
                .filter(Boolean);
              setBase(next.join("\n"));
              c.set(path, next.length ? next : undefined);
            }}
          />
        )}
      </Field>
    </Wrap>
  );
}

/** JSON value edited as text (priors, albedo_map); invalid JSON is reported, not saved. */
export function CfgJson({ path, label, hint, rows = 3, className }: Base & { rows?: number }) {
  const c = useCfg();
  const v = c.own(path);
  const shown = v === undefined || v === null ? "" : JSON.stringify(v);
  const [text, setText] = useState(shown);
  const [base, setBase] = useState(shown);
  const [error, setError] = useState<string | null>(null);
  if (shown !== base) {
    setBase(shown);
    setText(shown);
  }
  const local: Issue[] = error ? [{ level: "error", path: typeof path === "string" ? path : joinPath(path), code: "json", message: error }] : [];
  return (
    <Wrap path={path}>
      <Field label={label} hint={hint} issues={[...local, ...c.issues(path, true)]} className={className}>
        {(id, describedBy, invalid) => (
          <textarea
            id={id}
            rows={rows}
            className="mono"
            aria-describedby={describedBy}
            aria-invalid={invalid || undefined}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => {
              if (text.trim() === "") {
                setError(null);
                c.set(path, undefined);
                return;
              }
              try {
                const parsed: unknown = JSON.parse(text);
                setError(null);
                c.set(path, parsed);
              } catch {
                setError("Not valid JSON (e.g. {\"from\": \"auto\", \"to\": [0.08, 0.25]})");
              }
            }}
          />
        )}
      </Field>
    </Wrap>
  );
}

/** Issue messages (level word + message) for controls that are not a foundation Field. */
export function IssueLines({ issues }: { issues: readonly Issue[] }) {
  if (!issues.length) return null;
  return (
    <div>
      {issues.map((i, k) => (
        <div key={k} className="issue" data-level={i.level} data-issue-path={i.path}>
          {i.level === "error" ? "Error: " : i.level === "warn" ? "Warning: " : ""}
          {i.message}
        </div>
      ))}
    </div>
  );
}
