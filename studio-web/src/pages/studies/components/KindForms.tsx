// Launch-form editors of every study and post-run action kind (SPEC §8): baselines model
// subset, planner package and thresholds, emulator patches, uncertainty sources, placebo
// kinds/coarse/seed, the simcheck design editor, the multiverse variant checklist with custom
// dotted-override variants, reproduction stages and tolerances, benchmark settings. Each
// editor is controlled (form state from model/params.ts); `paramsFor` builds the wire params.
import { useId, type ReactNode } from "react";
import { PLACEBO_KINDS, SIM_GENERATORS, type LaunchableKind, type SimGenerator, type Study } from "../../../api/studies";
import { Button } from "../../../components/ui/Button";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg } from "../../../components/ui/Seg";
import { fmtDate } from "../../../theme/format";
import {
  BASELINE_MODELS,
  DEFAULT_DESIGN,
  designTotal,
  GENERATOR_LABELS,
  PLACEBO_LABELS,
  REPRODUCE_STAGES,
  type BaselinesForm,
  type BenchmarkForm,
  type EmulatorForm,
  type Forms,
  type MultiverseForm,
  type PlaceboForm,
  type PlannerForm,
  type ReproduceForm,
  type SimcheckForm,
  type UncertaintyForm,
} from "../model/params";
import { BASELINE_VARIANT, BUILTIN_VARIANTS } from "../model/multiverse";
import { Check, FormField } from "./common";
import { CustomVariants } from "./CustomVariants";

/** What the editors may offer besides the form: the project's studies, the heavy thread budget, configured scenario names. */
export type FormContext = {
  studies: Study[];
  threadsHeavy: number | null;
  /** Configured scenario names of the run (planner package suggestions). */
  packages?: string[];
  /** The run whose studies are listed first (uncertainty sources, simcheck continue). */
  runId?: string | null;
};

type EditorProps<K extends LaunchableKind> = { form: Forms[K]; onChange: (f: Forms[K]) => void; ctx: FormContext };

const toggle = <T,>(list: readonly T[], v: T, on: boolean): T[] => (on ? (list.includes(v) ? [...list] : [...list, v]) : list.filter((x) => x !== v));

function studyOption(s: Study): string {
  const label = typeof s.summary?.label === "string" ? s.summary.label : null;
  return `${label ?? s.id} · ${s.status} · ${fmtDate(s.created_utc)}${s.target_run_id ? ` · run ${s.target_run_id}` : ""}`;
}

/** Studies of one kind usable as a source: finished first, this run's first. */
function studiesOf(ctx: FormContext, kind: string): Study[] {
  return ctx.studies
    .filter((s) => s.kind === kind)
    .sort((a, b) => Number(b.target_run_id === ctx.runId) - Number(a.target_run_id === ctx.runId) || b.created_utc.localeCompare(a.created_utc));
}

function CellSize({ id, fine, coarse, onChange, hint }: { id: string; fine: boolean; coarse: number; onChange: (fine: boolean, coarse: number) => void; hint?: ReactNode }) {
  return (
    <FormField id={id} label="Grid for the re-fits" hint={hint ?? "Coarse cells make each re-fit much faster; the fine grid matches the run."}>
      <div className="row">
        <NumberField id={id} value={coarse} min={10} max={2000} step={10} unit="m" onChange={(v) => onChange(fine, v ?? coarse)} disabled={fine} style={{ width: "6em" }} />
        <Check checked={fine} onChange={(v) => onChange(v, coarse)}>
          fine grid (no coarsening)
        </Check>
      </div>
    </FormField>
  );
}

function Pool({ id, workers, threads, threadsHeavy, onChange }: { id: string; workers: number; threads: number; threadsHeavy: number | null; onChange: (w: number, t: number) => void }) {
  const over = threadsHeavy !== null && workers * threads > threadsHeavy;
  return (
    <>
      <FormField id={`${id}-w`} label="Workers" hint={threadsHeavy !== null ? `workers × threads ≤ ${threadsHeavy} (heavy-job budget)` : undefined}>
        <NumberField id={`${id}-w`} value={workers} min={1} max={64} step={1} onChange={(v) => onChange(Math.round(v ?? 1), threads)} aria-invalid={over || undefined} />
      </FormField>
      <FormField id={`${id}-t`} label="Threads per worker">
        <NumberField id={`${id}-t`} value={threads} min={1} max={64} step={1} onChange={(v) => onChange(workers, Math.round(v ?? 1))} aria-invalid={over || undefined} />
      </FormField>
    </>
  );
}

// ---------------------------------------------------------------- per kind

function BaselinesEditor({ form, onChange }: EditorProps<"baselines">) {
  const f = form as BaselinesForm;
  return (
    <fieldset className="sx-checks column" aria-label="Baseline models">
      <legend className="cap">Reference models refitted on the run's folds</legend>
      {BASELINE_MODELS.map((m) => (
        <Check key={m.id} checked={f.models.includes(m.id)} onChange={(on) => onChange({ models: toggle(f.models, m.id, on) })}>
          {m.label} <span className="sx-mono">{m.id}</span>
        </Check>
      ))}
    </fieldset>
  );
}

function PlannerEditor({ form, onChange, ctx }: EditorProps<"planner">) {
  const f = form as PlannerForm;
  const id = useId();
  return (
    <div className="sx-formgrid">
      <FormField id={`${id}-pkg`} label="Scenario package" hint="A configured scenario's slug or exact name; empty uses the headline scenario.">
        <input id={`${id}-pkg`} type="text" list={`${id}-pkgs`} value={f.package} onChange={(e) => onChange({ ...f, package: e.target.value })} />
        <datalist id={`${id}-pkgs`}>
          {(ctx.packages ?? []).map((p) => (
            <option key={p} value={p} />
          ))}
        </datalist>
      </FormField>
      <FormField id={`${id}-th`} label="Hot-day thresholds" hint="Temperatures in the target's units, comma separated; empty uses the config.">
        <input id={`${id}-th`} type="text" inputMode="decimal" value={f.thresholds} placeholder="90, 95" onChange={(e) => onChange({ ...f, thresholds: e.target.value })} />
      </FormField>
      <FormField id={`${id}-hex`} label="Hexagon sizes (m)">
        <input id={`${id}-hex`} type="text" inputMode="decimal" value={f.hex_sizes} onChange={(e) => onChange({ ...f, hex_sizes: e.target.value })} />
      </FormField>
      <Check checked={f.export} onChange={(v) => onChange({ ...f, export: v })}>
        Write GeoTIFF and GeoPackage files
      </Check>
    </div>
  );
}

function EmulatorEditor({ form, onChange }: EditorProps<"emulator">) {
  const f = form as EmulatorForm;
  const id = useId();
  return (
    <FormField id={id} label="Validation patches per lever" hint="Random patches edited exactly and with the emulator to measure its error (more = slower, tighter check).">
      <NumberField id={id} value={f.patches} min={1} max={64} step={1} onChange={(v) => onChange({ patches: Math.round(v ?? 8) })} style={{ width: "6em" }} />
    </FormField>
  );
}

function UncertaintyEditor({ form, onChange, ctx }: EditorProps<"uncertainty">) {
  const f = form as UncertaintyForm;
  const id = useId();
  const mv = studiesOf(ctx, "multiverse");
  const sc = studiesOf(ctx, "simcheck");
  const pl = studiesOf(ctx, "placebo");
  return (
    <div className="stack" style={{ gap: 10 }}>
      <Seg
        label="Evidence sources"
        size="small"
        value={f.mode}
        onChange={(mode) => onChange({ ...f, mode })}
        options={[
          { value: "attached", label: "Studies attached to this run" },
          { value: "choose", label: "Choose studies" },
        ]}
      />
      {f.mode === "choose" ? (
        <div className="sx-formgrid">
          <FormField id={`${id}-mv`} label="Multiverse (specification)">
            <select id={`${id}-mv`} value={f.multiverse_study} onChange={(e) => onChange({ ...f, multiverse_study: e.target.value })}>
              <option value="">None</option>
              {mv.map((s) => (
                <option key={s.id} value={s.id}>
                  {studyOption(s)}
                </option>
              ))}
            </select>
          </FormField>
          <FormField id={`${id}-pl`} label="Placebo suite">
            <select id={`${id}-pl`} value={f.placebo_study} onChange={(e) => onChange({ ...f, placebo_study: e.target.value })}>
              <option value="">None</option>
              {pl.map((s) => (
                <option key={s.id} value={s.id}>
                  {studyOption(s)}
                </option>
              ))}
            </select>
          </FormField>
          <fieldset className="sx-checks column" aria-label="Simulation checks (attribution)">
            <legend className="cap">Simulation checks (attribution)</legend>
            {sc.length ? (
              sc.map((s) => (
                <Check key={s.id} checked={f.simcheck_studies.includes(s.id)} onChange={(on) => onChange({ ...f, simcheck_studies: toggle(f.simcheck_studies, s.id, on) })}>
                  {studyOption(s)}
                </Check>
              ))
            ) : (
              <span className="cap">No simulation-check studies in this project.</span>
            )}
          </fieldset>
        </div>
      ) : (
        <p className="cap">The report reads the multiverse, simulation-check and placebo studies attached to this run (attach them on their cards).</p>
      )}
      <Check checked={f.real_r2_gate} onChange={(v) => onChange({ ...f, real_r2_gate: v })} title="Only keep simulation replicates whose held-out R² is close to the real run's">
        Gate simulation replicates on the real run's R²
      </Check>
    </div>
  );
}

function PlaceboEditor({ form, onChange }: EditorProps<"placebo">) {
  const f = form as PlaceboForm;
  const id = useId();
  return (
    <div className="stack" style={{ gap: 10 }}>
      <fieldset className="sx-checks column" aria-label="Placebo kinds">
        <legend className="cap">Placebo kinds (one re-fit each)</legend>
        {PLACEBO_KINDS.map((k) => (
          <Check key={k} checked={f.kinds.includes(k)} onChange={(on) => onChange({ ...f, kinds: PLACEBO_KINDS.filter((x) => (x === k ? on : f.kinds.includes(x))) })}>
            {PLACEBO_LABELS[k]} <span className="sx-mono">{k}</span>
          </Check>
        ))}
      </fieldset>
      <div className="sx-formgrid">
        <CellSize id={`${id}-cell`} fine={f.fine} coarse={f.coarse_m} onChange={(fine, coarse_m) => onChange({ ...f, fine, coarse_m })} />
        <FormField id={`${id}-seed`} label="Seed">
          <NumberField id={`${id}-seed`} value={f.seed} min={0} step={1} onChange={(v) => onChange({ ...f, seed: Math.round(v ?? 0) })} />
        </FormField>
        <FormField id={`${id}-range`} label="Random-field range" hint="Correlation length of the fake layer (grf only).">
          <NumberField id={`${id}-range`} value={f.grf_range_m} min={30} step={50} unit="m" onChange={(v) => onChange({ ...f, grf_range_m: v ?? 600 })} disabled={!f.kinds.includes("grf")} />
        </FormField>
      </div>
    </div>
  );
}

/** Simcheck design editor: replicates per generator, presets, the total, and continuing a study. */
export function SimcheckEditor({ form, onChange, ctx }: EditorProps<"simcheck">) {
  const f = form as SimcheckForm;
  const id = useId();
  const total = designTotal(f.design);
  const setN = (g: SimGenerator, n: number) => onChange({ ...f, design: { ...f.design, [g]: Math.max(0, Math.round(n)) } });
  const prior = studiesOf(ctx, "simcheck").filter((s) => !ctx.runId || s.target_run_id === ctx.runId);
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="tablewrap">
        <table className="tbl" aria-label="Simulation design (replicates per generator)">
          <thead>
            <tr>
              <th scope="col">Generator</th>
              <th scope="col" className="r">
                Replicates
              </th>
            </tr>
          </thead>
          <tbody>
            {SIM_GENERATORS.map((g) => (
              <tr key={g}>
                <td>
                  <label htmlFor={`${id}-${g}`}>{GENERATOR_LABELS[g]}</label> <span className="sx-mono">{g}</span>
                </td>
                <td className="r">
                  <NumberField id={`${id}-${g}`} value={f.design[g] ?? 0} min={0} max={500} step={1} onChange={(v) => setN(g, v ?? 0)} style={{ width: "5em" }} />
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row">Total</th>
              <td className="r num" data-testid="design-total">
                {total}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      <div className="sx-actions">
        <Button size="small" onClick={() => onChange({ ...f, design: { ...DEFAULT_DESIGN } })}>
          CLI default ({designTotal(DEFAULT_DESIGN)})
        </Button>
        <Button size="small" onClick={() => onChange({ ...f, design: Object.fromEntries(SIM_GENERATORS.map((g) => [g, 2])) as Record<SimGenerator, number> })}>
          Quick check (2 each)
        </Button>
        <Button size="small" onClick={() => onChange({ ...f, design: Object.fromEntries(SIM_GENERATORS.map((g) => [g, g === "null" ? 20 : 0])) as Record<SimGenerator, number> })}>
          False positives only
        </Button>
      </div>
      <div className="sx-formgrid">
        <CellSize id={`${id}-cell`} fine={f.fine} coarse={f.coarse_m} onChange={(fine, coarse_m) => onChange({ ...f, fine, coarse_m })} hint="Each replicate re-fits the stack on a synthetic target; coarse cells keep it affordable." />
        <FormField id={`${id}-ep`} label="Epochs">
          <NumberField id={`${id}-ep`} value={f.epochs} min={1} max={5000} step={10} onChange={(v) => onChange({ ...f, epochs: Math.round(v ?? 200) })} />
        </FormField>
        <Pool id={`${id}-pool`} workers={f.workers} threads={f.threads} threadsHeavy={ctx.threadsHeavy} onChange={(workers, threads) => onChange({ ...f, workers, threads })} />
      </div>
      {prior.length ? (
        <FormField id={`${id}-cont`} label="Continue a study" hint="Adds the design's missing replicates to that study's simcheck.jsonl instead of starting a new one.">
          <select id={`${id}-cont`} value={f.continue_study_id} onChange={(e) => onChange({ ...f, continue_study_id: e.target.value })}>
            <option value="">New study</option>
            {prior.map((s) => (
              <option key={s.id} value={s.id}>
                {studyOption(s)}
              </option>
            ))}
          </select>
        </FormField>
      ) : null}
    </div>
  );
}

/** Multiverse: built-in variant checklist (baseline always included) plus custom dotted-override variants. */
export function MultiverseEditor({ form, onChange, ctx }: EditorProps<"multiverse">) {
  const f = form as MultiverseForm;
  const id = useId();
  const all = BUILTIN_VARIANTS.map((v) => v.name);
  return (
    <div className="stack" style={{ gap: 10 }}>
      <fieldset className="sx-checks column" aria-label="Built-in variants">
        <legend className="cap">Built-in variants (one re-fit each)</legend>
        {BUILTIN_VARIANTS.map((v) => (
          <Check
            key={v.name}
            checked={v.name === BASELINE_VARIANT || f.variants.includes(v.name)}
            disabled={v.name === BASELINE_VARIANT}
            onChange={(on) => onChange({ ...f, variants: all.filter((n) => n === BASELINE_VARIANT || (n === v.name ? on : f.variants.includes(n))) })}
          >
            {v.label} <span className="cap">({v.changes})</span>
          </Check>
        ))}
        <div className="sx-actions">
          <Button size="small" onClick={() => onChange({ ...f, variants: [...all] })}>
            All
          </Button>
          <Button size="small" onClick={() => onChange({ ...f, variants: [BASELINE_VARIANT] })}>
            Baseline only
          </Button>
        </div>
      </fieldset>
      <section aria-label="Custom variants" className="stack" style={{ gap: 6 }}>
        <p className="cap" style={{ margin: 0 }}>
          Custom variants change any config key of the run's snapshot, written as a dotted path (<span className="sx-mono">cv.block_m</span>,{" "}
          <span className="sx-mono">models.gwrf</span>).
        </p>
        <CustomVariants value={f.custom} onChange={(custom) => onChange({ ...f, custom })} />
      </section>
      <div className="sx-formgrid">
        <CellSize id={`${id}-cell`} fine={f.fine} coarse={f.coarse_m} onChange={(fine, coarse_m) => onChange({ ...f, fine, coarse_m })} />
        <Pool id={`${id}-pool`} workers={f.workers} threads={f.threads} threadsHeavy={ctx.threadsHeavy} onChange={(workers, threads) => onChange({ ...f, workers, threads })} />
      </div>
    </div>
  );
}

function ReproduceEditor({ form, onChange }: EditorProps<"reproduce">) {
  const f = form as ReproduceForm;
  const id = useId();
  return (
    <div className="stack" style={{ gap: 10 }}>
      <fieldset className="sx-checks" aria-label="Stages to re-run">
        <legend className="cap">Stages re-run from the run's launch snapshot</legend>
        {REPRODUCE_STAGES.map((s) => (
          <Check key={s} checked={s === "S0" || f.stages.includes(s)} disabled={s === "S0"} onChange={(on) => onChange({ ...f, stages: toggle(f.stages, s, on) })}>
            {s}
          </Check>
        ))}
      </fieldset>
      <div className="sx-formgrid">
        <FormField id={`${id}-r2`} label="R² tolerance">
          <NumberField id={`${id}-r2`} value={f.tol_r2} min={0} max={1} step={0.005} onChange={(v) => onChange({ ...f, tol_r2: v ?? 0.01 })} />
        </FormField>
        <FormField id={`${id}-eff`} label="Effect tolerance (relative)">
          <NumberField id={`${id}-eff`} value={f.tol_effect} min={0} max={1} step={0.01} onChange={(v) => onChange({ ...f, tol_effect: v ?? 0.05 })} />
        </FormField>
      </div>
    </div>
  );
}

function BenchmarkEditor({ form, onChange }: EditorProps<"benchmark">) {
  const f = form as BenchmarkForm;
  const id = useId();
  return (
    <div className="sx-formgrid">
      <FormField id={`${id}-n`} label="City size (cells per side)">
        <NumberField id={`${id}-n`} value={f.n} min={16} max={400} step={8} onChange={(v) => onChange({ ...f, n: Math.round(v ?? 96) })} />
      </FormField>
      <FormField id={`${id}-ep`} label="Epochs">
        <NumberField id={`${id}-ep`} value={f.epochs} min={1} max={5000} step={10} onChange={(v) => onChange({ ...f, epochs: Math.round(v ?? 150) })} />
      </FormField>
      <FormField id={`${id}-seed`} label="Seed">
        <NumberField id={`${id}-seed`} value={f.seed} min={0} step={1} onChange={(v) => onChange({ ...f, seed: Math.round(v ?? 0) })} />
      </FormField>
      <Check checked={f.ab} onChange={(v) => onChange({ ...f, ab: v })}>
        Also run with Spatial+ (A/B)
      </Check>
    </div>
  );
}

/** The editor of a kind (writeup has no parameters). */
export function KindForm<K extends LaunchableKind>({ kind, form, onChange, ctx }: { kind: K } & EditorProps<K>) {
  const p = { form, onChange, ctx } as unknown;
  switch (kind) {
    case "baselines":
      return <BaselinesEditor {...(p as EditorProps<"baselines">)} />;
    case "planner":
      return <PlannerEditor {...(p as EditorProps<"planner">)} />;
    case "emulator":
      return <EmulatorEditor {...(p as EditorProps<"emulator">)} />;
    case "uncertainty":
      return <UncertaintyEditor {...(p as EditorProps<"uncertainty">)} />;
    case "placebo":
      return <PlaceboEditor {...(p as EditorProps<"placebo">)} />;
    case "simcheck":
      return <SimcheckEditor {...(p as EditorProps<"simcheck">)} />;
    case "multiverse":
      return <MultiverseEditor {...(p as EditorProps<"multiverse">)} />;
    case "reproduce":
      return <ReproduceEditor {...(p as EditorProps<"reproduce">)} />;
    case "benchmark":
      return <BenchmarkEditor {...(p as EditorProps<"benchmark">)} />;
    default:
      return <p className="cap">No parameters: methods.md and model_card.md are rewritten from the run's merged manifest.</p>;
  }
}
