// Shared bits of the studies pages: cost and requirement wording, launch announcements, the
// attach/detach toggle, the run picker and small layout helpers.
import { useState, type ReactNode } from "react";
import { errorMessage, isApiError } from "../../../api/client";
import { invalidate } from "../../../api/resource";
import { attachStudy, type StudyCost, type StudyEstimate } from "../../../api/studies";
import type { Job, RunSummary } from "../../../api/types";
import { Button } from "../../../components/ui/Button";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { fmtDate, fmtDurationRange, fmtNum } from "../../../theme/format";

/** "≈ 4–6 min" for a status-row estimate. */
export function estimateText(e: StudyEstimate | null | undefined): string | null {
  if (!e) return null;
  const range = fmtDurationRange(e.est_lo, e.est_hi);
  return range && range !== "—" ? `≈ ${range}` : null;
}

/** "≈ 12–18 min · peak 2.1 GB RAM · 0.3 GB disk · 3 child runs". */
export function costText(c: StudyCost): string {
  const parts = [estimateText(c) ?? "time unknown"];
  if (Number.isFinite(c.est_peak_rss_gb) && c.est_peak_rss_gb > 0) parts.push(`peak ${fmtNum(c.est_peak_rss_gb, 1)} GB RAM`);
  if (Number.isFinite(c.est_disk_gb) && c.est_disk_gb > 0) parts.push(`${fmtNum(c.est_disk_gb, c.est_disk_gb < 0.1 ? 2 : 1)} GB disk`);
  if (c.n_children > 0) parts.push(`${c.n_children} child run${c.n_children === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

const REQUIREMENTS: Record<string, string> = {
  checkpoint: "a checkpoint (the run must reach S3 and keep checkpoint.pkl)",
  manifest: "a finished run (its manifest.json)",
  "manifest.config": "the run's config recorded in its manifest",
  predictions: "the run's predictions (predictions.parquet, written by S2–S3)",
  "data.path": "the run's input data file (a run made from an in-memory table has none to re-read)",
  "planner.layers": "people and land-cover layers (planner.layers; fetch them in Setup → Inputs)",
  "roles.canopy": "a canopy role in the physics roles (Setup → Physics)",
  "roles.impervious": "an impervious role in the physics roles (Setup → Physics)",
  "physics.roles.canopy": "a canopy role in the physics roles (Setup → Physics)",
  "physics.roles.impervious": "an impervious role in the physics roles (Setup → Physics)",
  crs: "a coordinate reference system for the data (Setup → Data)",
  config_dir: "the run's config folder (launch snapshot or import record)",
  file_input: "a file-based run (reproduction refuses runs made from an in-memory table)",
  studies: "at least one finished placebo, simulation-check or multiverse study",
  S4: "response curves (stage S4)",
  S5: "configured scenarios (stage S5)",
};

/** Plain wording for a requirement the server reports missing (unknown ones are shown as given). */
export function requirementText(code: string): string {
  return REQUIREMENTS[code] ?? code;
}

/** Missing requirements carried by a `422 requirements` error (`detail.missing`). */
export function missingFromError(e: unknown): string[] {
  if (!isApiError(e)) return [];
  const m = e.detail?.missing;
  return Array.isArray(m) ? m.map(String) : [];
}

/** Put a started job in the tray and say so, with a link to Mission Control. */
export function announceJob(job: Job, what: string): void {
  useJobs.getState().upsert(job);
  toast("info", `${what} started`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
}

/** Attach or detach a study (feeds the run's uncertainty report; attaching re-runs it). */
export function AttachToggle({ studyId, runId, attached, label, onDone }: { studyId: string; runId: string; attached: boolean; label: string; onDone?: () => void }) {
  const [busy, setBusy] = useState(false);
  const toggle = async () => {
    setBusy(true);
    try {
      const r = await attachStudy(studyId, runId, !attached);
      if (r.job) announceJob(r.job, "Uncertainty report");
      else toast("success", attached ? `${label} detached` : `${label} attached`);
      invalidate(`run:${runId}:studies`);
      invalidate(`run:${runId}:views`);
      invalidate(`study:${studyId}`);
      invalidate("studies");
      onDone?.();
    } catch (e) {
      toast("error", `Could not ${attached ? "detach" : "attach"} ${label}`, { body: errorMessage(e), action: isApiError(e) && e.action ? e.action : undefined });
    } finally {
      setBusy(false);
    }
  };
  return (
    <Button
      size="small"
      busy={busy}
      aria-pressed={attached}
      title={attached ? "Feeds this run's uncertainty report. Detach to stop." : "Attach to feed this run's uncertainty report (re-runs it)."}
      onClick={() => void toggle()}
    >
      {attached ? "Attached" : "Attach to run"}
    </Button>
  );
}

export function runLabel(r: Pick<RunSummary, "id" | "label" | "mode" | "created_utc" | "coarse_m">): string {
  const mode = r.mode === "coarse" && r.coarse_m ? `coarse ${Math.round(r.coarse_m)} m` : r.mode;
  return `${r.label || r.id} · ${mode} · ${fmtDate(r.created_utc)}`;
}

/** Run select (value = run id). */
export function RunPicker({ runs, value, onChange, id, label = "Run" }: { runs: RunSummary[]; value: string | null; onChange: (rid: string) => void; id?: string; label?: string }) {
  return (
    <select id={id} aria-label={id ? undefined : label} value={value ?? ""} onChange={(e) => e.target.value && onChange(e.target.value)}>
      {value ? null : <option value="">Choose a run…</option>}
      {runs.map((r) => (
        <option key={r.id} value={r.id}>
          {runLabel(r)}
        </option>
      ))}
    </select>
  );
}

/** A labelled control in the launch forms (the control gets `id`). */
export function FormField({ id, label, hint, children }: { id: string; label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}

/** A checkbox with its label. */
export function Check({ checked, onChange, children, disabled, title }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean; title?: string }) {
  return (
    <label title={title}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}
