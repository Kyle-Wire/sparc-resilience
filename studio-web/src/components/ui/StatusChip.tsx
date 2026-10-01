import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

/** Status token families (CSS --st-*). */
export type StatusTone = "queued" | "running" | "done" | "cached" | "skipped" | "failed" | "cancelled" | "interrupted" | "stale" | "blocked" | "missing";

export type StatusInfo = { tone: StatusTone; icon: IconName; text: string; pulse?: boolean };

/**
 * Every status word Studio shows (jobs, stages, runs, outputs, tab availability, plan nodes,
 * status-board cells, engine states) → token, icon and text. Status is never colour alone.
 */
export const STATUS: Record<string, StatusInfo> = {
  // jobs
  queued: { tone: "queued", icon: "clock", text: "queued" },
  blocked: { tone: "blocked", icon: "ban", text: "blocked" },
  starting: { tone: "running", icon: "play", text: "starting", pulse: true },
  running: { tone: "running", icon: "play", text: "running", pulse: true },
  cancelling: { tone: "cancelled", icon: "hourglass", text: "cancelling", pulse: true },
  succeeded: { tone: "done", icon: "check", text: "succeeded" },
  failed: { tone: "failed", icon: "x", text: "failed" },
  cancelled: { tone: "cancelled", icon: "stop", text: "cancelled" },
  interrupted: { tone: "interrupted", icon: "alert", text: "interrupted" },
  killed: { tone: "cancelled", icon: "stop", text: "killed" },
  // stage UI states
  planned: { tone: "queued", icon: "dot", text: "planned" },
  done: { tone: "done", icon: "check", text: "done" },
  not_reached: { tone: "skipped", icon: "minus", text: "not reached" },
  skipped: { tone: "skipped", icon: "skip", text: "skipped" },
  cached: { tone: "cached", icon: "cached", text: "cached" },
  disabled: { tone: "skipped", icon: "ban", text: "disabled" },
  not_requested: { tone: "skipped", icon: "minus", text: "not requested" },
  will_run: { tone: "queued", icon: "play", text: "will run" },
  // runs
  complete: { tone: "done", icon: "check", text: "complete" },
  partial: { tone: "interrupted", icon: "pause", text: "partial" },
  external_live: { tone: "running", icon: "activity", text: "external (live)", pulse: true },
  imported: { tone: "cached", icon: "folder", text: "imported" },
  // outputs and tabs
  present: { tone: "done", icon: "check", text: "present" },
  ready: { tone: "done", icon: "check", text: "ready" },
  writing: { tone: "running", icon: "pulse", text: "writing", pulse: true },
  missing: { tone: "missing", icon: "minus", text: "missing" },
  stale: { tone: "stale", icon: "alert", text: "stale" },
  not_run: { tone: "missing", icon: "minus", text: "not run" },
  // engine
  no_checkpoint: { tone: "missing", icon: "ban", text: "no checkpoint" },
  cold: { tone: "queued", icon: "dot", text: "cold" },
  loading: { tone: "running", icon: "hourglass", text: "loading", pulse: true },
  busy: { tone: "running", icon: "bolt", text: "busy", pulse: true },
  incompatible: { tone: "failed", icon: "alert", text: "incompatible" },
  error: { tone: "failed", icon: "x", text: "error" },
  absent: { tone: "queued", icon: "dot", text: "not started" },
  recycling: { tone: "running", icon: "refresh", text: "recycling", pulse: true },
};

export function statusInfo(status: string | null | undefined): StatusInfo {
  if (!status) return { tone: "queued", icon: "dot", text: "unknown" };
  return STATUS[status] ?? { tone: "queued", icon: "dot", text: status.replace(/_/g, " ") };
}

export type StatusChipProps = {
  status: string | null | undefined;
  /** Replaces the default text (e.g. "skipped: no budget"). */
  text?: ReactNode;
  /** Extra detail after the text: duration, percentage, reason. */
  meta?: ReactNode;
  title?: string;
};

/** Icon + text status chip coloured by the --st-* tokens. Pulses only without reduced motion. */
export function StatusChip({ status, text, meta, title }: StatusChipProps) {
  const info = statusInfo(status);
  const reduced = prefersReducedMotion();
  return (
    <span className="status" data-st={info.tone} data-pulse={info.pulse && !reduced ? "true" : undefined} title={title}>
      <Icon name={info.icon} />
      <span>{text ?? info.text}</span>
      {meta !== undefined && meta !== null && meta !== "" ? <span className="meta">{meta}</span> : null}
    </span>
  );
}

export function prefersReducedMotion(): boolean {
  try {
    return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
