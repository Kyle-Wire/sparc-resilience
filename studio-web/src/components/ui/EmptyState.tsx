import { useState, type ReactNode } from "react";
import { ApiError, errorMessage, request } from "../../api/client";
import type { Action, Job } from "../../api/types";
import { navigate } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { Button } from "./Button";

function isJob(v: unknown): v is Job {
  return !!v && typeof v === "object" && typeof (v as Job).id === "string" && (v as Job).id.startsWith("j_") && typeof (v as Job).status === "string";
}

/** The job a remedy started, if its response carries one ({…Job} or {job: Job}). */
export function jobFromResponse(v: unknown): Job | null {
  if (isJob(v)) return v;
  if (v && typeof v === "object" && isJob((v as { job?: unknown }).job)) return (v as { job: Job }).job;
  return null;
}

/**
 * Execute a server-provided Action (api.md §0.3).
 * - A path outside /api (or `kind: "open"` with a GET) navigates within the app.
 * - Otherwise the request is sent (POST by default, with `body`); a returned Job is added
 *   to the job tray and announced with a link to Mission Control.
 */
export async function runAction(action: Action): Promise<unknown> {
  const path = action.path;
  if (!path) return null;
  const method = action.method ?? (path.startsWith("/api/") ? "POST" : "GET");
  if (method === "GET" && !path.startsWith("/api/")) {
    navigate(path);
    return null;
  }
  if (method === "GET" && /\/(raw|export|logs\/raw)\b/.test(path)) {
    window.open(path, "_blank", "noopener");
    return null;
  }
  const res = await request<unknown>(method, path, method === "GET" ? {} : { body: action.body ?? {} });
  const job = jobFromResponse(res);
  if (job) {
    useJobs.getState().upsert(job);
    toast("info", `${job.label || action.label} started`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
  }
  return res;
}

export function ActionButton({
  action,
  size = "normal",
  variant = "primary",
  onDone,
}: {
  action: Action;
  size?: "normal" | "small";
  variant?: "primary" | "default";
  onDone?: (result: unknown) => void;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <Button
      variant={variant}
      size={size}
      busy={busy}
      onClick={async () => {
        setBusy(true);
        try {
          const r = await runAction(action);
          onDone?.(r);
        } catch (e) {
          toast("error", `${action.label} failed`, { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
        } finally {
          setBusy(false);
        }
      }}
    >
      {action.label}
    </Button>
  );
}

export type EmptyStateProps = {
  title?: ReactNode;
  body?: ReactNode;
  /** An error to explain (ApiError message, producing stage, and its action). */
  error?: unknown;
  /** One-click remedy; defaults to the ApiError's action. */
  action?: Action | null;
  onAction?: (result: unknown) => void;
  children?: ReactNode;
};

function stageText(producedBy: unknown): string | null {
  if (typeof producedBy !== "string" || !producedBy) return null;
  const [kind, name] = producedBy.split(":");
  if (kind === "stage") return `stage ${name}`;
  if (kind === "post") return `the ${name} post-run action`;
  if (kind === "study") return `the ${name} study`;
  if (kind === "studio") return `Studio (${name})`;
  return producedBy;
}

/**
 * Typed empty state: names what is missing and what produces it, and renders the server's
 * remedy as a button ("Resume to compute S6", "Run planner pack").
 */
export function EmptyState({ title, body, error, action, onAction, children }: EmptyStateProps) {
  const err = error instanceof ApiError ? error : null;
  const act = action !== undefined ? action : err?.action ?? null;
  let t: ReactNode = title;
  let b: ReactNode = body;
  if (err && t === undefined) {
    if (err.code === "output_missing") t = "Not in this run yet";
    else if (err.code === "unauthorized") t = "Session expired";
    else if (err.code === "network") t = "Studio server not reachable";
    else if (err.status === 404) t = "Not found";
    else t = "Could not load this view";
  }
  if (err && b === undefined) {
    const producer = stageText(err.detail?.produced_by);
    b = (
      <>
        {err.message}
        {producer ? ` It is produced by ${producer}.` : ""}
        {err.code === "unauthorized" ? " Open the link printed by `sparc studio` in the terminal to sign in again." : ""}
      </>
    );
  } else if (!err && error && b === undefined) {
    b = errorMessage(error);
  }
  return (
    <div className="empty" role={err || error ? "alert" : "status"}>
      {t ? <div className="empty-title">{t}</div> : null}
      {b ? <div className="empty-body">{b}</div> : null}
      {children}
      {act ? <ActionButton action={act} onDone={onAction} /> : null}
    </div>
  );
}
