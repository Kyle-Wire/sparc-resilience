// Launch panel of a study or post-run action (SPEC §8, api.md §8–9): the kind's form, what the
// form still needs, the server's missing requirements (the launch stays disabled while any is
// listed), the cost estimate (`POST /api/studies/estimate`, debounced) and the Launch button.
// A `422 requirements` refusal adds the server's `detail.missing` to the list.
import { useEffect, useMemo, useState } from "react";
import { errorMessage, isApiError, type ApiError } from "../../../api/client";
import { invalidate } from "../../../api/resource";
import {
  estimateStudy,
  KIND_LABELS,
  launchStudy,
  useThreadsHeavy,
  type LaunchableKind,
  type LaunchResult,
  type StudyCost,
} from "../../../api/studies";
import { Button } from "../../../components/ui/Button";
import { ActionButton } from "../../../components/ui/EmptyState";
import { defaultForm, paramProblems, paramsFor, type Forms } from "../model/params";
import { announceJob, costText, missingFromError, requirementText } from "./common";
import { KindForm, type FormContext } from "./KindForms";

export type LaunchPanelProps<K extends LaunchableKind> = {
  kind: K;
  rid: string | null;
  pid: string | null;
  /** The status row's requirements; `ok: false` disables the launch. */
  requirements?: { ok: boolean; missing: string[] };
  ctx: FormContext;
  /** Start from these form values (e.g. a study's previous params). */
  initial?: Forms[K];
  launchLabel?: string;
  onLaunched?: (r: LaunchResult) => void;
};

/** The missing requirements, worded, with the server's codes. */
export function RequirementList({ missing }: { missing: string[] }) {
  if (!missing.length) return null;
  return (
    <div className="vt-req callout" data-tone="crit" role="note" aria-label="Missing requirements">
      <strong>Needs before it can run:</strong>
      <ul data-testid="missing-requirements">
        {missing.map((m) => (
          <li key={m} data-requirement={m}>
            {requirementText(m)} {requirementText(m) !== m ? <span className="sx-mono">({m})</span> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function LaunchPanel<K extends LaunchableKind>({ kind, rid, pid, requirements, ctx, initial, launchLabel, onLaunched }: LaunchPanelProps<K>) {
  const [form, setForm] = useState<Forms[K]>(() => initial ?? defaultForm(kind));
  const [busy, setBusy] = useState(false);
  const [serverMissing, setServerMissing] = useState<string[]>([]);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [cost, setCost] = useState<StudyCost | null>(null);
  const [costError, setCostError] = useState<string | null>(null);
  const threadsHeavy = useThreadsHeavy();
  const fullCtx: FormContext = { ...ctx, threadsHeavy: ctx.threadsHeavy ?? threadsHeavy };
  const params = useMemo(() => paramsFor(kind, form), [kind, form]);
  const paramsKey = JSON.stringify(params);
  const problems = paramProblems(kind, form, fullCtx.threadsHeavy);
  const missing = [...new Set([...(requirements && !requirements.ok ? requirements.missing : []), ...serverMissing])];
  const blocked = (requirements !== undefined && !requirements.ok) || serverMissing.length > 0;

  // Cost estimate for the current params (debounced; skipped while the form is invalid or blocked).
  useEffect(() => {
    if (problems.length || blocked || (!rid && kind !== "benchmark")) {
      setCost(null);
      return;
    }
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      // The benchmark is project-wide: its estimate takes no run.
      estimateStudy(kind, kind === "benchmark" ? null : rid, JSON.parse(paramsKey) as object, ctrl.signal).then(
        (c) => {
          setCost(c);
          setCostError(null);
        },
        (e: unknown) => {
          if (isApiError(e) && e.code === "aborted") return;
          setCost(null);
          setCostError(errorMessage(e));
        },
      );
    }, 400);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [kind, rid, paramsKey, problems.length, blocked]);

  // A refusal's missing list holds until the server's requirements change (refetched row).
  const reqKey = requirements ? `${requirements.ok}:${requirements.missing.join(",")}` : "";
  useEffect(() => setServerMissing([]), [reqKey]);

  const launch = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await launchStudy(kind, { rid, pid }, params);
      announceJob(r.job, KIND_LABELS[kind]);
      if (rid) invalidate(`run:${rid}:studies`);
      invalidate("studies");
      if (pid) invalidate(`project:${pid}:studies`);
      onLaunched?.(r);
    } catch (e) {
      const miss = missingFromError(e);
      if (miss.length) setServerMissing(miss);
      else setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack" style={{ gap: 10 }} data-kind={kind}>
      <KindForm kind={kind} form={form} onChange={(f) => setForm(f)} ctx={fullCtx} />
      <RequirementList missing={missing} />
      {problems.length ? (
        <ul className="sx-problems" aria-label="Fix before launching">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      ) : null}
      {cost ? (
        <p className="vt-cost" aria-label="Estimated cost">
          Estimated cost: <b>{costText(cost)}</b>
        </p>
      ) : costError ? (
        <p className="vt-cost">Estimate unavailable: {costError}</p>
      ) : null}
      {error ? (
        <div className="callout" data-tone="crit" role="alert">
          {error.message}
          {isApiError(error) && error.validationErrors.length ? (
            <ul>
              {error.validationErrors.map((v, i) => (
                <li key={i}>
                  <span className="sx-mono">{v.path}</span>: {v.message}
                </li>
              ))}
            </ul>
          ) : null}
          {isApiError(error) && error.action ? (
            <div style={{ marginTop: 6 }}>
              <ActionButton action={error.action} size="small" variant="default" />
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="sx-actions">
        <Button variant="primary" icon="play" busy={busy} disabled={blocked || problems.length > 0} onClick={() => void launch()} data-launch={kind}>
          {launchLabel ?? `Run ${KIND_LABELS[kind].toLowerCase()}`}
        </Button>
        {blocked ? <span className="cap">Disabled until the requirements above are met.</span> : null}
      </div>
    </div>
  );
}
