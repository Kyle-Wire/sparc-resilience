// Engine status chip (SPEC §7.6): cold / loading x% ("Loading checkpoint 312/525 MB…") / ready /
// busy / incompatible / no checkpoint, with "Open engine". Opening can be refused:
// - 409 untrusted_pickle: the run was imported without pickle trust. The trust dialog names
//   the risk (checkpoint files execute code when loaded) before re-registering the run with
//   trust_pickles, then opens the engine;
// - 409 engine_memory: needed vs available memory and the holders, with the server's action
//   (evict a run or stop a job).
import { useState } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import { mutate } from "../../../api/resource";
import { openEngine, trustRunCheckpoint, useRunEngine, type EngineState, type RunEngineStatus } from "../../../api/lab";
import { Button } from "../../../components/ui/Button";
import { Dialog } from "../../../components/ui/Dialog";
import { ActionButton, runAction } from "../../../components/ui/EmptyState";
import { StatusChip } from "../../../components/ui/StatusChip";
import { useRunDetail } from "../../../layouts/resources";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { fmtNum, fmtPct } from "../../../theme/format";

export const ENGINE_TEXT: Record<EngineState, string> = {
  no_checkpoint: "no checkpoint (preview only)",
  cold: "engine cold",
  queued: "engine queued",
  loading: "engine loading",
  ready: "engine ready",
  busy: "engine busy",
  incompatible: "checkpoint incompatible",
  error: "engine error",
};

/** The run's engine state, live from `engine.status` events when there are any. */
export function useEngineState(rid: string): { state: EngineState | null; progress: number | null; status: RunEngineStatus | undefined; reload: () => Promise<void> } {
  const res = useRunEngine(rid);
  const live = useJobs((s) => s.engineByRun[rid]);
  const liveState = live?.state && live.state in ENGINE_TEXT ? (live.state as EngineState) : null;
  return { state: liveState ?? res.data?.state ?? null, progress: live?.progress ?? res.data?.progress ?? null, status: res.data, reload: res.reload };
}

/** Start loading the run into the engine host; returns a refusal to explain, if any. */
export async function requestOpen(rid: string): Promise<ApiError | null> {
  try {
    const r = await openEngine(rid);
    if (r.job) {
      useJobs.getState().upsert(r.job);
      toast("info", "Loading the run into the engine", { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
    }
    if (r.status) mutate(`run:${rid}:engine`, r.status);
    return null;
  } catch (e) {
    if (e instanceof ApiError && (e.code === "untrusted_pickle" || e.code === "engine_memory")) return e;
    toast("error", "Could not open the engine", { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
    return null;
  }
}

export function TrustDialog({ rid, error, onClose, onTrusted }: { rid: string; error: ApiError; onClose: () => void; onTrusted: () => void }) {
  const detail = useRunDetail(rid);
  const [busy, setBusy] = useState(false);
  const dir = detail.data?.header.run_dir ?? null;
  const pid = detail.data?.run.project_id ?? null;
  const trust = async () => {
    setBusy(true);
    try {
      if (error.action?.path) await runAction(error.action);
      else if (dir) await trustRunCheckpoint(dir, pid);
      else throw new Error("The run folder is not known yet; try again in a moment.");
      onTrusted();
    } catch (e) {
      toast("error", "Could not mark the checkpoint as trusted", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onClose={onClose}
      busy={busy}
      title="Trust this run's checkpoint?"
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Use preview only
          </Button>
          <Button variant="danger" busy={busy} onClick={() => void trust()} disabled={!dir && !error.action?.path}>
            {error.action?.label ?? "Trust and open the engine"}
          </Button>
        </>
      }
    >
      <p>{error.message}</p>
      <div className="callout" data-tone="crit" role="note">
        <strong>Checkpoint files execute code when they are loaded.</strong> Trust only run folders you (or your team) produced. A checkpoint from an unknown source could run anything on this
        computer.
      </div>
      {dir ? (
        <p className="cap">
          Run folder: <span className="mono">{dir}</span>
        </p>
      ) : null}
      <p className="cap">The preview (emulator) and every stored output stay available without trusting the checkpoint; exact results need it.</p>
    </Dialog>
  );
}

export function MemoryDialog({ error, onClose }: { error: ApiError; onClose: () => void }) {
  const d = (error.detail ?? {}) as { needed_gb?: number; available_gb?: number; holders?: { job_id?: string; run_id?: string; rss_gb?: number }[] };
  return (
    <Dialog open onClose={onClose} title="Not enough memory to load this run" footer={error.action ? <ActionButton action={error.action} onDone={onClose} /> : <Button onClick={onClose}>Close</Button>}>
      <p>{error.message}</p>
      <dl className="kv">
        <dt>Needed</dt>
        <dd className="num">{d.needed_gb !== undefined ? `${fmtNum(d.needed_gb, 1)} GB` : "—"}</dd>
        <dt>Available</dt>
        <dd className="num">{d.available_gb !== undefined ? `${fmtNum(d.available_gb, 1)} GB` : "—"}</dd>
      </dl>
      {d.holders?.length ? (
        <>
          <p className="cap">Using memory now:</p>
          <ul>
            {d.holders.map((h, i) => (
              <li key={i} className="num">
                {h.job_id ? `job ${h.job_id}` : `run ${h.run_id ?? "?"}`} · {fmtNum(h.rss_gb ?? null, 1)} GB
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </Dialog>
  );
}

/** The engine chip with "Open engine" (and its refusal dialogs). */
export function EngineChip({ rid }: { rid: string }) {
  const { state, progress, status, reload } = useEngineState(rid);
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const open = async () => {
    setBusy(true);
    const err = await requestOpen(rid);
    setBusy(false);
    setRefusal(err);
    void reload();
  };
  const text = state ? ENGINE_TEXT[state] : "engine";
  const meta = state === "loading" && progress !== null ? fmtPct(progress) : undefined;
  const canOpen = state === "cold" || state === "error" || (state === null && !!status);
  return (
    <span className="lab-engine row" data-state={state ?? "unknown"}>
      <StatusChip status={state ?? "absent"} text={text} meta={meta} title={status?.step ?? (status?.error ? status.error.message : undefined)} />
      {state === "loading" && status?.step ? <span className="cap">{status.step}</span> : null}
      {state === "incompatible" && status?.action ? <ActionButton action={status.action} size="small" variant="default" /> : null}
      {status?.code_match === false ? (
        <span className="cap" title="The checkpoint was written by different core code; exact results are marked stale.">
          code changed
        </span>
      ) : null}
      {canOpen ? (
        <Button size="small" busy={busy} onClick={() => void open()} title="Load the run's checkpoint for exact results">
          Open engine
        </Button>
      ) : null}
      {refusal?.code === "untrusted_pickle" ? (
        <TrustDialog
          rid={rid}
          error={refusal}
          onClose={() => setRefusal(null)}
          onTrusted={() => {
            setRefusal(null);
            void open();
          }}
        />
      ) : null}
      {refusal?.code === "engine_memory" ? <MemoryDialog error={refusal} onClose={() => setRefusal(null)} /> : null}
    </span>
  );
}
