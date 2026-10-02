// Library dialogs (SPEC §7.4, §7.12):
// - Ladder: fork N child revisions of one edit at increasing doses (tagged ladder:<id>) and
//   batch-run them;
// - Check across runs: pick project runs, see the estimated time (Σ load + exact) and peak RAM
//   from the estimate endpoint BEFORE Start is enabled; runs without checkpoints are refused
//   with their reason;
// - Promote to config: eligibility, the exact generated scenario names and the YAML diff,
//   then write a new project config version (the current run is never touched).
import { useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "../../../api/client";
import { invalidate } from "../../../api/resource";
import {
  estimateAcrossRuns,
  makeLadder,
  promoteScenario,
  startAcrossRuns,
  useProjectRunList,
  type AcrossRunsEstimate,
  type Edit,
  type PromoteResult,
} from "../../../api/lab";
import { Button } from "../../../components/ui/Button";
import { Dialog } from "../../../components/ui/Dialog";
import { Table } from "../../../components/ui/Table";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { fmtDuration, fmtNum } from "../../../theme/format";
import { MODE_INFO } from "../model/doc";
import { parseNumberList } from "../model/library";

// ---------------------------------------------------------------- ladder

export function LadderDialog({ sid, rid, edits, editIndex, onClose, onDone }: { sid: string; rid: string; edits: Edit[]; editIndex: number; onClose: () => void; onDone?: () => void }) {
  const [index, setIndex] = useState(editIndex);
  const [text, setText] = useState("5, 10, 20, 30");
  const [runNow, setRunNow] = useState(true);
  const [busy, setBusy] = useState(false);
  const edit = edits[index];
  const parsed = parseNumberList(text);
  const field = edit ? MODE_INFO[edit.mode].fieldLabel.toLowerCase() : "amount";
  const go = async () => {
    setBusy(true);
    try {
      const r = await makeLadder(sid, index, parsed.values, runNow ? rid : undefined);
      if (r.job) {
        useJobs.getState().upsert(r.job);
        toast("info", `Ladder of ${r.scenarios.length} revisions started`, { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
      } else toast("success", `Created a ladder of ${r.scenarios.length} revisions`);
      invalidate("scenarios");
      onDone?.();
      onClose();
    } catch (e) {
      toast("error", "Could not make the ladder", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onClose={onClose}
      busy={busy}
      title="Make a ladder"
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" busy={busy} disabled={!edit || parsed.values.length < 2 || parsed.bad.length > 0} onClick={() => void go()}>
            Create {parsed.values.length} revisions
          </Button>
        </>
      }
    >
      <p className="cap">Each rung is a child revision with one value of the chosen edit, tagged with the ladder's id. The library plots ΔT against the dose with its likely range.</p>
      <label className="field">
        <span className="field-label">Edit</span>
        <select value={index} onChange={(e) => setIndex(Number(e.target.value))}>
          {edits.map((e, i) => (
            <option key={i} value={i} disabled={e.mode === "per_cell"}>
              Edit {i + 1}: {e.lever} · {MODE_INFO[e.mode].label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span className="field-label">Values of the {field}</span>
        <input value={text} onChange={(e) => setText(e.target.value)} aria-invalid={parsed.bad.length > 0 || undefined} />
        <span className="hint">{parsed.bad.length ? `Not numbers: ${parsed.bad.join(", ")}` : `Comma separated, e.g. 5, 10, 20, 30.`}</span>
      </label>
      <label className="row">
        <input type="checkbox" checked={runNow} onChange={(e) => setRunNow(e.target.checked)} /> Run them exactly on this run now
      </label>
    </Dialog>
  );
}

// ---------------------------------------------------------------- check across runs

export function AcrossRunsDialog({ sid, pid, rid, onClose }: { sid: string; pid: string; rid: string; onClose: () => void }) {
  const runs = useProjectRunList(pid);
  const list = useMemo(() => (runs.data?.items ?? []).filter((r) => r.status !== "running" && r.status !== "queued"), [runs.data]);
  const [picked, setPicked] = useState<string[] | null>(null);
  const [est, setEst] = useState<{ key: string; value: AcrossRunsEstimate } | null>(null);
  const [estError, setEstError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const seq = useRef(0);
  const chosen = picked ?? list.filter((r) => (r.checkpoint_bytes ?? 0) > 0 || r.id === rid).map((r) => r.id);
  const key = [...chosen].sort().join(",");

  useEffect(() => {
    if (!chosen.length) {
      setEst(null);
      return;
    }
    const my = ++seq.current;
    setEstError(null);
    estimateAcrossRuns(sid, chosen).then(
      (value) => my === seq.current && setEst({ key, value }),
      (e: unknown) => my === seq.current && setEstError(errorMessage(e)),
    );
    // `key` stands for the chosen run ids.
  }, [sid, key]);

  const current = est && est.key === key ? est.value : null;
  const okRuns = current ? current.runs.filter((r) => r.ok).map((r) => r.run_id) : [];
  const start = async () => {
    setBusy(true);
    try {
      const job = await startAcrossRuns(sid, okRuns);
      useJobs.getState().upsert(job);
      toast("info", `Checking across ${okRuns.length} runs`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
      onClose();
    } catch (e) {
      toast("error", "Could not start the check", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  const toggle = (id: string, on: boolean) => setPicked(on ? [...chosen, id] : chosen.filter((x) => x !== id));
  return (
    <Dialog
      open
      wide
      onClose={onClose}
      busy={busy}
      title="Check across runs"
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" busy={busy} disabled={!current || okRuns.length === 0} onClick={() => void start()} data-testid="across-start">
            Start
          </Button>
        </>
      }
    >
      <p className="cap">
        Evaluates this scenario on each chosen run, one engine at a time inside a heavy job (not the engine host). The spread feeds the result's specification band.
      </p>
      {runs.error ? <p className="callout" data-tone="crit">{errorMessage(runs.error)}</p> : null}
      <fieldset className="stack" style={{ gap: 4, border: 0, padding: 0 }}>
        <legend className="eyebrow">Runs</legend>
        {list.map((r) => {
          const row = current?.runs.find((x) => x.run_id === r.id);
          return (
            <label key={r.id} className="row">
              <input type="checkbox" checked={chosen.includes(r.id)} onChange={(e) => toggle(r.id, e.target.checked)} />
              <span className="mono">{r.label || r.id}</span>
              <span className="cap">
                {r.mode}
                {r.coarse_m ? ` ${r.coarse_m} m` : ""}
                {(r.checkpoint_bytes ?? 0) > 0 ? "" : " · no checkpoint"}
              </span>
              {row && !row.ok ? <span className="cap" data-refused="true">refused: {row.reason}</span> : null}
            </label>
          );
        })}
        {!list.length && !runs.error ? <p className="cap">{runs.data ? "No finished runs in this project." : "Loading runs…"}</p> : null}
      </fieldset>
      {estError ? <p className="callout" data-tone="crit">Could not estimate: {estError}</p> : null}
      {current ? (
        <div className="stack" style={{ gap: 6 }} data-testid="across-estimate">
          <p>
            Estimated time <strong className="num" data-testid="across-total">{fmtDuration(current.total_s)}</strong> · peak memory{" "}
            <strong className="num" data-testid="across-peak">{fmtNum(current.peak_rss_gb, 1)} GB</strong> · {okRuns.length} of {current.runs.length} runs usable
          </p>
          <Table
            csvName={null}
            rowKey={(r) => r.run_id}
            columns={[
              { key: "run", label: "Run", value: (r) => r.run_id },
              { key: "ok", label: "Usable", value: (r) => (r.ok ? "yes" : `no: ${r.reason ?? ""}`) },
              { key: "load", label: "Load", align: "right", value: (r) => r.load_s, render: (r) => fmtDuration(r.load_s) },
              { key: "exact", label: "Exact", align: "right", value: (r) => r.exact_s, render: (r) => fmtDuration(r.exact_s) },
              { key: "rss", label: "Memory", unit: "GB", align: "right", value: (r) => r.rss_gb, render: (r) => fmtNum(r.rss_gb, 1) },
            ]}
            rows={current.runs}
          />
        </div>
      ) : chosen.length ? (
        <p className="cap" role="status">
          Estimating time and memory…
        </p>
      ) : (
        <p className="cap">Choose at least one run.</p>
      )}
    </Dialog>
  );
}

// ---------------------------------------------------------------- promote

function DiffText({ text }: { text: string }) {
  return (
    <pre className="yaml-diff" aria-label="Config changes">
      {text.split("\n").map((l, i) => (
        <span key={i} data-op={l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : undefined}>
          {l}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

export function PromoteDialog({ sid, onClose }: { sid: string; onClose: () => void }) {
  const [info, setInfo] = useState<PromoteResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true;
    promoteScenario(sid, false).then(
      (r) => live && setInfo(r),
      (e: unknown) => live && setError(errorMessage(e)),
    );
    return () => {
      live = false;
    };
  }, [sid]);
  const apply = async () => {
    setBusy(true);
    try {
      const r = await promoteScenario(sid, true);
      toast("success", `Wrote config version ${r.version ?? ""}`.trim(), { body: r.names.length ? `New scenarios: ${r.names.join(", ")}` : undefined });
      invalidate("projects");
      onClose();
    } catch (e) {
      toast("error", "Could not promote the scenario", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      wide
      onClose={onClose}
      busy={busy}
      title="Promote to the project config"
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" busy={busy} disabled={!info?.eligible} onClick={() => void apply()}>
            Write a new config version
          </Button>
        </>
      }
    >
      {error ? <p className="callout" data-tone="crit">{error}</p> : null}
      {!info && !error ? <p className="cap">Checking…</p> : null}
      {info && !info.eligible ? (
        <p className="callout" data-promote="ineligible">
          This scenario cannot be written to the config: {info.reason ?? "only whole-city add edits on actionable levers can be promoted"}.
        </p>
      ) : null}
      {info?.eligible ? (
        <>
          <p>
            Future runs will evaluate {info.names.length === 1 ? "this scenario" : "these scenarios"} at the end of S5 as:{" "}
            {info.names.map((n) => (
              <code key={n} style={{ marginRight: 6 }}>
                {n}
              </code>
            ))}
          </p>
          {info.yaml_diff ? <DiffText text={info.yaml_diff} /> : null}
          <p className="cap">Runs that already exist are not changed.</p>
        </>
      ) : null}
    </Dialog>
  );
}

