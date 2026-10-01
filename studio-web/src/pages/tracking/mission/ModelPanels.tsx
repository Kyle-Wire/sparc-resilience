// S2_S3 and cv_curve live panels (SPEC §5.10): the fold × model heatmap (cell = seconds, colour =
// held-out fold RMSE, the running cell pulses unless reduced motion is set), physics per fold,
// the advection decision chip, the stacker candidate leaderboard with the 0.1% tie rule, the
// debug-level epoch loss sparkline and the final skill against the 0.90 coverage target; and the
// CV curve (R² vs block size) with the nested fold heatmap of the active partition.
import { useState } from "react";
import { Card } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { Pill } from "../../../components/ui/Pill";
import { StatusChip } from "../../../components/ui/StatusChip";
import { LineBand, Sparkline } from "../../../charts";
import { useDark } from "../../../stores/ui";
import type { TrackerState } from "../../../stores/tracker";
import { fmtDuration, fmtNum, fmtPct, fmtSigned } from "../../../theme/format";
import { hexToRgb01, rampColor } from "../../../theme/palette";
import { useReducedMotion } from "../hooks";
import {
  advectionDecision,
  candidateLabel,
  cvCurvePoints,
  cvPartitions,
  foldModelGrid,
  idleNote,
  metricValue,
  physicsPerFold,
  stackerLeaderboard,
  type FoldGrid,
} from "../model";

function ink(fill: string): string {
  if (!fill.startsWith("#")) return "var(--ink)";
  const [r, g, b] = hexToRgb01(fill);
  return 0.299 * r + 0.587 * g + 0.114 * b < 0.45 ? "#ffffff" : "#0b0b0b";
}

/** Fold × model grid: seconds in each cell, colour by held-out RMSE (lighter = lower error). */
export function FoldModelHeatmap({ grid, title }: { grid: FoldGrid; title: string }) {
  const dark = useDark();
  const reduced = useReducedMotion();
  const vals = grid.cells.flat().map((c) => c.rmse).filter((v): v is number => v !== null);
  const lo = vals.length ? Math.min(...vals) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  if (!grid.models.length) return <p className="mc-note">No base-model fits in this job.</p>;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="tablewrap">
        <table className="foldgrid" aria-label={title}>
          <thead>
            <tr>
              <th scope="col">
                <span className="sr-only">Fold</span>
              </th>
              {grid.models.map((m) => (
                <th key={m} scope="col">
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.folds.map((f, i) => (
              <tr key={f}>
                <th scope="row">{f}</th>
                {grid.models.map((m, j) => {
                  const c = grid.cells[i][j];
                  const done = c.status === "ok" && c.rmse !== null;
                  const t = hi > lo && c.rmse !== null ? (c.rmse - lo) / (hi - lo) : 0.5;
                  const fill = done ? rampColor("seq", 0.15 + 0.75 * t, dark) : undefined;
                  const label =
                    c.status === "pending"
                      ? `${f}, ${m}: not fitted yet`
                      : c.status === "running"
                        ? `${f}, ${m}: fitting now`
                        : `${f}, ${m}: ${c.status === "ok" ? "fitted" : c.status} in ${fmtDuration(c.seconds)}${c.rmse !== null ? `, held-out RMSE ${fmtNum(c.rmse, 3)}` : ""}${c.r2 !== null ? `, R² ${fmtNum(c.r2, 2)}` : ""}`;
                  return (
                    <td
                      key={m}
                      data-status={c.status}
                      data-pulse={c.status === "running" && !reduced ? "true" : undefined}
                      style={fill ? { background: fill, color: ink(fill) } : undefined}
                      aria-label={label}
                      title={label}
                    >
                      {c.status === "pending" ? (
                        <span aria-hidden="true">·</span>
                      ) : c.status === "running" ? (
                        <span>running</span>
                      ) : (
                        <>
                          <span>{fmtDuration(c.seconds)}</span>
                          {c.rmse !== null ? <small>{fmtNum(c.rmse, 3)}</small> : null}
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend-inline" aria-hidden="true">
        <span>cell: fit time, held-out RMSE below</span>
        <span className="row" style={{ gap: 4 }}>
          <span className="num">{fmtNum(lo, 3)}</span>
          <svg width="80" height="9">
            {Array.from({ length: 16 }, (_, i) => (
              <rect key={i} x={i * 5} width={5.5} height={9} fill={rampColor("seq", 0.15 + (0.75 * i) / 15, dark)} />
            ))}
          </svg>
          <span className="num">{fmtNum(hi, 3)}</span>
        </span>
        <span>grey dot: not fitted yet · outlined: running</span>
      </div>
    </div>
  );
}

/** Stacker candidates by OOF RMSE with the winner badge (ties within 0.1% go to the simpler one). */
export function StackerLeaderboard({ state, stage = "S2_S3" }: { state: TrackerState; stage?: "S2_S3" | "cv_curve" }) {
  const rows = stackerLeaderboard(state, stage);
  if (!rows.length) return <p className="mc-note">Stacker candidates are scored after every fold is fitted.</p>;
  const best = rows[0];
  const winner = rows.find((r) => r.winner);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <table className="checklist" aria-label="Stacker candidate leaderboard">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Candidate</th>
            <th scope="col" className="r">
              OOF RMSE
            </th>
            <th scope="col" className="r">
              vs best
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.candidate} data-winner={r.winner || undefined}>
              <td className="num">{i + 1}</td>
              <td>
                {candidateLabel(r.candidate)}{" "}
                {r.winner ? (
                  <Badge tone="accent" title="Chosen stacker">
                    winner
                  </Badge>
                ) : null}
              </td>
              <td className="num r">{fmtNum(r.rmse, 4)}</td>
              <td className="num r">{r.relToBest === 0 ? "best" : `+${fmtPct(r.relToBest, 2)}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {winner && winner.candidate !== best.candidate ? (
        <p className="mc-note">
          {candidateLabel(best.candidate)} has the lowest RMSE but is within 0.1% of {candidateLabel(winner.candidate)}, so the simpler candidate wins.
        </p>
      ) : null}
    </div>
  );
}

function AdvectionChip({ state }: { state: TrackerState }) {
  const a = advectionDecision(state);
  const s23 = state.stages?.S2_S3?.state;
  if (a.state === "none")
    return s23 === "done" || s23 === "cached" ? <Pill icon="minus">advection not fitted (no wind field)</Pill> : <Pill icon="clock">advection check pending</Pill>;
  if (a.state === "checking") return <Pill icon="hourglass">checking advection {a.total ? `${a.done}/${a.total}` : `${a.done} refits`}</Pill>;
  const text = `advection ${a.kept ? "kept" : "dropped"}: ΔRMSE ${fmtSigned(a.mean, 4)} ± ${fmtNum(a.se, 4)} (${a.folds} folds)`;
  return (
    <Pill tone={a.kept ? "good" : "neutral"} icon={a.kept ? "check" : "x"} title="Kept only if refits without advection are worse by more than one standard error">
      {text}
    </Pill>
  );
}

function PhysicsFolds({ state }: { state: TrackerState }) {
  const rows = physicsPerFold(state);
  if (!rows.length) return null;
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r.metrics)))];
  const label = (k: string) => ({ fit_s: "fit (s)", heldout_rmse: "RMSE", heldout_r2: "R²" })[k] ?? k;
  return (
    <details>
      <summary className="cap">Physics model per fold</summary>
      <div className="tablewrap">
        <table className="checklist">
          <thead>
            <tr>
              <th scope="col">Fold</th>
              {keys.map((k) => (
                <th key={k} scope="col" className="r">
                  {label(k)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.fold}>
                <td>{r.fold}</td>
                {keys.map((k) => (
                  <td key={k} className="num r">
                    {fmtNum(r.metrics[k], 3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function EpochLoss({ epochs, onLoad, canLoad }: { epochs: { ts: number; value: number }[]; onLoad?: () => Promise<void>; canLoad: boolean }) {
  const [busy, setBusy] = useState(false);
  if (epochs.length)
    return <Sparkline title="Neural residual validation loss per epoch" values={epochs.map((e) => e.value)} units="MSE" decimals={4} pin={false} caption={`${epochs.length} epochs logged at debug level`} />;
  return (
    <p className="mc-note">
      Epoch losses are logged at debug level only.{" "}
      {canLoad && onLoad ? (
        <Button
          size="small"
          variant="ghost"
          busy={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await onLoad();
            } finally {
              setBusy(false);
            }
          }}
        >
          Look in the event log
        </Button>
      ) : null}
    </p>
  );
}

export function S2S3Panel({ state, epochs = [], onLoadEpochs }: { state: TrackerState; epochs?: { ts: number; value: number }[]; onLoadEpochs?: () => Promise<void> }) {
  const grid = foldModelGrid(state, "S2_S3");
  const r2 = metricValue(state, "stacker_r2");
  const rmse = metricValue(state, "stacker_rmse");
  const cov = metricValue(state, "interval_coverage");
  const hw = metricValue(state, "interval_halfwidth");
  const cached = state.stages?.S2_S3?.state === "cached";
  return (
    <div className="stack">
      {grid.models.length ? null : <p className="mc-note">{idleNote(state, "S2_S3", "Fits appear fold by fold as the base models train.")}</p>}
      <Card variant="tight" title="Fold × model fits" eyebrow="S2–S3">
        <FoldModelHeatmap grid={grid} title="Fold by model fits: seconds, coloured by held-out RMSE" />
        <PhysicsFolds state={state} />
      </Card>
      <div className="row">
        <AdvectionChip state={state} />
      </div>
      <Card variant="tight" title="Stacker candidates">
        <StackerLeaderboard state={state} />
        <EpochLoss epochs={epochs} onLoad={onLoadEpochs} canLoad={!cached && state.stages?.S2_S3?.state !== "planned"} />
      </Card>
      {r2 !== null || rmse !== null || cov !== null ? (
        <KpiRow label="Stacked model skill">
          <Kpi label="Held-out R²" value={fmtNum(r2, 3)} />
          <Kpi label="Held-out RMSE" value={fmtNum(rmse, 3)} />
          <Kpi label="90% interval coverage" value={fmtPct(cov, 1)} note="target 90%" tone={cov !== null && cov < 0.9 ? "warn" : cov !== null ? "good" : undefined} />
          {hw !== null ? <Kpi label="Interval half-width" value={fmtNum(hw, 3)} /> : null}
        </KpiRow>
      ) : null}
    </div>
  );
}

export function CvCurvePanel({ state }: { state: TrackerState }) {
  const pts = cvCurvePoints(state);
  const parts = cvPartitions(state);
  const active = parts.find((p) => p.active) ?? null;
  const grid = foldModelGrid(state, "cv_curve", active?.label ?? null);
  const withX = pts.filter((p) => p.block_m !== null);
  return (
    <div className="stack">
      {parts.length ? (
        <div className="row" aria-label="CV partitions">
          {parts.map((p) => (
            <StatusChip key={p.label} status={p.status === "ok" ? "done" : p.status === "error" ? "failed" : p.status} text={p.label} />
          ))}
        </div>
      ) : null}
      {withX.length ? (
        <LineBand
          title="Skill vs block size"
          caption={`${withX.length} partition${withX.length === 1 ? "" : "s"} scored so far`}
          series={[{ id: "r2", label: "Held-out R²", x: withX.map((p) => p.block_m as number), y: withX.map((p) => p.r2), points: true }]}
          xLabel="CV block size"
          xUnit="m"
          yLabel="Held-out R²"
          pin={false}
          height={200}
        />
      ) : (
        <p className="mc-note">{idleNote(state, "cv_curve", "The curve gains a point as each partition finishes.")}</p>
      )}
      {active ? (
        <Card variant="tight" title={`Fold × model fits: ${active.label}`} eyebrow="active partition">
          <FoldModelHeatmap grid={grid} title={`Fold by model fits for ${active.label}`} />
        </Card>
      ) : null}
    </div>
  );
}
