// Live stage panels of Mission Control (SPEC §5.10), driven by `metric` / `task` events through
// the reducer state: S0 QA tiles and flags, S1 influence table, baselines forest, S4 dose–response
// small multiples, S5 scenario feed (+ CMIP6 ticker), S6 treatment × step checklist, S7 planned vs
// realised, finish documents; S2_S3 and cv_curve live in ModelPanels. Jobs without a stage plan
// (fetches, exports, engine requests) get the generic panel.
import type { ReactNode } from "react";
import type { Job, StageId } from "../../../api/types";
import { Bars, Forest, LineBand, SmallMultiples } from "../../../charts";
import { Card } from "../../../components/ui/Card";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { Pill } from "../../../components/ui/Pill";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link } from "../../../router";
import type { TrackerState } from "../../../stores/tracker";
import { fmtBytes, fmtInt, fmtNum, fmtPct, fmtRange, fmtSigned, fmtTempChange, fmtValue } from "../../../theme/format";
import {
  CAUSAL_STEPS,
  STAGE_LABELS,
  baselineRows,
  budgetTotals,
  causalChecklist,
  climateTicker,
  doseCurves,
  idleNote,
  influenceRows,
  meanSe,
  metricValue,
  reasonText,
  s0Summary,
  scenarioFeed,
  verdictStatus,
} from "../model";
import { CvCurvePanel, S2S3Panel } from "./ModelPanels";

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function S0Panel({ state }: { state: TrackerState }) {
  const { summary, flags } = s0Summary(state);
  const clipped = summary && summary.clipped && typeof summary.clipped === "object" ? Object.values(summary.clipped as Record<string, number>).reduce((a, b) => a + Number(b || 0), 0) : null;
  return (
    <div className="stack">
      {summary ? (
        <KpiRow label="Data and QA">
          <Kpi label="Cells" value={fmtInt(num(summary.n_points))} note={num(summary.n_input) !== null ? `from ${fmtInt(num(summary.n_input))} input rows` : undefined} />
          <Kpi label="Dropped" value={fmtInt(num(summary.n_dropped))} tone={(num(summary.n_dropped) ?? 0) > 0 ? "warn" : undefined} />
          <Kpi label="Clipped values" value={fmtInt(clipped)} />
          <Kpi label="Grid fill" value={fmtPct(num(summary.fill_fraction), 1)} note={Array.isArray(summary.grid_shape) ? `grid ${(summary.grid_shape as number[]).join(" × ")} at ${fmtNum(num(summary.cell_m), 0)} m` : undefined} />
          <Kpi label="Background" value={fmtNum(num(summary.background), 2)} />
          <Kpi label="Rounding-noise floor" value={fmtNum(num(summary.noise_floor), 3)} />
        </KpiRow>
      ) : (
        <p className="mc-note">{idleNote(state, "S0", "The QA tiles appear when S0 finishes.")}</p>
      )}
      <div className="row" aria-label="QA flags">
        {flags.length ? (
          flags.map((f) => (
            <Pill key={f.code + f.message} tone="warn" icon="alert" title={f.message}>
              {f.code.replace(/^qa\./, "")}
              {f.count > 1 ? ` ×${f.count}` : ""}
            </Pill>
          ))
        ) : (
          <span className="mc-note">No QA flags.</span>
        )}
      </div>
    </div>
  );
}

function S1Panel({ state }: { state: TrackerState }) {
  const rows = influenceRows(state);
  const sum = Object.values(state.spans).find((sp) => sp.kind === "stage" && sp.name === "S1" && sp.status === "ok")?.metrics ?? null;
  return (
    <div className="stack">
      {rows.length ? (
        <table className="checklist" aria-label="Influence range per predictor">
          <thead>
            <tr>
              <th scope="col">Predictor</th>
              <th scope="col" className="r">
                Influence range (m)
              </th>
              <th scope="col">Anisotropy</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.predictor}>
                <td>{r.predictor}</td>
                <td className="num r">{fmtNum(r.range_m, 0)}</td>
                <td>{r.anisotropy !== null ? <Pill tone={r.anisotropy < 0.5 ? "warn" : "neutral"} title="Minor / major axis ratio of the directional range (1 = isotropic)">{`ratio ${fmtNum(r.anisotropy, 2)}`}</Pill> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mc-note">{idleNote(state, "S1", "Ranges fill in per predictor as S1 computes them.")}</p>
      )}
      {sum ? (
        <div className="kv-inline">
          <span>
            CV block <b>{fmtNum(num(sum.block_size_m), 0)} m</b>
          </span>
          <span>
            physics length prior <b>{fmtNum(num(sum.L_prior_m), 0)} m</b>
          </span>
        </div>
      ) : null}
    </div>
  );
}

function BaselinesPanel({ state }: { state: TrackerState }) {
  const rows = baselineRows(state);
  const stack = metricValue(state, "stacker_rmse");
  if (!rows.length) return <p className="mc-note">{idleNote(state, "baselines", "Baseline folds appear as they finish.")}</p>;
  if (stack === null) {
    return (
      <Forest
        title="Baseline held-out RMSE"
        caption="Mean over the folds fitted so far, ± 2 SE"
        rows={rows.map((r) => {
          const ms = meanSe(r.rmse);
          return { id: r.model, label: `${r.model} (${r.rmse.length} folds)`, est: ms.mean, se: ms.se };
        })}
        valueLabel="Held-out RMSE"
        z={2}
        pin={false}
      />
    );
  }
  return (
    <Forest
      title="Baselines vs the stack: ΔMSE ± 2 SE"
      caption={`Per-fold baseline MSE minus the stack's pooled out-of-fold MSE (${fmtNum(stack * stack, 4)}); positive means the stack is better`}
      rows={rows.map((r) => {
        const ms = meanSe(r.rmse.map((x) => x * x - stack * stack));
        return { id: r.model, label: `${r.model} (${r.rmse.length} folds)`, est: ms.mean, se: ms.se };
      })}
      valueLabel="ΔMSE"
      z={2}
      better={{ side: "positive", label: "stack better" }}
      decimals={4}
      pin={false}
    />
  );
}

function S4Panel({ state }: { state: TrackerState }) {
  const curves = doseCurves(state);
  if (!curves.length) return <p className="mc-note">{idleNote(state, "S4", "Dose–response points draw as each dose finishes.")}</p>;
  const all = curves.flatMap((c) => c.points.flatMap((p) => (p.mean !== null ? [p.mean - 1.96 * (p.se ?? 0), p.mean + 1.96 * (p.se ?? 0)] : [])));
  const domain: [number, number] = [Math.min(0, ...all), Math.max(0, ...all)];
  return (
    <SmallMultiples
      title="Dose–response per lever"
      caption="Mean cooling benefit per dose; ribbon ±1.96 SE; hollow points have more than 20% of cells extrapolated"
      items={curves}
      panelTitle={(c) => `${c.lever}${c.running ? " (running)" : ""}`}
      minPanelWidth={180}
      pin={false}
      table={{
        columns: [{ key: "lever", label: "Lever" }, { key: "dose", label: "Dose" }, { key: "mean", label: "Mean benefit" }, { key: "se", label: "SE" }, { key: "fx", label: "Extrapolated", unit: "%" }],
        rows: curves.flatMap((c) => c.points.map((p) => [c.lever, p.dose, p.mean, p.se, p.frac_extrapolated === null ? null : Number((p.frac_extrapolated * 100).toFixed(1))])),
      }}
      renderPanel={(c) => (
        <LineBand
          bare
          title={`${c.lever} dose–response`}
          series={[
            {
              id: c.lever,
              label: c.lever,
              x: c.points.map((p) => p.dose),
              y: c.points.map((p) => p.mean),
              lo: c.points.map((p) => (p.mean !== null && p.se !== null ? p.mean - 1.96 * p.se : null)),
              hi: c.points.map((p) => (p.mean !== null && p.se !== null ? p.mean + 1.96 * p.se : null)),
              points: true,
              hollow: c.points.map((p) => (p.frac_extrapolated ?? 0) > 0.2),
            },
          ]}
          xLabel="Dose"
          yLabel="Mean benefit"
          yDomain={domain}
          height={150}
        />
      )}
    />
  );
}

function S5Panel({ state, tickerOnly }: { state: TrackerState; tickerOnly?: boolean }) {
  const rows = tickerOnly ? [] : scenarioFeed(state);
  const cmip = climateTicker(state);
  return (
    <div className="stack">
      {tickerOnly ? (
        cmip.done.length || cmip.running.length ? null : <p className="mc-note">{idleNote(state, "climate", "CMIP6 model fetches appear here while the climate stage runs.")}</p>
      ) : rows.length ? (
        <table className="checklist" aria-label="Scenario feed">
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              <th scope="col" className="r">
                Mean Δ
              </th>
              <th scope="col" className="r">
                Likely range
              </th>
              <th scope="col" className="r">
                Extrapolated
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td className="num r" title={fmtTempChange(r.mean_delta)}>
                  {fmtSigned(r.mean_delta, 2)}
                </td>
                <td className="num r">{r.mean_delta !== null && r.se !== null ? fmtRange(r.mean_delta - 1.96 * r.se, r.mean_delta + 1.96 * r.se) : "—"}</td>
                <td className="num r">{fmtPct(r.frac_extrapolated)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mc-note">{idleNote(state, "S5", "Scenarios appear as each one is computed.")}</p>
      )}
      {cmip.done.length || cmip.running.length ? (
        <div className="kv-inline" aria-label="CMIP6 models">
          <span>
            CMIP6 models <b>{cmip.done.length}</b>
            {cmip.planned ? ` of ${fmtInt(cmip.planned)}` : ""} fetched
          </span>
          {cmip.running.map((m) => (
            <StatusChip key={m} status="running" text={m} />
          ))}
          {cmip.done.slice(-3).map((m) => (
            <StatusChip key={m} status="done" text={m} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function S6Panel({ state }: { state: TrackerState }) {
  const rows = causalChecklist(state);
  if (!rows.length) return <p className="mc-note">{idleNote(state, "S6", "Treatments appear as the causal checks start.")}</p>;
  const stepStatus = (st: { status: string; frac: number | null } | undefined) => {
    if (!st) return <StatusChip status="planned" text="pending" />;
    const s = st.status === "ok" ? "done" : st.status === "error" ? "failed" : st.status;
    return <StatusChip status={s} meta={st.frac !== null ? fmtPct(st.frac) : undefined} />;
  };
  return (
    <div className="tablewrap">
      <table className="checklist" aria-label="Treatment by step checklist">
        <thead>
          <tr>
            <th scope="col">Treatment</th>
            {CAUSAL_STEPS.map((c) => (
              <th key={c.name} scope="col">
                {c.label}
              </th>
            ))}
            <th scope="col" className="r">
              θ ± SE
            </th>
            <th scope="col">Verdicts</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.treatment}>
              <th scope="row">{r.treatment}</th>
              {CAUSAL_STEPS.map((c) => (
                <td key={c.name}>{stepStatus(r.steps[c.name])}</td>
              ))}
              <td className="num r">{r.theta !== null ? `${fmtSigned(r.theta, 4)} ± ${fmtNum(r.theta_se, 4)}` : "—"}</td>
              <td>
                <div className="row" style={{ gap: 4 }}>
                  {r.verdicts.length ? (
                    r.verdicts.map((v, i) => {
                      const vs = verdictStatus(v.verdict);
                      return <StatusChip key={i} status={vs.status} text={vs.text} title={v.check} />;
                    })
                  ) : r.steps.audit?.status === "ok" ? (
                    <StatusChip status="done" text="consistent" />
                  ) : (
                    <span className="mc-note">—</span>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function S7Panel({ state, runId }: { state: TrackerState; runId: string | null }) {
  const rows = budgetTotals(state);
  const pareto = Object.values(state.spans).find((sp) => sp.name === "pareto" && sp.kind === "task");
  return (
    <div className="stack">
      {rows.length ? (
        <Bars
          title="Planned vs realised cooling"
          caption="Open-loop planned total against the closed-loop (exact) realised total"
          categories={rows.map((r) => r.variable)}
          series={[
            { id: "planned", label: "planned (open-loop)", values: rows.map((r) => r.planned) },
            { id: "realised", label: "realised (closed-loop)", values: rows.map((r) => r.realised) },
          ]}
          valueLabel="Total cooling (target units × cells)"
          pin={false}
        />
      ) : (
        <p className="mc-note">{idleNote(state, "S7", "Totals appear once the allocation is computed.")}</p>
      )}
      {pareto ? (
        <div className="row">
          <StatusChip status={pareto.status === "ok" ? "done" : pareto.status} text="Pareto front" />
          {pareto.status === "ok" && runId ? <Link to={`/r/${encodeURIComponent(runId)}/budget`}>Open the Budget view</Link> : null}
        </div>
      ) : null}
    </div>
  );
}

function FinishPanel({ state, runId }: { state: TrackerState; runId: string | null }) {
  const docs = Object.values(state.artifacts).filter((a) => a.stage === "finish");
  if (!docs.length) return <p className="mc-note">{idleNote(state, "finish", "Documents are written when the run finishes.")}</p>;
  return (
    <ul className="roots">
      {docs.map((d) => (
        <li key={d.relpath}>
          <StatusChip status="present" text="written" />
          {runId && /\.md$/.test(d.relpath) ? <Link to={`/r/${encodeURIComponent(runId)}/docs`}>{d.relpath}</Link> : <span>{d.relpath}</span>}
          <span className="cap">{fmtBytes(d.bytes)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Panel of a job without a stage plan: progress, current task and the latest metrics. */
export function GenericPanel({ state, job }: { state: TrackerState; job: Job }) {
  const metrics = Object.entries(state.metrics_latest).slice(-24);
  return (
    <div className="stack">
      {job.result ? (
        <details open>
          <summary className="cap">Result</summary>
          <pre className="raw-pre">{JSON.stringify(job.result, null, 2)}</pre>
        </details>
      ) : null}
      {metrics.length ? (
        <table className="checklist" aria-label="Latest metrics">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="r">
                Value
              </th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(([k, v]) => (
              <tr key={k}>
                <td className="mono">{k}</td>
                <td className="num r">{typeof v.value === "number" ? fmtValue(v.value, v.unit, 4) : String(v.value ?? "—")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mc-note">This job reports progress and files, without stage panels.</p>
      )}
    </div>
  );
}

/** The live panel of one stage. */
export function StagePanel({
  stage,
  state,
  runId,
  epochs,
  onLoadEpochs,
}: {
  stage: StageId;
  state: TrackerState;
  runId: string | null;
  epochs?: { ts: number; value: number }[];
  onLoadEpochs?: () => Promise<void>;
}) {
  const st = state.stages?.[stage];
  let body: ReactNode;
  switch (stage) {
    case "S0":
      body = <S0Panel state={state} />;
      break;
    case "S1":
      body = <S1Panel state={state} />;
      break;
    case "S2_S3":
      body = <S2S3Panel state={state} epochs={epochs} onLoadEpochs={onLoadEpochs} />;
      break;
    case "baselines":
      body = <BaselinesPanel state={state} />;
      break;
    case "cv_curve":
      body = <CvCurvePanel state={state} />;
      break;
    case "S4":
      body = <S4Panel state={state} />;
      break;
    case "S5":
      body = <S5Panel state={state} />;
      break;
    case "climate":
      body = <S5Panel state={state} tickerOnly />;
      break;
    case "S6":
      body = <S6Panel state={state} />;
      break;
    case "S7":
      body = <S7Panel state={state} runId={runId} />;
      break;
    default:
      body = <FinishPanel state={state} runId={runId} />;
  }
  return (
    <Card
      title={STAGE_LABELS[stage]}
      eyebrow={stage}
      actions={<StatusChip status={st?.state ?? "planned"} meta={st?.reason ? reasonText(st.reason) : undefined} />}
      aria-label={`${STAGE_LABELS[stage]} live panel`}
    >
      {body}
    </Card>
  );
}
