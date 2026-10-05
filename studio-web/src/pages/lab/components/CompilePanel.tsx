// Right pane of the Design workbench (SPEC §7.3, §7.5, §7.6): the live compile of the draft
// (cells per lever, requested vs predicted realised dose, clipped share, cost, warnings),
// the emulator trust badges, an automatic "run exact" recommendation, the engine chip with
// "Open engine", and Run exact with its fold-tick progress.
import type { CompileResult, Lever, PreviewSummary } from "../../../api/lab";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { JobStrip } from "../../../components/ui/JobStrip";
import { Table } from "../../../components/ui/Table";
import { fmtDuration, fmtInt, fmtNum, fmtPct, unitLabel } from "../../../theme/format";
import type { HeatPreview } from "../model/heatPreview";
import { EngineChip } from "./EngineChip";
import { TrustBadge } from "./TrustBadge";

export type ExactAdvice = { recommend: boolean; reasons: string[] };

/** Why the draft should be run exactly (SPEC §7.5 hatching rules, plus trust and blocking). */
export function exactAdvice(compile: CompileResult | null, preview: PreviewSummary | null, levers: Lever[]): ExactAdvice {
  const reasons: string[] = [];
  if (!compile) return { recommend: false, reasons };
  if (!compile.emulator.usable) reasons.push("no usable preview for these edits");
  if (compile.emulator.hatched || preview?.hatched) reasons.push(...(compile.emulator.reasons.length ? compile.emulator.reasons : preview?.reasons ?? ["the preview is unreliable at this scale"]));
  for (const v of Object.keys(compile.levers)) {
    const l = levers.find((x) => x.var === v);
    if (l && l.emulator.trust !== "good") reasons.push(`${l.label}: preview ${l.emulator.trust === "rough" ? "is rough" : "is unavailable"}`);
  }
  if (Object.values(compile.levers).some((x) => x.clipped_share > 0.2)) reasons.push("more than 20% of the requested change is clipped (the linear preview ignores clipping)");
  return { recommend: reasons.length > 0, reasons: [...new Set(reasons)] };
}

export type CompilePanelProps = {
  rid: string;
  compile: CompileResult | null;
  compileError: string | null;
  compiling: boolean;
  preview: PreviewSummary | null;
  levers: Lever[];
  unit: string;
  canRun: boolean;
  runReason: string | null;
  running: boolean;
  jobId: string | null;
  onRunExact: () => void;
  /** Live heat risk of the preview (runs with campaign humidity). */
  heat?: HeatPreview | null;
};

/** The preview's heat risk: residents the draft moves across the NWS Extreme caution and Danger edges. */
export function HeatPreviewCard({ heat }: { heat: HeatPreview }) {
  const s = heat.shift;
  const who = s.measure === "people" ? "residents" : "cells";
  const n = (v: number) => fmtInt(Math.round(v));
  const moved = s.ecBefore - s.ecAfter;
  return (
    <section className="card" aria-label="Heat risk preview">
      <header>
        <h3>Heat risk</h3>
        <Badge tone="accent">preview</Badge>
      </header>
      <p style={{ margin: "0 0 6px", fontWeight: 600 }}>
        {Math.round(moved) > 0
          ? `Moves about ${n(moved)} ${who} out of Extreme caution or worse.`
          : Math.round(moved) < 0
            ? `Puts about ${n(-moved)} more ${who} at Extreme caution or worse.`
            : `No change in ${who} at Extreme caution or worse.`}
      </p>
      <dl className="kv">
        <dt>Extreme caution or worse</dt>
        <dd className="num">
          {n(s.ecBefore)} → {n(s.ecAfter)}
        </dd>
        <dt>Danger or worse</dt>
        <dd className="num">
          {n(s.dangerBefore)} → {n(s.dangerAfter)}
        </dd>
        <dt>Heat index felt</dt>
        <dd className="num">{`${s.meanHiChange <= 0 ? "−" : "+"}${fmtNum(Math.abs(s.meanHiChange), 2)} °F`}</dd>
      </dl>
      {heat.caveats.length ? (
        <ul className="edit-issues" aria-label="Evidence caveats">
          {heat.caveats.map((c, i) => (
            <li key={i} data-level="warn">
              {c}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="cap">
        NWS heat index on the campaign afternoon (dewpoint {fmtNum(heat.dewpointC, 1)} °C). From the linear preview; the exact result&apos;s impacts verify it and add the climate
        futures.
      </p>
    </section>
  );
}

export function CompilePanel(p: CompilePanelProps) {
  const c = p.compile;
  const advice = exactAdvice(c, p.preview, p.levers);
  const blocking = c?.warnings.filter((w) => w.blocking) ?? [];
  const rows = c ? Object.entries(c.levers).map(([v, x]) => ({ v, ...x, lever: p.levers.find((l) => l.var === v) })) : [];
  return (
    <div className="compile-panel stack" aria-label="Compile">
      <section className="card">
        <header>
          <h3>Compile</h3>
          {p.compiling ? <span className="spinner" aria-label="Compiling" /> : null}
        </header>
        {p.compileError ? <p className="callout" data-tone="crit">{p.compileError}</p> : null}
        {c ? (
          <>
            <dl className="kv">
              <dt>Cells edited</dt>
              <dd className="num">{fmtInt(c.union_cells)}</dd>
              {c.people !== null ? (
                <>
                  <dt>Residents there</dt>
                  <dd className="num">{fmtInt(Math.round(c.people))}</dd>
                </>
              ) : null}
              <dt>Estimated cost</dt>
              <dd className="num">{fmtNum(rows.reduce((a, r) => a + r.est_cost, 0), 0)}</dd>
              <dt>Exact run</dt>
              <dd className="num">≈ {fmtDuration(c.est_exact_s)}</dd>
              <dt>Portable</dt>
              <dd>{c.portable ? "yes (lon/lat selections)" : "this run only"}</dd>
            </dl>
            {rows.length ? (
              <Table
                caption="Per lever: requested vs predicted realised"
                csvName={null}
                rowKey={(r) => r.v}
                columns={[
                  { key: "v", label: "Lever", value: (r) => r.lever?.label ?? r.v, render: (r) => (
                    <span className="row" style={{ gap: 4 }}>
                      {r.lever?.label ?? r.v}
                      {r.lever ? <TrustBadge trust={r.lever.emulator.trust} relErr={r.lever.emulator.uniform_rel_err} /> : null}
                    </span>
                  ) },
                  { key: "n", label: "Cells", align: "right", value: (r) => r.n_cells, render: (r) => fmtInt(r.n_cells) },
                  { key: "req", label: "Requested mean", align: "right", value: (r) => r.mean_requested, render: (r) => `${fmtNum(r.mean_requested, 2)} ${unitLabel(r.lever?.unit ?? "")}` },
                  { key: "real", label: "Predicted realised", align: "right", value: (r) => r.predicted_mean_realised, render: (r) => `${fmtNum(r.predicted_mean_realised, 2)} ${unitLabel(r.lever?.unit ?? "")}` },
                  { key: "clip", label: "Clipped", align: "right", value: (r) => r.clipped_share, render: (r) => fmtPct(r.clipped_share) },
                  { key: "cost", label: "Cost", align: "right", value: (r) => r.est_cost, render: (r) => fmtNum(r.est_cost, 0) },
                ]}
                rows={rows}
              />
            ) : null}
            {c.warnings.length ? (
              <ul className="edit-issues">
                {c.warnings.map((w, i) => (
                  <li key={i} data-level={w.blocking ? "error" : "warn"}>
                    {w.edit_index !== null && !/^edit \d/i.test(w.message) ? `Edit ${w.edit_index + 1}: ` : ""}
                    {w.blocking ? "Blocked — " : ""}
                    {w.message}
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        ) : !p.compileError ? (
          <p className="cap">Add an edit to compile it.</p>
        ) : null}
      </section>
      {p.heat ? <HeatPreviewCard heat={p.heat} /> : null}
      <section className="card">
        <header>
          <h3>Exact</h3>
        </header>
        <EngineChip rid={p.rid} />
        {advice.recommend ? (
          <div className="callout" data-exact-advice="true">
            <Badge tone="warn">Run exact</Badge> recommended: {advice.reasons.join("; ")}.
          </div>
        ) : c ? (
          <p className="cap">The preview is a good guide here. Uncertainty, comparisons, impacts and decision packs still come only from exact results.</p>
        ) : null}
        <Button variant="primary" icon="play" busy={p.running} disabled={!p.canRun || blocking.length > 0} onClick={p.onRunExact} title={p.runReason ?? undefined}>
          Run exact
        </Button>
        {blocking.length ? <p className="cap">Fix the blocked edits first (or enable expert mode where allowed).</p> : p.runReason ? <p className="cap">{p.runReason}</p> : null}
        {p.jobId ? <JobStrip jobId={p.jobId} /> : null}
      </section>
    </div>
  );
}
