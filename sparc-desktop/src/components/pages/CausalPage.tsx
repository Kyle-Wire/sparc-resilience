/**
 * CausalPage — Phase C visualization hub.
 *
 * Surfaces the four new Phase-C artifacts:
 *
 *   1. KernelField  → per-predictor anisotropy ellipses + κ summary
 *   2. Causal PDP   → Bayesian dose-response curves with 89% HDI ribbons
 *   3. Divergence   → CATE-vs-GWR scoreboard (Pearson, sign-agreement, flagged)
 *   4. Routing audit→ scenario simulator routing (causal vs heuristic)
 *
 * All four endpoints gracefully 404 when the underlying artifact has not
 * been produced yet; the page renders empty-state hints in that case.
 */
import { useEffect, useState } from "react";
import { Card, KeyVal, SectionHeader, Stat, Tag } from "@/components/ui/DesignSystem";
import {
  getKernelFieldData,
  getCausalPDPCurves,
  getCateVsGwrDivergence,
  getScenarioRoutingAudit,
} from "@/lib/api";
import type {
  KernelFieldData,
  CausalPDPData,
  CausalPDPCurve,
  DivergenceReport,
  ScenarioRoutingAuditData,
} from "@/lib/types";

// ------------------------------------------------------------------
// 1) Anisotropy-ellipse visualisation
// ------------------------------------------------------------------

interface KernelFieldVizProps {
  data: KernelFieldData;
}

function KernelFieldViz({ data }: KernelFieldVizProps) {
  const W = 360;
  const H = 360;
  const cx = W / 2;
  const cy = H / 2;

  // Compute a common scale so the longest axis fits within ~85% of the box.
  const maxRange = Math.max(
    1e-9,
    ...data.predictors.flatMap((p) => {
      const rx = p.kappa_x && p.kappa_x > 0 ? 1 / p.kappa_x : 0;
      const ry = p.kappa_y && p.kappa_y > 0 ? 1 / p.kappa_y : 0;
      const ri = p.kappa && p.kappa > 0 ? 1 / p.kappa : 0;
      return [rx, ry, ri];
    }),
  );
  const scale = (W * 0.42) / maxRange;
  const palette = [
    "#7a3a3a", "#3a5e7a", "#5e7a3a", "#7a5e3a",
    "#5e3a7a", "#3a7a5e", "#7a3a5e", "#3a3a7a",
  ];

  return (
    <Card title="Anisotropy ellipses" subtitle={`outcome: ${data.outcome_name}`}>
      <div style={{ display: "flex", gap: 24, alignItems: "flex-start" }}>
        <svg width={W} height={H} style={{ background: "#fafaf7", border: "1px solid var(--line)" }}>
          {/* gridlines */}
          {[-1, 0, 1].map((g) => (
            <g key={g}>
              <line x1={0} x2={W} y1={cy + g * 80} y2={cy + g * 80}
                    stroke="#e5e5dd" strokeWidth={1} />
              <line x1={cx + g * 80} x2={cx + g * 80} y1={0} y2={H}
                    stroke="#e5e5dd" strokeWidth={1} />
            </g>
          ))}
          <line x1={cx} x2={cx} y1={0} y2={H} stroke="#bbb" />
          <line x1={0} x2={W} y1={cy} y2={cy} stroke="#bbb" />
          {/* one ellipse per predictor */}
          {data.predictors.map((p, i) => {
            const rx = p.kappa_x && p.kappa_x > 0 ? (1 / p.kappa_x) * scale
                       : p.kappa && p.kappa > 0 ? (1 / p.kappa) * scale : 0;
            const ry = p.kappa_y && p.kappa_y > 0 ? (1 / p.kappa_y) * scale
                       : p.kappa && p.kappa > 0 ? (1 / p.kappa) * scale : 0;
            if (rx <= 0 || ry <= 0) return null;
            const theta = ((p.theta_rad ?? 0) * 180) / Math.PI;
            const color = palette[i % palette.length];
            return (
              <g key={p.name} transform={`translate(${cx},${cy}) rotate(${theta})`}>
                <ellipse cx={0} cy={0} rx={rx} ry={ry}
                          fill={color} fillOpacity={0.10}
                          stroke={color} strokeWidth={1.5} />
                {/* principal-axis arrow */}
                <line x1={0} y1={0} x2={rx} y2={0}
                       stroke={color} strokeWidth={2} />
              </g>
            );
          })}
          <text x={8} y={H - 8} fontSize={10} fontFamily="JetBrains Mono"
                fill="var(--muted)">
            scale: 1 unit = {(maxRange / scale).toFixed(1)} m
          </text>
        </svg>

        <div style={{ flex: 1, minWidth: 260 }}>
          <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--line)" }}>
                <th style={{ textAlign: "left", padding: "4px 6px" }}>predictor</th>
                <th style={{ textAlign: "right", padding: "4px 6px" }}>κ</th>
                <th style={{ textAlign: "right", padding: "4px 6px" }}>κ_x</th>
                <th style={{ textAlign: "right", padding: "4px 6px" }}>κ_y</th>
                <th style={{ textAlign: "right", padding: "4px 6px" }}>θ°</th>
                <th style={{ textAlign: "right", padding: "4px 6px" }}>aniso</th>
              </tr>
            </thead>
            <tbody>
              {data.predictors.map((p, i) => {
                const color = palette[i % palette.length];
                const aniso = !!p.is_anisotropic
                  || (p.kappa_x != null && p.kappa_y != null && p.kappa_x !== p.kappa_y);
                return (
                  <tr key={p.name} style={{ borderBottom: "1px solid #f0f0e8" }}>
                    <td style={{ padding: "4px 6px" }}>
                      <span style={{
                        display: "inline-block", width: 8, height: 8,
                        background: color, marginRight: 6, borderRadius: 1,
                      }} />
                      {p.name}
                    </td>
                    <td style={{ textAlign: "right", padding: "4px 6px" }}>
                      {p.kappa != null ? p.kappa.toExponential(2) : "—"}
                    </td>
                    <td style={{ textAlign: "right", padding: "4px 6px" }}>
                      {p.kappa_x != null ? p.kappa_x.toExponential(2) : "—"}
                    </td>
                    <td style={{ textAlign: "right", padding: "4px 6px" }}>
                      {p.kappa_y != null ? p.kappa_y.toExponential(2) : "—"}
                    </td>
                    <td style={{ textAlign: "right", padding: "4px 6px" }}>
                      {p.theta_rad != null
                        ? ((p.theta_rad * 180) / Math.PI).toFixed(1)
                        : "—"}
                    </td>
                    <td style={{ textAlign: "right", padding: "4px 6px" }}>
                      {aniso ? <Tag color="#7a3a3a">aniso</Tag> : <Tag color="#3a5e7a">iso</Tag>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {data.bandwidth_mismatch_warnings?.length ? (
            <div style={{ marginTop: 12, fontSize: 11, color: "#7a3a3a" }}>
              ⚠ {data.bandwidth_mismatch_warnings.length} bandwidth-mismatch warning(s)
            </div>
          ) : null}
        </div>
      </div>
    </Card>
  );
}

// ------------------------------------------------------------------
// 2) Causal PDP curve (single curve)
// ------------------------------------------------------------------

function PDPSparkline({ curve }: { curve: CausalPDPCurve }) {
  const W = 320, H = 140, pad = 28;
  const xs = curve.dose_grid;
  const ys = curve.response_mean;
  const lo = curve.response_hdi_lo;
  const hi = curve.response_hdi_hi;
  if (!xs?.length) return null;

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...lo, ...ys);
  const yMax = Math.max(...hi, ...ys);
  const xR = xMax - xMin || 1;
  const yR = yMax - yMin || 1;
  const px = (x: number) => pad + ((x - xMin) / xR) * (W - 2 * pad);
  const py = (y: number) => H - pad - ((y - yMin) / yR) * (H - 2 * pad);

  // HDI ribbon polygon
  const ribbon = [
    ...xs.map((x, i) => `${px(x)},${py(hi[i])}`),
    ...xs.map((x, i) => `${px(x)},${py(lo[i])}`).reverse(),
  ].join(" ");
  const meanPath = xs.map((x, i) => `${i === 0 ? "M" : "L"}${px(x)},${py(ys[i])}`).join(" ");

  return (
    <Card title={`PDP — ${curve.treatment}`}
          subtitle={`source: ${curve.source}, peak |slope|=${curve.peak_marginal_slope?.toFixed(3) ?? "—"}`}>
      <svg width={W} height={H} style={{ background: "#fafaf7" }}>
        <polygon points={ribbon} fill="#7a3a3a" fillOpacity={0.18} stroke="none" />
        <path d={meanPath} fill="none" stroke="#7a3a3a" strokeWidth={1.5} />
        <line x1={pad} x2={W - pad} y1={py(0)} y2={py(0)}
              stroke="#bbb" strokeDasharray="2 3" />
        <text x={pad} y={H - 4} fontSize={9} fill="var(--muted)">
          dose ∈ [{xMin.toFixed(2)}, {xMax.toFixed(2)}]
        </text>
        <text x={W - pad} y={H - 4} fontSize={9} fill="var(--muted)" textAnchor="end">
          89% HDI
        </text>
        {curve.saturation_dose != null ? (
          <line x1={px(curve.saturation_dose)} x2={px(curve.saturation_dose)}
                y1={pad} y2={H - pad}
                stroke="#3a5e7a" strokeDasharray="3 3" />
        ) : null}
      </svg>
    </Card>
  );
}

// ------------------------------------------------------------------
// 3) Divergence summary
// ------------------------------------------------------------------

function DivergenceCard({ rep }: { rep: DivergenceReport }) {
  return (
    <Card title={`Divergence — ${rep.treatment}`}
          subtitle={`${rep.flagged_count}/${rep.cell_count} cells flagged`}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <Stat label="Pearson(β, τ)" value={rep.pearson_correlation.toFixed(3)} />
        <Stat label="Sign-agreement" value={`${(rep.sign_agreement_rate * 100).toFixed(1)}%`} />
        <Stat label="Mean |Δ|" value={rep.mean_abs_divergence.toFixed(3)} />
        <Stat label="Median |Δ|" value={rep.median_abs_divergence.toFixed(3)} />
        <Stat label="Max |Δ|" value={rep.max_abs_divergence.toFixed(3)} />
        <Stat label="Threshold" value={rep.threshold_used.toFixed(3)} />
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--muted)" }}>
        Flagged fraction: {(rep.flagged_fraction * 100).toFixed(1)}%
      </div>
    </Card>
  );
}

// ------------------------------------------------------------------
// 4) Scenario routing audit
// ------------------------------------------------------------------

function RoutingAuditCard({ data }: { data: ScenarioRoutingAuditData }) {
  return (
    <Card title="Scenario routing audit"
          subtitle={`${data.summary.causal_routed}/${data.summary.total_variables} variables routed through causal CATE`}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <Stat label="Total" value={String(data.summary.total_variables)} />
        <Stat label="Causal" value={String(data.summary.causal_routed)} tint="#3a7a5e" />
        <Stat label="Heuristic" value={String(data.summary.correlational_routed)} tint="#7a5e3a" />
        <Stat label="Anisotropic preds" value={String(data.summary.anisotropic_predictor_count)} tint="#7a3a3a" />
      </div>
      <table className="mono" style={{ width: "100%", marginTop: 12, fontSize: 11, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--line)" }}>
            <th style={{ textAlign: "left", padding: "4px 6px" }}>variable</th>
            <th style={{ textAlign: "left", padding: "4px 6px" }}>source</th>
            <th style={{ textAlign: "right", padding: "4px 6px" }}>mean mult.</th>
            <th style={{ textAlign: "right", padding: "4px 6px" }}>n cells</th>
          </tr>
        </thead>
        <tbody>
          {data.variables.map((v) => {
            const r = data.routing[v];
            const tag = r.source === "causal_cate"
              ? <Tag color="#3a7a5e">causal</Tag>
              : r.source === "length_mismatch"
                ? <Tag color="#7a3a3a">mismatch</Tag>
                : <Tag color="#7a5e3a">heuristic</Tag>;
            return (
              <tr key={v} style={{ borderBottom: "1px solid #f0f0e8" }}>
                <td style={{ padding: "4px 6px" }}>{v}</td>
                <td style={{ padding: "4px 6px" }}>{tag}</td>
                <td style={{ textAlign: "right", padding: "4px 6px" }}>
                  {r.mean_multiplier != null ? r.mean_multiplier.toFixed(3) : "—"}
                </td>
                <td style={{ textAlign: "right", padding: "4px 6px" }}>
                  {r.n_cells ?? r.n_cells_actual ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

// ------------------------------------------------------------------
// Page
// ------------------------------------------------------------------

interface State<T> { data: T | null; error: string | null; loading: boolean }

const initialState = <T,>(): State<T> => ({ data: null, error: null, loading: true });

export default function CausalPage() {
  const [kf, setKf] = useState<State<KernelFieldData>>(initialState());
  const [pdp, setPdp] = useState<State<CausalPDPData>>(initialState());
  const [div, setDiv] = useState<
    State<{ reports: Record<string, DivergenceReport> } | DivergenceReport>
  >(initialState());
  const [aud, setAud] = useState<State<ScenarioRoutingAuditData>>(initialState());

  useEffect(() => {
    let cancelled = false;
    const grab = async <T,>(
      fn: () => Promise<T>,
      setter: (s: State<T>) => void,
    ) => {
      try {
        const data = await fn();
        if (!cancelled) setter({ data, error: null, loading: false });
      } catch (e: any) {
        if (!cancelled) {
          setter({
            data: null,
            error: e?.message ?? String(e),
            loading: false,
          });
        }
      }
    };
    grab(getKernelFieldData, setKf);
    grab(getCausalPDPCurves, setPdp);
    grab(getCateVsGwrDivergence, setDiv);
    grab(getScenarioRoutingAudit, setAud);
    return () => { cancelled = true; };
  }, []);

  const divReports: DivergenceReport[] = (() => {
    const d = div.data;
    if (!d) return [];
    if ((d as any).reports) {
      return Object.values((d as { reports: Record<string, DivergenceReport> }).reports);
    }
    return [d as DivergenceReport];
  })();

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 16, overflowY: "auto", height: "100%" }}>
      <SectionHeader
        kicker="Phase C"
        label="Causal — KernelField, PDP, divergence, routing"
      />

      {/* KernelField */}
      {kf.loading ? (
        <Card title="Kernel field"><div className="mono" style={{ fontSize: 11 }}>Loading…</div></Card>
      ) : kf.error ? (
        <Card title="Kernel field">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
            No KernelField artifact available yet. Run Stages 0/1 to produce one.
          </div>
        </Card>
      ) : kf.data ? <KernelFieldViz data={kf.data} /> : null}

      {/* PDP curves */}
      {pdp.loading ? (
        <Card title="Causal PDP"><div className="mono" style={{ fontSize: 11 }}>Loading…</div></Card>
      ) : pdp.error ? (
        <Card title="Causal PDP">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
            No causal_pdp_curves artifact yet. Run Stage 3 with Bayesian-CATE enabled.
          </div>
        </Card>
      ) : pdp.data?.curves?.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 12 }}>
          {pdp.data.curves.map((c) => <PDPSparkline key={c.treatment} curve={c} />)}
        </div>
      ) : (
        <Card title="Causal PDP">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
            Artifact present but contains no curves.
          </div>
        </Card>
      )}

      {/* Divergence */}
      {div.loading ? (
        <Card title="CATE vs GWR divergence"><div className="mono" style={{ fontSize: 11 }}>Loading…</div></Card>
      ) : div.error ? (
        <Card title="CATE vs GWR divergence">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
            No cate_vs_gwr_divergence artifact yet.
          </div>
        </Card>
      ) : divReports.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(360px, 1fr))", gap: 12 }}>
          {divReports.map((r) => <DivergenceCard key={r.treatment} rep={r} />)}
        </div>
      ) : null}

      {/* Routing audit */}
      {aud.loading ? (
        <Card title="Scenario routing audit"><div className="mono" style={{ fontSize: 11 }}>Loading…</div></Card>
      ) : aud.error ? (
        <Card title="Scenario routing audit">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
            No scenario_routing_audit artifact yet. Run Stage 4 (scenarios).
          </div>
        </Card>
      ) : aud.data ? <RoutingAuditCard data={aud.data} /> : null}

      <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 8 }}>
        Endpoints: /results/kernel_field · /results/causal/pdp_curves ·
        /results/causal/divergence · /results/scenarios/routing_audit
      </div>

      {kf.data ? (
        <Card title="KernelField metadata">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
            <KeyVal label="outcome" value={kf.data.outcome_name} />
            <KeyVal label="predictors" value={String(kf.data.predictors.length)} />
            <KeyVal label="default ν" value={String(kf.data.matern_default_nu ?? "—")} />
            <KeyVal label="source" value={kf.data.source ?? "—"} />
          </div>
        </Card>
      ) : null}
    </div>
  );
}
