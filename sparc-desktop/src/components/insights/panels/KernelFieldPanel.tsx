/**
 * KernelFieldPanel — bandwidth + anisotropy summary per predictor.
 *
 * Renders whichever spatial-structure artifact is available, in this
 * order of preference:
 *   1. Stage 1/0 ``kernel_field``                  (legacy MGWR-derived)
 *   2. Stage 0  ``cross_correlogram_kernel_field`` (Phase-3 cross summary)
 *
 * When the cross-correlogram fallback is in play, we additionally
 * surface the supporting Stage-0 artifacts (anisotropy ellipses,
 * Matérn fit, effective-range matrix) so the panel becomes a single
 * spatial-structure briefing instead of a one-table affair.
 */
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { getKernelFieldData, getArtifactJson } from "@/lib/api";
import type { AnisotropyEntry, BandwidthMismatchWarning, KernelFieldData, KernelFieldPair } from "@/lib/types";

interface MaternFit {
  variable_names?: string[];
  per_variable?: Record<string, {
    nu?: number | null;
    range_m?: number | null;
    sigma2?: number | null;
    rmse?: number | null;
  }>;
  stationarity_warnings?: string[];
}

interface AnisotropyArtifact {
  per_variable?: AnisotropyEntry[];
  strong_anisotropy_variables?: string[];
}

interface EffectiveRangeMatrix {
  variable_names: string[];
  range_matrix: (number | null)[][];
  mismatch_factor?: number;
  mismatch_warnings?: { pair: string; range_pair_m?: number; marginal_i_m?: number | null; marginal_j_m?: number | null; ratio_max?: number }[];
  significant_pair_count?: number;
}

export default function KernelFieldPanel() {
  const manifest = useManifest();
  const hasLegacy =
    !!manifest.lookup("1", "kernel_field") || !!manifest.lookup("0", "kernel_field");
  const hasCross = !!manifest.lookup("0", "cross_correlogram_kernel_field");
  const present = hasLegacy || hasCross;

  const { data } = useResult<KernelFieldData>(
    present ? "s0:kernel_field" : null,
    getKernelFieldData,
  );
  const hasAniso = present && !!manifest.lookup("0", "correlogram_anisotropy");
  const hasMatern = present && !!manifest.lookup("0", "correlogram_matern_fit");
  const hasErm = present && !!manifest.lookup("0", "effective_range_matrix");
  const { data: aniso } = useResult<AnisotropyArtifact>(
    hasAniso ? "s0:anisotropy" : null,
    () => getArtifactJson<AnisotropyArtifact>("0", "correlogram_anisotropy"),
  );
  const { data: matern } = useResult<MaternFit>(
    hasMatern ? "s0:matern_fit" : null,
    () => getArtifactJson<MaternFit>("0", "correlogram_matern_fit"),
  );
  const { data: erm } = useResult<EffectiveRangeMatrix>(
    hasErm ? "s0:effective_range_matrix" : null,
    () => getArtifactJson<EffectiveRangeMatrix>("0", "effective_range_matrix"),
  );

  if (!present) {
    return (
      <PanelGate panelId="kernel_field" title="Kernel field" subtitle="stage 0/2 · spatial structure">
        <Panel title="Kernel field" subtitle="stage 0/2 · spatial structure">
          <PanelEmpty
            reason="No kernel-field output"
            hint="Run Stage 0 (correlogram) or Stage 2 with a kernel/MGWR-capable model."
          />
        </Panel>
      </PanelGate>
    );
  }
  if (!data) {
    return (
      <PanelGate panelId="kernel_field" title="Kernel field" subtitle="stage 0/2 · spatial structure">
        <Panel title="Kernel field" subtitle="stage 0/2 · spatial structure">
          <PanelEmpty reason="Loading…" />
        </Panel>
      </PanelGate>
    );
  }

  const isCross = data.source === "cross_correlogram" || (!!data.pairs && !data.predictors);
  const subtitle = isCross
    ? `cross-correlogram · ${data.variable_names?.length ?? 0} variables`
    : `outcome · ${data.outcome_name ?? "—"}`;

  return (
    <Panel
      title="Kernel field"
      subtitle={subtitle}
      actions={
        data.source ? (
          <Pill color={isCross ? "var(--ink-2)" : "var(--purple)"}>
            {isCross ? "cross-correlogram" : "MGWR kernel_field"}
          </Pill>
        ) : null
      }
    >
      {isCross
        ? <CrossCorrelogramSections data={data} aniso={aniso} matern={matern} erm={erm} />
        : <LegacyPredictorTable data={data} />}
    </Panel>
  );
}

// ------------------------------------------------------------------
// Legacy MGWR-derived KernelField table
// ------------------------------------------------------------------

function formatMismatchWarning(w: string | BandwidthMismatchWarning): string {
  if (typeof w === "string") return w;
  const rangeStr = w.range_pair_m != null ? ` pair=${fmtLag(w.range_pair_m)}m` : "";
  const margStr =
    w.marginal_i_m != null || w.marginal_j_m != null
      ? ` marginals=${fmtLag(w.marginal_i_m)}/${fmtLag(w.marginal_j_m)}m`
      : "";
  const ratioStr = w.ratio_max != null ? ` ratio=${w.ratio_max.toFixed(2)}×` : "";
  return `${w.pair}:${rangeStr}${margStr}${ratioStr}`.trim();
}

function LegacyPredictorTable({ data }: { data: KernelFieldData }) {
  const warn = data.bandwidth_mismatch_warnings ?? [];
  const predictors = data.predictors ?? [];
  if (predictors.length === 0) {
    return <PanelEmpty reason="No predictors in kernel_field artifact." />;
  }
  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Predictor</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>κ</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>κx</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>κy</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>θ°</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>ν</th>
              <th style={{ padding: "6px 8px" }}>aniso</th>
            </tr>
          </thead>
          <tbody>
            {predictors.map((p) => (
              <tr key={p.name} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                <td style={{ padding: "6px 8px" }}>{p.name}</td>
                <td style={cell}>{fmt(p.kappa)}</td>
                <td style={cell}>{fmt(p.kappa_x)}</td>
                <td style={cell}>{fmt(p.kappa_y)}</td>
                <td style={cell}>
                  {p.theta_rad != null ? ((p.theta_rad * 180) / Math.PI).toFixed(0) : "—"}
                </td>
                <td style={cell}>{fmt(p.nu)}</td>
                <td style={{ padding: "6px 8px" }}>
                  {p.is_anisotropic ? <Pill color="var(--purple)">anisotropic</Pill> : <Pill>iso</Pill>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {warn.length > 0 && (
        <ul style={{ marginTop: 10, fontSize: 11, color: "var(--ink-2)", paddingLeft: 18 }}>
          {warn.slice(0, 5).map((w, i) => <li key={i}>{formatMismatchWarning(w)}</li>)}
        </ul>
      )}
    </>
  );
}

// ------------------------------------------------------------------
// Cross-correlogram fallback: pairs + anisotropy + Matérn + range matrix
// ------------------------------------------------------------------

interface CrossSectionsProps {
  data: KernelFieldData;
  aniso: AnisotropyArtifact | null;
  matern: MaternFit | null;
  erm: EffectiveRangeMatrix | null;
}

function CrossCorrelogramSections({ data, aniso, matern, erm }: CrossSectionsProps) {
  const pairs = Object.values(data.pairs ?? {}) as KernelFieldPair[];
  const significant = pairs.filter((p) => p.significant_sym || p.significant_anti);
  const topPairs = (significant.length > 0 ? significant : pairs)
    .slice()
    .sort((a, b) => Math.abs(b.peak_sym) - Math.abs(a.peak_sym))
    .slice(0, 12);
  const antisymFlags = data.antisymmetric_flags ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* --- Pair table ---------------------------------------------- */}
      <Section
        title="Top variable pairs"
        meta={
          <span style={{ fontSize: 10, color: "var(--ink-2)" }}>
            {pairs.length} pairs · {significant.length} significant
            {antisymFlags.length > 0 ? ` · ${antisymFlags.length} antisym` : ""}
          </span>
        }
      >
        {pairs.length === 0 ? (
          <PanelEmpty reason="No cross-pair entries in payload." />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                  <th style={{ padding: "6px 8px" }}>Pair (i → j)</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>peak sym</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>lag sym (m)</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>eff. range (m)</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>peak anti</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>θ°</th>
                  <th style={{ padding: "6px 8px" }}>flags</th>
                </tr>
              </thead>
              <tbody>
                {topPairs.map((p) => (
                  <tr key={`${p.i_name}|${p.j_name}`} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                    <td style={{ padding: "6px 8px" }}>{p.i_name} → {p.j_name}</td>
                    <td style={cell}>{fmt(p.peak_sym, 3)}</td>
                    <td style={cell}>{fmtLag(p.peak_lag_sym_m)}</td>
                    <td style={cell}>{p.effective_range_sym_m == null ? "—" : fmtLag(p.effective_range_sym_m)}</td>
                    <td style={cell}>{fmt(p.peak_anti, 3)}</td>
                    <td style={cell}>{p.peak_angle_anti_deg == null ? "—" : p.peak_angle_anti_deg.toFixed(0)}</td>
                    <td style={{ padding: "6px 8px" }}>
                      {p.significant_sym && <Pill color="var(--green, #2f7d32)">sym</Pill>}
                      {p.significant_anti && <Pill color="var(--purple)">anti</Pill>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {pairs.length > topPairs.length && (
              <div style={{ fontSize: 10, color: "var(--ink-2)", marginTop: 6 }}>
                Showing top {topPairs.length} of {pairs.length} pairs (ranked by |peak sym|).
              </div>
            )}
          </div>
        )}
      </Section>

      {/* --- Effective-range matrix --------------------------------- */}
      {erm && erm.variable_names?.length > 0 && (
        <Section
          title="Effective-range matrix"
          meta={
            <span style={{ fontSize: 10, color: "var(--ink-2)" }}>
              {erm.significant_pair_count ?? "?"} significant ·
              mismatch ≥ {erm.mismatch_factor ?? "—"}× → {erm.mismatch_warnings?.length ?? 0} warnings
            </span>
          }
        >
          <RangeMatrix data={erm} />
        </Section>
      )}

      {/* --- Anisotropy snapshot ----------------------------------- */}
      {aniso?.per_variable?.length ? (
        <Section
          title="Anisotropy"
          meta={
            <span style={{ fontSize: 10, color: "var(--ink-2)" }}>
              {aniso.strong_anisotropy_variables?.length ?? 0} strongly anisotropic
            </span>
          }
        >
          <AnisotropyTable rows={aniso.per_variable} />
        </Section>
      ) : null}

      {/* --- Matérn fit summary ------------------------------------ */}
      {matern?.per_variable && Object.keys(matern.per_variable).length > 0 && (
        <Section title="Matérn fit per variable">
          <MaternTable matern={matern} />
        </Section>
      )}
    </div>
  );
}

function Section({ title, meta, children }: {
  title: string;
  meta?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "baseline",
        marginBottom: 6, paddingBottom: 4, borderBottom: "1px solid var(--line)",
      }}>
        <strong style={{ fontSize: 11, letterSpacing: 0.4, textTransform: "uppercase", color: "var(--ink-2)" }}>
          {title}
        </strong>
        {meta}
      </div>
      {children}
    </section>
  );
}

function RangeMatrix({ data }: { data: EffectiveRangeMatrix }) {
  const { variable_names, range_matrix } = data;
  const allVals = range_matrix.flat().filter((v): v is number => v != null && Number.isFinite(v));
  const max = allVals.length ? Math.max(...allVals) : 1;
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="mono" style={{ borderCollapse: "collapse", fontSize: 10 }}>
        <thead>
          <tr>
            <th style={{ padding: "4px 6px" }} />
            {variable_names.map((n) => (
              <th key={n} style={{ padding: "4px 6px", textAlign: "right", color: "var(--ink-2)" }}>
                {n}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {variable_names.map((rn, i) => (
            <tr key={rn}>
              <th style={{ padding: "4px 6px", textAlign: "right", color: "var(--ink-2)" }}>{rn}</th>
              {variable_names.map((cn, j) => {
                const v = range_matrix[i]?.[j];
                const intensity = v != null && Number.isFinite(v) ? v / max : 0;
                const bg = v == null
                  ? "transparent"
                  : `rgba(122, 58, 58, ${0.08 + 0.55 * intensity})`;
                return (
                  <td
                    key={cn}
                    style={{
                      padding: "4px 6px",
                      textAlign: "right",
                      background: bg,
                      fontVariantNumeric: "tabular-nums",
                      color: "var(--ink-1)",
                      minWidth: 56,
                    }}
                  >
                    {v == null ? "—" : fmtLag(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AnisotropyTable({ rows }: { rows: AnisotropyEntry[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
            <th style={{ padding: "6px 8px" }}>Variable</th>
            <th style={{ padding: "6px 8px", textAlign: "right" }}>range_x</th>
            <th style={{ padding: "6px 8px", textAlign: "right" }}>range_y</th>
            <th style={{ padding: "6px 8px", textAlign: "right" }}>aspect</th>
            <th style={{ padding: "6px 8px", textAlign: "right" }}>θ°</th>
            <th style={{ padding: "6px 8px" }}>flag</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
              <td style={{ padding: "6px 8px" }}>{r.name}</td>
              <td style={cell}>{r.range_x == null ? "—" : fmtLag(r.range_x)}</td>
              <td style={cell}>{r.range_y == null ? "—" : fmtLag(r.range_y)}</td>
              <td style={cell}>{r.aspect_ratio == null ? "—" : r.aspect_ratio.toFixed(2)}</td>
              <td style={cell}>
                {r.theta_deg != null
                  ? r.theta_deg.toFixed(0)
                  : r.theta_rad != null ? ((r.theta_rad * 180) / Math.PI).toFixed(0) : "—"}
              </td>
              <td style={{ padding: "6px 8px" }}>
                {r.is_anisotropic
                  ? <Pill color="var(--purple)">anisotropic</Pill>
                  : <Pill>iso</Pill>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MaternTable({ matern }: { matern: MaternFit }) {
  const entries = Object.entries(matern.per_variable ?? {});
  const warns = matern.stationarity_warnings ?? [];
  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Variable</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>ν</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>range (m)</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>σ²</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, fit]) => (
              <tr key={name} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                <td style={{ padding: "6px 8px" }}>{name}</td>
                <td style={cell}>{fmt(fit.nu)}</td>
                <td style={cell}>{fit.range_m == null ? "—" : fmtLag(fit.range_m)}</td>
                <td style={cell}>{fmt(fit.sigma2, 3)}</td>
                <td style={cell}>{fmt(fit.rmse, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {warns.length > 0 && (
        <ul style={{ marginTop: 8, fontSize: 11, color: "var(--ink-2)", paddingLeft: 18 }}>
          {warns.slice(0, 5).map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}
    </>
  );
}

const cell = {
  padding: "6px 8px",
  textAlign: "right" as const,
  fontVariantNumeric: "tabular-nums" as const,
  color: "var(--ink-2)",
};

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
}

function fmtLag(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(0);
}
