/**
 * KernelFieldPanel — bandwidth + anisotropy summary per predictor.
 * Stage 2 spatial structure diagnostic.
 */
import { useEffect, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { getKernelFieldData } from "@/lib/api";
import type { KernelFieldData } from "@/lib/types";

export default function KernelFieldPanel() {
  const manifest = useManifest();
  const stage2 = manifest.stage("2");
  const present =
    !!stage2 && Object.keys(stage2.artifacts ?? {}).some((a) => a.includes("kernel_field"));
  const [data, setData] = useState<KernelFieldData | null>(null);

  useEffect(() => {
    if (!present) {
      setData(null);
      return;
    }
    getKernelFieldData().then(setData).catch(() => setData(null));
  }, [present, manifest.lastUpdated]);

  if (!present) {
    return (
      <Panel title="Kernel field" subtitle="stage 2 · spatial structure">
        <PanelEmpty
          reason="No kernel-field output"
          hint="Run stage 2 with a kernel/MGWR-capable model."
        />
      </Panel>
    );
  }
  if (!data?.predictors?.length) {
    return (
      <Panel title="Kernel field" subtitle="stage 2 · spatial structure">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  const warn = data.bandwidth_mismatch_warnings ?? [];

  return (
    <Panel
      title="Kernel field"
      subtitle={`outcome · ${data.outcome_name}`}
      actions={
        warn.length > 0 ? (
          <Pill color="var(--crimson, #b91c1c)">
            {warn.length} warning{warn.length === 1 ? "" : "s"}
          </Pill>
        ) : null
      }
    >
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
            {data.predictors.map((p) => (
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
          {warn.slice(0, 5).map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

const cell = {
  padding: "6px 8px",
  textAlign: "right" as const,
  fontVariantNumeric: "tabular-nums" as const,
  color: "var(--ink-2)",
};

function fmt(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(2);
}
