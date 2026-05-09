/**
 * ArtifactBrowserPanel — flat list of every manifest artifact, with
 * stage tag and size.
 */
import { useMemo } from "react";
import { Panel, PanelEmpty, Pill, Btn } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";

interface Row {
  stage: string;
  name: string;
  size?: number | null;
  partial: boolean;
}

export default function ArtifactBrowserPanel() {
  const manifest = useManifest();
  const navigate = useInsightsNavigate();

  const rows = useMemo<Row[]>(() => {
    const stages = manifest.manifest?.stages ?? {};
    const out: Row[] = [];
    for (const st of Object.values(stages)) {
      for (const [name, meta] of Object.entries(st.artifacts ?? {})) {
        out.push({ stage: st.stage, name, size: meta.size_bytes ?? null, partial: meta.partial });
      }
    }
    return out.sort((a, b) => (a.stage + a.name).localeCompare(b.stage + b.name));
  }, [manifest.manifest]);

  if (!rows.length) {
    return (
      <Panel title="Artifact browser" subtitle="manifest">
        <PanelEmpty
          reason="No artifacts yet"
          hint="Run any stage to populate the manifest."
          action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
        />
      </Panel>
    );
  }

  return (
    <Panel title="Artifact browser" subtitle={`${rows.length} artifact${rows.length === 1 ? "" : "s"}`}>
      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Stage</th>
              <th style={{ padding: "6px 8px" }}>Artifact</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Size</th>
              <th style={{ padding: "6px 8px" }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.stage}-${r.name}`} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                <td style={{ padding: "6px 8px" }}>
                  <Pill>stage {r.stage}</Pill>
                </td>
                <td style={{ padding: "6px 8px" }}>{r.name}</td>
                <td
                  style={{
                    padding: "6px 8px",
                    textAlign: "right",
                    color: "var(--ink-2)",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {r.size != null ? formatBytes(r.size) : "—"}
                </td>
                <td style={{ padding: "6px 8px" }}>
                  {r.partial ? <Pill color="var(--crimson)">partial</Pill> : <Pill>ok</Pill>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
