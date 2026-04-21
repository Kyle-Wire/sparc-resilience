import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, Stat, Tag, Btn, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { dataSummary, uploadData, getConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { DataSummary } from "@/lib/types";

export default function DataPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [target, setTarget] = useState<string>("AAT_z");
  const { notify } = useNotification();

  useEffect(() => {
    dataSummary().then(setSummary).catch(() => {});
    getConfig()
      .then((c) => {
        if (c.data?.target_column) setTarget(c.data.target_column);
      })
      .catch(() => {});
  }, []);

  const handleUpload = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({
        filters: [
          { name: "Data", extensions: ["csv", "parquet", "gpkg"] },
        ],
        multiple: false,
      });
      if (!path) return;
      // Create a File object from path for upload
      const resp = await fetch(`http://127.0.0.1:8008/data/select?path=${encodeURIComponent(path as string)}`, { method: "POST" });
      if (!resp.ok) throw new Error(await resp.text());
      const refreshed = await dataSummary();
      setSummary(refreshed);
      notify("success", "Data loaded successfully");
    } catch {
      // Fallback: file input
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".csv,.parquet,.gpkg";
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;
        try {
          await uploadData(file);
          const refreshed = await dataSummary();
          setSummary(refreshed);
          notify("success", "Data uploaded successfully");
        } catch (err) {
          notify("error", err instanceof Error ? err.message : "Upload failed");
        }
      };
      input.click();
    }
  }, [notify]);

  const columns = summary?.columns ?? [];
  const numericSummary = summary?.numeric_summary ?? {};
  const nRows = summary?.row_count ?? 0;
  const nCols = columns.length;
  const missingPct = 0; // computed from numeric_summary if needed

  const getRole = (col: string) => {
    if (col === target) return { tag: "Target", color: "var(--crimson)" };
    if (col === "x" || col === "y" || col === "POINT_X" || col === "POINT_Y") return { tag: "Coord", color: "var(--purple)" };
    if (col === "id" || col === "FID") return { tag: "ID", color: "var(--muted)" };
    return { tag: "Predictor", color: "var(--ink-2)" };
  };

  const getType = (col: string) => {
    const s = numericSummary[col];
    if (!s) return "str";
    if (Number.isInteger(s.min) && Number.isInteger(s.max)) return "int";
    return "float";
  };

  const getSummaryText = (col: string) => {
    const s = numericSummary[col];
    if (!s) return "—";
    return `${s.min?.toFixed?.(2) ?? s.min} … ${s.max?.toFixed?.(2) ?? s.max}`;
  };

  return (
    <div>
      <SectionHeader kicker="02 · setup" label="Data" right={<Btn small onClick={handleUpload}>Upload CSV</Btn>} />

      <StatGrid>
        <Stat label="Rows" value={nRows.toLocaleString()} tint="var(--ink)" />
        <Stat label="Columns" value={String(nCols)} tint="var(--ink)" />
        <Stat label="Missing" value={`${missingPct.toFixed(2)}%`} tint="var(--purple)" />
        <Stat label="Spatial density" value={nRows > 0 ? `≈${(nRows / 50000).toFixed(1)} / 30m²` : "—"} tint="var(--crimson)" />
      </StatGrid>

      <Card title="Schema" subtitle={summary ? `${summary.row_count} rows · ${summary.column_count} columns` : "no data loaded"}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--muted)" }}>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Column</th>
              <th style={thStyle}>Type</th>
              <th style={thStyle}>Summary</th>
              <th style={thStyle}>Role</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col, i) => {
              const role = getRole(col);
              return (
                <tr key={col} style={{ borderTop: "1px solid var(--line)" }}>
                  <td style={{ ...tdStyle, color: "var(--muted)" }} className="mono">
                    {String(i).padStart(2, "0")}
                  </td>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{col}</td>
                  <td style={tdStyle}>
                    <span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>
                      {getType(col)}
                    </span>
                  </td>
                  <td style={{ ...tdStyle, color: "var(--muted)" }} className="mono">
                    {getSummaryText(col)}
                  </td>
                  <td style={tdStyle}>
                    <Tag color={role.color}>{role.tag}</Tag>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
