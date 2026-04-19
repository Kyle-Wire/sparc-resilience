import { useState, useEffect } from "react";
import { Btn, Card, SectionHeader, Stat, Tag } from "@/components/ui/DesignSystem";
import { dataSummary, getConfig, uploadData } from "@/lib/api";
import type { DataSummary, ProjectConfig } from "@/lib/types";

const DEMO_COLS: [string, string, string][] = [
  ["id", "int", "54,701 unique"],
  ["x", "float", "RI SP · 237,112 … 258,441"],
  ["y", "float", "RI SP · 236,504 … 269,880"],
  ["AAT_z", "float", "target · −2.41 … +3.08"],
  ["Pct_Canopy", "float", "0 … 100"],
  ["Pct_Impervious", "float", "0 … 100"],
  ["NDVI", "float", "−0.12 … 0.91"],
  ["Albedo", "float", "0.04 … 0.48"],
  ["Elevation_m", "float", "0.4 … 88.2"],
  ["Distance_from_water_m", "float", "0 … 4,421"],
];

const TH = {
  padding: "6px 8px",
  fontWeight: 600,
  fontSize: 10,
  letterSpacing: "0.1em",
  textTransform: "uppercase" as const,
};
const TD = { padding: "7px 8px", fontSize: 12 };

function formatNumericSummary(stats: { min: number; max: number } | undefined): string {
  if (!stats) return "—";
  const fmt = (n: number) => {
    if (Number.isInteger(n)) return n.toLocaleString();
    return n.toFixed(2);
  };
  return `${fmt(stats.min)} … ${fmt(stats.max)}`;
}

function inferRole(
  col: string,
  config: ProjectConfig | null,
): { label: string; color: string } {
  if (config?.data?.target_column === col) return { label: "Target", color: "var(--crimson)" };
  if (config?.data?.coord_columns?.includes(col)) return { label: "Coord", color: "var(--purple)" };
  if (config?.data?.identifier_column === col) return { label: "ID", color: "var(--muted)" };
  if (col === "x" || col === "y") return { label: "Coord", color: "var(--purple)" };
  if (col === "id") return { label: "ID", color: "var(--muted)" };
  return { label: "Predictor", color: "var(--ink-2)" };
}

export default function DataPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [_loading, setLoading] = useState(true);
  const fileInputId = "data-upload-input";

  const load = () => {
    Promise.all([
      dataSummary().catch(() => null),
      getConfig().catch(() => null),
    ]).then(([sum, cfg]) => {
      setSummary(sum);
      setConfig(cfg);
      setLoading(false);
    });
  };

  useEffect(load, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadData(file);
      setLoading(true);
      load();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Upload failed");
    }
  };

  // Build column list from live data or fallback
  const cols: [string, string, string][] = summary
    ? summary.columns.map((col) => {
        const dtype = summary.dtypes[col] ?? "unknown";
        const shortType = dtype.includes("float") ? "float" : dtype.includes("int") ? "int" : dtype;
        const stats = summary.numeric_summary[col];
        const desc = stats ? formatNumericSummary(stats) : `${summary.row_count.toLocaleString()} entries`;
        return [col, shortType, desc];
      })
    : DEMO_COLS;

  const rowCount = summary?.row_count?.toLocaleString() ?? "54,701";
  const colCount = summary?.column_count ?? cols.length;
  const dataFile = config?.data?.file_path ?? "examples/brown4.csv";

  return (
    <div>
      <SectionHeader
        kicker="02 · setup"
        label="Data"
        right={
          <>
            <Btn small onClick={() => document.getElementById(fileInputId)?.click()}>
              Upload CSV
            </Btn>
            <input
              id={fileInputId}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={handleUpload}
            />
          </>
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <Stat label="Rows" value={rowCount} tint="var(--ink)" />
        <Stat label="Columns" value={String(colCount)} tint="var(--ink)" />
        <Stat label="Missing" value="0.00%" tint="var(--purple)" />
        <Stat label="Spatial density" value="≈1.1 / 30m²" tint="var(--crimson)" />
      </div>

      <Card title="Schema" subtitle={dataFile}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--muted)" }}>
              <th style={TH}>#</th>
              <th style={TH}>Column</th>
              <th style={TH}>Type</th>
              <th style={TH}>Summary</th>
              <th style={TH}>Role</th>
            </tr>
          </thead>
          <tbody>
            {cols.map((c, i) => {
              const role = inferRole(c[0], config);
              return (
                <tr key={c[0]} style={{ borderTop: "1px solid var(--line)" }}>
                  <td style={{ ...TD, color: "var(--muted)" }} className="mono">
                    {String(i).padStart(2, "0")}
                  </td>
                  <td style={{ ...TD, fontWeight: 600 }}>{c[0]}</td>
                  <td style={TD}>
                    <span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>
                      {c[1]}
                    </span>
                  </td>
                  <td style={{ ...TD, color: "var(--muted)" }} className="mono">
                    {c[2]}
                  </td>
                  <td style={TD}>
                    <Tag color={role.color}>{role.label}</Tag>
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
