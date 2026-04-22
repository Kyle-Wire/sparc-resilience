import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Stat, Tag, Btn, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { getConfig, dataSummary, dataPreview, dataGeoJson, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import SpatialMap from "@/components/map/SpatialMap";
import type { GeoJsonData } from "@/lib/types";

interface VariableInfo {
  name: string;
  type: "float" | "int" | "str";
  role: "target" | "predictor" | "coord" | "id";
  selected: boolean;
  actionable: boolean;
  mean: number;
  std: number;
  min: number;
  max: number;
  corr: number;
  sparkline: number[];
}

export default function VariablesPage() {
  const [variables, setVariables] = useState<VariableInfo[]>([]);
  const [filter, setFilter] = useState<string>("");
  const [sortBy, setSortBy] = useState<"name" | "corr">("corr");
  const [selectedVar, setSelectedVar] = useState<string>("");
  const [selectedGeo, setSelectedGeo] = useState<GeoJsonData | null>(null);
  const [geoLoading, setGeoLoading] = useState(false);
  const { notify } = useNotification();

  useEffect(() => {
    Promise.all([dataSummary(), getConfig(), dataPreview(200)])
      .then(([summary, config, preview]) => {
        const target = config.data?.target_column ?? "";
        const cols = summary.columns ?? [];
        const ns = summary.numeric_summary ?? {};
        const predictors = config.predictors ?? cols;
        const rows = preview?.rows ?? [];

        // Build column value arrays from preview rows
        const colVals: Record<string, number[]> = {};
        for (const col of cols) {
          colVals[col] = rows
            .map((r) => Number(r[col]))
            .filter((v) => Number.isFinite(v));
        }

        // Compute Pearson correlation with target column
        const targetVals = colVals[target] ?? [];
        const tMean = targetVals.length ? targetVals.reduce((a, b) => a + b, 0) / targetVals.length : 0;
        const tStd = targetVals.length
          ? Math.sqrt(targetVals.reduce((a, b) => a + (b - tMean) ** 2, 0) / targetVals.length)
          : 0;

        function pearson(col: string): number {
          const vals = colVals[col] ?? [];
          if (!vals.length || !targetVals.length || tStd === 0) return 0;
          // Align by index (same rows)
          const n = Math.min(vals.length, targetVals.length);
          const vMean = vals.slice(0, n).reduce((a, b) => a + b, 0) / n;
          const vStd = Math.sqrt(vals.slice(0, n).reduce((a, b) => a + (b - vMean) ** 2, 0) / n);
          if (vStd === 0) return 0;
          let cov = 0;
          for (let i = 0; i < n; i++) cov += (vals[i] - vMean) * (targetVals[i] - tMean);
          return cov / (n * vStd * tStd);
        }

        const vars: VariableInfo[] = cols.map((col) => {
          const s = ns[col];
          const isTarget = col === target;
          const isCoord = /^(x|y|POINT_X|POINT_Y|longitude|latitude)$/i.test(col);
          const isId = /^(id|fid|objectid)$/i.test(col);
          return {
            name: col,
            type: s ? (Number.isInteger(s.min) && Number.isInteger(s.max) ? "int" : "float") : "str",
            role: isTarget ? "target" : isCoord ? "coord" : isId ? "id" : "predictor",
            selected: predictors.includes(col) || isTarget,
            actionable: !isTarget && !isCoord && !isId,
            mean: s?.mean ?? 0,
            std: s?.std ?? 0,
            min: s?.min ?? 0,
            max: s?.max ?? 0,
            corr: isTarget ? 1 : pearson(col),
            sparkline: colVals[col] ?? [],
          };
        });
        setVariables(vars);
      })
      .catch(() => {});
  }, []);

  const handleToggle = useCallback((name: string) => {
    setVariables((prev) => prev.map((v) => v.name === name ? { ...v, selected: !v.selected } : v));
  }, []);

  // Auto-select first numeric predictor for the spatial preview.
  useEffect(() => {
    if (selectedVar || variables.length === 0) return;
    const firstNumeric = variables.find((v) => v.type !== "str" && v.role !== "id");
    if (firstNumeric) setSelectedVar(firstNumeric.name);
  }, [variables, selectedVar]);

  // Refetch GeoJSON whenever the highlighted variable changes.
  useEffect(() => {
    if (!selectedVar) { setSelectedGeo(null); return; }
    setGeoLoading(true);
    dataGeoJson(selectedVar)
      .then((g) => setSelectedGeo(g))
      .catch(() => setSelectedGeo(null))
      .finally(() => setGeoLoading(false));
  }, [selectedVar]);

  const handleSave = useCallback(async () => {
    const selected = variables.filter((v) => v.selected && v.role === "predictor").map((v) => v.name);
    try {
      await saveConfig({ predictors: selected });
      notify("success", `${selected.length} predictors saved`);
    } catch {
      notify("error", "Failed to save predictor selection");
    }
  }, [variables, notify]);

  const filtered = variables.filter((v) => !filter || v.name.toLowerCase().includes(filter.toLowerCase()));
  const sorted = [...filtered].sort((a, b) =>
    sortBy === "corr" ? Math.abs(b.corr) - Math.abs(a.corr) : a.name.localeCompare(b.name),
  );

  const selected = variables.filter((v) => v.selected);
  const actionable = variables.filter((v) => v.actionable && v.selected);
  const maxCorr = Math.max(0, ...variables.filter((v) => v.role === "predictor").map((v) => Math.abs(v.corr)));

  return (
    <div>
      <SectionHeader
        kicker="05 · analysis"
        label="Variables"
        right={<Btn primary small onClick={handleSave}>Save selection</Btn>}
      />

      <StatGrid>
        <Stat label="Total" value={String(variables.length)} tint="var(--ink)" />
        <Stat label="Selected" value={String(selected.length)} tint="var(--crimson)" />
        <Stat label="Actionable" value={String(actionable.length)} tint="var(--purple)" />
        <Stat label="Max |ρ|" value={maxCorr.toFixed(3)} tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.6fr) minmax(280px, 1fr)", gap: 14 }}>
      <Card
        title="Variable ledger"
        subtitle={
          variables.length > 0 ? (
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              type="text"
              placeholder="Filter columns..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{
                border: "1px solid var(--line)",
                borderRadius: 4,
                padding: "4px 8px",
                fontSize: 11,
                fontFamily: "inherit",
                width: 160,
                background: "#fff",
              }}
            />
            <button
              className="mono"
              onClick={() => setSortBy(sortBy === "corr" ? "name" : "corr")}
              style={{
                fontSize: 9.5,
                color: "var(--muted)",
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "inherit",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              sort: {sortBy} {sortBy === "corr" ? "↓" : "↑"}
            </button>
          </div>
          ) : undefined
        }
      >
        {variables.length === 0 ? (
          <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--muted)", fontSize: 13 }}>
            Load a dataset to see variable details
          </div>
        ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--muted)" }}>
              <th style={{ ...thStyle, width: 30 }}>☑</th>
              <th style={thStyle}>Variable</th>
              <th style={thStyle}>Type</th>
              <th style={thStyle}>Role</th>
              <th style={thStyle}>Mean</th>
              <th style={thStyle}>Std</th>
              <th style={thStyle}>Range</th>
              <th style={{ ...thStyle, textAlign: "right" }} title="Absolute Pearson correlation with the target variable">|ρ|</th>
              <th style={{ ...thStyle, width: 80 }}>Distribution</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((v) => (
              <tr
                key={v.name}
                onClick={() => setSelectedVar(v.name)}
                style={{
                  borderTop: "1px solid var(--line)",
                  opacity: v.selected ? 1 : 0.5,
                  background: v.name === selectedVar ? "#fff8ef" : undefined,
                  cursor: "pointer",
                }}
              >
                <td style={tdStyle}>
                  <input
                    type="checkbox"
                    checked={v.selected}
                    onChange={() => handleToggle(v.name)}
                    onClick={(e) => e.stopPropagation()}
                    disabled={v.role === "target"}
                    style={{ accentColor: "var(--crimson)" }}
                  />
                </td>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{v.name}</td>
                <td style={tdStyle}>
                  <span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>{v.type}</span>
                </td>
                <td style={tdStyle}>
                  <Tag
                    color={
                      v.role === "target"
                        ? "var(--crimson)"
                        : v.role === "coord"
                        ? "var(--purple)"
                        : v.role === "id"
                        ? "var(--muted)"
                        : "var(--ink-2)"
                    }
                  >
                    {v.role}
                  </Tag>
                </td>
                <td style={tdStyle} className="mono">{v.mean.toFixed(2)}</td>
                <td style={tdStyle} className="mono">{v.std.toFixed(2)}</td>
                <td style={tdStyle} className="mono" title={`${v.min} … ${v.max}`}>
                  {v.min.toFixed(1)}–{v.max.toFixed(1)}
                </td>
                <td className="mono" style={{ ...tdStyle, textAlign: "right", fontWeight: 700, color: Math.abs(v.corr) > 0.3 ? "var(--crimson)" : "var(--muted)" }}>
                  {Math.abs(v.corr).toFixed(3)}
                </td>
                <td style={tdStyle}>
                  <MiniHistogram data={v.sparkline} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        )}
      </Card>

      <Card
        title="Spatial preview"
        subtitle={selectedVar ? `${selectedVar}${geoLoading ? " · loading…" : ""}` : "click a row to colour the map"}
      >
        <div style={{ height: 360, borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)" }}>
          {selectedGeo && selectedVar ? (
            <SpatialMap
              geojson={selectedGeo}
              colorField={selectedVar}
              mode="scatter"
              height="100%"
              palette="sparc"
            />
          ) : (
            <div style={{
              display: "flex", height: "100%", alignItems: "center", justifyContent: "center",
              color: "var(--muted)", fontSize: 12, background: "#faf8f4",
            }}>
              {selectedVar ? "No spatial data for this variable" : "Select a variable from the ledger"}
            </div>
          )}
        </div>
      </Card>
      </div>
    </div>
  );
}

function MiniHistogram({ data }: { data: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);

    // Bin data into 7 buckets
    const nBins = 7;
    const min = Math.min(...data), max = Math.max(...data);
    const range = max - min || 1;
    const bins = new Array(nBins).fill(0);
    for (const v of data) {
      const idx = Math.min(nBins - 1, Math.floor(((v - min) / range) * nBins));
      bins[idx]++;
    }
    const maxBin = Math.max(1, ...bins);
    const barW = w / nBins;

    for (let i = 0; i < nBins; i++) {
      const barH = (bins[i] / maxBin) * (h - 2);
      ctx.fillStyle = "var(--crimson)";
      ctx.globalAlpha = 0.5 + 0.5 * (bins[i] / maxBin);
      ctx.fillRect(i * barW + 1, h - barH, barW - 2, barH);
    }
    ctx.globalAlpha = 1;
  }, [data]);

  return <canvas ref={ref} style={{ width: 70, height: 20, display: "block" }} />;
}
