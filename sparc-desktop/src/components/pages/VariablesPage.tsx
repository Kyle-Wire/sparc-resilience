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
  treatment: boolean;
  actionable: boolean;
  mean: number;
  std: number;
  min: number;
  max: number;
  corr: number;
}

function computePearson(aVals: number[], bVals: number[]): number {
  const n = Math.min(aVals.length, bVals.length);
  if (n === 0) return 0;
  const aSlice = aVals.slice(0, n);
  const bSlice = bVals.slice(0, n);
  const aMean = aSlice.reduce((s, v) => s + v, 0) / n;
  const bMean = bSlice.reduce((s, v) => s + v, 0) / n;
  const aStd = Math.sqrt(aSlice.reduce((s, v) => s + (v - aMean) ** 2, 0) / n);
  const bStd = Math.sqrt(bSlice.reduce((s, v) => s + (v - bMean) ** 2, 0) / n);
  if (aStd === 0 || bStd === 0) return 0;
  let cov = 0;
  for (let i = 0; i < n; i++) cov += (aSlice[i] - aMean) * (bSlice[i] - bMean);
  return cov / (n * aStd * bStd);
}

function roleToggleStyle(active: boolean, activeColor: string) {
  return {
    width: 20,
    height: 20,
    borderRadius: 3,
    border: `1.5px solid ${active ? activeColor : "var(--line)"}`,
    background: active ? activeColor : "transparent",
    color: active ? "#fff" : "var(--muted)",
    fontSize: 9,
    fontWeight: 800,
    cursor: "pointer",
    fontFamily: "var(--font-mono, monospace)",
    letterSpacing: 0,
    lineHeight: "1",
    display: "inline-flex" as const,
    alignItems: "center",
    justifyContent: "center",
    padding: 0,
    transition: "all 0.12s",
  };
}

export default function VariablesPage() {
  const [variables, setVariables] = useState<VariableInfo[]>([]);
  const [target, setTarget] = useState<string>("");
  const [filter, setFilter] = useState<string>("");
  const [sortBy, setSortBy] = useState<"name" | "corr">("corr");
  const [selectedVar, setSelectedVar] = useState<string>("");
  const [selectedGeo, setSelectedGeo] = useState<GeoJsonData | null>(null);
  const [geoLoading, setGeoLoading] = useState(false);
  const colValsRef = useRef<Record<string, number[]>>({});
  const origDataRef = useRef<Record<string, unknown>>({});
  const origCausalRef = useRef<Record<string, unknown>>({});
  const { notify } = useNotification();

  useEffect(() => {
    Promise.all([dataSummary(), getConfig(), dataPreview(200)])
      .then(([summary, config, preview]) => {
        const tgt = config.data?.target_column ?? "";
        setTarget(tgt);
        origDataRef.current = (config.data ?? {}) as Record<string, unknown>;
        origCausalRef.current = (config.causal ?? {}) as Record<string, unknown>;

        const cols = summary.columns ?? [];
        const ns = summary.numeric_summary ?? {};

        // Support predictors as string[] or { base_model: string[], treatment: string[] }
        const rawPreds = config.predictors as unknown;
        const predictorList: string[] = Array.isArray(rawPreds)
          ? rawPreds
          : Array.isArray((rawPreds as Record<string, unknown>)?.base_model)
          ? ((rawPreds as Record<string, unknown>).base_model as string[])
          : cols;
        const treatmentList: string[] = Array.isArray(
          (rawPreds as Record<string, unknown>)?.treatment,
        )
          ? ((rawPreds as Record<string, unknown>).treatment as string[])
          : [];
        const actionableList: string[] = config.causal?.actionable_variables ?? [];

        const rows = preview?.rows ?? [];
        const colVals: Record<string, number[]> = {};
        for (const col of cols) {
          colVals[col] = rows
            .map((r) => Number(r[col]))
            .filter((v) => Number.isFinite(v));
        }
        colValsRef.current = colVals;

        const targetVals = colVals[tgt] ?? [];

        const vars: VariableInfo[] = cols.map((col) => {
          const s = ns[col];
          const isTarget = col === tgt;
          const isCoord = /^(x|y|POINT_X|POINT_Y|longitude|latitude)$/i.test(col);
          const isId = /^(id|fid|objectid)$/i.test(col);
          const defaultActionable = !isTarget && !isCoord && !isId;
          return {
            name: col,
            type: s ? (Number.isInteger(s.min) && Number.isInteger(s.max) ? "int" : "float") : "str",
            role: isTarget ? "target" : isCoord ? "coord" : isId ? "id" : "predictor",
            selected: predictorList.includes(col) || isTarget,
            treatment: treatmentList.includes(col),
            actionable: actionableList.length > 0 ? actionableList.includes(col) : defaultActionable,
            mean: s?.mean ?? 0,
            std: s?.std ?? 0,
            min: s?.min ?? 0,
            max: s?.max ?? 0,
            corr: isTarget ? 1 : computePearson(colVals[col] ?? [], targetVals),
          };
        });
        setVariables(vars);
      })
      .catch(() => {});
  }, []);

  const handleTargetChange = useCallback((newTarget: string) => {
    setTarget(newTarget);
    setVariables((prev) =>
      prev.map((v) => {
        const isNewTarget = v.name === newTarget;
        const isCoord = /^(x|y|POINT_X|POINT_Y|longitude|latitude)$/i.test(v.name);
        const isId = /^(id|fid|objectid)$/i.test(v.name);
        const newRole = isNewTarget ? "target" : isCoord ? "coord" : isId ? "id" : "predictor";
        const newCorr = isNewTarget
          ? 1
          : computePearson(
              colValsRef.current[v.name] ?? [],
              colValsRef.current[newTarget] ?? [],
            );
        return {
          ...v,
          role: newRole,
          corr: newCorr,
          selected: isNewTarget ? true : v.role === "target" ? false : v.selected,
          actionable: isNewTarget || isCoord || isId ? false : v.actionable,
          treatment: isNewTarget ? false : v.treatment,
        };
      }),
    );
  }, []);

  const handleToggle = useCallback((name: string) => {
    setVariables((prev) => prev.map((v) => v.name === name ? { ...v, selected: !v.selected } : v));
  }, []);

  const handleToggleActionable = useCallback((name: string) => {
    setVariables((prev) =>
      prev.map((v) =>
        v.name === name
          ? { ...v, actionable: !v.actionable, treatment: !v.actionable }
          : v,
      ),
    );
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
    const predictorList = variables
      .filter((v) => v.selected && v.role === "predictor")
      .map((v) => v.name);
    const actionableList = variables.filter((v) => v.actionable).map((v) => v.name);
    const treatmentList = actionableList;
    try {
      await (saveConfig as (c: Record<string, unknown>) => Promise<{ status: string }>)({
        data: { ...origDataRef.current, target_column: target },
        predictors:
          treatmentList.length > 0
            ? { base_model: predictorList, treatment: treatmentList }
            : predictorList,
        causal: { ...origCausalRef.current, actionable_variables: actionableList },
      });
      notify(
        "success",
        `Variables saved — ${predictorList.length} predictor${predictorList.length !== 1 ? "s" : ""}${
          treatmentList.length ? `, ${treatmentList.length} treatment${treatmentList.length > 1 ? "s" : ""}` : ""
        }`,
      );
    } catch {
      notify("error", "Failed to save variables");
    }
  }, [variables, target, notify]);

  const filtered = variables.filter(
    (v) => !filter || v.name.toLowerCase().includes(filter.toLowerCase()),
  );
  const sorted = [...filtered].sort((a, b) =>
    sortBy === "corr" ? Math.abs(b.corr) - Math.abs(a.corr) : a.name.localeCompare(b.name),
  );

  const predictorCount = variables.filter((v) => v.selected && v.role === "predictor").length;
  const actionableCount = variables.filter((v) => v.actionable).length;
  const maxCorr = Math.max(
    0,
    ...variables.filter((v) => v.role === "predictor").map((v) => Math.abs(v.corr)),
  );
  const numericCols = variables.filter((v) => v.type !== "str");

  return (
    <div>
      <SectionHeader
        kicker="05 · analysis"
        label="Variables"
        right={<Btn primary small onClick={handleSave}>Save variables</Btn>}
      />

      <StatGrid>
        <Stat label="Total" value={String(variables.length)} tint="var(--ink)" />
        <Stat label="Predictors" value={String(predictorCount)} tint="var(--crimson)" />
        <Stat label="Actionable" value={String(actionableCount)} tint="var(--amber)" />
        <Stat label="Max |ρ|" value={maxCorr.toFixed(3)} tint="var(--purple)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.6fr) minmax(280px, 1fr)", gap: 14 }}>
        <Card
          title="Variable ledger"
          subtitle={
            variables.length > 0 ? (
              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <label
                  className="mono"
                  style={{
                    fontSize: 9.5,
                    color: "var(--muted)",
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    whiteSpace: "nowrap",
                  }}
                >
                  Target
                </label>
                <select
                  value={target}
                  onChange={(e) => handleTargetChange(e.target.value)}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "3px 6px",
                    fontSize: 11,
                    fontFamily: "inherit",
                    background: "#fff",
                    color: "var(--crimson)",
                    fontWeight: 600,
                    maxWidth: 160,
                  }}
                >
                  {numericCols.map((v) => (
                    <option key={v.name} value={v.name}>{v.name}</option>
                  ))}
                </select>
                <span style={{ color: "var(--line)", userSelect: "none" }}>·</span>
                <input
                  type="text"
                  placeholder="Filter columns…"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "4px 8px",
                    fontSize: 11,
                    fontFamily: "inherit",
                    width: 140,
                    background: "#fff",
                  }}
                />
                <Btn small onClick={() => setSortBy(sortBy === "corr" ? "name" : "corr")}>
                  sort: {sortBy} {sortBy === "corr" ? "↓" : "↑"}
                </Btn>
              </div>
            ) : undefined
          }
        >
          {variables.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--muted)", fontSize: 13 }}>
              Load a dataset to see variable details
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                  <th style={{ ...thStyle, width: 28 }}>☑</th>
                  <th style={thStyle}>Variable</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Role</th>
                  <th
                    style={{ ...thStyle, textAlign: "center", width: 28 }}
                    title="Actionable — used for CATE estimation and scenario sliders"
                  >
                    A
                  </th>
                  <th style={thStyle}>Range</th>
                  <th
                    style={{ ...thStyle, textAlign: "right" }}
                    title="Absolute Pearson correlation with the target variable"
                  >
                    |ρ|
                  </th>
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
                      <span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>
                        {v.type}
                      </span>
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
                    <td style={{ ...tdStyle, textAlign: "center" }}>
                      {v.role === "predictor" && (
                        <button
                          title={
                            v.actionable
                              ? "Remove actionable/treatment designation"
                              : "Mark as actionable — used for CATE estimation and scenario sliders"
                          }
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleActionable(v.name);
                          }}
                          style={roleToggleStyle(v.actionable, "var(--purple)")}
                        >
                          A
                        </button>
                      )}
                    </td>
                    <td
                      style={tdStyle}
                      className="mono"
                      title={`${v.min} … ${v.max}`}
                    >
                      {v.min.toFixed(1)}–{v.max.toFixed(1)}
                    </td>
                    <td
                      className="mono"
                      style={{
                        ...tdStyle,
                        textAlign: "right",
                        fontWeight: 700,
                        color: Math.abs(v.corr) > 0.3 ? "var(--crimson)" : "var(--muted)",
                      }}
                    >
                      {Math.abs(v.corr).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
        </Card>

        <Card
          title="Spatial preview"
          subtitle={
            selectedVar
              ? `${selectedVar}${geoLoading ? " · loading…" : ""}`
              : "click a row to colour the map"
          }
        >
          <div
            style={{ height: 360, borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)" }}
          >
            {selectedGeo && selectedVar ? (
              <SpatialMap
                geojson={selectedGeo}
                colorField={selectedVar}
                mode="scatter"
                height="100%"
                palette="sparc"
              />
            ) : (
              <div
                style={{
                  display: "flex",
                  height: "100%",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "var(--muted)",
                  fontSize: 12,
                  background: "#faf8f4",
                }}
              >
                {selectedVar ? "No spatial data for this variable" : "Select a variable from the ledger"}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

