import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import ExplainButton from "@/components/common/ExplainButton";
import {
  getModelPerformance, getScenarioDetail, getPdpCurves, getGwenData,
  getCorrelogramData, getPredictions, getScenarioIncrement,
  getCateMapVariables, getCateMap, getDoseResponseCurves,
  getCausalSensitivity,
  getCausalNegativeControl,
  freezeCurrentRun,
  downloadStandaloneSnapshot,
  type CausalSensitivity,
  type NegativeControlResponse,
} from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { usePipeline } from "@/hooks/PipelineProvider";
import SpatialMap from "@/components/map/SpatialMap";
import LayerManager, { useContextLayers } from "@/components/map/LayerManager";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import type { ScenarioDetail, PdpCurves, CorrelogramData, GeoJsonData, DoseResponseData } from "@/lib/types";

// Human-readable labels for PDE-derived field columns
const PDE_LABELS: Record<string, string> = {
  alpha_field: "α thermal diffusivity",
  lap_t: "∇²T Laplacian",
  grad_mag: "|∇T| gradient magnitude",
  heat_source: "Q heat source",
  diffusion_coef: "κ diffusion coeff",
  residual_field: "PDE residual",
  div_flux: "∇·F flux divergence",
};

type ViewMode = "map" | "scenarios" | "histogram" | "correlogram" | "causal";

interface ModelInfo {
  name: string;
  r2: number;
  color: string;
}

export default function ResultsPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [activeScenarioIdx, setActiveScenarioIdx] = useState(0);
  const [activeScenarioVar, setActiveScenarioVar] = useState<string>("predicted");
  const [mapLayer, setMapLayer] = useState<string>("");
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const histCanvasRef = useRef<HTMLCanvasElement>(null);
  const r2CanvasRef = useRef<HTMLCanvasElement>(null);
  const curveCanvasRef = useRef<HTMLCanvasElement>(null);
  const corrCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();
  const pipeline = usePipeline();
  const layerCtx = useContextLayers();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [scenarioDetail, setScenarioDetail] = useState<ScenarioDetail | null>(null);
  const [predictionsGeoJson, setPredictionsGeoJson] = useState<GeoJsonData | null>(null);
  const [pdpData, setPdpData] = useState<PdpCurves | null>(null);
  const [correlogram, setCorrelogram] = useState<CorrelogramData | null>(null);
  const [activePdpVar, setActivePdpVar] = useState<string>("");
  const [incrVar, setIncrVar] = useState("");
  const [incrPct, setIncrPct] = useState(10);
  const [incrResult, setIncrResult] = useState<ScenarioDetail | null>(null);

  // Causal-tab state: CATE variable list + selected variable + spatial geojson
  // and dose-response curves keyed by treatment variable.
  const [cateVars, setCateVars] = useState<string[]>([]);
  const [activeCateVar, setActiveCateVar] = useState<string>("");
  const [cateGeo, setCateGeo] = useState<GeoJsonData | null>(null);
  const [doseResponse, setDoseResponse] = useState<DoseResponseData | null>(null);
  const [sensitivity, setSensitivity] = useState<CausalSensitivity | null>(null);
  const [negControl, setNegControl] = useState<NegativeControlResponse | null>(null);
  const doseCanvasRef = useRef<HTMLCanvasElement>(null);

  // Load and refresh data from API. We re-run this after pipeline completion
  // so the Results page updates automatically when artifacts are finished.
  const loadResults = useCallback(() => {
    // Model RÂ² from dedicated endpoint
    getModelPerformance().then((data) => {
      const mods: ModelInfo[] = data.models.map((m, i) => ({
        name: m.name,
        r2: m.r2,
        color: SPARC_RAMP_HEX[i * 2] ?? SPARC_RAMP_HEX[0],
      }));
      setModels(mods);
    }).catch(() => {});

    // Scenario detail
    getScenarioDetail().then(setScenarioDetail).catch(() => {});

    // Model predictions GeoJSON (stage 3 = inference; fallback to stage 2)
    // Retry briefly because files may appear moments after completion event.
    const loadPredictions = (attempt = 0) => {
      (getPredictions(3) as Promise<GeoJsonData>)
        .catch(() => getPredictions(2) as Promise<GeoJsonData>)
        .then(setPredictionsGeoJson)
        .catch(() => {
          if (attempt < 3) {
            setTimeout(() => loadPredictions(attempt + 1), 1200);
          }
        });
    };
    loadPredictions();

    // PDP curves for intervention response
    getPdpCurves().then((pdp) => {
      setPdpData(pdp);
      if (pdp) {
        const vars = Object.keys(pdp);
        if (vars.length > 0) setActivePdpVar(vars[0]);
      }
    }).catch(() => {});

    // Correlogram
    getCorrelogramData().then(setCorrelogram).catch(() => {});

    // GWEN weights (update model RÂ²)
    getGwenData().then((gwen) => {
      if (!gwen?.rows?.length) return;
      setModels((prev) => {
        if (!prev.length) return prev;
        return prev.map((m) => {
          const row = gwen.rows.find((r: any) => r.model === m.name || r.name === m.name);
          if (row && typeof (row as any).r2 === "number") return { ...m, r2: (row as any).r2 };
          return m;
        });
      });
    }).catch(() => {});
  }, []);

  // Initial load
  useEffect(() => {
    loadResults();
  }, [loadResults]);

  // Refresh automatically once a run finishes (uses runEndedAt for an exact
  // edge transition rather than the noisier isRunning flag).
  useEffect(() => {
    if (pipeline.runEndedAt) {
      loadResults();
    }
  }, [pipeline.runEndedAt, loadResults]);

  // Causal results: load CATE variable list + dose-response curves once results
  // are available; the discovery / mediation results live behind getCateMap and
  // getDoseResponseCurves and are populated by the causal stage.
  const loadCausal = useCallback(() => {
    getCateMapVariables()
      .then((res) => {
        const vars = (res?.variables ?? []) as string[];
        setCateVars(vars);
        if (vars.length > 0 && !activeCateVar) setActiveCateVar(vars[0]);
      })
      .catch(() => setCateVars([]));
    getDoseResponseCurves()
      .then((d) => setDoseResponse(d as DoseResponseData))
      .catch(() => setDoseResponse(null));
    getCausalSensitivity()
      .then((s) => setSensitivity(s))
      .catch(() => setSensitivity(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { loadCausal(); }, [loadCausal, pipeline.runEndedAt]);

  // Refetch CATE surface when the user picks a different treatment variable.
  useEffect(() => {
    if (!activeCateVar) { setCateGeo(null); return; }
    getCateMap(activeCateVar)
      .then((g) => setCateGeo(g as GeoJsonData))
      .catch(() => setCateGeo(null));
  }, [activeCateVar]);

  // Permutation negative-control test for the active CATE variable.
  useEffect(() => {
    if (!activeCateVar) { setNegControl(null); return; }
    getCausalNegativeControl(activeCateVar, 1000)
      .then((nc) => setNegControl(nc))
      .catch(() => setNegControl(null));
  }, [activeCateVar]);

  const hasData = models.length > 0;
  const bestModel = hasData ? models.reduce((a, b) => (b.r2 > a.r2 ? b : a)) : null;
  const scenarioRows = scenarioDetail?.summary ?? [];

  // Derive scenario column names for variable selector
  const scenarioFeatures = (incrResult ?? scenarioDetail)?.geojson?.features ?? scenarioDetail?.geojson?.features ?? [];
  const scenarioCols = scenarioFeatures.length > 0
    ? Object.keys(scenarioFeatures[0].properties ?? {}).filter((k) =>
        k.startsWith("delta_") || k.startsWith("pred_") || k === "predicted" || k === "target"
      )
    : [];

  // Variables available for increment slider (feature columns that have a delta_ counterpart)
  const incrVarOptions = scenarioCols
    .filter((c) => c.startsWith("delta_"))
    .map((c) => c.replace(/^delta_/, ""));

  // Derive scenario names from summary rows
  const scenarioNames: string[] = scenarioRows.map((r: any, i: number) =>
    (r.scenario_name ?? r.scenario ?? r.name ?? `Scenario ${i + 1}`) as string
  );

  // Diverging blue→white→red color scale centered at zero (for delta_* columns)
  function divergeColor(val: number, absMax: number): string {
    const t = Math.max(-1, Math.min(1, val / (absMax || 1)));
    const lp = (a: number, b: number, s: number) => Math.round(a + (b - a) * s);
    if (t <= 0) {
      const s = 1 + t;
      return `rgb(${lp(33, 247, s)},${lp(102, 247, s)},${lp(172, 247, s)})`;
    }
    return `rgb(${lp(247, 178, t)},${lp(247, 24, t)},${lp(247, 43, t)})`;
  }

  // Helpers for canvas drawing
  function drawEmptyState(canvas: HTMLCanvasElement, message: string) {
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = "#f5f1eb";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#6e6358";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message, w / 2, h / 2);
  }

  function drawPointMap(canvas: HTMLCanvasElement, features: GeoJsonData["features"], propKey: string) {
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    if (!features.length) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No spatial data available", w / 2, h / 2);
      return;
    }

    ctx.fillStyle = "#f5f1eb";
    ctx.fillRect(0, 0, w, h);

    const pts: { x: number; y: number; val: number }[] = [];
    for (const f of features) {
      const coords = f.geometry?.coordinates as number[] | undefined;
      if (!coords || coords.length < 2) continue;
      const raw = f.properties?.[propKey];
      const val = Number(raw ?? 0);
      if (!Number.isFinite(val)) continue;
      pts.push({ x: coords[0], y: coords[1], val });
    }
    if (!pts.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No valid coordinates in data", w / 2, h / 2);
      return;
    }

    const minX = Math.min(...pts.map((p) => p.x));
    const maxX = Math.max(...pts.map((p) => p.x));
    const minY = Math.min(...pts.map((p) => p.y));
    const maxY = Math.max(...pts.map((p) => p.y));
    const minV = Math.min(...pts.map((p) => p.val));
    const maxV = Math.max(...pts.map((p) => p.val));
    const rangeV = maxV - minV || 1;
    const isDelta = propKey.startsWith("delta_");
    const absMax = isDelta ? Math.max(Math.abs(minV), Math.abs(maxV)) : 0;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    // Uniform scaling preserves aspect ratio
    const pad = 16;
    const scaleX = (w - 2 * pad) / rangeX;
    const scaleY = (h - 2 * pad) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (w - scale * rangeX) / 2;
    const offsetY = (h - scale * rangeY) / 2;

    for (const p of pts) {
      const px = offsetX + (p.x - minX) * scale;
      const py = h - offsetY - (p.y - minY) * scale;
      if (isDelta) {
        ctx.fillStyle = divergeColor(p.val, absMax);
      } else {
        const t = (p.val - minV) / rangeV;
        const ci = Math.floor(t * (SPARC_RAMP_HEX.length - 1));
        ctx.fillStyle = SPARC_RAMP_HEX[ci];
      }
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Derive available numeric layers from predictions GeoJSON for the layer selector
  const mapLayerOptions = (() => {
    const features = predictionsGeoJson?.features ?? [];
    if (!features.length) return [];
    const props = features[0]?.properties ?? {};
    const skip = new Set(["lon", "lat", "geometry", "id"]);
    return Object.entries(props)
      .filter(([k, v]) => !skip.has(k) && typeof v === "number" && Number.isFinite(v as number))
      .map(([k]) => k);
  })();

  // Default map layer selection: prefer predicted_full, then first available
  const resolvedMapLayer = mapLayer ||
    (mapLayerOptions.includes("predicted_full") ? "predicted_full" :
      mapLayerOptions.includes("predicted") ? "predicted" :
      mapLayerOptions[0] ?? "predicted_full");

  // Draw predictions map
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || viewMode !== "map") return;
    const features = predictionsGeoJson?.features ?? [];
    if (!features.length) {
      drawEmptyState(canvas, "Run pipeline to see predictions map");
      return;
    }
    drawPointMap(canvas, features, resolvedMapLayer);
  }, [viewMode, predictionsGeoJson, resolvedMapLayer]);

  // Draw scenario map
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || viewMode !== "scenarios") return;
    const features = (incrResult ?? scenarioDetail)?.geojson?.features ?? scenarioFeatures;
    if (!features.length) {
      drawEmptyState(canvas, "Run pipeline with scenarios to see scenario map");
      return;
    }
    // Filter to selected scenario if possible via pred_ScenarioN columns
    const col = activeScenarioVar || scenarioCols[0] || "predicted";
    drawPointMap(canvas, features, col);
  }, [viewMode, scenarioDetail, incrResult, activeScenarioIdx, activeScenarioVar]);

  // Draw histogram from real prediction values
  useEffect(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || viewMode !== "histogram") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    // Use predictions GeoJSON preferentially, fall back to scenario detail
    const features = (predictionsGeoJson?.features ?? scenarioDetail?.geojson?.features ?? []);
    if (!features.length) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see histogram", w / 2, h / 2);
      return;
    }

    const vals = features
      .map((f) => Number(f.properties?.predicted ?? f.properties?.target))
      .filter((v) => Number.isFinite(v));
    if (!vals.length) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No valid prediction values in data", w / 2, h / 2);
      return;
    }

    ctx.fillStyle = "#f5f1eb";
    ctx.fillRect(0, 0, w, h);

    const nBins = 40;
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const range = maxV - minV || 1;
    const bins = new Array(nBins).fill(0);
    for (const v of vals) {
      const idx = Math.min(Math.floor(((v - minV) / range) * nBins), nBins - 1);
      bins[idx]++;
    }
    const maxBin = Math.max(...bins);

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 30); ctx.lineTo(w - 10, h - 30);
    ctx.stroke();

    // Y-axis ticks (4 evenly spaced) and label
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const tickVal = Math.round((maxBin * (4 - i)) / 4);
      const yPx = 10 + ((h - 40) * i) / 4;
      ctx.fillText(String(tickVal), 36, yPx + 3);
      ctx.strokeStyle = "rgba(0,0,0,0.04)";
      ctx.beginPath(); ctx.moveTo(40, yPx); ctx.lineTo(w - 10, yPx); ctx.stroke();
    }
    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("count", 0, 0);
    ctx.restore();

    // X-axis labels (min / max values in original scale)
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(minV.toFixed(2), 40, h - 14);
    ctx.fillText(maxV.toFixed(2), w - 10, h - 14);
    ctx.fillText(((minV + maxV) / 2).toFixed(2), w / 2, h - 14);

    // Bars
    const barW = (w - 55) / nBins;
    bins.forEach((v, i) => {
      const barH = (v / maxBin) * (h - 50);
      const x = 42 + i * barW;
      const colorIdx = Math.floor((i / nBins) * (SPARC_RAMP_HEX.length - 1));
      ctx.fillStyle = SPARC_RAMP_HEX[colorIdx] + "cc";
      ctx.fillRect(x, h - 30 - barH, barW - 1, barH);
    });

    // Title
    ctx.fillStyle = "#6e6358";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`Predicted value distribution  n=${vals.length}`, w / 2, h - 2);
  }, [viewMode, predictionsGeoJson, scenarioDetail]);

  // Model RÂ² bars (from real data)
  useEffect(() => {
    const canvas = r2CanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    if (!models.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see model RÂ²", w / 2, h / 2);
      return;
    }

    const barH = 16;
    const gap = 6;
    const labelW = 60;

    models.forEach((m, i) => {
      const y = i * (barH + gap) + 4;
      ctx.fillStyle = "#6e6358";
      ctx.font = "10px 'JetBrains Mono'";
      ctx.textAlign = "right";
      ctx.fillText(m.name, labelW - 6, y + barH / 2 + 4);

      ctx.fillStyle = "rgba(0,0,0,0.04)";
      ctx.fillRect(labelW, y, w - labelW - 40, barH);

      ctx.fillStyle = m.color;
      ctx.fillRect(labelW, y, (w - labelW - 40) * m.r2, barH);

      ctx.fillStyle = "#1a1416";
      ctx.font = "bold 10px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(m.r2.toFixed(3), w - 36, y + barH / 2 + 4);
    });
  }, [models]);

  // Intervention response curve (from real PDP data) â€” single variable with selector
  useEffect(() => {
    const canvas = curveCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const variables = pdpData ? Object.keys(pdpData) : [];
    if (!variables.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see response curves", w / 2, h / 2);
      return;
    }

    const selectedVar = activePdpVar || variables[0];
    const curve = (pdpData as any)?.[selectedVar];
    const gridVals: number[] = curve?.grid_values ?? [];
    const pdpVals: number[] = curve?.pdp_values ?? [];
    const pdpStd: number[] = curve?.pdp_std ?? [];

    if (!gridVals.length || !pdpVals.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No data for selected variable", w / 2, h / 2);
      return;
    }

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    const minG = Math.min(...gridVals), maxG = Math.max(...gridVals);
    const minP = Math.min(...pdpVals.map((v, i) => v - (pdpStd[i] ?? 0)));
    const maxP = Math.max(...pdpVals.map((v, i) => v + (pdpStd[i] ?? 0)));
    const rG = maxG - minG || 1, rP = maxP - minP || 1;

    // Confidence band
    if (pdpStd.length === pdpVals.length) {
      ctx.fillStyle = SPARC_RAMP_HEX[0] + "22";
      ctx.beginPath();
      gridVals.forEach((g, i) => {
        const x = 42 + ((g - minG) / rG) * (w - 54);
        const y = (h - 35) - ((pdpVals[i] + pdpStd[i] - minP) / rP) * (h - 45) + 10;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      for (let i = gridVals.length - 1; i >= 0; i--) {
        const x = 42 + ((gridVals[i] - minG) / rG) * (w - 54);
        const y = (h - 35) - ((pdpVals[i] - pdpStd[i] - minP) / rP) * (h - 45) + 10;
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Main curve
    ctx.strokeStyle = SPARC_RAMP_HEX[0];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    gridVals.forEach((g, i) => {
      const x = 42 + ((g - minG) / rG) * (w - 54);
      const y = (h - 35) - ((pdpVals[i] - minP) / rP) * (h - 45) + 10;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Axis labels in original scale
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(minG.toFixed(2), 42, h - 10);
    ctx.fillText(maxG.toFixed(2), w - 10, h - 10);
    ctx.textAlign = "right";
    ctx.fillText(maxP.toFixed(2), 38, 18);
    ctx.fillText(minP.toFixed(2), 38, h - 28);

    // Saturation point marker
    const satPoint: number | null | undefined = (pdpData as any)?.[selectedVar]?.curve_fit?.saturation_point;
    if (satPoint != null && Number.isFinite(satPoint) && satPoint >= minG && satPoint <= maxG) {
      const sx = 42 + ((satPoint - minG) / rG) * (w - 54);
      ctx.save();
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, 12);
      ctx.lineTo(sx, h - 28);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#dc2626";
      ctx.font = "bold 8px 'JetBrains Mono'";
      ctx.textAlign = "center";
      ctx.fillText("sat.", sx, 10);
      ctx.restore();
    }
  }, [pdpData, activePdpVar]);

  // Correlogram canvas
  useEffect(() => {
    const canvas = corrCanvasRef.current;
    if (!canvas || viewMode !== "correlogram") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const results = correlogram?.individual_results;
    const varNames = results ? Object.keys(results) : [];
    if (!varNames.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see correlogram", w / 2, h / 2);
      return;
    }

    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Lag distance", w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Moran's I", 0, 0);
    ctx.restore();

    // Zero line
    ctx.strokeStyle = "#c9c2b3";
    ctx.setLineDash([4, 3]);
    const zeroY = pad.top + plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(w - pad.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    let allMinI = 0, allMaxI = 0, allMaxLag = 0;
    for (const name of varNames) {
      const cr = results![name].correlogram_results;
      const morans = cr.morans_i_values;
      const lags = cr.lag_distances;
      allMinI = Math.min(allMinI, ...morans);
      allMaxI = Math.max(allMaxI, ...morans);
      allMaxLag = Math.max(allMaxLag, ...lags);
    }
    const rangeI = allMaxI - allMinI || 1;

    // Tick labels (x: 5 ticks across lag, y: 5 ticks across Moran's I range)
    ctx.fillStyle = "#6e6358";
    ctx.font = "8px 'JetBrains Mono'";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const lagVal = (allMaxLag * i) / 4;
      const x = pad.left + (plotW * i) / 4;
      ctx.fillText(lagVal.toFixed(0), x, h - pad.bottom + 12);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const iVal = allMinI + (rangeI * (4 - i)) / 4;
      const y = pad.top + (plotH * i) / 4;
      ctx.fillText(iVal.toFixed(2), pad.left - 4, y + 3);
      ctx.strokeStyle = "rgba(0,0,0,0.04)";
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    const top = Math.min(varNames.length, 6);
    for (let vi = 0; vi < top; vi++) {
      const name = varNames[vi];
      const cr = results![name].correlogram_results;
      const lags = cr.lag_distances;
      const morans = cr.morans_i_values;
      const pvals = cr.p_values;

      const color = SPARC_RAMP_HEX[vi % SPARC_RAMP_HEX.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < lags.length; i++) {
        const x = pad.left + (lags[i] / allMaxLag) * plotW;
        const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      for (let i = 0; i < lags.length; i++) {
        if (pvals[i] < 0.05) {
          const x = pad.left + (lags[i] / allMaxLag) * plotW;
          const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Legend
    ctx.font = "9px 'JetBrains Mono'";
    const legendY = pad.top + 4;
    for (let vi = 0; vi < top; vi++) {
      const name = varNames[vi];
      const x = pad.left + 6 + vi * 85;
      ctx.fillStyle = SPARC_RAMP_HEX[vi % SPARC_RAMP_HEX.length];
      ctx.fillRect(x, legendY, 10, 3);
      ctx.fillStyle = "#6e6358";
      ctx.textAlign = "left";
      ctx.fillText(name.replace(/_/g, " ").slice(0, 10), x + 14, legendY + 4);
    }
  }, [viewMode, correlogram]);

  // Dose-response curves: marginal causal effect vs treatment level for the
  // currently selected CATE variable. Falls back to the first available curve.
  useEffect(() => {
    const canvas = doseCanvasRef.current;
    if (!canvas || viewMode !== "causal" || !doseResponse) return;
    const treatments = Object.keys(doseResponse);
    if (treatments.length === 0) return;
    const treatmentName = treatments.find((t) => t === activeCateVar) ?? treatments[0];
    const curve = doseResponse[treatmentName];
    if (!curve || !curve.dose_levels?.length || !curve.marginal_effects?.length) return;

    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 400, h = canvas.clientHeight || 320;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 16, right: 14, bottom: 30, left: 44 };
    const xs = curve.dose_levels;
    const ys = curve.marginal_effects;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys, 0), yMax = Math.max(...ys, 0);
    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
    const toX = (v: number) => pad.left + ((v - xMin) / xRange) * (w - pad.left - pad.right);
    const toY = (v: number) => h - pad.bottom - ((v - yMin) / yRange) * (h - pad.top - pad.bottom);

    // Grid
    ctx.strokeStyle = "rgba(0,0,0,0.06)"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * (h - pad.top - pad.bottom);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }
    // Zero line
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = "rgba(0,0,0,0.25)"; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(pad.left, toY(0)); ctx.lineTo(w - pad.right, toY(0)); ctx.stroke();
    }
    // Curve
    ctx.strokeStyle = "var(--purple)"; ctx.lineWidth = 1.8;
    ctx.beginPath();
    xs.forEach((x, i) => { if (i === 0) ctx.moveTo(toX(x), toY(ys[i])); else ctx.lineTo(toX(x), toY(ys[i])); });
    ctx.stroke();
    // Points
    ctx.fillStyle = "var(--purple)";
    xs.forEach((x, i) => { ctx.beginPath(); ctx.arc(toX(x), toY(ys[i]), 2.5, 0, Math.PI * 2); ctx.fill(); });

    // Axes labels
    ctx.fillStyle = "#6e6358"; ctx.font = "9px 'JetBrains Mono'"; ctx.textAlign = "center";
    ctx.fillText(treatmentName.replace(/_/g, " "), (pad.left + w - pad.right) / 2, h - 8);
    ctx.save();
    ctx.translate(10, (pad.top + h - pad.bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("dY/dT", 0, 0);
    ctx.restore();
    // Tick numbers
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const v = yMin + (i / 4) * yRange;
      const y = h - pad.bottom - (i / 4) * (h - pad.top - pad.bottom);
      ctx.fillText(v.toFixed(2), pad.left - 4, y + 3);
    }
    ctx.textAlign = "center";
    [0, 0.5, 1].forEach((t) => {
      const v = xMin + t * xRange;
      ctx.fillText(v.toFixed(2), pad.left + t * (w - pad.left - pad.right), h - pad.bottom + 12);
    });

    // Nonlinearity tag
    if (curve.is_nonlinear) {
      ctx.fillStyle = "var(--crimson)"; ctx.textAlign = "right";
      ctx.fillText("nonlinear", w - pad.right - 2, pad.top + 8);
    }
  }, [viewMode, doseResponse, activeCateVar]);

  const handleExport = useCallback(async (format: string) => {
    notify("success", `${format} export started`);
  }, [notify]);

  const pdpVariables = pdpData ? Object.keys(pdpData) : [];

  return (
    <div>
      <SectionHeader
        kicker="11 Â· pipeline"
        label="Results"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={async () => {
              try {
                const r = await freezeCurrentRun();
                notify("success", `Froze run → ${r.path.split(/[\\/]/).pop()}`);
              } catch (err) {
                notify("error", err instanceof Error ? err.message : String(err));
              }
            }}>Freeze run</Btn>
            <Btn small onClick={async () => {
              const r = await downloadStandaloneSnapshot({});
              if (r.ok) notify("success", `Saved ${r.filename}`);
              else notify("error", r.error || "snapshot failed");
            }}>Share snapshot</Btn>
            <Btn small onClick={() => handleExport("CSV")}>Export CSV</Btn>
            <Btn small onClick={() => handleExport("GeoPackage")}>Export GPKG</Btn>
          </div>
        }
      />

      <StatGrid>
        <Stat label="Best RÂ²" value={bestModel ? bestModel.r2.toFixed(3) : "â€”"} tint="var(--crimson)" sub={bestModel?.name} />
        <Stat label="Models" value={models.length ? String(models.length) : "â€”"} tint="var(--purple)" />
        <Stat label="Scenarios" value={scenarioRows.length ? String(scenarioRows.length) : "â€”"} tint="var(--amber)" />
        <Stat label="Features" value={predictionsGeoJson?.features?.length ? String(predictionsGeoJson.features.length) : "â€”"} tint="var(--ink)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        {/* Main visualization */}
        <Card
          title="Predictions"
          subtitle={
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              {/* View mode toggle */}
              {(["map", "scenarios", "histogram", "correlogram", "causal"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  style={{
                    padding: "3px 8px",
                    fontSize: 10,
                    borderRadius: 3,
                    border: "1px solid " + (viewMode === mode ? "var(--crimson)" : "var(--line)"),
                    background: viewMode === mode ? "var(--crimson)" : "#fff",
                    color: viewMode === mode ? "#fff" : "var(--ink-2)",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    fontWeight: 600,
                    textTransform: "capitalize",
                  }}
                >
                  {mode === "map" ? "Map" : mode === "scenarios" ? "Scenarios" : mode === "histogram" ? "Histogram" : mode === "correlogram" ? "Correlogram" : "Causal"}
                </button>
              ))}
              {/* Map layer selector (only shown in map mode with predictions data) */}
              {viewMode === "map" && mapLayerOptions.length > 0 && (
                <select
                  value={resolvedMapLayer}
                  onChange={(e) => setMapLayer(e.target.value)}
                  style={{ fontSize: 10, padding: "2px 4px", borderRadius: 3, border: "1px solid var(--line)", fontFamily: "inherit" }}
                >
                  {mapLayerOptions.map((col) => (
                    <option key={col} value={col}>
                      {PDE_LABELS[col] ? `${col.replace(/_/g, " ")}  (${PDE_LABELS[col]})` : col.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              )}
              {/* Scenario selectors (only shown in scenarios mode) */}
              {viewMode === "scenarios" && scenarioNames.length > 0 && (
                <select
                  value={activeScenarioIdx}
                  onChange={(e) => setActiveScenarioIdx(Number(e.target.value))}
                  style={{ fontSize: 10, padding: "2px 4px", borderRadius: 3, border: "1px solid var(--line)", fontFamily: "inherit" }}
                >
                  {scenarioNames.map((name, i) => (
                    <option key={i} value={i}>{name}</option>
                  ))}
                </select>
              )}
              {viewMode === "scenarios" && scenarioCols.length > 0 && (
                <select
                  value={activeScenarioVar}
                  onChange={(e) => setActiveScenarioVar(e.target.value)}
                  style={{ fontSize: 10, padding: "2px 4px", borderRadius: 3, border: "1px solid var(--line)", fontFamily: "inherit" }}
                >
                  {scenarioCols.map((col) => (
                    <option key={col} value={col}>{col.replace(/_/g, " ")}</option>
                  ))}
                </select>
              )}
              {/* Scenario increment slider */}
              {viewMode === "scenarios" && incrVarOptions.length > 0 && (
                <>
                  <select
                    value={incrVar}
                    onChange={(e) => { setIncrVar(e.target.value); setIncrResult(null); }}
                    style={{ fontSize: 10, padding: "2px 4px", borderRadius: 3, border: "1px solid var(--line)", fontFamily: "inherit" }}
                    title="Variable to increment"
                  >
                    <option value="">Δ variable…</option>
                    {incrVarOptions.map((v) => (
                      <option key={v} value={v}>{v.replace(/_/g, " ")}</option>
                    ))}
                  </select>
                  {incrVar && (
                    <>
                      <input
                        type="range"
                        min={-50}
                        max={50}
                        step={5}
                        value={incrPct}
                        onChange={(e) => {
                          const pct = Number(e.target.value);
                          setIncrPct(pct);
                          getScenarioIncrement(incrVar, pct)
                            .then((data) => setIncrResult(data as ScenarioDetail))
                            .catch(() => {});
                        }}
                        style={{ width: 72, cursor: "pointer" }}
                      />
                      <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 34, textAlign: "right", fontFamily: "'JetBrains Mono', monospace" }}>
                        {incrPct > 0 ? "+" : ""}{incrPct}%
                      </span>
                    </>
                  )}
                </>
              )}
            </div>
          }
        >
          {(viewMode === "map" || viewMode === "scenarios") ? (
            <div style={{ position: "relative", height: 380, borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)" }}>
              <div style={{ position: "absolute", top: 8, left: 8, zIndex: 5, maxHeight: "calc(100% - 16px)", overflowY: "auto" }}>
                <LayerManager ctx={layerCtx} compact />
              </div>
              {(() => {
                // Pick the geojson + field for the active mode.
                const geo = viewMode === "scenarios"
                  ? ((incrResult ?? scenarioDetail)?.geojson ?? null)
                  : predictionsGeoJson;
                const field = viewMode === "scenarios"
                  ? (activeScenarioVar || scenarioCols[0] || "")
                  : resolvedMapLayer;
                const isDiverg = field.startsWith("delta_");
                const featureCount = (geo as GeoJsonData | null)?.features?.length ?? 0;
                if (featureCount === 0 || !field) {
                  return (
                    <div style={{ display: "flex", height: "100%", alignItems: "center", justifyContent: "center", background: "#faf8f4", color: "var(--muted)", fontSize: 12 }}>
                      {viewMode === "scenarios" ? "Run a scenario to see the spatial response" : "No predictions yet \u2014 run the pipeline"}
                    </div>
                  );
                }
                return (
                  <SpatialMap
                    geojson={geo as GeoJsonData}
                    colorField={field}
                    mode="scatter"
                    height="100%"
                    palette={isDiverg ? "puor" : "sparc"}
                    contextLayers={layerCtx.active}
                  />
                );
              })()}
              {/* Hidden canvas kept so legacy refs/effects do not crash */}
              <canvas ref={mapCanvasRef} style={{ display: "none" }} />
              {(predictionsGeoJson?.features?.length || scenarioFeatures.length) ? (() => {
                const activeCol = viewMode === "scenarios" ? (activeScenarioVar || scenarioCols[0] || "") : resolvedMapLayer;
                const isDiverg = activeCol.startsWith("delta_");
                const pdeLabel = PDE_LABELS[activeCol];
                return (
                  <div
                    style={{
                      position: "absolute",
                      bottom: 12,
                      right: 12,
                      background: "rgba(255,255,255,0.92)",
                      borderRadius: 6,
                      padding: "8px 10px",
                      backdropFilter: "blur(6px)",
                      minWidth: 120,
                    }}
                  >
                    {pdeLabel && (
                      <div className="mono" style={{ fontSize: 7, color: "var(--crimson)", marginBottom: 2, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                        {pdeLabel}
                      </div>
                    )}
                    <div className="mono" style={{ fontSize: 8, color: "var(--muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                      {viewMode === "scenarios" ? (activeScenarioVar || "value") : "predicted value"}
                    </div>
                    {isDiverg ? (
                      <div style={{ height: 10, borderRadius: 2, width: 120, background: "linear-gradient(to right, rgb(33,102,172), rgb(247,247,247), rgb(178,24,43))" }} />
                    ) : (
                      <div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden", width: 120 }}>
                        {SPARC_RAMP_HEX.map((c, i) => (
                          <div key={i} style={{ flex: 1, background: c }} />
                        ))}
                      </div>
                    )}
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                      {isDiverg ? (
                        <>
                          <span className="mono" style={{ fontSize: 8, color: "rgb(33,102,172)" }}>cooling</span>
                          <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>0</span>
                          <span className="mono" style={{ fontSize: 8, color: "rgb(178,24,43)" }}>warming</span>
                        </>
                      ) : (
                        <>
                          <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>low</span>
                          <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>high</span>
                        </>
                      )}
                    </div>
                  </div>
                );
              })() : null}
            </div>
          ) : viewMode === "histogram" ? (
            <canvas
              ref={histCanvasRef}
              style={{ width: "100%", height: 380, display: "block" }}
            />
          ) : viewMode === "correlogram" ? (
            <canvas
              ref={corrCanvasRef}
              style={{ width: "100%", height: 380, display: "block" }}
            />
          ) : (
            // Causal view: CATE map (left) + dose-response curve (right)
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, height: 380 }}>
              <div style={{ borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)", position: "relative" }}>
                {cateVars.length > 0 && (
                  <select
                    value={activeCateVar}
                    onChange={(e) => setActiveCateVar(e.target.value)}
                    style={{
                      position: "absolute", top: 6, left: 6, zIndex: 10,
                      fontSize: 10, padding: "2px 6px", borderRadius: 3,
                      border: "1px solid var(--line)", background: "rgba(255,255,255,0.92)",
                      fontFamily: "inherit",
                    }}
                  >
                    {cateVars.map((v) => (
                      <option key={v} value={v}>{v.replace(/_/g, " ")}</option>
                    ))}
                  </select>
                )}
                {cateGeo ? (
                  <SpatialMap
                    geojson={cateGeo}
                    colorField={`cate_${activeCateVar}`}
                    mode="scatter"
                    height="100%"
                    palette="puor"
                    contextLayers={layerCtx.active}
                  />
                ) : (
                  <div style={{ display: "flex", height: "100%", alignItems: "center", justifyContent: "center", color: "var(--muted)", fontSize: 11, background: "#faf8f4" }}>
                    {cateVars.length === 0 ? "Run causal stage to see local treatment effects (CATE)" : "Loading CATE surface…"}
                  </div>
                )}
                <div className="mono" style={{ position: "absolute", bottom: 8, left: 8, fontSize: 8, padding: "3px 6px", background: "rgba(255,255,255,0.92)", border: "1px solid var(--line)", borderRadius: 3, color: "var(--ink-2)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  CATE · local causal effect
                </div>
              </div>
              <div style={{ borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)", padding: 10, position: "relative" }}>
                <div className="mono" style={{ fontSize: 8, color: "var(--muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                  Dose–response · marginal causal curve
                </div>
                <canvas
                  ref={doseCanvasRef}
                  style={{ width: "100%", height: 340, display: "block" }}
                />
              </div>
            </div>
          )}
          {viewMode === "causal" && sensitivity && sensitivity.results.length > 0 && (
            <div style={{ marginTop: 14, padding: 12, border: "1px solid var(--line)", borderRadius: 4, background: "#faf8f4" }}>
              <div className="mono" style={{ fontSize: 9, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Robustness · {sensitivity.method}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}>
                {sensitivity.results.map((s) => {
                  const e = s.e_value_point;
                  const color = e >= 2.5 ? "var(--purple)" : e >= 1.5 ? "var(--amber)" : "var(--crimson)";
                  return (
                    <div key={s.effect_label} style={{ padding: 10, border: `1px solid ${color}`, borderRadius: 4, background: "#fff" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                        <span className="mono" style={{ fontSize: 10, fontWeight: 600 }}>{s.effect_label}</span>
                        <span className="mono" style={{ fontSize: 14, fontWeight: 700, color }}>
                          E={e.toFixed(2)}
                        </span>
                      </div>
                      <div className="mono" style={{ fontSize: 9, color: "var(--muted)", marginBottom: 4 }}>
                        β={s.point_estimate.toFixed(3)}
                        {s.ci_lower !== null && s.ci_upper !== null && ` · CI [${s.ci_lower.toFixed(3)}, ${s.ci_upper.toFixed(3)}]`}
                        {s.e_value_ci !== null && ` · tip ${s.e_value_ci.toFixed(2)}`}
                      </div>
                      <div style={{ fontSize: 10, color: "var(--ink)", lineHeight: 1.4, marginBottom: 6 }}>
                        {s.interpretation}
                      </div>
                      <ExplainButton
                        compact
                        prompt={
                          `Explain the causal effect labelled "${s.effect_label}". ` +
                          `Point estimate β=${s.point_estimate.toFixed(3)}` +
                          (s.ci_lower !== null && s.ci_upper !== null
                            ? `, 95% CI [${s.ci_lower.toFixed(3)}, ${s.ci_upper.toFixed(3)}]`
                            : "") +
                          `. E-value at point=${e.toFixed(2)}` +
                          (s.e_value_ci !== null ? `, tipping E=${s.e_value_ci.toFixed(2)}` : "") +
                          `. What does this mean for the project, and how confident should the user be?`
                        }
                      />
                    </div>
                  );
                })}
              </div>
              <div style={{ marginTop: 8, fontSize: 9.5, color: "var(--muted)", lineHeight: 1.4 }}>
                <strong>E-value</strong>: minimum strength (on the risk-ratio scale) an unmeasured confounder
                would need with both treatment and outcome to fully explain away the effect. Higher = more robust.
              </div>
            </div>
          )}
          {viewMode === "causal" && negControl && (
            <div style={{
              marginTop: 14, padding: 12,
              border: `1px solid ${negControl.passed ? "var(--amber)" : "var(--purple)"}`,
              borderRadius: 4, background: "#fff",
            }}>
              <div className="mono" style={{ fontSize: 9, color: "var(--muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Validation · permutation test on CATE({negControl.variable}) · n={negControl.n}, {negControl.n_permutations} perms
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
                <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: negControl.passed ? "var(--amber)" : "var(--purple)" }}>
                  p = {negControl.p_value < 0.0001 ? "< 0.0001" : negControl.p_value.toFixed(4)}
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-2)" }}>
                  z = {Number.isFinite(negControl.z_score) ? negControl.z_score.toFixed(2) : "—"}
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-2)" }}>
                  μ̂ = {negControl.mean_observed.toFixed(4)} · null μ = {negControl.mean_null.toFixed(4)} ± {negControl.std_null.toFixed(4)}
                </div>
              </div>
              <div style={{ marginTop: 6, fontSize: 11, color: "var(--ink)", lineHeight: 1.5 }}>
                {negControl.interpretation}
              </div>
              <div style={{ marginTop: 6, fontSize: 9.5, color: "var(--muted)", lineHeight: 1.4 }}>
                Run this on a known <em>negative-control</em> variable (something a priori unrelated to the outcome).
                A passing test on a real treatment is a red flag — the model isn't picking up signal.
              </div>
            </div>
          )}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Model R²" subtitle={models.length ? "cross-validated performance" : undefined}>
            <canvas
              ref={r2CanvasRef}
              style={{ width: "100%", height: 120, display: "block" }}
            />
          </Card>

          <Card
            title="Intervention response"
            subtitle={
              pdpVariables.length > 0 ? (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: "var(--muted)" }}>marginal effect on target</span>
                  <select
                    value={activePdpVar}
                    onChange={(e) => setActivePdpVar(e.target.value)}
                    style={{ fontSize: 10, padding: "2px 4px", borderRadius: 3, border: "1px solid var(--line)", fontFamily: "inherit", marginLeft: 4 }}
                  >
                    {pdpVariables.map((v) => (
                      <option key={v} value={v}>{v.replace(/_/g, " ")}</option>
                    ))}
                  </select>
                </div>
              ) : "marginal effect on target"
            }
          >
            <canvas
              ref={curveCanvasRef}
              style={{ width: "100%", height: 160, display: "block" }}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
