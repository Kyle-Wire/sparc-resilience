import { useState, useEffect } from "react";
import { SectionHeader, Card } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig } from "@/lib/types";

// -----------------------------------------------------------------------
// Types for the models config section
// -----------------------------------------------------------------------
interface GwrParams { bandwidth?: number; kernel?: string; alpha?: number; }
interface GwrfParams { n_estimators?: number; k_neighbors?: number; min_samples_leaf?: number; }
interface GgpgamParams { n_splines?: number; n_spatial_bases?: number; lam?: number; }
interface NeuralParams { hidden_dim?: number; dropout?: number; n_heads?: number; max_neighbors?: number; }
interface TrainingParams { n_epochs?: number; pretrain_epochs?: number; learning_rate?: number; }
interface SpatialCvParams { block_size?: number | null; buffer_size?: number; buffer_size_auto?: boolean; }

// -----------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------
const inputStyle: React.CSSProperties = {
  padding: "5px 8px",
  border: "1px solid var(--line)",
  borderRadius: 5,
  fontSize: 12,
  fontFamily: "var(--font-mono, monospace)",
  width: "100%",
  background: "var(--paper)",
  color: "var(--ink)",
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: "var(--muted)",
  marginBottom: 3,
  display: "block",
};

function NumInput({
  label, value, onChange, min, max, step = 1,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div style={{ marginBottom: 10 }}>
      <label style={labelStyle}>{label}</label>
      <input
        type="number"
        style={inputStyle}
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function SelectInput({
  label, value, options, onChange,
}: {
  label: string;
  value: string | undefined;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ marginBottom: 10 }}>
      <label style={labelStyle}>{label}</label>
      <select style={{ ...inputStyle }} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

function ModelToggle({
  id, label, description, enabled, onToggle, children,
}: {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
}) {
  return (
    <div
      style={{
        border: "1px solid var(--line)",
        borderRadius: 7,
        marginBottom: 10,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          background: enabled ? "rgba(231,60,37,0.04)" : "transparent",
          cursor: "pointer",
        }}
        onClick={onToggle}
      >
        <div
          style={{
            width: 36,
            height: 20,
            borderRadius: 10,
            background: enabled ? "var(--crimson, #e73c25)" : "var(--line)",
            position: "relative",
            transition: "background 0.2s",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: 3,
              left: enabled ? 18 : 3,
              width: 14,
              height: 14,
              borderRadius: "50%",
              background: "#fff",
              transition: "left 0.2s",
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12.5, fontWeight: 600 }}>{label}</div>
          <div style={{ fontSize: 10.5, color: "var(--muted)" }}>{description}</div>
        </div>
        {id === "ols" && (
          <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--muted)", fontStyle: "italic" }}>
            always on
          </span>
        )}
      </div>
      {enabled && children && (
        <div
          style={{
            padding: "12px 14px",
            borderTop: "1px solid var(--line)",
            background: "#fafaf8",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0 16px",
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------
export default function ModelsPage() {
  const { notify } = useNotification();

  const [config, setConfig] = useState<ProjectConfig | null>(null);

  // Local form state — derived from config
  const [gwr, setGwr] = useState<GwrParams>({});
  const [gwrf, setGwrf] = useState<GwrfParams>({});
  const [ggpgam, setGgpgam] = useState<GgpgamParams>({});
  const [neural, setNeural] = useState<NeuralParams>({});
  const [training, setTraining] = useState<TrainingParams>({});
  const [spatialCv, setSpatialCv] = useState<SpatialCvParams>({});
  const [nFolds, setNFolds] = useState(5);

  // Enabled model flags
  const [gwrEnabled, setGwrEnabled] = useState(true);
  const [gwrfEnabled, setGwrfEnabled] = useState(true);
  const [ggpgamEnabled, setGgpgamEnabled] = useState(true);
  const [neuralEnabled, setNeuralEnabled] = useState(true);

  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        setConfig(cfg);
        const m = cfg.models ?? {};
        setGwrEnabled("gwr" in m);
        setGwrfEnabled("gwrf" in m);
        setGgpgamEnabled("ggpgam" in m);
        setNeuralEnabled("neural" in m || !("gwr" in m || "gwrf" in m || "ggpgam" in m));
        setGwr((m.gwr as GwrParams) ?? { bandwidth: 200, kernel: "gaussian" });
        setGwrf((m.gwrf as GwrfParams) ?? { n_estimators: 100, k_neighbors: 100, min_samples_leaf: 5 });
        setGgpgam((m.ggpgam as GgpgamParams) ?? { n_splines: 10, n_spatial_bases: 12 });
        setNeural((m.neural as NeuralParams) ?? { hidden_dim: 256, dropout: 0.1, n_heads: 4, max_neighbors: 128 });
        const tr = (cfg as any).training ?? {};
        setTraining({ n_epochs: tr.n_epochs ?? 100, pretrain_epochs: tr.pretrain_epochs ?? 200, learning_rate: tr.learning_rate ?? 0.001 });
        const scv = (m as any).spatial_cv ?? {};
        setSpatialCv({
          block_size: scv.block_size ?? null,
          buffer_size: scv.buffer_size ?? 0,
          buffer_size_auto: scv.buffer_size_auto ?? false,
        });
        setNFolds(cfg.pipeline?.n_spatial_folds ?? 5);
      })
      .catch(() => {});
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      const modelsSection: Record<string, unknown> = {};
      if (gwrEnabled) modelsSection.gwr = gwr;
      if (gwrfEnabled) modelsSection.gwrf = gwrf;
      if (ggpgamEnabled) modelsSection.ggpgam = ggpgam;
      if (neuralEnabled) modelsSection.neural = neural;
      modelsSection.spatial_cv = {
        ...spatialCv,
        block_size: spatialCv.block_size ?? null,
      };

      await saveConfig({
        ...config,
        models: modelsSection,
        training: training as Record<string, unknown>,
        pipeline: {
          ...(config?.pipeline ?? {}),
          n_spatial_folds: nFolds,
        },
      });
      notify("Model parameters saved.", "success");
    } catch {
      notify("Failed to save parameters.", "error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <SectionHeader kicker="09 · analysis" label="Models" />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        {/* Model selection + params */}
        <Card title="Model selection" subtitle="enable models and set parameters">
          <ModelToggle
            id="ols"
            label="OLS"
            description="Ordinary Least Squares — global baseline"
            enabled={true}
            onToggle={() => {}}
          />
          <ModelToggle
            id="gwr"
            label="GWR"
            description="Geographically Weighted Regression"
            enabled={gwrEnabled}
            onToggle={() => setGwrEnabled((v) => !v)}
          >
            <NumInput label="Bandwidth (m)" value={gwr.bandwidth} onChange={(v) => setGwr({ ...gwr, bandwidth: v })} min={50} />
            <SelectInput
              label="Kernel"
              value={gwr.kernel}
              options={[
                { value: "gaussian", label: "Gaussian" },
                { value: "bisquare", label: "Bisquare" },
                { value: "exponential", label: "Exponential" },
              ]}
              onChange={(v) => setGwr({ ...gwr, kernel: v })}
            />
            <NumInput label="Alpha (regularisation)" value={gwr.alpha} onChange={(v) => setGwr({ ...gwr, alpha: v })} min={0} step={0.01} />
          </ModelToggle>
          <ModelToggle
            id="gwrf"
            label="GWRF"
            description="Geographically Weighted Random Forest"
            enabled={gwrfEnabled}
            onToggle={() => setGwrfEnabled((v) => !v)}
          >
            <NumInput label="N Estimators" value={gwrf.n_estimators} onChange={(v) => setGwrf({ ...gwrf, n_estimators: v })} min={10} />
            <NumInput label="K Neighbors" value={gwrf.k_neighbors} onChange={(v) => setGwrf({ ...gwrf, k_neighbors: v })} min={5} />
            <NumInput label="Min Samples Leaf" value={gwrf.min_samples_leaf} onChange={(v) => setGwrf({ ...gwrf, min_samples_leaf: v })} min={1} />
          </ModelToggle>
          <ModelToggle
            id="ggpgam"
            label="GGPGAM"
            description="Gaussian Process + Generalized Additive Model"
            enabled={ggpgamEnabled}
            onToggle={() => setGgpgamEnabled((v) => !v)}
          >
            <NumInput label="N Splines" value={ggpgam.n_splines} onChange={(v) => setGgpgam({ ...ggpgam, n_splines: v })} min={3} />
            <NumInput label="N Spatial Bases" value={ggpgam.n_spatial_bases} onChange={(v) => setGgpgam({ ...ggpgam, n_spatial_bases: v })} min={4} />
            <NumInput label="Lambda (smoothing)" value={ggpgam.lam} onChange={(v) => setGgpgam({ ...ggpgam, lam: v })} min={0} step={0.1} />
          </ModelToggle>
          <ModelToggle
            id="neural"
            label="Neural Network"
            description="V2 Physics-Guided Meta-Learner"
            enabled={neuralEnabled}
            onToggle={() => setNeuralEnabled((v) => !v)}
          >
            <NumInput label="Hidden Dim" value={neural.hidden_dim} onChange={(v) => setNeural({ ...neural, hidden_dim: v })} min={32} step={32} />
            <NumInput label="Dropout" value={neural.dropout} onChange={(v) => setNeural({ ...neural, dropout: v })} min={0} max={0.9} step={0.05} />
            <NumInput label="Attention Heads" value={neural.n_heads} onChange={(v) => setNeural({ ...neural, n_heads: v })} min={1} />
            <NumInput label="Max Neighbors" value={neural.max_neighbors} onChange={(v) => setNeural({ ...neural, max_neighbors: v })} min={8} />
          </ModelToggle>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Neural training parameters */}
          {neuralEnabled && (
            <Card title="Neural training" subtitle="epoch and optimiser settings">
              <NumInput label="Epochs" value={training.n_epochs} onChange={(v) => setTraining({ ...training, n_epochs: v })} min={10} />
              <NumInput label="Pretrain Epochs" value={training.pretrain_epochs} onChange={(v) => setTraining({ ...training, pretrain_epochs: v })} min={10} />
              <NumInput label="Learning Rate" value={training.learning_rate} onChange={(v) => setTraining({ ...training, learning_rate: v })} min={0} step={0.0001} />
            </Card>
          )}

          {/* Spatial CV settings */}
          <Card title="Spatial cross-validation" subtitle="block CV configuration">
            <NumInput label="N Spatial Folds" value={nFolds} onChange={setNFolds} min={2} max={20} />
            <div style={{ marginBottom: 10 }}>
              <label style={labelStyle}>Block Size (m)</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <input
                  type="checkbox"
                  id="block-auto"
                  checked={spatialCv.block_size === null || spatialCv.block_size === undefined}
                  onChange={(e) => setSpatialCv({ ...spatialCv, block_size: e.target.checked ? null : 500 })}
                />
                <label htmlFor="block-auto" style={{ ...labelStyle, marginBottom: 0, cursor: "pointer" }}>
                  Auto (from correlogram)
                </label>
              </div>
              {spatialCv.block_size !== null && spatialCv.block_size !== undefined && (
                <input
                  type="number"
                  style={inputStyle}
                  value={spatialCv.block_size}
                  min={100}
                  step={100}
                  onChange={(e) => setSpatialCv({ ...spatialCv, block_size: Number(e.target.value) })}
                />
              )}
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={labelStyle}>Buffer Size (m)</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <input
                  type="checkbox"
                  id="buffer-auto"
                  checked={spatialCv.buffer_size_auto ?? false}
                  onChange={(e) => setSpatialCv({ ...spatialCv, buffer_size_auto: e.target.checked })}
                />
                <label htmlFor="buffer-auto" style={{ ...labelStyle, marginBottom: 0, cursor: "pointer" }}>
                  Auto (⅓ of block size)
                </label>
              </div>
              {!(spatialCv.buffer_size_auto) && (
                <input
                  type="number"
                  style={inputStyle}
                  value={spatialCv.buffer_size ?? 0}
                  min={0}
                  step={50}
                  onChange={(e) => setSpatialCv({ ...spatialCv, buffer_size: Number(e.target.value) })}
                />
              )}
            </div>
          </Card>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button
          onClick={handleSave}
          disabled={saving}
          style={{
            padding: "8px 22px",
            background: saving ? "var(--muted)" : "var(--crimson, #e73c25)",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontSize: 12.5,
            fontWeight: 600,
            cursor: saving ? "not-allowed" : "pointer",
          }}
        >
          {saving ? "Saving…" : "Save Parameters"}
        </button>
      </div>
    </div>
  );
}


interface ModelInfo {
  name: string;
  fullName: string;
  r2: number;
  rmse: number;
  mae: number;
  weight: number;
  color: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("Ensemble");
  const residualsRef = useRef<HTMLCanvasElement>(null);
  const weightsRef = useRef<HTMLCanvasElement>(null);
  const { notify: _notify } = useNotification();

  useEffect(() => {
    // Load results from each pipeline stage
    const modelNames = ["OLS", "GWR", "GWRF", "GGPGAM", "Ensemble"];
    const colorMap: Record<string, string> = {
      OLS: SPARC_RAMP_HEX[0],
      GWR: SPARC_RAMP_HEX[2],
      GWRF: SPARC_RAMP_HEX[4],
      GGPGAM: SPARC_RAMP_HEX[6],
      Ensemble: SPARC_RAMP_HEX[8],
    };

    // Try stages 1-5
    Promise.all(
      [1, 2, 3, 4, 5].map((stage) =>
        getResults(stage).catch(() => null),
      ),
    ).then((results) => {
      const loaded: ModelInfo[] = [];
      results.forEach((r: any, i: number) => {
        if (!r) return;
        const name = r.model_name ?? modelNames[i] ?? `Stage ${i + 1}`;
        loaded.push({
          name: r.model_name ?? modelNames[i] ?? name,
          fullName: r.full_name ?? name,
          r2: r.r2 ?? r.metrics?.r2 ?? 0,
          rmse: r.rmse ?? r.metrics?.rmse ?? 0,
          mae: r.mae ?? r.metrics?.mae ?? 0,
          weight: r.weight ?? r.gwen_weight ?? 0,
          color: colorMap[name] ?? SPARC_RAMP_HEX[i % SPARC_RAMP_HEX.length],
        });
      });
      if (loaded.length > 0) setModels(loaded);
    });

    // Load GWEN weights
    getGwenData()
      .then((data) => {
        if (data?.rows) {
          setModels((prev) => {
            if (prev.length === 0) return prev;
            return prev.map((m) => {
              const row = data.rows.find((r: any) => r.model === m.name || r.name === m.name);
              if (row && typeof (row as any).weight === "number") {
                return { ...m, weight: (row as any).weight };
              }
              return m;
            });
          });
        }
      })
      .catch(() => {});
  }, []);

  // Draw residuals scatter
  useEffect(() => {
    const canvas = residualsRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const model = models.find((m) => m.name === selectedModel);
    if (!model) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see residuals", w / 2, h / 2);
      return;
    }

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Zero line
    const yMid = (h - 35) / 2 + 10;
    ctx.strokeStyle = "rgba(0,0,0,0.1)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(40, yMid); ctx.lineTo(w - 10, yMid); ctx.stroke();
    ctx.setLineDash([]);

    // Show RMSE band (real residual scatter requires prediction-level data)
    const rmseScale = model.rmse;
    if (rmseScale > 0) {
      const bandH = ((rmseScale) / (rmseScale * 2)) * (h - 45);
      ctx.fillStyle = model.color;
      ctx.globalAlpha = 0.1;
      ctx.fillRect(40, yMid - bandH, w - 50, bandH * 2);
      ctx.globalAlpha = 1;

      // RMSE label
      ctx.fillStyle = model.color;
      ctx.font = "bold 10px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(`±RMSE = ${rmseScale.toFixed(3)}`, 45, yMid - bandH - 4);
    } else {
      ctx.fillStyle = "#6e6358";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No residual data available", w / 2, h / 2);
    }

    // Labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Predicted", w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Residual", 0, 0);
    ctx.restore();
  }, [selectedModel, models]);

  // Draw ensemble weights bar
  useEffect(() => {
    const canvas = weightsRef.current;
    if (!canvas || models.length === 0) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const barH = 20;
    let x = 0;
    const total = models.reduce((s, m) => s + m.weight, 0);
    models.forEach((m) => {
      const barW = (m.weight / total) * w;
      ctx.fillStyle = m.color + "cc";
      ctx.fillRect(x, (h - barH) / 2, barW, barH);
      if (barW > 30) {
        ctx.fillStyle = "#fff";
        ctx.font = "bold 9px 'JetBrains Mono'";
        ctx.textAlign = "center";
        ctx.fillText(m.name, x + barW / 2, (h - barH) / 2 + barH / 2 + 3);
      }
      x += barW;
    });
  }, [models]);

  const best = models.length > 0
    ? models.reduce((a, b) => (a.r2 > b.r2 ? a : b), models[0])
    : null;

  if (models.length === 0) {
    return (
      <div>
        <SectionHeader kicker="09 · analysis" label="Models" />
        <Card title="Model results" subtitle="no models available">
          <div style={{ padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
            Run the pipeline to see model results.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <SectionHeader kicker="09 · analysis" label="Models" />

      <StatGrid>
        <Stat label="Models" value={String(models.length)} tint="var(--ink)" />
        <Stat label="Best R²" value={best?.r2?.toFixed(3) ?? "—"} tint="var(--crimson)" sub={best?.name ?? ""} />
        <Stat label="RMSE (best)" value={best?.rmse?.toFixed(3) ?? "—"} tint="var(--purple)" />
        <Stat label="Ensemble" value={models.find((m) => m.name === "Ensemble")?.r2?.toFixed(3) ?? "—"} tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Model zoo" subtitle="click to view residuals">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                <th style={thStyle}>Model</th>
                <th style={{ ...thStyle, textAlign: "right" }}>R²</th>
                <th style={{ ...thStyle, textAlign: "right" }}>RMSE</th>
                <th style={{ ...thStyle, textAlign: "right" }}>MAE</th>
                <th style={{ ...thStyle, width: 90 }}>R² bar</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.name}
                  onClick={() => setSelectedModel(m.name)}
                  style={{
                    borderTop: "1px solid var(--line)",
                    cursor: "pointer",
                    background: selectedModel === m.name ? "rgba(231,60,37,0.04)" : "transparent",
                  }}
                >
                  <td style={{ ...tdStyle, fontWeight: 600 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
                      <div>
                        <div>{m.name}</div>
                        <div className="mono" style={{ fontSize: 9, color: "var(--muted)" }}>{m.fullName}</div>
                      </div>
                    </div>
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right", fontWeight: 700 }}>
                    {m.r2.toFixed(3)}
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                    {m.rmse.toFixed(3)}
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                    {m.mae.toFixed(3)}
                  </td>
                  <td style={tdStyle}>
                    <div style={{ height: 10, background: "rgba(0,0,0,0.04)", borderRadius: 3, overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${m.r2 * 100}%`,
                          height: "100%",
                          background: m.color,
                          transition: "width 0.3s",
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Residuals scatter" subtitle={`${selectedModel} · predicted vs residual`}>
            <canvas
              ref={residualsRef}
              style={{ width: "100%", height: 200, display: "block" }}
            />
          </Card>

          <Card title="Ensemble weights" subtitle="GWEN-based stacking">
            <canvas
              ref={weightsRef}
              style={{ width: "100%", height: 50, display: "block", marginBottom: 8 }}
            />
            <div style={{ display: "flex", justifyContent: "center", gap: 14, flexWrap: "wrap" }}>
              {models.map((m) => (
                <div key={m.name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
                  <span className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>
                    {m.name} {(m.weight * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
