import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { usePipeline, type StageStatus } from "@/hooks/PipelineProvider";
import { useNotification } from "@/hooks/useNotifications";
import type { PipelineEvent } from "@/lib/types";

const STAGE_NAMES: Record<number, string> = {
  0: "Correlogram",
  1: "GWEN",
  2: "Validation",
  3: "Inference",
  4: "Simulation",
};

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Spatial autocorrelation analysis + pipeline config",
  1: "Variable selection via GWEN weighting",
  2: "Enhanced spatial cross-validation",
  3: "Causal validation (MC3 / NUTS)",
  4: "Scenario simulation + counterfactuals",
};

const STAGE_IDS = [0, 1, 2, 3, 4];

const LEVEL_COLORS: Record<string, string> = {
  info: "#a0a0a0",
  success: "#66bb6a",
  warn: "#ffa726",
  error: "#ef5350",
  debug: "#78909c",
};

function eventToLogLine(evt: PipelineEvent) {
  const ts = new Date().toTimeString().slice(0, 8);
  const type = (evt as any).type ?? "";

  if (type === "stage_status") {
    const ss = evt as any;
    const name = STAGE_NAMES[ss.stage] ?? `Stage ${ss.stage}`;
    const desc = STAGE_DESCRIPTIONS[ss.stage] ?? "";
    if (ss.status === "running") return { text: `▸ Starting ${name} — ${desc}`, level: "info" as const, ts };
    if (ss.status === "complete") {
      const dur = ss.elapsed_seconds ? ` (${ss.elapsed_seconds.toFixed(1)}s)` : "";
      return { text: `✓ ${name} complete${dur}`, level: "success" as const, ts };
    }
    if (ss.status === "failed") return { text: `✕ ${name} failed: ${ss.error ?? "unknown error"}`, level: "error" as const, ts };
    return null;
  }
  if (type === "epoch_update") {
    const e = evt as any;
    const eta = e.eta_seconds ? `  eta=${Math.round(e.eta_seconds)}s` : "";
    const phase = e.train_phase ? `[${e.train_phase}] ` : "";
    return { text: `  ${phase}epoch ${e.epoch}/${e.n_epochs}  loss=${e.total_loss?.toFixed(4) ?? "?"}${eta}`, level: "debug" as const, ts };
  }
  if (type === "metric") {
    const e = evt as any;
    const fold = e.fold != null ? ` fold ${e.fold}` : "";
    return { text: `  📊${fold} ${e.metric ?? "metric"} = ${typeof e.value === "number" ? e.value.toFixed(4) : e.value}`, level: "info" as const, ts };
  }
  if (type === "fold_start") {
    const e = evt as any;
    return { text: `  ── Fold ${e.fold}/${e.n_folds} ──`, level: "info" as const, ts };
  }
  if (type === "fold_complete") {
    const e = evt as any;
    const r2 = e.r2 != null ? `  R²=${e.r2.toFixed(4)}` : "";
    return { text: `  ✓ Fold ${e.fold} complete${r2}`, level: "success" as const, ts };
  }
  if (type === "model_start") {
    const e = evt as any;
    return { text: `  ▸ Training ${e.model_name ?? "model"}...`, level: "info" as const, ts };
  }
  if (type === "model_complete") {
    const e = evt as any;
    const r2 = e.r2 != null ? ` R²=${e.r2.toFixed(4)}` : "";
    return { text: `  ✓ ${e.model_name ?? "Model"} done${r2}`, level: "success" as const, ts };
  }
  if (type === "convergence") {
    return { text: `  convergence: ${(evt as any).status ?? ""}`, level: "info" as const, ts };
  }
  if (type === "curriculum_stage") {
    const e = evt as any;
    return { text: `  curriculum → ${e.label ?? e.curriculum ?? "next phase"}`, level: "info" as const, ts };
  }
  if (type === "capacity_result") {
    const e = evt as any;
    return { text: `  capacity check: dim=${e.hidden_dim} R²=${e.r2?.toFixed(4) ?? "?"}`, level: "debug" as const, ts };
  }
  if (type === "error") {
    return { text: (evt as any).message ?? "Error", level: "error" as const, ts };
  }
  if (type === "complete") {
    return { text: "Pipeline complete!", level: "success" as const, ts };
  }
  if (type === "dag_approval_requested") {
    return { text: "⏸ DAG approval requested — review on the DAG page", level: "warn" as const, ts };
  }
  if (type === "training_health") {
    return { text: `⚠ ${(evt as any).warning ?? "health warning"}`, level: "warn" as const, ts };
  }
  if (type === "progress") {
    const e = evt as any;
    return { text: `  ${e.message ?? `progress: ${e.pct ?? ""}%`}`, level: "debug" as const, ts };
  }
  // Catch-all for unknown event types with a message field
  if ((evt as any).message) {
    return { text: `  ${(evt as any).message}`, level: "info" as const, ts };
  }
  return null;
}

export default function RunPage() {
  const pipeline = usePipeline();
  const { notify } = useNotification();
  const logEndRef = useRef<HTMLDivElement>(null);
  const [enabledStages, setEnabledStages] = useState<Set<number>>(new Set(STAGE_IDS));
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Curated log lines derived from pipeline events
  const logLines = pipeline.events
    .map(eventToLogLine)
    .filter(Boolean) as { text: string; level: string; ts: string }[];

  // Auto-scroll
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines.length]);

  // Timer
  useEffect(() => {
    if (pipeline.isRunning) {
      const start = Date.now();
      timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [pipeline.isRunning]);

  const toggleStage = useCallback((id: number) => {
    setEnabledStages((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleStartAll = useCallback(() => {
    const stages = STAGE_IDS.filter((id) => enabledStages.has(id));
    if (stages.length === 0) {
      notify("warning", "No stages selected");
      return;
    }
    pipeline.startPipeline(stages, { fast: false });
    setElapsed(0);
    notify("info", `Pipeline started (${stages.length} stage${stages.length > 1 ? "s" : ""})`);
  }, [pipeline, enabledStages, notify]);

  const handleRunFromHere = useCallback((stage: number) => {
    const stages = STAGE_IDS.filter((id) => id >= stage && enabledStages.has(id));
    if (stages.length === 0) {
      pipeline.startStage(stage, { fast: false });
    } else {
      pipeline.startPipeline(stages, { fast: false });
    }
    setElapsed(0);
    notify("info", `Running from ${STAGE_NAMES[stage] ?? `stage ${stage}`}`);
  }, [pipeline, enabledStages, notify]);

  const handleStop = useCallback(() => {
    pipeline.cancel();
    notify("info", "Pipeline stopped");
  }, [pipeline, notify]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  const stageStatus = (id: number): StageStatus["status"] =>
    pipeline.stageStatuses[id]?.status ?? "pending";

  const currentStageName = pipeline.currentStage != null
    ? STAGE_NAMES[pipeline.currentStage] ?? `Stage ${pipeline.currentStage}`
    : "Idle";

  const doneCount = STAGE_IDS.filter((id) => stageStatus(id) === "complete").length;

  return (
    <div>
      <SectionHeader
        kicker="10 · pipeline"
        label="Run"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            {!pipeline.isRunning ? (
              <Btn primary onClick={handleStartAll}>▶ Start pipeline</Btn>
            ) : (
              <Btn onClick={handleStop}>◼ Stop</Btn>
            )}
          </div>
        }
      />

      <StatGrid>
        <Stat
          label="Status"
          value={pipeline.isRunning ? "Running" : doneCount === 5 ? "Complete" : "Idle"}
          tint={pipeline.isRunning ? "var(--crimson)" : doneCount === 5 ? "var(--purple)" : "var(--muted)"}
        />
        <Stat label="Stage" value={currentStageName} tint="var(--ink)" />
        <Stat label="Progress" value={`${doneCount}/5`} tint="var(--amber)" />
        <Stat label="Elapsed" value={formatTime(elapsed)} tint="var(--ink)" />
      </StatGrid>

      {pipeline.error && (
        <div style={{ padding: "10px 14px", background: "#fff0f0", border: "1px solid #ef5350", borderRadius: 6, marginBottom: 14, fontSize: 12, color: "#c62828" }}>
          {pipeline.error}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 14 }}>
        {/* Terminal output */}
        <Card title="Terminal" subtitle={`${logLines.length} lines · ${pipeline.isRunning ? "streaming..." : "idle"}`}>
          <div
            style={{
              background: "#1a1416",
              borderRadius: 6,
              padding: "12px 14px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              lineHeight: 1.65,
              height: 400,
              overflowY: "auto",
              color: "#ccc",
            }}
            className="scroll"
          >
            {logLines.length === 0 && !pipeline.isRunning && (
              <div style={{ color: "#555", padding: 20, textAlign: "center" }}>
                Press ▶ Start pipeline to begin
              </div>
            )}
            {logLines.map((line, i) => (
              <div key={i} style={{ display: "flex", gap: 10 }}>
                <span style={{ color: "#555", flexShrink: 0 }}>{line.ts}</span>
                <span
                  style={{
                    color: LEVEL_COLORS[line.level] ?? "#a0a0a0",
                    flexShrink: 0,
                    width: 50,
                    textTransform: "uppercase",
                    fontSize: 9,
                    lineHeight: "18px",
                  }}
                >
                  {line.level}
                </span>
                <span style={{ color: line.level === "error" ? "#ef5350" : line.level === "success" ? "#66bb6a" : "#d0ccc5" }}>
                  {line.text}
                </span>
              </div>
            ))}
            {pipeline.isRunning && (
              <div style={{ color: "var(--crimson)" }}>
                <span style={{ animation: "blink 1s infinite" }}>▍</span>
              </div>
            )}
            <div ref={logEndRef} />
          </div>
        </Card>

        {/* Stages */}
        <Card title="Stages" subtitle={`${doneCount}/5 complete`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {STAGE_IDS.map((id, i) => {
              const status = stageStatus(id);
              const enabled = enabledStages.has(id);
              return (
                <div
                  key={id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "10px 0",
                    borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                    opacity: enabled ? 1 : 0.5,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => toggleStage(id)}
                    disabled={pipeline.isRunning}
                    style={{ accentColor: "var(--crimson)", flexShrink: 0 }}
                  />
                  <span
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 12,
                      fontWeight: 700,
                      background:
                        status === "complete"
                          ? "var(--ink)"
                          : status === "running"
                          ? "var(--crimson)"
                          : status === "failed"
                          ? "#ef5350"
                          : "rgba(0,0,0,0.05)",
                      color: status === "pending" ? "var(--muted)" : "#fff",
                    }}
                    className="mono"
                  >
                    {status === "complete" ? "✓" : status === "failed" ? "✕" : String(i + 1)}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 600 }}>{STAGE_NAMES[id]}</div>
                    <div className="mono" style={{ fontSize: 9, color: "var(--muted)", marginTop: 1 }}>
                      {STAGE_DESCRIPTIONS[id]}
                    </div>
                    {status === "running" && (
                      <div
                        style={{
                          height: 3,
                          background: "rgba(0,0,0,0.06)",
                          borderRadius: 2,
                          marginTop: 4,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: "70%",
                            height: "100%",
                            background: "var(--crimson)",
                            borderRadius: 2,
                            animation: "loadBar 1.5s ease-in-out infinite",
                          }}
                        />
                      </div>
                    )}
                  </div>
                  {!pipeline.isRunning && status !== "complete" && (
                    <button
                      onClick={() => handleRunFromHere(id)}
                      className="mono"
                      style={{
                        fontSize: 9,
                        color: "var(--muted)",
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        fontFamily: "inherit",
                        textDecoration: "underline",
                        padding: 0,
                      }}
                      title="Run all from here"
                    >
                      from here
                    </button>
                  )}
                  <Tag
                    color={
                      status === "complete"
                        ? "var(--ink)"
                        : status === "running"
                        ? "var(--crimson)"
                        : status === "failed"
                        ? "#ef5350"
                        : "var(--muted)"
                    }
                  >
                    {status}
                  </Tag>
                </div>
              );
            })}
          </div>

          {/* Pipeline flow diagram */}
          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <div
              className="mono"
              style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}
            >
              flow
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {STAGE_IDS.map((id, i) => {
                const status = stageStatus(id);
                return (
                  <div key={id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <div
                      style={{
                        width: 32,
                        height: 8,
                        borderRadius: 4,
                        background:
                          status === "complete"
                            ? "var(--ink)"
                            : status === "running"
                            ? "var(--crimson)"
                            : "rgba(0,0,0,0.06)",
                        transition: "background 0.3s",
                      }}
                    />
                    {i < STAGE_IDS.length - 1 && (
                      <span style={{ color: "var(--line)", fontSize: 10 }}>→</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
