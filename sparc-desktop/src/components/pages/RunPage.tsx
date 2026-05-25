import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { usePipeline, type StageStatus } from "@/hooks/PipelineProvider";
import { useNotification } from "@/hooks/useNotifications";
import { GwenApprovalModal } from "@/components/run/GwenApprovalModal";
import RunCompletionCard from "@/components/run/RunCompletionCard";
import { eventToLogLine } from "@/lib/pipelineEventFormat";
import { useNavigationStore } from "@/stores/navigationStore";

const STAGE_NAMES: Record<number, string> = {
  0: "Spatial Patterns",
  1: "Variable Selection",
  2: "Spatial Validation",
  3: "Causal Modelling",
  4: "Scenario Simulation",
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
  milestone: "#e79024",
  log: "#607d8b",
};

// eventToLogLine is defined in @/lib/pipelineEventFormat (imported above)

const PHASE_LABELS: Record<string, string> = {
  cv: "Cross-validation",
  retrain: "Full retrain",
  swa: "SWA averaging",
  jepa_pretrain: "JEPA pre-training",
  jepa_finetune: "JEPA fine-tune",
};

function fmtEta(secs: number): string {
  const s = Math.round(secs);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r > 0 ? `${m}m ${r}s` : `${m}m`;
}

function TrainingPanel({ training, isRunning, currentStage }: {
  training: import("@/hooks/PipelineProvider").TrainingTelemetry;
  isRunning: boolean;
  currentStage: number | null;
}) {
  const isTraining = isRunning && currentStage === 3;
  const lastEpoch = training.epochHistory[training.epochHistory.length - 1];

  if (!isTraining && !lastEpoch) return null;

  const pct = lastEpoch ? Math.round((lastEpoch.epoch / lastEpoch.n_epochs) * 100) : 0;
  const etaEvt = training.epochHistory.slice().reverse().find((e) => (e as any).eta_seconds != null);
  const etaSec = etaEvt ? Math.round((etaEvt as any).eta_seconds) : null;

  return (
    <Card
      title="Training"
      subtitle={isTraining ? "stage 3 · inference" : "stage 3 complete"}
      style={{ marginBottom: 14 }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {/* Progress bar */}
        {lastEpoch && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span className="mono" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                {PHASE_LABELS[lastEpoch.train_phase] ?? lastEpoch.train_phase}
              </span>
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-2)" }}>
                {lastEpoch.epoch} / {lastEpoch.n_epochs} epochs · {pct}%
              </span>
            </div>
            <div style={{ height: 6, background: "rgba(0,0,0,0.06)", borderRadius: 3, overflow: "hidden" }}>
              <div
                style={{
                  height: "100%",
                  width: `${pct}%`,
                  background: isTraining ? "var(--crimson)" : "var(--ink)",
                  borderRadius: 3,
                  transition: "width 0.4s ease",
                }}
              />
            </div>
          </div>
        )}

        {/* Metrics row */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {lastEpoch && (
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Loss</div>
              <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                {lastEpoch.total_loss.toFixed(4)}
              </div>
            </div>
          )}
          {etaSec != null && isTraining && (
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>ETA</div>
              <div style={{ fontSize: 13, fontWeight: 700 }}>
                {etaSec < 60 ? `${etaSec}s` : `${Math.floor(etaSec / 60)}m ${etaSec % 60}s`}
              </div>
            </div>
          )}
          {training.curriculumLabel && (
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Phase</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{training.curriculumLabel}</div>
            </div>
          )}
          {training.convergenceStatus && (
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Convergence</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{training.convergenceStatus}</div>
            </div>
          )}
        </div>

        {/* Health warnings */}
        {training.healthWarnings.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {training.healthWarnings.slice(-3).map((w, i) => (
              <div key={i} style={{ fontSize: 11, color: "#7c4a00", background: "#fffbf0", border: "1px solid var(--amber)", borderRadius: 4, padding: "4px 8px" }}>
                ⚠ {w.warning}
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}

export default function RunPage() {
  const pipeline = usePipeline();
  const { navigate } = useNavigationStore();
  const { notify } = useNotification();
  const logEndRef = useRef<HTMLDivElement>(null);
  const [enabledStages, setEnabledStages] = useState<Set<number>>(new Set(STAGE_IDS));
  const [verbosity, setVerbosity] = useState<"summary" | "normal" | "debug">("normal");
  const [fastMode, setFastMode] = useState(false);
  const [skipGwen, setSkipGwen] = useState(false);
  const [showGwenModal, setShowGwenModal] = useState(false);
  // clearedAt tracks how many events existed when user last hit Clear,
  // so we can slice them away without mutating pipeline state.
  const [clearedAt, setClearedAt] = useState(0);
  // elapsed is derived from context so it survives page navigation
  const [, forceUpdate] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const elapsed = pipeline.runStartedAt
    ? Math.floor(((pipeline.runEndedAt ?? Date.now()) - pipeline.runStartedAt) / 1000)
    : 0;

  // Curated log lines derived from pipeline events (sliced after last clear)
  const allLogLines = pipeline.events
    .slice(clearedAt)
    .map(eventToLogLine)
    .filter(Boolean) as { text: string; level: string; ts: string }[];

  // Filter by verbosity tier:
  //   summary  — milestones, errors, warnings, success only
  //   normal   — all structured events except debug-level and raw log lines
  //   debug    — everything including raw stdout log lines
  const logLines = allLogLines.filter((line) => {
    if (verbosity === "debug") return true;
    if (verbosity === "normal") return line.level !== "debug" && line.level !== "log";
    return ["milestone", "error", "warn", "success"].includes(line.level);
  });

  const copyLogs = useCallback(async () => {
    const text = allLogLines
      .map((l) => `[${l.ts}] ${l.level.toUpperCase().padEnd(8)} ${l.text}`)
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      notify("success", `Copied ${allLogLines.length} log lines`);
    } catch (e) {
      notify("error", "Could not copy to clipboard");
    }
  }, [allLogLines, notify]);

  const copyError = useCallback(async () => {
    if (!pipeline.error) return;
    try {
      await navigator.clipboard.writeText(pipeline.error);
      notify("success", "Error copied to clipboard");
    } catch (e) {
      notify("error", "Could not copy to clipboard");
    }
  }, [pipeline.error, notify]);

  // Auto-scroll
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines.length]);

  // Timer — tick every second to recompute derived elapsed from context.
  // Stops once runEndedAt is set so total time freezes at completion.
  useEffect(() => {
    if (pipeline.runStartedAt && !pipeline.runEndedAt) {
      timerRef.current = setInterval(() => forceUpdate((n) => n + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [pipeline.runStartedAt, pipeline.runEndedAt]);

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
    pipeline.startPipeline(stages, { fast: fastMode, skip_gwen: skipGwen });
    notify("info", `Pipeline started (${stages.length} stage${stages.length > 1 ? "s" : ""})`);
  }, [pipeline, enabledStages, fastMode, skipGwen, notify]);

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

  // Advance workflow step when all stages complete
  useEffect(() => {
    if (doneCount === STAGE_IDS.length && !pipeline.isRunning) {
      useNavigationStore.getState().markPipelineRun();
    }
  }, [doneCount, pipeline.isRunning]);
  const wasStopped =
    pipeline.runStartedAt !== null &&
    !pipeline.isRunning &&
    doneCount < STAGE_IDS.length &&
    !pipeline.error;

  const statusLabel = pipeline.isRunning
    ? "Running"
    : doneCount === STAGE_IDS.length
    ? "Complete"
    : wasStopped
    ? "Stopped"
    : "Idle";

  const statusTint = pipeline.isRunning
    ? "var(--crimson)"
    : doneCount === STAGE_IDS.length
    ? "var(--purple)"
    : wasStopped
    ? "var(--amber)"
    : "var(--muted)";

  return (
    <div>
      {showGwenModal && (
        <GwenApprovalModal
          onApproved={() => { setShowGwenModal(false); }}
          onDismiss={() => setShowGwenModal(false)}
        />
      )}
      <SectionHeader
        kicker="10 · pipeline"
        label="Run"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            {!pipeline.isRunning ? (
              <Btn primary onClick={handleStartAll}>
                {wasStopped ? "↺ Restart pipeline" : "▶ Start pipeline"}
              </Btn>
            ) : (
              <Btn onClick={handleStop}>◼ Stop</Btn>
            )}
          </div>
        }
      />

      <StatGrid>
        <Stat
          label="Status"
          value={statusLabel}
          tint={statusTint}
        />
        <Stat label="Stage" value={currentStageName} tint="var(--ink)" />
        <Stat label="Progress" value={`${doneCount}/${STAGE_IDS.length}`} tint="var(--amber)" />
        <Stat
          label={pipeline.runEndedAt ? "Total time" : "Elapsed"}
          value={formatTime(elapsed)}
          tint={pipeline.runEndedAt ? "var(--purple)" : "var(--ink)"}
        />
      </StatGrid>

      {doneCount === STAGE_IDS.length && pipeline.runEndedAt && !pipeline.error && (
        <RunCompletionCard
          elapsedSeconds={elapsed}
          stagesComplete={doneCount}
          totalStages={STAGE_IDS.length}
          onViewResults={() => navigate("Results")}
        />
      )}

      {pipeline.dagApprovalPending && (
        <div
          style={{
            padding: "10px 14px",
            background: "#fffbf0",
            border: "1px solid var(--amber)",
            borderRadius: 6,
            marginBottom: 14,
            fontSize: 12,
            color: "#7c4a00",
            display: "flex",
            alignItems: "center",
            gap: 12,
            justifyContent: "space-between",
          }}
        >
          <span>⚠ DAG approval required — the pipeline is paused until you review and approve the discovered causal graph.</span>
          <Btn small onClick={() => navigate("Configure")}>Review configuration →</Btn>
        </div>
      )}

      {pipeline.gwenApprovalPending && (
        <div
          style={{
            padding: "10px 14px",
            background: "#f0f7ff",
            border: "1px solid var(--blue, #1976d2)",
            borderRadius: 6,
            marginBottom: 14,
            fontSize: 12,
            color: "#0d47a1",
            display: "flex",
            alignItems: "center",
            gap: 12,
            justifyContent: "space-between",
          }}
        >
          <span>ℹ️ Stage 1 complete — review the GWEN variable rankings before Stage 2 begins.</span>
          <Btn small onClick={() => setShowGwenModal(true)}>Review variables →</Btn>
        </div>
      )}

      {pipeline.error && (
        <div
          style={{
            padding: "10px 14px",
            background: "#fff0f0",
            border: "1px solid #ef5350",
            borderRadius: 6,
            marginBottom: 14,
            fontSize: 12,
            color: "#c62828",
            display: "flex",
            alignItems: "flex-start",
            gap: 12,
            justifyContent: "space-between",
          }}
        >
          <div style={{ whiteSpace: "pre-wrap", fontFamily: "'JetBrains Mono', monospace", flex: 1 }}>
            {pipeline.error}
          </div>
          <Btn small onClick={copyError}>Copy</Btn>
        </div>
      )}

      <TrainingPanel
        training={pipeline.training}
        isRunning={pipeline.isRunning}
        currentStage={pipeline.currentStage}
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 14 }}>
        {/* Terminal output */}
        <Card
          title="Terminal"
          subtitle={
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span>{logLines.length} lines (of {allLogLines.length}) · {pipeline.isRunning ? "streaming…" : "idle"}</span>
              <div style={{ display: "flex", gap: 4 }}>
                {(["summary", "normal", "debug"] as const).map((tier) => (
                  <button
                    key={tier}
                    onClick={() => setVerbosity(tier)}
                    style={{
                      padding: "2px 8px",
                      fontSize: 9,
                      borderRadius: 3,
                      border: "1px solid " + (verbosity === tier ? "var(--crimson)" : "var(--line)"),
                      background: verbosity === tier ? "var(--crimson)" : "#fff",
                      color: verbosity === tier ? "#fff" : "var(--ink-2)",
                      cursor: "pointer",
                      fontFamily: "inherit",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      fontWeight: 600,
                    }}
                  >
                    {tier}
                  </button>
                ))}
              </div>
              <Btn small onClick={copyLogs} disabled={allLogLines.length === 0}>Copy</Btn>
              <Btn small onClick={() => setClearedAt(pipeline.events.length)} disabled={allLogLines.length === 0}>Clear</Btn>
            </div>
          }
        >
          <div
            style={{
              background: "#1a1416",
              borderRadius: 6,
              padding: "12px 14px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              lineHeight: 1.65,
              minHeight: 400,
              maxHeight: "calc(100vh - 280px)",
              overflowY: "auto",
              overflowX: "auto",
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
                <span style={{ color: line.level === "error" ? "#ef5350" : line.level === "success" ? "#66bb6a" : line.level === "milestone" ? "#e79024" : "#d0ccc5" }}>
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
        <Card title="Stages" subtitle={`${doneCount}/${STAGE_IDS.length} complete`}>
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
                    <div style={{ fontSize: 12.5, fontWeight: 600, display: "flex", alignItems: "center", gap: 5 }}>
                      {STAGE_NAMES[id]}
                      <span
                        title={STAGE_DESCRIPTIONS[id]}
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          justifyContent: "center",
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          border: "1px solid var(--muted)",
                          color: "var(--muted)",
                          fontSize: 9,
                          fontWeight: 700,
                          cursor: "default",
                          lineHeight: 1,
                          flexShrink: 0,
                        }}
                      >?</span>
                    </div>
                    {status === "running" && pipeline.stageProgress[id] != null && (
                      <div style={{ marginTop: 4, height: 3, borderRadius: 2, background: "rgba(0,0,0,0.08)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${pipeline.stageProgress[id]}%`, background: "var(--crimson)", transition: "width 0.3s ease" }} />
                      </div>
                    )}
                    {status === "running" && (pipeline.stageProgress[id] != null || pipeline.stageStatuses[id]?.eta_seconds != null) && (
                      <div style={{ fontSize: 9, color: "var(--muted)", marginTop: 2, display: "flex", gap: 5 }}>
                        {pipeline.stageProgress[id] != null && <span>{pipeline.stageProgress[id]}%</span>}
                        {pipeline.stageStatuses[id]?.eta_seconds != null && (
                          <span>· ETA {fmtEta(pipeline.stageStatuses[id]!.eta_seconds!)}</span>
                        )}
                      </div>
                    )}
                  </div>

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
              style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}
            >
              flow
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
              {STAGE_IDS.map((id, i) => {
                const status = stageStatus(id);
                const isActive = status === "running";
                const isDone = status === "complete";
                const isFailed = status === "failed";
                const bg = isDone
                  ? "var(--ink)"
                  : isActive
                  ? "var(--crimson)"
                  : isFailed
                  ? "#ef5350"
                  : "rgba(0,0,0,0.07)";
                const labelColor = isDone || isActive || isFailed ? "#fff" : "var(--muted)";
                return (
                  <div key={id} style={{ display: "flex", alignItems: "center", flex: 1, minWidth: 0 }}>
                    <div
                      title={STAGE_NAMES[id]}
                      style={{
                        flex: 1,
                        minWidth: 0,
                        height: 24,
                        borderRadius: 4,
                        background: bg,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        transition: "background 0.3s",
                        position: "relative",
                        overflow: "hidden",
                      }}
                    >
                      {/* Animated shimmer for running stage */}
                      {isActive && (
                        <span
                          style={{
                            position: "absolute",
                            inset: 0,
                            background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.18) 50%, transparent 100%)",
                            animation: "shimmer 1.4s infinite",
                          }}
                        />
                      )}
                      <span
                        className="mono"
                        style={{
                          fontSize: 8.5,
                          fontWeight: 700,
                          color: labelColor,
                          letterSpacing: "0.04em",
                          textTransform: "uppercase",
                          position: "relative",
                          zIndex: 1,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          padding: "0 4px",
                        }}
                      >
                        {isDone ? "✓" : isFailed ? "✕" : String(i + 1)}
                      </span>
                    </div>
                    {i < STAGE_IDS.length - 1 && (
                      <div
                        style={{
                          width: 10,
                          height: 2,
                          background: isDone ? "var(--ink)" : "rgba(0,0,0,0.1)",
                          flexShrink: 0,
                          transition: "background 0.3s",
                        }}
                      />
                    )}
                  </div>
                );
              })}
            </div>
            {/* Stage name labels */}
            <div style={{ display: "flex", gap: 0, marginTop: 5 }}>
              {STAGE_IDS.map((id, i) => (
                <div key={id} style={{ display: "flex", flex: 1, minWidth: 0 }}>
                  <div
                    className="mono"
                    style={{
                      flex: 1,
                      fontSize: 8,
                      color: stageStatus(id) === "pending" ? "var(--muted)" : "var(--ink-2)",
                      textAlign: "center",
                      letterSpacing: "0.03em",
                      textTransform: "uppercase",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {STAGE_NAMES[id]}
                  </div>
                  {i < STAGE_IDS.length - 1 && <div style={{ width: 10, flexShrink: 0 }} />}
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline run options */}
          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <div
              className="mono"
              style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}
            >
              options
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <label
                style={{ display: "flex", alignItems: "flex-start", gap: 8, cursor: pipeline.isRunning ? "not-allowed" : "pointer", opacity: pipeline.isRunning ? 0.5 : 1 }}
                title="Skip long secondary diagnostics for fast iteration runs"
              >
                <input
                  type="checkbox"
                  checked={fastMode}
                  onChange={(e) => setFastMode(e.target.checked)}
                  disabled={pipeline.isRunning}
                  style={{ accentColor: "var(--crimson)", marginTop: 2, flexShrink: 0 }}
                />
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600 }}>Fast mode</div>
                  <div style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 1.4 }}>Skip secondary diagnostics — faster iteration, less detail</div>
                </div>
              </label>
              <label
                style={{ display: "flex", alignItems: "flex-start", gap: 8, cursor: pipeline.isRunning ? "not-allowed" : "pointer", opacity: pipeline.isRunning ? 0.5 : 1 }}
                title="Bypass GWEN variable selection — use all predictors"
              >
                <input
                  type="checkbox"
                  checked={skipGwen}
                  onChange={(e) => setSkipGwen(e.target.checked)}
                  disabled={pipeline.isRunning}
                  style={{ accentColor: "var(--crimson)", marginTop: 2, flexShrink: 0 }}
                />
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600 }}>Skip GWEN</div>
                  <div style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 1.4 }}>Use all predictors without GWEN variable selection</div>
                </div>
              </label>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
