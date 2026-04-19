import { useState, useEffect, useRef } from "react";
import { usePipeline } from "@/hooks/PipelineProvider";
import type { PipelineEvent } from "@/lib/types";
import { CapacitySweepView } from "@/components/training/CapacitySweepView";
import { EpochLossChart } from "@/components/training/EpochLossChart";
import { GroupedLossChart } from "@/components/training/GroupedLossChart";
import { ConvergenceBadge } from "@/components/training/ConvergenceBadge";
import { TrainingHealthBadge } from "@/components/training/TrainingHealthBadge";
import PipelineFlow from "@/components/pipeline/PipelineFlow";
import { SectionHeader, Card, Btn, Stat } from "@/components/ui/DesignSystem";

const STAGES = [
  { value: 0, label: "0 — Correlogram" },
  { value: 1, label: "1 — GWEN" },
  { value: 2, label: "2 — Spatial CV" },
  { value: 3, label: "3 — Causal" },
  { value: 4, label: "4 — Scenarios" },
];

/**
 * Model-level weight map for Stage 2 progress.
 * Each model contributes a slice of the overall stage progress.
 */
const STAGE2_MODEL_WEIGHTS: Record<string, [number, number]> = {
  ols:     [5,  15],
  gwr:     [15, 35],
  gwrf:    [35, 55],
  ggpgam:  [55, 70],
};

/** Heuristic phase ordering for non-Stage-2 stages. */
const STAGE_PHASES: Record<number, string[]> = {
  0: ["Correlogram analysis", "Analyzing variable", "Pipeline configuration"],
  1: ["GWEN variable selection", "GWEN results"],
  3: ["Causal validation"],
  4: ["Scenario simulation"],
};

function phaseProgress(stage: number | undefined, phase: string | undefined): number {
  if (stage == null || !phase) return 0;
  const phases = STAGE_PHASES[stage];
  if (!phases) return 0;
  const idx = phases.findIndex((p) => phase.startsWith(p));
  if (idx < 0) return 0;
  return Math.round(((idx + 1) / phases.length) * 100);
}

/** Extract model-checkpoint progress for Stage 2. */
function modelCheckpointProgress(events: PipelineEvent[]): number | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.progress_pct !== undefined && e.model) {
      return e.progress_pct;
    }
  }
  return null;
}

/** Build a list of model milestones for the progress bar visualization. */
function getModelMilestones(events: PipelineEvent[]): { name: string; done: boolean; pct: number }[] {
  const models = Object.entries(STAGE2_MODEL_WEIGHTS).map(([name, [, endPct]]) => ({
    name: name.toUpperCase(),
    done: false,
    pct: endPct,
  }));

  const doneModels = new Set<string>();
  for (const e of events) {
    if (e.phase?.includes("complete") && e.model) {
      doneModels.add(e.model);
    }
  }
  for (const m of models) {
    if (doneModels.has(m.name.toLowerCase())) {
      m.done = true;
    }
  }

  return models;
}

const STAGE_LABELS = ["Correlogram", "GWEN", "Spatial CV", "Causal", "Scenarios"];

/** Return a CSS color for a terminal log line based on its content. */
function termColor(msg: string): string {
  if (/✓|complete|done|success|saved/i.test(msg)) return "var(--color-sparc-gold)";
  if (/⚠|warn/i.test(msg)) return "var(--color-sparc-amber)";
  if (/✗|error|fail|exception/i.test(msg)) return "var(--color-sparc-crimson)";
  if (/^stage|^─|^═|pipeline/i.test(msg)) return "var(--color-sparc-crimson)";
  if (/r²|rmse|metric|fold/i.test(msg)) return "var(--color-sparc-amber)";
  return "#8a8278";
}

/** Blinking block cursor for the terminal. */
function Blink() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 7,
        height: 12,
        background: "#e6ddcb",
        animation: "blink 1s steps(1) infinite",
        verticalAlign: "text-bottom",
        marginLeft: 2,
      }}
    />
  );
}

export default function PipelineRun() {
  const {
    events, isRunning, error, currentStage, training, stageStatuses,
    dagApprovalPending, startStage, cancel, handleApproveDag, handleRejectDag,
  } = usePipeline();
  const [logOpen, setLogOpen] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const handleRun = (stage: number) => {
    setLogOpen(true);
    startStage(stage);
  };

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events]);

  // Derived state
  const metrics = events.filter((e): e is PipelineEvent & { type: "metric" } => e.type === "metric");
  const lastMetric = metrics[metrics.length - 1];
  const logs = events.filter((e) => e.type === "log");
  const complete = events.find((e) => e.type === "complete");

  // Progress: prefer model checkpoint progress for Stage 2, else phase heuristic
  const latestPhaseEvent = [...events].reverse().find((e) => e.phase);
  const currentPhase = latestPhaseEvent?.phase ?? null;
  const latestStage = currentStage ?? latestPhaseEvent?.stage ?? lastMetric?.stage;

  const checkpointPct = latestStage === 2 ? modelCheckpointProgress(events) : null;
  const explicitPct = checkpointPct ?? [...events].reverse().find((e) => e.progress_pct !== undefined)?.progress_pct;
  const progressPct = explicitPct ?? phaseProgress(latestStage, currentPhase ?? undefined);

  // Model milestones for Stage 2
  const modelMilestones = latestStage === 2 ? getModelMilestones(events) : [];

  // Current model info
  const latestModelEvent = [...events].reverse().find((e) => e.model);
  const currentModel = latestModelEvent?.model?.toUpperCase() ?? null;

  // Latest epoch info for progress display
  const latestEpoch = training.epochHistory.length > 0
    ? training.epochHistory[training.epochHistory.length - 1]
    : null;

  // Build a descriptive phase label
  let progressLabel = complete ? "Complete" : (currentPhase ?? "Running…");
  if (!complete && isRunning) {
    if (latestEpoch && training.epochHistory.length > 0) {
      progressLabel = `${currentPhase ?? "Neural meta-learner"} — Epoch ${latestEpoch.epoch}/${latestEpoch.n_epochs} · loss ${latestEpoch.total_loss.toFixed(4)}`;
    } else if (currentModel) {
      progressLabel += ` — ${currentModel}`;
    }
  }

  return (
    <div>
      <SectionHeader
        kicker="11 · pipeline"
        label="Run"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={() => handleRun(-1)} disabled={isRunning} primary>
              Run All
            </Btn>
            {isRunning ? (
              <Btn small onClick={cancel}>Cancel</Btn>
            ) : (
              <Btn small disabled>Cancel</Btn>
            )}
          </div>
        }
      />

      {/* Stage buttons */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
        {STAGES.map((s) => (
          <Btn key={s.value} small onClick={() => handleRun(s.value)} disabled={isRunning}>
            {s.label}
          </Btn>
        ))}
      </div>

      {/* Progress bar */}
      {(isRunning || complete) && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
            <span style={{ fontWeight: 600, color: "var(--color-sparc-ink-2)" }}>{progressLabel}</span>
            <span className="mono" style={{ color: "var(--color-sparc-muted)" }}>
              {complete ? "100" : progressPct}%
            </span>
          </div>
          <div style={{ position: "relative", height: 6, width: "100%", overflow: "hidden", borderRadius: 4, background: "rgba(0,0,0,0.06)" }}>
            <div
              style={{
                height: "100%",
                borderRadius: 4,
                transition: "width 500ms ease",
                width: `${complete ? 100 : progressPct}%`,
                background: complete ? "var(--color-sparc-crimson)" : "var(--color-sparc-purple)",
              }}
            />
            {modelMilestones.map((m) => (
              <div
                key={m.name}
                style={{ position: "absolute", top: 0, height: "100%", width: 1, background: "rgba(0,0,0,0.25)", left: `${m.pct}%` }}
                title={m.name}
              />
            ))}
          </div>
          {modelMilestones.length > 0 && (
            <div className="mono" style={{ position: "relative", marginTop: 4, height: 14, fontSize: 9, color: "var(--color-sparc-muted)" }}>
              {modelMilestones.map((m) => (
                <span
                  key={m.name}
                  style={{
                    position: "absolute",
                    transform: "translateX(-50%)",
                    fontWeight: m.done ? 700 : 500,
                    color: m.done ? "var(--color-sparc-crimson)" : "var(--color-sparc-muted)",
                    left: `${m.pct}%`,
                  }}
                >
                  {m.done ? "✓ " : ""}{m.name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* PipelineFlow — horizontal node-edge diagram */}
      {(Object.keys(stageStatuses).length > 0 || isRunning || complete) && (
        <div style={{ marginBottom: 14 }}>
          <Card title="Stage flow" subtitle="pipeline execution graph" padding={14}>
            <PipelineFlow
              stages={STAGE_LABELS.map((label, i) => {
                const ss = stageStatuses[i]?.status;
                let status: "done" | "running" | "queued" = "queued";
                if (ss === "complete") status = "done";
                else if (ss === "running" || currentStage === i) status = "running";
                else if (ss === "failed") status = "done";
                if (currentStage !== null && currentStage !== undefined && currentStage > i && !ss) status = "done";
                return {
                  label,
                  status,
                  progress: status === "running" ? (progressPct ?? 0) / 100 : status === "done" ? 1 : 0,
                  duration: ss === "complete" ? "done" : undefined,
                };
              })}
              currentStageIndex={currentStage ?? 0}
            />
          </Card>
        </div>
      )}

      {/* Live metric dashboard */}
      {lastMetric && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
          <Stat label="Stage" value={String(lastMetric.stage ?? "—")} tint="var(--color-sparc-ink)" />
          <Stat label="Fold" value={String(lastMetric.fold ?? "—")} tint="var(--color-sparc-purple)" />
          <Stat
            label={lastMetric.metric?.toUpperCase() ?? "Metric"}
            value={lastMetric.value?.toFixed(4) ?? "—"}
            tint="var(--color-sparc-crimson)"
          />
          <Stat label="Progress" value={`${explicitPct ?? progressPct}%`} tint="var(--color-sparc-amber)" />
        </div>
      )}

      {/* Training telemetry — visible during Stage 2 */}
      {(latestStage === 2 || training.epochHistory.length > 0 || training.capacityResults.length > 0) && (
        <div style={{ marginBottom: 14 }}>
          <Card
            title="Neural training"
            subtitle="capacity sweep · loss curves · health"
            padding={14}
            actions={
              <ConvergenceBadge
                status={training.convergenceStatus ?? (isRunning && training.epochHistory.length > 0 ? "training" : null)}
              />
            }
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <CapacitySweepView results={training.capacityResults} />
              <GroupedLossChart
                epochHistory={training.epochHistory}
                curriculumStage={training.curriculumStage}
                curriculumLabel={training.curriculumLabel}
              />
              <EpochLossChart
                epochHistory={training.epochHistory}
                curriculumStage={training.curriculumStage}
                curriculumLabel={training.curriculumLabel}
              />
              <TrainingHealthBadge warnings={training.healthWarnings} />
            </div>
          </Card>
        </div>
      )}

      {/* DAG approval gate banner */}
      {dagApprovalPending && (
        <div
          style={{
            marginBottom: 14,
            border: "1px solid var(--color-sparc-amber)",
            background: "rgba(231,144,36,0.08)",
            borderRadius: 8,
            padding: 14,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--color-sparc-ink)" }}>
              DAG review required — pipeline paused
            </div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--color-sparc-muted)", marginTop: 3 }}>
              MC³ structure learning complete. Review in DAG tab, then approve to continue to NUTS sampling.
            </div>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <Btn small onClick={handleRejectDag}>Reject</Btn>
            <Btn small primary onClick={handleApproveDag}>Approve DAG</Btn>
          </div>
        </div>
      )}

      {/* Status banners */}
      {complete && (
        <div
          className="mono"
          style={{
            marginBottom: 14,
            border: "1px solid var(--color-sparc-crimson)",
            background: "rgba(231,60,37,0.05)",
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 11,
            color: "var(--color-sparc-crimson)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          Pipeline complete
        </div>
      )}
      {error && (
        <div
          className="mono"
          style={{
            marginBottom: 14,
            border: "1px solid var(--color-sparc-crimson)",
            background: "rgba(231,60,37,0.08)",
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 11,
            color: "var(--color-sparc-crimson)",
          }}
        >
          {error}
        </div>
      )}

      {/* Completion summary strip */}
      {complete && lastMetric && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
          <Stat label="R²" value={lastMetric.value?.toFixed(3) ?? "—"} tint="var(--color-sparc-crimson)" />
          <Stat label="RMSE" value={lastMetric.metric === "rmse" ? (lastMetric.value?.toFixed(3) ?? "—") : "—"} tint="var(--color-sparc-ink)" />
          <Stat label="Stage" value={String(lastMetric.stage ?? "—")} tint="var(--color-sparc-purple)" />
          <Stat label="Logs" value={String(logs.length)} tint="var(--color-sparc-amber)" />
        </div>
      )}

      {/* Dark terminal output */}
      <div style={{ borderRadius: 8, overflow: "hidden", border: "1px solid rgba(255,255,255,0.06)" }}>
        <button
          onClick={() => setLogOpen(!logOpen)}
          style={{
            display: "flex",
            width: "100%",
            alignItems: "center",
            justifyContent: "space-between",
            background: "#1a1416",
            border: "none",
            padding: "8px 14px",
            fontSize: 10.5,
            fontWeight: 600,
            color: "#8a8278",
            cursor: "pointer",
            fontFamily: "var(--font-mono)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          <span>Terminal Output ({logs.length} lines)</span>
          <span style={{ color: "#6e6358" }}>{logOpen ? "▲ Hide" : "▼ Show"}</span>
        </button>
        {logOpen && (
          <div
            ref={logRef}
            className="scroll"
            style={{
              height: 360,
              overflowY: "auto",
              background: "#1a1416",
              padding: "12px 14px",
              fontFamily: "var(--font-mono)",
              fontSize: 10.5,
              lineHeight: 1.55,
              color: "#e6ddcb",
            }}
          >
            {logs.length === 0 && !isRunning && (
              <span style={{ color: "#6e6358" }}>No output yet. Start a stage to begin.</span>
            )}
            {logs.map((e, i) => (
              <div key={i} style={{ padding: "1px 0", color: termColor(e.message ?? "") }}>
                {e.message}
              </div>
            ))}
            {isRunning && (
              <div style={{ padding: "1px 0", color: "#e6ddcb" }}>
                <Blink />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
