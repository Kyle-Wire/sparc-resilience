import { useRef, useEffect } from "react";
import type { ReactNode } from "react";
import { Btn, Card, SectionHeader, Tag } from "@/components/ui/DesignSystem";
import { usePipeline } from "@/hooks/PipelineProvider";
import type { PipelineEvent } from "@/lib/types";

type Status = "done" | "running" | "queued";

const STAGE_LABELS = [
  "Correlogram Analysis",
  "GWEN Variable Selection",
  "Spatial Cross-Validation",
  "Causal Validation",
  "Scenario Simulation",
];

const DEMO_STAGES = [
  { n: 0, name: "Correlogram Analysis", status: "done" as Status, time: "00:14" },
  { n: 1, name: "GWEN Variable Selection", status: "done" as Status, time: "02:37" },
  { n: 2, name: "Spatial Cross-Validation", status: "done" as Status, time: "18:42" },
  { n: 3, name: "Causal Validation", status: "running" as Status, time: "04:12", progress: 0.62 },
  { n: 4, name: "Scenario Simulation", status: "queued" as Status, time: "—" },
];

const DEMO_TERMINAL = [
  { t: "var(--muted)", text: "[00:00:02] loading project.yml …" },
  { t: "var(--amber)", text: "[00:00:03] target: AAT_z ← 6 predictors" },
  { t: "#e6ddcb", text: "[stage 0] Moran's I @ lag 30m … 0.842" },
  { t: "#e6ddcb", text: "[stage 0] ↳ auto bandwidth = 180 m · block = 420 m" },
  { t: "#e6ddcb", text: "[stage 1] GWEN rank  1 Pct_Impervious   0.512" },
  { t: "#e6ddcb", text: "[stage 1] GWEN rank  2 Pct_Canopy       0.378" },
  { t: "#e6ddcb", text: "[stage 1] GWEN rank  3 NDVI             0.301" },
  { t: "var(--gold)", text: "[stage 2] OOF R² → OLS 0.294 · GWR 0.828 · GWRF 0.898" },
  { t: "var(--gold)", text: "[stage 2] meta-ensemble (enhanced) = 0.915 ✓" },
  { t: "#e6ddcb", text: "[stage 3] DML fold 3/5 ATE(Canopy) = −0.015" },
  { t: "#e6ddcb", text: "[stage 3] refutation: placebo   p=0.83 ✓" },
  { t: "#e6ddcb", text: "[stage 3] refutation: subset    p=0.71 ✓" },
  { t: "var(--crimson)", text: "[stage 3] running causal forest …" },
];

function formatTime(seconds: number | undefined): string {
  if (seconds === undefined) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function logColor(event: PipelineEvent): string {
  if (event.type === "error") return "var(--crimson)";
  if (event.message?.includes("stage 2") || event.message?.includes("R²")) return "var(--gold)";
  if (event.message?.includes("target:")) return "var(--amber)";
  if (event.message?.includes("loading")) return "var(--muted)";
  return "#e6ddcb";
}

function TermLine({ t = "#e6ddcb", children }: { t?: string; children: ReactNode }) {
  return <div style={{ color: t }}>{children}</div>;
}

function Blink() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 7,
        height: 12,
        background: "currentColor",
        verticalAlign: "middle",
        marginLeft: 2,
        animation: "blink 1s steps(1) infinite",
      }}
    />
  );
}

function statusBg(s: Status): string {
  if (s === "done") return "var(--ink)";
  if (s === "running") return "var(--crimson)";
  return "rgba(0,0,0,0.05)";
}

function statusTagColor(s: Status): string {
  if (s === "done") return "var(--ink)";
  if (s === "running") return "var(--crimson)";
  return "var(--muted)";
}

export default function RunPage() {
  const { events, isRunning, stageStatuses, currentStage, startStage, cancel } = usePipeline();
  const termRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal
  useEffect(() => {
    if (termRef.current) {
      termRef.current.scrollTop = termRef.current.scrollHeight;
    }
  }, [events.length]);

  const hasLiveData = events.length > 0 || isRunning;

  // Build stages from live data or fallback
  const stages = hasLiveData
    ? STAGE_LABELS.map((name, i) => {
        const ss = stageStatuses[i];
        let status: Status = "queued";
        if (ss?.status === "complete" || ss?.status === "failed") status = "done";
        else if (ss?.status === "running" || currentStage === i) status = "running";
        else if (currentStage !== null && currentStage !== undefined && currentStage > i) status = "done";

        return {
          n: i,
          name,
          status,
          time: formatTime(ss?.elapsed_seconds),
          progress: status === "running" ? (ss?.eta_seconds !== undefined ? 0.5 : undefined) : undefined,
        };
      })
    : DEMO_STAGES;

  // Build terminal lines from live events or fallback
  const logEvents = events.filter((e) => e.type === "log" && e.message);
  const termLines = hasLiveData
    ? logEvents.map((e, i) => ({ key: i, t: logColor(e), text: e.message ?? "" }))
    : DEMO_TERMINAL.map((l, i) => ({ key: i, ...l }));

  const currentStageName = currentStage !== null && currentStage !== undefined
    ? STAGE_LABELS[currentStage]?.toLowerCase() ?? "—"
    : "dml backdoor";

  return (
    <div>
      <SectionHeader
        kicker="11 · pipeline"
        label="Run"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={cancel} disabled={!isRunning}>
              Cancel
            </Btn>
            <Btn primary small onClick={() => startStage(0)} disabled={isRunning}>
              Re-run
            </Btn>
          </div>
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 14 }}>
        <Card title="Stages" subtitle="5-stage pipeline">
          {stages.map((s) => (
            <div
              key={s.n}
              style={{
                display: "grid",
                gridTemplateColumns: "30px 1fr 68px 70px",
                alignItems: "center",
                gap: 10,
                padding: "8px 0",
                borderTop: s.n > 0 ? "1px dashed var(--line)" : "none",
              }}
            >
              <span
                className="mono"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 26,
                  height: 26,
                  borderRadius: 4,
                  background: statusBg(s.status),
                  color: s.status === "queued" ? "var(--muted)" : "#fff",
                  fontSize: 11,
                  fontWeight: 700,
                }}
              >
                {s.n}
              </span>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>{s.name}</div>
                {s.status === "running" && s.progress !== undefined && (
                  <div
                    style={{
                      height: 4,
                      background: "rgba(0,0,0,0.05)",
                      borderRadius: 2,
                      marginTop: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${(s.progress ?? 0) * 100}%`,
                        height: "100%",
                        background: "var(--crimson)",
                      }}
                    />
                  </div>
                )}
              </div>
              <Tag color={statusTagColor(s.status)}>{s.status}</Tag>
              <span
                className="mono"
                style={{ fontSize: 10.5, textAlign: "right", color: "var(--muted)" }}
              >
                {s.time}
              </span>
            </div>
          ))}
        </Card>

        <Card
          title="Live terminal"
          subtitle={`stage ${currentStage ?? 3} · ${currentStageName}`}
          padding={0}
          style={{ minHeight: 280 }}
        >
          <div
            ref={termRef}
            style={{
              background: "#1a1416",
              color: "#e6ddcb",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 10.5,
              lineHeight: 1.55,
              padding: "12px 14px",
              height: "100%",
              overflow: "auto",
              borderRadius: "0 0 8px 8px",
            }}
          >
            {termLines.map((l) => (
              <TermLine key={l.key} t={l.t}>
                {l.text}
              </TermLine>
            ))}
            {isRunning && (
              <TermLine t="var(--crimson)">
                running … <Blink />
              </TermLine>
            )}
            {!isRunning && !hasLiveData && (
              <TermLine t="var(--crimson)">
                [stage 3] running causal forest … <Blink />
              </TermLine>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
