import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";

interface PipelineStage {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "error";
  elapsed?: string;
}

const STAGE_DEFS: PipelineStage[] = [
  { id: "data", name: "Data prep", status: "pending" },
  { id: "dag", name: "DAG / causal", status: "pending" },
  { id: "train", name: "Training", status: "pending" },
  { id: "eval", name: "Evaluation", status: "pending" },
  { id: "scenario", name: "Scenarios", status: "pending" },
];

interface LogLine {
  text: string;
  level: "info" | "success" | "warn" | "error" | "debug";
  timestamp: string;
  stage?: string;
}

const LEVEL_COLORS: Record<string, string> = {
  info: "#a0a0a0",
  success: "#66bb6a",
  warn: "#ffa726",
  error: "#ef5350",
  debug: "#78909c",
};

export default function RunPage() {
  const [stages, setStages] = useState<PipelineStage[]>(STAGE_DEFS);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const logEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const wsRef = useRef<WebSocket | null>(null);
  const { notify } = useNotification();

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Timer
  useEffect(() => {
    if (isRunning) {
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRunning]);

  const addLog = useCallback((text: string, level: LogLine["level"] = "info", stage?: string) => {
    const now = new Date();
    const timestamp = now.toTimeString().slice(0, 8);
    setLogs((prev) => [...prev, { text, level, timestamp, stage }]);
  }, []);

  const handleStart = useCallback(async () => {
    setIsRunning(true);
    setElapsed(0);
    setLogs([]);
    setStages(STAGE_DEFS);

    addLog("Pipeline started", "info");

    try {
      const ws = new WebSocket("ws://127.0.0.1:8008/run/stream");
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ stage: 0, fast: false }));
      };

      ws.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data);
          if (evt.type === "stage_start") {
            addLog(`Stage: ${evt.stage}`, "info", evt.stage);
            setStages((prev) =>
              prev.map((s) =>
                s.id === evt.stage
                  ? { ...s, status: "running" }
                  : s.status === "running"
                  ? { ...s, status: "done" }
                  : s,
              ),
            );
          } else if (evt.type === "log") {
            const level = evt.level ?? "info";
            addLog(evt.message ?? String(evt.data ?? ""), level, evt.stage);
          } else if (evt.type === "stage_end") {
            setStages((prev) =>
              prev.map((s) => (s.id === evt.stage ? { ...s, status: "done" } : s)),
            );
            addLog(`${evt.stage} complete`, "success", evt.stage);
          } else if (evt.type === "error") {
            addLog(evt.message ?? "Error", "error");
            setStages((prev) =>
              prev.map((s) => (s.status === "running" ? { ...s, status: "error" } : s)),
            );
          } else if (evt.type === "complete" || evt.type === "done") {
            addLog("Pipeline complete!", "success");
            setIsRunning(false);
            setStages((prev) => prev.map((s) => ({ ...s, status: s.status === "pending" ? "done" : s.status })));
            notify("success", "Pipeline run complete");
          }
        } catch {
          addLog(msg.data, "info");
        }
      };

      ws.onerror = () => {
        addLog("WebSocket connection failed — running simulation", "warn");
        ws.close();
        runSimulation();
      };

      ws.onclose = () => {
        setIsRunning(false);
      };
    } catch {
      runSimulation();
    }
  }, [addLog, notify]);

  const runSimulation = useCallback(async () => {
    setIsRunning(true);
    addLog("Simulating pipeline (backend not connected)", "warn");
    for (let i = 0; i < STAGE_DEFS.length; i++) {
      setStages((prev) =>
        prev.map((s, j) => ({
          ...s,
          status: j < i ? "done" : j === i ? "running" : "pending",
        })),
      );
      addLog(`▸ ${STAGE_DEFS[i].name}...`, "info", STAGE_DEFS[i].id);
      await new Promise((r) => setTimeout(r, 600));
      addLog(`  ✓ ${STAGE_DEFS[i].name} complete`, "success", STAGE_DEFS[i].id);
    }
    setStages((prev) => prev.map((s) => ({ ...s, status: "done" })));
    addLog("Pipeline complete!", "success");
    setIsRunning(false);
    notify("success", "Pipeline run complete (simulated)");
  }, [addLog, notify]);

  const handleStop = useCallback(() => {
    wsRef.current?.close();
    setIsRunning(false);
    addLog("Pipeline stopped by user", "warn");
    notify("info", "Pipeline stopped");
  }, [addLog, notify]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  const currentStage = stages.find((s) => s.status === "running")?.name ?? "Idle";
  const doneCount = stages.filter((s) => s.status === "done").length;

  return (
    <div>
      <SectionHeader
        kicker="10 · pipeline"
        label="Run"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            {!isRunning ? (
              <Btn primary onClick={handleStart}>▶ Start pipeline</Btn>
            ) : (
              <Btn onClick={handleStop}>◼ Stop</Btn>
            )}
          </div>
        }
      />

      <StatGrid>
        <Stat label="Status" value={isRunning ? "Running" : doneCount === 5 ? "Complete" : "Idle"} tint={isRunning ? "var(--crimson)" : doneCount === 5 ? "var(--purple)" : "var(--muted)"} />
        <Stat label="Stage" value={currentStage} tint="var(--ink)" />
        <Stat label="Progress" value={`${doneCount}/5`} tint="var(--amber)" />
        <Stat label="Elapsed" value={formatTime(elapsed)} tint="var(--ink)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 14 }}>
        {/* Terminal output */}
        <Card title="Terminal" subtitle={`${logs.length} lines · ${isRunning ? "streaming..." : "idle"}`}>
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
            {logs.map((line, i) => (
              <div key={i} style={{ display: "flex", gap: 10 }}>
                <span style={{ color: "#555", flexShrink: 0 }}>{line.timestamp}</span>
                <span
                  style={{
                    color: LEVEL_COLORS[line.level],
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
            {isRunning && (
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
            {stages.map((stage, i) => (
              <div
                key={stage.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
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
                      stage.status === "done"
                        ? "var(--ink)"
                        : stage.status === "running"
                        ? "var(--crimson)"
                        : stage.status === "error"
                        ? "#ef5350"
                        : "rgba(0,0,0,0.05)",
                    color: stage.status === "pending" ? "var(--muted)" : "#fff",
                  }}
                  className="mono"
                >
                  {stage.status === "done" ? "✓" : stage.status === "error" ? "✕" : String(i + 1)}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{stage.name}</div>
                  {stage.status === "running" && (
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
                <Tag
                  color={
                    stage.status === "done"
                      ? "var(--ink)"
                      : stage.status === "running"
                      ? "var(--crimson)"
                      : stage.status === "error"
                      ? "#ef5350"
                      : "var(--muted)"
                  }
                >
                  {stage.status}
                </Tag>
              </div>
            ))}
          </div>

          {/* Pipeline flow diagram (simplified) */}
          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <div
              className="mono"
              style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}
            >
              flow
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {stages.map((stage, i) => (
                <div key={stage.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <div
                    style={{
                      width: 32,
                      height: 8,
                      borderRadius: 4,
                      background:
                        stage.status === "done"
                          ? "var(--ink)"
                          : stage.status === "running"
                          ? "var(--crimson)"
                          : "rgba(0,0,0,0.06)",
                      transition: "background 0.3s",
                    }}
                  />
                  {i < stages.length - 1 && (
                    <span style={{ color: "var(--line)", fontSize: 10 }}>→</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
