import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, KeyVal, Tag, Btn } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, dataSummary, getRunEvents } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { pickYaml } from "@/lib/fileDialogs";
import ProjectCreationWizard from "@/components/project/ProjectCreationWizard";
import type { ProjectConfig, DataSummary, PipelineEvent } from "@/lib/types";
import { useProjectStore } from "@/stores/projectStore";
import { useNavigationStore } from "@/stores/navigationStore";


const DOMAIN_COLORS: Record<string, string> = {
  uhi: "var(--crimson)",
  forcesmip: "var(--purple)",
  groundwater: "#9e337d",
  air_quality: "var(--amber)",
  stormwater: "#e94d9b",
  coastal: "#e94461",
  geotechnical: "#e76c25",
  seismic: "var(--gold)",
  noise: "var(--muted)",
  wildfire: "var(--crimson)",
  drought: "var(--amber)",
  water_quality: "var(--purple)",
};

/** Pull a list of recent pipeline runs out of the buffered event log. */
function summarizeRuns(events: PipelineEvent[]): { time: string; stage: string; status: string; tint: string }[] {
  if (!events.length) return [];
  const stageNames: Record<number, string> = {
    0: "Correlogram", 1: "GWEN", 2: "Spatial CV", 3: "Causal", 4: "Scenarios",
  };
  const out: { time: string; stage: string; status: string; tint: string }[] = [];
  for (const e of events) {
    if (e.type === "stage_status" && (e.status === "complete" || e.status === "failed")) {
      const ts = e.completed_at ?? e.started_at ?? Date.now();
      out.push({
        time: new Date(ts * (ts < 1e12 ? 1000 : 1)).toLocaleString(),
        stage: stageNames[e.stage ?? -1] ?? `Stage ${e.stage}`,
        status: e.status === "complete" ? "✓ complete" : "✕ failed",
        tint: e.status === "complete" ? "var(--purple)" : "var(--crimson)",
      });
    }
  }
  return out.slice(-6).reverse();
}

export default function ProjectPage() {
  const { projectPath, openProject } = useProjectStore();
  const { navigate } = useNavigationStore();
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [runs, setRuns] = useState<{ time: string; stage: string; status: string; tint: string }[]>([]);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [flashField, setFlashField] = useState<string | null>(null);
  const { notify } = useNotification();

  useEffect(() => {
    if (projectPath) {
      getConfig().then(setConfig).catch(() => {});
      dataSummary().then(setSummary).catch(() => setSummary(null));
      getRunEvents()
        .then((r) => setRuns(summarizeRuns(r.events)))
        .catch(() => setRuns([]));
    }
  }, [projectPath]);

  // Esc closes the wizard modal
  useEffect(() => {
    if (!wizardOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setWizardOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [wizardOpen]);

  const handleOpenYml = useCallback(async () => {
    const path = await pickYaml();
    if (path) {
      await openProject(path);
      notify("success", "Project loaded successfully");
      navigate("Data");
    }
  }, [openProject, notify, navigate]);

  const project = config?.project ?? {};
  const data = config?.data ?? {};
  const domain = project.domain ?? "";

  /** Persist a single project-section field and refresh local state. */
  const saveProjectField = useCallback(
    async (field: keyof NonNullable<ProjectConfig["project"]>, value: string) => {
      try {
        await saveConfig({ project: { ...project, [field]: value } });
        const updated = await getConfig();
        setConfig(updated);
      } catch (e) {
        notify("error", e instanceof Error ? e.message : "Failed to save");
        return;
      }
      setFlashField(field);
      window.setTimeout(() => setFlashField(null), 700);
    },
    [project, notify],
  );

  return (
    <div>
      <SectionHeader
        kicker="01 · setup"
        label="Project"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn onClick={handleOpenYml}>Open existing…</Btn>
            <Btn primary onClick={() => setWizardOpen(true)}>New project…</Btn>
          </div>
        }
      />

      {wizardOpen && (
        <div
          onMouseDown={(e) => {
            // Click on backdrop (not the modal itself) closes the wizard
            if (e.target === e.currentTarget) setWizardOpen(false);
          }}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            background: "rgba(26, 20, 22, 0.42)",
            backdropFilter: "blur(4px)",
            WebkitBackdropFilter: "blur(4px)",
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "center",
            padding: "6vh 24px 24px",
            overflowY: "auto",
            animation: "sparc-modal-fade 160ms ease-out",
          }}
        >
          <div
            style={{
              width: "100%",
              maxWidth: 880,
              animation: "sparc-modal-pop 200ms cubic-bezier(0.2, 0.8, 0.2, 1)",
              boxShadow: "0 24px 60px rgba(26,20,22,0.28), 0 4px 12px rgba(26,20,22,0.16)",
              borderRadius: 8,
            }}
          >
            <ProjectCreationWizard
              templates={[]}
              onCancel={() => setWizardOpen(false)}
              onCreated={async (path, meta) => {
                setWizardOpen(false);
                await openProject(path, meta);
                notify("success", "Project created successfully");
                navigate("Data");
              }}
            />
          </div>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {/* ---- Project Identity (full width) ---- */}
        <Card title="Project identity" subtitle={projectPath ?? "no project loaded"}>
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

            {/* Name — large, prominent */}
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
                Name
              </span>
              <input
                value={project.name ?? ""}
                placeholder="Project name"
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, name: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("name", e.target.value)}
                style={{
                  border: "none",
                  borderBottom: `1.5px solid ${flashField === "name" ? "var(--purple)" : "var(--line)"}`,
                  background: "transparent", padding: "2px 0",
                  fontSize: 20, fontWeight: 800, fontFamily: "inherit",
                  color: "var(--ink)", outline: "none",
                  letterSpacing: "-0.02em",
                  transition: "border-color 300ms ease",
                }}
              />
            </label>

            {/* Domain · Target variable · Response units — inline row */}
            <div style={{ display: "flex", gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
              {domain && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Domain</span>
                  <Tag color={DOMAIN_COLORS[domain] ?? "var(--muted)"}>{domain.replace(/_/g, " ")}</Tag>
                </div>
              )}
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Target variable</span>
                <Tag color="var(--purple)">{data.target_column ?? "—"}</Tag>
              </div>
              <label style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 130 }}>
                <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Response units</span>
                <input
                  value={project.response_units ?? ""}
                  placeholder="e.g. °C, mg/L, dB"
                  onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, response_units: e.target.value } } : c)}
                  onBlur={(e) => saveProjectField("response_units", e.target.value)}
                  style={{
                    border: "none",
                    borderBottom: `1.5px solid ${flashField === "response_units" ? "var(--purple)" : "var(--line)"}`,
                    background: "transparent", padding: "2px 0",
                    fontSize: 13, fontFamily: "inherit", color: "var(--ink)", outline: "none",
                    transition: "border-color 300ms ease",
                  }}
                />
              </label>
            </div>

            {/* Description */}
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Description</span>
              <textarea
                value={project.description ?? ""}
                placeholder="One-paragraph description of this project's purpose, scope, and target audience."
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, description: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("description", e.target.value)}
                rows={3}
                style={{
                  border: `1px solid ${flashField === "description" ? "var(--purple)" : "var(--line)"}`,
                  borderRadius: 4, background: "transparent",
                  padding: "6px 8px", fontSize: 12, fontFamily: "inherit",
                  color: "var(--ink-2)", outline: "none", resize: "vertical",
                  transition: "border-color 300ms ease",
                }}
              />
            </label>

            {/* Predictors chip list */}
            {config?.predictors && config.predictors.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>Predictors</span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 9.5, fontWeight: 700, color: "var(--purple)",
                      background: "rgba(96,36,104,0.08)", padding: "1px 7px", borderRadius: 10,
                    }}
                  >
                    {config.predictors.length}
                  </span>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", letterSpacing: "0.06em" }}>
                    · edit on Variables page
                  </span>
                </div>
                <div
                  style={{
                    display: "flex", flexWrap: "wrap", gap: 5,
                    maxHeight: 78, overflowY: "auto", paddingRight: 4,
                  }}
                >
                  {config.predictors.map((p) => (
                    <span
                      key={p}
                      className="mono"
                      style={{
                        fontSize: 10, padding: "2px 8px",
                        background: "var(--paper-2)", border: "1px solid var(--line)",
                        borderRadius: 3, color: "var(--ink-2)",
                      }}
                    >
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* ---- Dataset + Pipeline history (side-by-side) ---- */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Card title="Dataset snapshot" subtitle={data.file_path ? data.file_path.split(/[\\/]/).pop() : "no data linked"}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <KeyVal label="Rows" value={summary?.row_count?.toLocaleString() ?? "—"} />
              <KeyVal label="Columns" value={summary?.column_count?.toString() ?? "—"} />
              <KeyVal
                label="Bounding box"
                value={summary?.bbox
                  ? `${summary.bbox.minx.toFixed(2)}, ${summary.bbox.miny.toFixed(2)}  →  ${summary.bbox.maxx.toFixed(2)}, ${summary.bbox.maxy.toFixed(2)}`
                  : "—"}
              />
              <KeyVal
                label="File"
                value={
                  data.file_path ? (
                    <span
                      className="mono"
                      title={data.file_path}
                      style={{
                        fontSize: 10.5,
                        color: "var(--ink-2)",
                        display: "block",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        maxWidth: 220,
                      }}
                    >
                      {data.file_path}
                    </span>
                  ) : "—"
                }
              />
            </div>
          </Card>

          <Card title="Pipeline history" subtitle={runs.length ? `${runs.length} recent stage events` : "no runs yet"}>
          {runs.length === 0 ? (
            <div style={{ fontSize: 12, color: "var(--muted)", padding: "10px 0" }}>
              Run the pipeline to see stage completion history here.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {runs.map((r, i) => (
                <div
                  key={i}
                  style={{
                    display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 8,
                    fontSize: 11, padding: "5px 0",
                    borderBottom: i < runs.length - 1 ? "1px dotted var(--line)" : "none",
                  }}
                >
                  <span className="mono" style={{ color: "var(--muted)", fontSize: 10 }}>{r.time}</span>
                  <span style={{ color: r.tint === "var(--crimson)" ? "var(--crimson)" : "var(--ink-2)", fontWeight: r.tint === "var(--crimson)" ? 600 : 400 }}>{r.stage}</span>
                  <span className="mono" style={{ color: r.tint, fontSize: 10 }}>{r.status}</span>
                </div>
              ))}
              <div style={{ marginTop: 6, display: "flex", justifyContent: "flex-end" }}>
                <button
                  onClick={() => navigate("Run")}
                  style={{
                    background: "none", border: "none", cursor: "pointer",
                    fontFamily: "inherit", fontSize: 11, fontWeight: 600,
                    color: "var(--purple)", padding: 0, letterSpacing: "-0.01em",
                  }}
                >
                  → Run page
                </button>
              </div>
            </div>
          )}
        </Card>
        </div>
      </div>
    </div>
  );
}
