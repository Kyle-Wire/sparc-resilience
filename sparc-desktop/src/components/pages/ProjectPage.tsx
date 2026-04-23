import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, KeyVal, Tag, Btn } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, listTemplates, initProject, dataSummary, getRunEvents } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import ProjectCreationWizard from "@/components/project/ProjectCreationWizard";
import type { ProjectConfig, TemplateInfo, DataSummary, PipelineEvent } from "@/lib/types";

const TEMPLATE_COLORS: Record<string, string> = {
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

interface ProjectPageProps {
  projectPath: string | null;
  onProjectLoaded: (path: string, meta?: { name?: string; template?: string }) => Promise<void>;
}

/** Lightweight YAML emitter for the read-only preview block. */
function toYamlPreview(config: ProjectConfig | null): string {
  if (!config) return "# No project loaded";
  const project = config.project ?? {};
  const data = config.data ?? {};
  const crs = config.crs ?? {};
  const flags = config.flags ?? {};
  const predictors = config.predictors ?? [];
  return [
    "project:",
    `  name: ${project.name ?? ""}`,
    `  description: ${project.description ?? ""}`,
    `  domain: ${project.domain ?? ""}`,
    `  author: ${project.author ?? ""}`,
    `  version: ${project.version ?? ""}`,
    `  response_units: ${project.response_units ?? ""}`,
    "data:",
    `  file_path: ${data.file_path ?? ""}`,
    `  target_column: ${data.target_column ?? ""}`,
    "crs:",
    `  input: ${crs.input ?? ""}`,
    `  projected: ${crs.projected ?? ""}`,
    "predictors:",
    ...predictors.map((p) => `  - ${p}`),
    "flags:",
    ...Object.entries(flags).map(([k, v]) => `  ${k}: ${v}`),
  ].join("\n");
}

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

export default function ProjectPage({ projectPath, onProjectLoaded }: ProjectPageProps) {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [runs, setRuns] = useState<{ time: string; stage: string; status: string; tint: string }[]>([]);
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [yamlOpen, setYamlOpen] = useState(false);
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const { notify } = useNotification();

  useEffect(() => {
    if (projectPath) {
      getConfig().then(setConfig).catch(() => {});
      dataSummary().then(setSummary).catch(() => setSummary(null));
      getRunEvents()
        .then((r) => setRuns(summarizeRuns(r.events)))
        .catch(() => setRuns([]));
    }
    listTemplates()
      .then((r) => setTemplates(r.templates))
      .catch(() => {});
  }, [projectPath]);

  const handleTemplateClick = useCallback(
    async (t: TemplateInfo) => {
      const home = prompt("Choose output directory:", `${t.name}_project`);
      if (!home) return;
      try {
        const res = await initProject(t.name, home);
        await onProjectLoaded(res.project_yml, { template: t.name });
        notify("success", `Project created from ${t.name} template`);
      } catch (e) {
        notify("error", e instanceof Error ? e.message : "Failed to create project");
      }
    },
    [onProjectLoaded, notify],
  );

  const handleOpenYml = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({
        filters: [{ name: "YAML", extensions: ["yml", "yaml"] }],
        multiple: false,
      });
      if (path) await onProjectLoaded(path as string);
    } catch {
      const path = prompt("Enter path to project.yml:");
      if (path) await onProjectLoaded(path);
    }
  }, [onProjectLoaded]);

  const project = config?.project ?? {};
  const data = config?.data ?? {};
  const crs = config?.crs ?? {};
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
      }
    },
    [project, notify],
  );

  const yamlPreview = toYamlPreview(config);

  return (
    <div>
      <SectionHeader
        kicker="01 · setup"
        label="Project"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn onClick={handleOpenYml}>Open project.yml</Btn>
            <Btn onClick={() => setTemplatesOpen((o) => !o)}>
              {templatesOpen ? "Hide templates" : "From template"}
            </Btn>
            <Btn primary onClick={() => setWizardOpen(true)}>New project…</Btn>
          </div>
        }
      />

      {wizardOpen && (
        <ProjectCreationWizard
          templates={templates}
          onCancel={() => setWizardOpen(false)}
          onCreated={async (path, meta) => {
            setWizardOpen(false);
            await onProjectLoaded(path, meta);
          }}
        />
      )}

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        {/* ---- Project Identity ---- */}
        <Card title="Project identity" subtitle={projectPath ?? "no project loaded"}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Name
              </span>
              <input
                value={project.name ?? ""}
                placeholder="Project name"
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, name: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("name", e.target.value)}
                style={{
                  border: "none", borderBottom: "1px dashed var(--line)",
                  background: "transparent", padding: "2px 0",
                  fontSize: 13, fontFamily: "inherit", color: "var(--ink)", outline: "none",
                }}
              />
            </label>

            <KeyVal label="Domain" value={<Tag color="var(--crimson)">{domain || "—"}</Tag>} />

            <label style={{ display: "flex", flexDirection: "column", gap: 4, gridColumn: "1 / -1" }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Description
              </span>
              <textarea
                value={project.description ?? ""}
                placeholder="One-paragraph description of this project's purpose, scope, and target audience."
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, description: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("description", e.target.value)}
                rows={3}
                style={{
                  border: "1px dashed var(--line)", borderRadius: 4, background: "transparent",
                  padding: "6px 8px", fontSize: 12, fontFamily: "inherit", color: "var(--ink-2)",
                  outline: "none", resize: "vertical",
                }}
              />
            </label>

            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Author
              </span>
              <input
                value={project.author ?? ""}
                placeholder="Author name or organization"
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, author: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("author", e.target.value)}
                style={{
                  border: "none", borderBottom: "1px dashed var(--line)",
                  background: "transparent", padding: "2px 0",
                  fontSize: 13, fontFamily: "inherit", color: "var(--ink)", outline: "none",
                }}
              />
            </label>

            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Version
              </span>
              <input
                value={project.version ?? ""}
                placeholder="e.g. 1.0.0"
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, version: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("version", e.target.value)}
                style={{
                  border: "none", borderBottom: "1px dashed var(--line)",
                  background: "transparent", padding: "2px 0",
                  fontSize: 13, fontFamily: "inherit", color: "var(--ink)", outline: "none",
                }}
              />
            </label>

            <KeyVal label="Target column" value={data.target_column ?? "—"} />
            <KeyVal label="Response units" value={
              <input
                value={project.response_units ?? ""}
                placeholder="e.g. °C, mg/L, dB"
                onChange={(e) => setConfig((c) => c ? { ...c, project: { ...c.project, response_units: e.target.value } } : c)}
                onBlur={(e) => saveProjectField("response_units", e.target.value)}
                style={{
                  border: "none", borderBottom: "1px dashed var(--line)",
                  background: "transparent", padding: "2px 0", width: "100%",
                  fontSize: 12, fontFamily: "inherit", color: "var(--ink)", outline: "none",
                }}
              />
            } />
          </div>
        </Card>

        {/* ---- Coordinate System (read-only summary) ---- */}
        <Card title="Coordinate system" subtitle="edit on CRS page">
          <KeyVal label="Input EPSG" value={crs.input ? `EPSG:${crs.input}` : "—"} />
          <KeyVal label="Projected EPSG" value={crs.projected ? `EPSG:${crs.projected}` : "—"} />
          <KeyVal label="Source CRS" value={summary?.crs ?? "—"} />
        </Card>

        {/* ---- Dataset snapshot ---- */}
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
            <KeyVal label="Predictors" value={(config?.predictors?.length ?? 0).toString()} />
          </div>
        </Card>

        {/* ---- Pipeline history ---- */}
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
                  <span style={{ color: "var(--ink-2)" }}>{r.stage}</span>
                  <span className="mono" style={{ color: r.tint, fontSize: 10 }}>{r.status}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* ---- Templates (collapsible) ---- */}
      {templatesOpen && (
        <div style={{ marginTop: 14 }}>
          <Card title="Switch / re-init template" subtitle={`${templates.length || 12} domains available`}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 6 }}>
              {(templates.length > 0
                ? templates.map((t) => ({
                    key: t.name,
                    label: t.name,
                    color: TEMPLATE_COLORS[t.name] ?? "var(--muted)",
                  }))
                : Object.entries(TEMPLATE_COLORS).map(([k, c]) => ({
                    key: k,
                    label: k.replace(/_/g, " "),
                    color: c,
                  }))
              ).map((t) => (
                <button
                  key={t.key}
                  onClick={() => {
                    const info = templates.find((tt) => tt.name === t.key);
                    if (info) handleTemplateClick(info);
                  }}
                  style={{
                    textAlign: "left",
                    border: "1px solid var(--line)",
                    background: t.key === domain ? "#fff8ef" : "#fff",
                    borderColor: t.key === domain ? "var(--amber)" : "var(--line)",
                    borderRadius: 5,
                    padding: "7px 9px",
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  <div
                    className="mono"
                    style={{
                      fontSize: 9, color: t.color,
                      letterSpacing: "0.08em", textTransform: "uppercase",
                    }}
                  >
                    {t.key}
                  </div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--ink-2)", marginTop: 1 }}>
                    {t.label}
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ---- Raw config preview ---- */}
      <div style={{ marginTop: 14 }}>
        <button
          onClick={() => setYamlOpen(!yamlOpen)}
          className="mono"
          style={{
            fontSize: 10, color: "var(--muted)",
            textTransform: "uppercase", letterSpacing: "0.1em",
            background: "none", border: "none", cursor: "pointer",
            fontFamily: "inherit", padding: "4px 0",
          }}
        >
          project.yml · raw preview {yamlOpen ? "▲" : "▼"}
        </button>
        {yamlOpen && (
          <Card>
            <pre
              className="mono"
              style={{ fontSize: 10.5, margin: 0, lineHeight: 1.6, color: "var(--ink-2)", whiteSpace: "pre-wrap" }}
            >
              {yamlPreview}
            </pre>
          </Card>
        )}
      </div>
    </div>
  );
}
