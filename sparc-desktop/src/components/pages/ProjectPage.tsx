import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, KeyVal, Tag, Btn } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, listTemplates, initProject } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig, TemplateInfo } from "@/lib/types";

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

export default function ProjectPage({ projectPath, onProjectLoaded }: ProjectPageProps) {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [yamlOpen, setYamlOpen] = useState(false);
  const { notify } = useNotification();

  useEffect(() => {
    if (projectPath) {
      getConfig().then(setConfig).catch(() => {});
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

  const domain = config?.project?.domain ?? "";
  const name = config?.project?.name ?? "No project loaded";
  const target = config?.data?.target_column ?? "—";
  const inputEpsg = config?.crs?.input ?? "";
  const projEpsg = config?.crs?.projected ?? "";
  const predictors = config?.predictors ?? [];

  const handleFieldChange = useCallback(
    async (key: string, value: string) => {
      try {
        let patch: Partial<ProjectConfig> = {};
        if (key === "name") patch = { project: { ...config?.project, name: value } };
        else if (key === "domain") patch = { project: { ...config?.project, domain: value } };
        await saveConfig(patch);
        const updated = await getConfig();
        setConfig(updated);
        notify("success", "Project updated");
      } catch (e) {
        notify("error", e instanceof Error ? e.message : "Failed to save");
      }
    },
    [config, notify],
  );

  // Build YAML preview string
  const yamlPreview = config
    ? `project:
  name: ${config.project?.name ?? ""}
  domain: ${config.project?.domain ?? ""}
  version: ${config.project?.version ?? "2.1"}
data:
  path: ${config.data?.file_path ?? ""}
  target_column: ${target}
crs:
  input_epsg: ${inputEpsg}
  projected_epsg: ${projEpsg}
predictors:
${predictors.map((p: string) => `  - ${p}`).join("\n")}
flags:
  use_gwen: ${config.flags?.use_gwen ?? true}
  use_laplacian: ${config.flags?.use_laplacian ?? true}`
    : "# No project loaded";

  return (
    <div>
      <SectionHeader
        kicker="01 · setup"
        label="Project"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn onClick={handleOpenYml}>Open project.yml</Btn>
            <Btn primary onClick={() => templates.length > 0 && handleTemplateClick(templates[0])}>
              New from template
            </Btn>
          </div>
        }
      />
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        <Card title="Active project" subtitle={`${domain} · ${name}`}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <KeyVal
              label="Name"
              value={
                <span
                  contentEditable
                  suppressContentEditableWarning
                  onBlur={(e) => handleFieldChange("name", e.currentTarget.textContent ?? "")}
                  style={{ outline: "none", borderBottom: "1px dashed var(--line)", cursor: "text" }}
                >
                  {name}
                </span>
              }
            />
            <KeyVal
              label="Domain"
              value={<Tag color="var(--crimson)">{domain || "—"}</Tag>}
            />
            <KeyVal label="Target" value={`${target}`} />
            <KeyVal label="Data file" value={config?.data?.file_path ?? "—"} />
            <KeyVal label="CRS (in / proj)" value={`EPSG:${inputEpsg} → ${projEpsg}`} />
            <KeyVal label="Target" value={config?.data?.target_column ?? "—"} />
            <KeyVal label="Random seed" value={String(config?.pipeline?.random_seed ?? 42)} />
            <KeyVal
              label="Pipeline"
              value={
                <Tag color="var(--purple)">
                  fast_mode:{String(config?.pipeline?.fast_mode ?? false)}
                </Tag>
              }
            />
          </div>

          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <button
              onClick={() => setYamlOpen(!yamlOpen)}
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: 6,
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "inherit",
                padding: 0,
              }}
            >
              project.yml · preview {yamlOpen ? "▲" : "▼"}
            </button>
            {yamlOpen && (
              <pre
                className="mono"
                style={{ fontSize: 10.5, margin: 0, lineHeight: 1.6, color: "var(--ink-2)" }}
              >
                {yamlPreview}
              </pre>
            )}
          </div>
        </Card>

        <Card title="Templates" subtitle={`${templates.length || 12} domains available`}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
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
                    fontSize: 9,
                    color: t.color,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
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
    </div>
  );
}
