import { useState, useEffect } from "react";
import { Btn, Card, KeyVal, SectionHeader, Tag } from "@/components/ui/DesignSystem";
import { getConfig, listTemplates, dataSummary, initProject, loadProject } from "@/lib/api";
import { pickYaml, pickFolder } from "@/lib/fileDialogs";
import type { ProjectConfig, DataSummary, TemplateInfo } from "@/lib/types";

const DEMO_TEMPLATES: [string, string, string][] = [
  ["uhi", "Urban Heat Island", "var(--crimson)"],
  ["forcesmip", "Climate Forcing", "var(--purple)"],
  ["groundwater", "Hydrogeology", "var(--magenta)"],
  ["air_quality", "Air Quality", "var(--amber)"],
  ["stormwater", "Stormwater", "var(--pink)"],
  ["coastal", "Coastal Eng.", "var(--red)"],
  ["geotechnical", "Geotechnical", "var(--orange)"],
  ["seismic", "Seismic", "var(--gold)"],
  ["noise", "Noise", "var(--muted)"],
  ["wildfire", "Wildfire", "var(--crimson)"],
  ["drought", "Drought", "var(--amber)"],
  ["water_quality", "Water Quality", "var(--purple)"],
];

const DOMAIN_COLORS: Record<string, string> = {
  uhi: "var(--crimson)", forcesmip: "var(--purple)", groundwater: "var(--magenta)",
  air_quality: "var(--amber)", stormwater: "var(--pink)", coastal: "var(--red)",
  geotechnical: "var(--orange)", seismic: "var(--gold)", noise: "var(--muted)",
  wildfire: "var(--crimson)", drought: "var(--amber)", water_quality: "var(--purple)",
};

interface ProjectPageProps {
  projectPath?: string | null;
  onProjectLoaded?: (path: string, meta?: { name?: string; template?: string }) => Promise<void>;
}

export default function ProjectPage({ projectPath, onProjectLoaded }: ProjectPageProps) {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [templates, setTemplates] = useState<TemplateInfo[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getConfig().catch(() => null),
      dataSummary().catch(() => null),
      listTemplates().then((r) => r.templates).catch(() => null),
    ]).then(([cfg, sum, tpl]) => {
      setConfig(cfg);
      setSummary(sum);
      setTemplates(tpl);
      setLoading(false);
    });
  }, [projectPath]);

  const handleOpen = async () => {
    const path = await pickYaml();
    if (!path) return;
    try {
      await loadProject(path);
      await onProjectLoaded?.(path);
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to load project");
    }
  };

  const handleNewTemplate = async (templateName: string) => {
    const folder = await pickFolder();
    if (!folder) return;
    try {
      const res = await initProject(templateName, folder);
      await onProjectLoaded?.(res.project_yml, { template: templateName });
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to create project");
    }
  };

  const domain = config?.project?.domain ?? "uhi";
  const name = config?.project?.name ?? "Brown UHI — 30 m";
  const target = config?.data?.target_column ?? "AAT_z (°F, z-score)";
  const rows = summary?.row_count?.toLocaleString() ?? "54,701";
  const crsIn = config?.crs?.input ?? "EPSG:4326";
  const crsProj = config?.crs?.projected ?? "3438";
  const seed = config?.pipeline?.random_seed ?? 42;
  const fastMode = config?.pipeline?.fast_mode ?? false;

  const yamlPreview = config
    ? JSON.stringify(config, null, 2).slice(0, 400)
    : `project:\n  name: brown_uhi\n  domain: uhi\n  version: 2.1\ndata:\n  path: examples/brown4.csv\n  target_column: AAT_z\ncrs:\n  input_epsg: 4326\n  projected_epsg: 3438\nflags:\n  use_gwen: true\n  use_laplacian: true`;

  const templateList: [string, string, string][] = templates
    ? templates.map((t) => [
        t.name,
        t.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        DOMAIN_COLORS[t.name] ?? "var(--muted)",
      ])
    : DEMO_TEMPLATES;

  return (
    <div>
      <SectionHeader
        kicker="01 · setup"
        label="Project"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn onClick={handleOpen}>Open project.yml</Btn>
            <Btn primary onClick={() => handleNewTemplate(domain)}>New from template</Btn>
          </div>
        }
      />
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        <Card title="Active project" subtitle={loading ? "loading…" : `${domain} · ${name}`}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <KeyVal label="Name" value={name} />
            <KeyVal label="Domain" value={<Tag color={DOMAIN_COLORS[domain] ?? "var(--crimson)"}>{domain.toUpperCase()}</Tag>} />
            <KeyVal label="Target" value={target} />
            <KeyVal label="Observations" value={`${rows} pts`} />
            <KeyVal label="CRS (in / proj)" value={`${crsIn} → ${crsProj}`} />
            <KeyVal label="Resolution" value="≈30 m" />
            <KeyVal label="Random seed" value={String(seed)} />
            <KeyVal label="Pipeline" value={<Tag color="var(--purple)">{`fast_mode:${fastMode}`}</Tag>} />
          </div>
          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <div
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: 6,
              }}
            >
              project.yml · preview
            </div>
            <pre
              className="mono"
              style={{ fontSize: 10.5, margin: 0, lineHeight: 1.6, color: "var(--ink-2)", whiteSpace: "pre-wrap" }}
            >
              {yamlPreview}
            </pre>
          </div>
        </Card>

        <Card title="Templates" subtitle={`${templateList.length} domains available`}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {templateList.map(([k, label, col]) => (
              <button
                key={k}
                onClick={() => handleNewTemplate(k)}
                style={{
                  textAlign: "left",
                  border: "1px solid var(--line)",
                  background: k === domain ? "#fff8ef" : "#fff",
                  borderColor: k === domain ? "var(--amber)" : "var(--line)",
                  borderRadius: 5,
                  padding: "7px 9px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                <div
                  className="mono"
                  style={{ fontSize: 9, color: col, letterSpacing: "0.08em", textTransform: "uppercase" }}
                >
                  {k}
                </div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "var(--ink-2)", marginTop: 1 }}>{label}</div>
              </button>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
