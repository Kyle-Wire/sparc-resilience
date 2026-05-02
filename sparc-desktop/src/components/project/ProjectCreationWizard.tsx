import { useState, useEffect, useMemo } from "react";
import { Card, Btn, Tag } from "@/components/ui/DesignSystem";
import { listTemplates, createProject } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { getWorkspaceDir, setWorkspaceDir, joinPath } from "@/lib/workspacePrefs";
import type { TemplateInfo } from "@/lib/types";

/**
 * Project Creation Wizard — Phase 1
 *
 * Three honest steps:
 *   1. Identity   — name, description, domain template, author
 *   2. Spatial    — input CRS, projected CRS (with sensible presets)
 *   3. Confirm    — review + choose output directory + create
 *
 * Variables (target / predictors / object ID / coords) are *not* in this
 * wizard because they require data to exist. They are captured on the
 * Data page once the user uploads their CSV/Parquet/GeoPackage.
 *
 * Cross-cutting rule: nothing hardcoded, every default visible and
 * editable, no leaked template params.
 */

interface ProjectCreationWizardProps {
  templates: TemplateInfo[];
  onCreated: (path: string, meta: { name: string; template: string }) => void | Promise<void>;
  onCancel: () => void;
}

type Step = 0 | 1 | 2;

interface WizardState {
  // Identity
  name: string;
  description: string;
  template: string;
  author: string;
  responseUnits: string;
  // Spatial
  inputEpsg: string;
  projectedEpsg: string;
  // Output
  outputDir: string;
}

const COMMON_CRS = [
  { code: "EPSG:4326", label: "WGS 84 (lat/lon)", note: "Most CSVs with lat/lon" },
  { code: "EPSG:3857", label: "Web Mercator", note: "Web maps, deck.gl, MapLibre" },
  { code: "EPSG:5070", label: "NAD83 / Conus Albers", note: "Continental US analysis" },
  { code: "EPSG:32617", label: "UTM 17N (NAD83)", note: "Eastern US (Providence, NYC)" },
  { code: "EPSG:32618", label: "UTM 18N (NAD83)", note: "US East Coast" },
  { code: "EPSG:3035", label: "ETRS89 / LAEA Europe", note: "Pan-European analysis" },
];

const TEMPLATE_BLURBS: Record<string, string> = {
  blank: "Empty scaffold — bring your own variables and physics.",
  uhi: "Urban Heat Island — surface temperature, canopy, impervious cover.",
  coastal: "Coastal flooding — surge, sea-level, elevation, infrastructure.",
  drought: "Drought severity — precipitation, soil moisture, vegetation.",
  air_quality: "Air pollution — PM2.5, NO₂, traffic, meteorology.",
  stormwater: "Stormwater runoff — imperviousness, slope, rainfall intensity.",
  geotechnical: "Geotechnical hazard — soil class, slope, groundwater.",
  groundwater: "Groundwater levels — recharge, pumping, geology.",
  noise: "Urban noise — traffic, land use, building height.",
  wildfire: "Wildfire risk — fuel, weather, topography.",
  water_quality: "Water quality — nutrients, land use, hydrology.",
  seismic: "Seismic risk — site class, shaking intensity, structure age.",
  forcesmip: "ForceSMIP — climate forcing fingerprint detection.",
};

export default function ProjectCreationWizard({
  templates,
  onCreated,
  onCancel,
}: ProjectCreationWizardProps) {
  const { notify } = useNotification();
  const [step, setStep] = useState<Step>(0);
  const [creating, setCreating] = useState(false);
  const [discoveredTemplates, setDiscoveredTemplates] = useState(templates);

  useEffect(() => {
    if (templates.length === 0) {
      listTemplates()
        .then((r) => setDiscoveredTemplates(r.templates))
        .catch(() => {});
    } else {
      setDiscoveredTemplates(templates);
    }
  }, [templates]);

  const [s, setS] = useState<WizardState>({
    name: "",
    description: "",
    template: "blank",
    author: "",
    responseUnits: "",
    inputEpsg: "EPSG:4326",
    projectedEpsg: "EPSG:3857",
    outputDir: "",
  });

  const set = <K extends keyof WizardState>(k: K, v: WizardState[K]) =>
    setS((prev) => ({ ...prev, [k]: v }));

  // Default the output dir from the workspace pref + slugged project name.
  useEffect(() => {
    if (s.outputDir || !s.name) return;
    const safe = s.name
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "");
    if (!safe) return;
    const ws = getWorkspaceDir();
    set("outputDir", ws ? joinPath(ws, safe) : safe);
  }, [s.name]); // eslint-disable-line react-hooks/exhaustive-deps

  const stepValid = useMemo(() => {
    if (step === 0) return s.name.trim().length > 0 && s.template.length > 0;
    if (step === 1) return s.inputEpsg.length > 0 && s.projectedEpsg.length > 0;
    if (step === 2) return s.outputDir.trim().length > 0;
    return false;
  }, [step, s]);

  const handlePickOutputDir = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({ directory: true, multiple: false });
      if (typeof picked === "string") set("outputDir", picked);
    } catch {
      // Browser fallback — let the user type it
    }
  };

  const handleCreate = async () => {
    setCreating(true);
    try {
      const res = await createProject({
        template: s.template,
        output: s.outputDir,
        identity: {
          name: s.name,
          description: s.description,
          author: s.author,
          response_units: s.responseUnits,
        },
        crs: {
          input: s.inputEpsg,
          projected: s.projectedEpsg,
        },
      });
      await onCreated(res.project_yml, { name: s.name, template: s.template });
      notify("success", `Project "${s.name}" created`);
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const STEPS = ["Identity", "Spatial", "Create"] as const;

  return (
    <Card
      title="New project"
      subtitle="Three steps to a clean project. No hidden defaults."
      actions={
        <button
          onClick={onCancel}
          style={{
            border: "none",
            background: "none",
            color: "var(--muted)",
            fontSize: 11,
            cursor: "pointer",
            fontFamily: "inherit",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          ✕ cancel
        </button>
      }
    >
      {/* Stepper */}
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 18,
          paddingBottom: 14,
          borderBottom: "1px dashed var(--line)",
        }}
      >
        {STEPS.map((label, i) => {
          const active = i === step;
          const done = i < step;
          return (
            <div
              key={label}
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                borderRadius: 5,
                background: active ? "#fff8ef" : done ? "#f5f1e8" : "transparent",
                border: active ? "1px solid var(--amber)" : "1px solid transparent",
              }}
            >
              <div
                className="mono"
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  background: active ? "var(--amber)" : done ? "var(--ink)" : "var(--line)",
                  color: active || done ? "#fff" : "var(--muted)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 11,
                  fontWeight: 700,
                }}
              >
                {done ? "✓" : i + 1}
              </div>
              <div style={{ fontSize: 12, fontWeight: 600, color: active ? "var(--ink)" : "var(--muted)" }}>
                {label}
              </div>
            </div>
          );
        })}
      </div>

      {step === 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Field label="Project name" required>
              <TextInput value={s.name} onChange={(v) => set("name", v)} placeholder="Providence UHI 2024" autoFocus />
            </Field>
            <Field label="Description">
              <TextArea
                value={s.description}
                onChange={(v) => set("description", v)}
                placeholder="One sentence on what you're studying and why."
                rows={3}
              />
            </Field>
            <Field label="Author">
              <TextInput value={s.author} onChange={(v) => set("author", v)} placeholder="Your name" />
            </Field>
            <Field label="Response units" hint="Units of your target variable, e.g. °F, mg/L, dB">
              <TextInput value={s.responseUnits} onChange={(v) => set("responseUnits", v)} placeholder="°F" />
            </Field>
          </div>

          <div>
            <FieldLabel>Domain template</FieldLabel>
            <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 8 }}>
              Templates seed sensible physics priors, monotonic constraints, and a starting DAG.
              Everything seeded is visible and editable — no hidden parameters.
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 6,
                maxHeight: 320,
                overflowY: "auto",
                paddingRight: 4,
              }}
            >
              {discoveredTemplates.map((t) => {
                const active = s.template === t.name;
                return (
                  <button
                    key={t.name}
                    onClick={() => set("template", t.name)}
                    style={{
                      textAlign: "left",
                      border: `1px solid ${active ? "var(--amber)" : "var(--line)"}`,
                      background: active ? "#fff8ef" : "#fff",
                      borderRadius: 5,
                      padding: "8px 10px",
                      cursor: "pointer",
                      fontFamily: "inherit",
                    }}
                  >
                    <div
                      className="mono"
                      style={{
                        fontSize: 9,
                        color: active ? "var(--amber)" : "var(--muted)",
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                      }}
                    >
                      {t.name}
                    </div>
                    <div style={{ fontSize: 10.5, color: "var(--ink-2)", marginTop: 3, lineHeight: 1.4 }}>
                      {TEMPLATE_BLURBS[t.name] ?? "Custom template."}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {step === 1 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.55 }}>
            Pick the CRS your input data is in (usually <code>EPSG:4326</code> for lat/lon CSVs)
            and the projected CRS to use for distance-based spatial analysis. You can change these
            later on the CRS page.
          </div>

          <CrsPicker
            label="Input CRS"
            hint="Coordinate system of your raw data file"
            value={s.inputEpsg}
            onChange={(v) => set("inputEpsg", v)}
          />
          <CrsPicker
            label="Projected CRS"
            hint="Used for distances, neighborhoods, fishnet cells. Pick a meter-based CRS local to your study area."
            value={s.projectedEpsg}
            onChange={(v) => set("projectedEpsg", v)}
          />
        </div>
      )}

      {step === 2 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.55 }}>
            Review your choices. The project will be scaffolded into the output directory below.
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              padding: 14,
              border: "1px dashed var(--line)",
              borderRadius: 6,
              background: "#fdfbf7",
            }}
          >
            <Review label="Name" value={s.name || "—"} />
            <Review label="Template" value={<Tag color="var(--amber)">{s.template}</Tag>} />
            <Review label="Author" value={s.author || "—"} />
            <Review label="Response units" value={s.responseUnits || "—"} />
            <Review label="Input CRS" value={s.inputEpsg} />
            <Review label="Projected CRS" value={s.projectedEpsg} />
            {s.description && (
              <div style={{ gridColumn: "1 / -1" }}>
                <Review label="Description" value={s.description} />
              </div>
            )}
          </div>

          <Field label="Output directory" required hint="Where the project folder will be created">
            <div style={{ display: "flex", gap: 6 }}>
              <TextInput
                value={s.outputDir}
                onChange={(v) => set("outputDir", v)}
                placeholder="C:\path\to\my_project"
              />
              <Btn small onClick={handlePickOutputDir}>
                Browse…
              </Btn>
            </div>
          </Field>
        </div>
      )}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 22,
          paddingTop: 14,
          borderTop: "1px dashed var(--line)",
        }}
      >
        <Btn onClick={() => (step === 0 ? onCancel() : setStep((step - 1) as Step))}>
          {step === 0 ? "Cancel" : "← Back"}
        </Btn>

        {step < 2 ? (
          <Btn primary disabled={!stepValid} onClick={() => setStep((step + 1) as Step)}>
            Next →
          </Btn>
        ) : (
          <Btn primary disabled={!stepValid || creating} onClick={handleCreate}>
            {creating ? "Creating…" : "Create project"}
          </Btn>
        )}
      </div>
    </Card>
  );
}

/* ---------- field primitives ---------- */

function FieldLabel({ children, required }: { children: React.ReactNode; required?: boolean }) {
  return (
    <div
      className="mono"
      style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 5 }}
    >
      {children}
      {required && <span style={{ color: "var(--crimson)", marginLeft: 4 }}>*</span>}
    </div>
  );
}

function Field({
  label,
  required,
  hint,
  children,
}: {
  label: string;
  required?: boolean;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <FieldLabel required={required}>{label}</FieldLabel>
      {children}
      {hint && (
        <div style={{ fontSize: 10.5, color: "var(--muted)", marginTop: 4, lineHeight: 1.4 }}>{hint}</div>
      )}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  autoFocus,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      autoFocus={autoFocus}
      style={{
        width: "100%",
        border: "1px solid var(--line)",
        borderRadius: 5,
        padding: "7px 10px",
        fontSize: 12.5,
        fontFamily: "inherit",
        background: "#fff",
        color: "var(--ink)",
      }}
    />
  );
}

function TextArea({
  value,
  onChange,
  placeholder,
  rows = 3,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
}) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      rows={rows}
      style={{
        width: "100%",
        border: "1px solid var(--line)",
        borderRadius: 5,
        padding: "7px 10px",
        fontSize: 12.5,
        fontFamily: "inherit",
        resize: "vertical",
        background: "#fff",
        color: "var(--ink)",
      }}
    />
  );
}

function Review({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div
        className="mono"
        style={{ fontSize: 9, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 3 }}
      >
        {label}
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink)" }}>{value}</div>
    </div>
  );
}

function CrsPicker({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Field label={label} required hint={hint}>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <TextInput value={value} onChange={onChange} placeholder="EPSG:4326" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 5 }}>
          {COMMON_CRS.map((c) => {
            const active = value === c.code;
            return (
              <button
                key={c.code}
                onClick={() => onChange(c.code)}
                style={{
                  textAlign: "left",
                  border: `1px solid ${active ? "var(--amber)" : "var(--line)"}`,
                  background: active ? "#fff8ef" : "#fff",
                  borderRadius: 4,
                  padding: "5px 7px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                <div className="mono" style={{ fontSize: 9, color: active ? "var(--amber)" : "var(--ink-2)", fontWeight: 700 }}>
                  {c.code}
                </div>
                <div style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 2 }}>{c.label}</div>
              </button>
            );
          })}
        </div>
      </div>
    </Field>
  );
}
