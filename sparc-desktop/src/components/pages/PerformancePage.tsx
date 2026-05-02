import { useCallback, useEffect, useState } from "react";
import { Btn, Card, KeyVal, SectionHeader, Stat, Tag } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import { useHardwareProfile } from "@/hooks/useHardwareProfile";
import {
  getConfig,
  saveConfig,
  updatePreferences,
  validatePerformance,
  type HardwareTier,
  type PerformanceOverrides,
  type PerformancePreset,
  type PerformanceValidation,
  type ValidationIssue,
} from "@/lib/api";

type Scope = "global" | "project" | "both";

const PRESETS: { key: PerformancePreset; label: string; desc: string }[] = [
  { key: "eco", label: "Eco", desc: "Single worker, small batches. Safe to run alongside other apps." },
  { key: "balanced", label: "Balanced", desc: "Auto-detected defaults for this machine. Recommended." },
  { key: "performance", label: "Performance", desc: "Push closer to capacity. Leave the OS some headroom." },
  { key: "max", label: "Max", desc: "Saturate the machine. Close other apps before running." },
  { key: "custom", label: "Custom", desc: "Manage every knob yourself in the Advanced section." },
];

const TIERS: { key: HardwareTier; label: string }[] = [
  { key: "low", label: "Low (<12 GB)" },
  { key: "standard", label: "Standard (12–32 GB)" },
  { key: "high", label: "High (≥32 GB)" },
];

function fieldStyle(invalid: boolean): React.CSSProperties {
  return {
    border: `1px solid ${invalid ? "#c84545" : "var(--line)"}`,
    borderRadius: 4,
    padding: "6px 8px",
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
    width: 100,
    background: "#fff",
  };
}

function HelperText({ children, error }: { children: React.ReactNode; error?: boolean }) {
  return (
    <div
      className="mono"
      style={{
        fontSize: 10,
        color: error ? "#c84545" : "var(--muted)",
        marginTop: 4,
      }}
    >
      {children}
    </div>
  );
}

export default function PerformancePage() {
  const { notify } = useNotification();
  const { data, loading, error, refresh } = useHardwareProfile(true);

  const [scope, setScope] = useState<Scope>("global");
  const [draft, setDraft] = useState<PerformanceOverrides>({});
  const [validation, setValidation] = useState<PerformanceValidation | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  // Seed the draft from the active overrides whenever the server data changes.
  useEffect(() => {
    if (!data) return;
    const seed = scope === "project" ? data.project_overrides : data.global_prefs;
    setDraft({ preset: seed.preset ?? "balanced", ...seed });
  }, [data, scope]);

  const detected = data?.detected;
  const effective = data?.effective;
  const safeCaps = data?.safe_caps;

  const numericIssue = useCallback(
    (field: string): ValidationIssue | undefined => {
      if (!validation) return undefined;
      return (
        validation.errors.find((e) => e.field === field) ??
        validation.warnings.find((w) => w.field === field)
      );
    },
    [validation],
  );

  const updateField = useCallback(<K extends keyof PerformanceOverrides>(key: K, value: PerformanceOverrides[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
    setValidation(null);
  }, []);

  const handlePreset = useCallback((preset: PerformancePreset) => {
    setDraft((prev) => ({ ...prev, preset }));
    if (preset !== "custom") setShowAdvanced(false);
    setValidation(null);
  }, []);

  const runValidation = useCallback(async (): Promise<PerformanceValidation> => {
    const result = await validatePerformance(draft);
    setValidation(result);
    return result;
  }, [draft]);

  const persist = useCallback(async () => {
    setSaving(true);
    try {
      const tasks: Promise<unknown>[] = [];
      if (scope === "global" || scope === "both") {
        tasks.push(updatePreferences({ performance: draft }));
      }
      if (scope === "project" || scope === "both") {
        const cfg = await getConfig();
        const merged = { ...cfg, performance: draft };
        tasks.push(saveConfig(merged));
      }
      await Promise.all(tasks);
      await refresh();
      notify("success", "Performance settings saved. They apply on the next pipeline run.");
    } catch (err) {
      notify("error", `Save failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSaving(false);
      setConfirmOpen(false);
    }
  }, [draft, scope, notify, refresh]);

  const handleSave = useCallback(async () => {
    const result = await runValidation();
    if (result.errors.length > 0) {
      notify("error", `Cannot save: ${result.errors[0].message}`);
      return;
    }
    if (result.warnings.length > 0) {
      setConfirmOpen(true);
      return;
    }
    await persist();
  }, [runValidation, persist, notify]);

  const presetIsCustom = draft.preset === "custom";

  return (
    <div>
      <SectionHeader kicker="settings" label="Performance" />

      {error && (
        <Card title="Hardware profile unavailable" subtitle="see hint below">
          <div style={{ fontSize: 12, color: "#c84545", marginBottom: 8 }}>{error}</div>
          {error.includes("404") && (
            <div className="mono" style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.6 }}>
              The <code>/api/hardware</code> endpoint is missing. Your bundled sidecar may be out
              of date — rebuild it from the repo root with <code>python build-sidecar.py
              --target-dir sparc-desktop/src-tauri/binaries</code>, then relaunch the app.
            </div>
          )}
        </Card>
      )}

      {/* System overview */}
      {detected && safeCaps && effective && (
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
          <Card title="This machine" subtitle="auto-detected resources">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 10,
                marginBottom: 12,
              }}
            >
              <Stat label="Tier" value={detected.tier.toUpperCase()} />
              <Stat label="Total RAM" value={`${detected.total_ram_gb.toFixed(1)} GB`} sub={`${detected.available_ram_gb.toFixed(1)} GB free`} />
              <Stat label="CPU cores" value={detected.cpu_count} />
              <Stat label="GPU" value={detected.force_cpu ? "off" : "available"} />
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
              <Tag>effective</Tag>
              <KeyVal label="Workers" value={effective.max_workers} />
              <KeyVal label="Outer × inner" value={`${effective.outer_jobs} × ${effective.inner_jobs}`} />
              <KeyVal label="Batch" value={effective.batch_size} />
              <KeyVal label="NUTS thin" value={effective.nuts_thin} />
              <KeyVal label="Mem cap" value={`${effective.memory_limit_gb.toFixed(1)} GB`} />
              <KeyVal label="Source" value={effective.source} />
            </div>
          </Card>

          <Card title="Safe caps" subtitle="warnings trigger above these">
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <KeyVal label="Max workers" value={safeCaps.max_workers} />
              <KeyVal label="Batch size" value={safeCaps.batch_size} />
              <KeyVal label="Memory" value={`${safeCaps.memory_limit_gb.toFixed(1)} GB`} />
              <KeyVal label="NUTS thin" value={safeCaps.nuts_thin} />
            </div>
          </Card>
        </div>
      )}

      {/* Scope */}
      <div style={{ marginTop: 14 }}>
        <Card title="Scope" subtitle="where to save these settings">
          <div style={{ display: "flex", gap: 8 }}>
            {(["global", "project", "both"] as Scope[]).map((s) => (
              <button
                key={s}
                onClick={() => setScope(s)}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  fontWeight: 600,
                  borderRadius: 5,
                  border: `1px solid ${scope === s ? "var(--ink)" : "var(--line)"}`,
                  background: scope === s ? "var(--ink)" : "#fff",
                  color: scope === s ? "#fff" : "var(--ink-2)",
                  cursor: "pointer",
                }}
              >
                {s === "global" ? "Global default" : s === "project" ? "This project" : "Both"}
              </button>
            ))}
          </div>
          <HelperText>
            {scope === "global"
              ? "Saved to ~/.sparc/preferences.json. Applies to every project on this machine."
              : scope === "project"
                ? "Saved into project.yml under `performance`. Travels with the project for reproducibility."
                : "Saved as both the machine-wide default and a per-project override."}
          </HelperText>
        </Card>
      </div>

      {/* Preset */}
      <div style={{ marginTop: 14 }}>
        <Card title="Preset" subtitle="how aggressively SPARC uses your machine">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
            {PRESETS.map((p) => {
              const active = (draft.preset ?? "balanced") === p.key;
              return (
                <button
                  key={p.key}
                  onClick={() => handlePreset(p.key)}
                  style={{
                    padding: "10px 8px",
                    border: `2px solid ${active ? "var(--ink)" : "var(--line)"}`,
                    borderRadius: 6,
                    background: active ? "var(--ink)" : "#fff",
                    color: active ? "#fff" : "var(--ink-2)",
                    cursor: "pointer",
                    textAlign: "left",
                    fontFamily: "inherit",
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700 }}>{p.label}</div>
                  <div style={{ fontSize: 10, opacity: 0.85, marginTop: 4, lineHeight: 1.3 }}>{p.desc}</div>
                </button>
              );
            })}
          </div>
          <div style={{ marginTop: 12 }}>
            <Btn small onClick={() => setShowAdvanced((v) => !v)}>
              {showAdvanced ? "Hide advanced" : presetIsCustom ? "Edit advanced fields" : "Show advanced (override fields)"}
            </Btn>
          </div>
        </Card>
      </div>

      {/* Advanced */}
      {showAdvanced && safeCaps && (
        <div style={{ marginTop: 14 }}>
          <Card title="Advanced" subtitle="manual overrides — leave blank to inherit the preset">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: 14,
              }}
            >
              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>
                  Hardware tier override
                </div>
                <select
                  value={draft.hardware_tier_override ?? ""}
                  onChange={(e) =>
                    updateField(
                      "hardware_tier_override",
                      (e.target.value || undefined) as HardwareTier | undefined,
                    )
                  }
                  style={fieldStyle(false)}
                >
                  <option value="">auto</option>
                  {TIERS.map((t) => (
                    <option key={t.key} value={t.key}>
                      {t.label}
                    </option>
                  ))}
                </select>
                <HelperText>Skips the RAM-based tier classification.</HelperText>
              </div>

              <NumberField
                label="max_workers"
                cap={safeCaps.max_workers}
                value={draft.max_workers}
                onChange={(v) => updateField("max_workers", v)}
                issue={numericIssue("max_workers")}
              />
              <NumberField
                label="outer_jobs"
                cap={safeCaps.outer_jobs}
                value={draft.outer_jobs}
                onChange={(v) => updateField("outer_jobs", v)}
                issue={numericIssue("outer_jobs")}
              />
              <NumberField
                label="inner_jobs"
                cap={safeCaps.inner_jobs}
                value={draft.inner_jobs}
                onChange={(v) => updateField("inner_jobs", v)}
                issue={numericIssue("inner_jobs")}
              />
              <NumberField
                label="batch_size"
                cap={safeCaps.batch_size}
                value={draft.batch_size}
                onChange={(v) => updateField("batch_size", v)}
                issue={numericIssue("batch_size")}
              />
              <NumberField
                label="nuts_thin"
                cap={safeCaps.nuts_thin}
                value={draft.nuts_thin}
                onChange={(v) => updateField("nuts_thin", v)}
                issue={numericIssue("nuts_thin")}
              />
              <NumberField
                label="memory_limit_gb"
                cap={safeCaps.memory_limit_gb}
                value={draft.memory_limit_gb}
                onChange={(v) => updateField("memory_limit_gb", v)}
                issue={numericIssue("memory_limit_gb")}
                step={0.5}
              />

              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>
                  Toggles
                </div>
                <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12 }}>
                  <input
                    type="checkbox"
                    checked={draft.force_cpu ?? false}
                    onChange={(e) => updateField("force_cpu", e.target.checked)}
                  />
                  Force CPU only (disable GPU)
                </label>
                <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, marginTop: 6 }}>
                  <input
                    type="checkbox"
                    checked={draft.high_memory_mode ?? false}
                    onChange={(e) => updateField("high_memory_mode", e.target.checked)}
                  />
                  High-memory mode
                </label>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end", gap: 8 }}>
        <Btn small onClick={refresh} disabled={loading}>
          Reload from server
        </Btn>
        <Btn primary onClick={handleSave} disabled={saving}>
          {saving ? "Saving…" : "Save settings"}
        </Btn>
      </div>

      {/* Confirmation modal */}
      {confirmOpen && validation && (
        <ConfirmationModal
          warnings={validation.warnings}
          onCancel={() => setConfirmOpen(false)}
          onConfirm={persist}
          loading={saving}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function NumberField({
  label,
  cap,
  value,
  onChange,
  issue,
  step = 1,
}: {
  label: string;
  cap: number;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  issue?: ValidationIssue;
  step?: number;
}) {
  const isError = issue && (issue.cap === undefined);
  const isWarning = issue && issue.cap !== undefined;
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          type="number"
          step={step}
          min={1}
          value={value ?? ""}
          onChange={(e) => {
            const t = e.target.value;
            if (t === "") onChange(undefined);
            else {
              const n = step < 1 ? Number(t) : Number.parseInt(t, 10);
              onChange(Number.isFinite(n) ? n : undefined);
            }
          }}
          placeholder="auto"
          style={fieldStyle(Boolean(isError || isWarning))}
        />
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>
          safe ≤ {cap}
        </span>
      </div>
      {issue && <HelperText error={isError || isWarning}>{issue.message}</HelperText>}
    </div>
  );
}

function ConfirmationModal({
  warnings,
  onCancel,
  onConfirm,
  loading,
}: {
  warnings: ValidationIssue[];
  onCancel: () => void;
  onConfirm: () => void;
  loading: boolean;
}) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={onCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--paper)",
          border: "1px solid var(--line)",
          borderRadius: 8,
          padding: 20,
          maxWidth: 520,
          width: "90%",
          boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 10 }}>
          These settings exceed your hardware's safe capacity
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, marginBottom: 12 }}>
          Pipeline runs with these values may exhaust memory or cause the OS to kill the
          process. SPARC will not stop you — but please review the warnings below.
        </div>
        <ul style={{ paddingLeft: 18, fontSize: 12, lineHeight: 1.6, marginBottom: 16 }}>
          {warnings.map((w, idx) => (
            <li key={idx} style={{ color: "#a05a00" }}>
              {w.message}
            </li>
          ))}
        </ul>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <Btn small onClick={onCancel}>
            Cancel
          </Btn>
          <Btn primary onClick={onConfirm} disabled={loading}>
            {loading ? "Saving…" : "Save anyway"}
          </Btn>
        </div>
      </div>
    </div>
  );
}
