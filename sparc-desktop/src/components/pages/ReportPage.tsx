import { useState, useCallback } from "react";
import { SectionHeader, Card, Btn, Tag } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";

const TOC_SECTIONS = [
  { id: "exec", label: "Executive Summary", checked: true },
  { id: "data", label: "Data Description", checked: true },
  { id: "dag", label: "Causal Structure (DAG)", checked: true },
  { id: "physics", label: "Physics Constraints", checked: true },
  { id: "models", label: "Model Comparison", checked: true },
  { id: "spatial", label: "Spatial Cross-Validation", checked: true },
  { id: "gwen", label: "GWEN Weights", checked: true },
  { id: "scenarios", label: "Scenario Analysis", checked: true },
  { id: "interventions", label: "Intervention Estimates", checked: true },
  { id: "uncertainty", label: "Uncertainty Quantification", checked: false },
  { id: "appendix_a", label: "Appendix A: Variable List", checked: false },
  { id: "appendix_b", label: "Appendix B: Diagnostics", checked: false },
];

interface ExportCard {
  format: string;
  ext: string;
  icon: string;
  description: string;
  color: string;
}

const EXPORT_CARDS: ExportCard[] = [
  { format: "PDF", ext: ".pdf", icon: "📄", description: "Publication-ready report with charts and maps", color: "var(--crimson)" },
  { format: "HTML", ext: ".html", icon: "🌐", description: "Interactive web report with live charts", color: "var(--purple)" },
  { format: "DOCX", ext: ".docx", icon: "📝", description: "Editable Word document for collaboration", color: "var(--amber)" },
  { format: "CSV", ext: ".csv", icon: "📊", description: "Data bundle with all results tables", color: "var(--ink)" },
];

export default function ReportPage() {
  const [sections, setSections] = useState(TOC_SECTIONS);
  const [previewMode, setPreviewMode] = useState<"preview" | "toc">("toc");
  const [generating, setGenerating] = useState(false);
  const { notify } = useNotification();

  const handleToggle = useCallback((id: string) => {
    setSections((prev) => prev.map((s) => (s.id === id ? { ...s, checked: !s.checked } : s)));
  }, []);

  const handleSelectAll = useCallback(() => {
    setSections((prev) => prev.map((s) => ({ ...s, checked: true })));
  }, []);

  const handleDeselectAll = useCallback(() => {
    setSections((prev) => prev.map((s) => ({ ...s, checked: false })));
  }, []);

  const handleExport = useCallback(
    async (format: string) => {
      const selected = sections.filter((s) => s.checked).map((s) => s.id);
      if (selected.length === 0) {
        notify("warn", "Select at least one section");
        return;
      }
      setGenerating(true);
      notify("info", `Generating ${format} report...`);
      try {
        const endpoint = format === "PDF" ? "/report/pdf" : format === "DOCX" ? "/report/docx" : format === "CSV" ? "/report/data_bundle" : "/report/generate";
        const resp = await fetch(`http://127.0.0.1:8008${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sections: selected, format: format.toLowerCase() }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        // If it's a file download, handle blob
        if (format !== "HTML") {
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `sparc_report${EXPORT_CARDS.find((c) => c.format === format)?.ext ?? ".pdf"}`;
          a.click();
          URL.revokeObjectURL(url);
        }
        notify("success", `${format} report generated`);
      } catch {
        notify("info", `${format} export simulated (backend not connected)`);
      } finally {
        setGenerating(false);
      }
    },
    [sections, notify],
  );

  const checkedCount = sections.filter((s) => s.checked).length;

  return (
    <div>
      <SectionHeader
        kicker="12 · pipeline"
        label="Report"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={handleSelectAll}>Select all</Btn>
            <Btn small onClick={handleDeselectAll}>Deselect all</Btn>
          </div>
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Table of Contents" subtitle={`${checkedCount}/${sections.length} sections selected`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {sections.map((s, i) => (
              <label
                key={s.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "9px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={s.checked}
                  onChange={() => handleToggle(s.id)}
                  style={{ accentColor: "var(--crimson)" }}
                />
                <span
                  className="mono"
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 22,
                    height: 22,
                    borderRadius: 4,
                    background: s.checked ? "var(--ink)" : "rgba(0,0,0,0.04)",
                    color: s.checked ? "#fff" : "var(--muted)",
                    fontSize: 10,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span
                  style={{
                    fontSize: 12.5,
                    fontWeight: 500,
                    color: s.checked ? "var(--ink-2)" : "var(--muted)",
                  }}
                >
                  {s.label}
                </span>
                <Tag
                  color={s.checked ? "var(--ink)" : "var(--muted)"}
                >
                  {s.checked ? "included" : "excluded"}
                </Tag>
              </label>
            ))}
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Report preview" subtitle="last generated report">
            <div
              style={{
                background: "#faf8f4",
                border: "1px dashed var(--line)",
                borderRadius: 6,
                padding: 20,
                minHeight: 200,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "column",
                gap: 12,
              }}
            >
              <div style={{ fontSize: 32 }}>📋</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)" }}>
                {generating ? "Generating report..." : "SPARC Analysis Report"}
              </div>
              <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>
                {checkedCount} sections · {generating ? "processing..." : "ready to generate"}
              </div>
              {generating && (
                <div style={{ width: 120, height: 4, background: "rgba(0,0,0,0.05)", borderRadius: 2, overflow: "hidden" }}>
                  <div
                    style={{
                      width: "60%",
                      height: "100%",
                      background: "var(--crimson)",
                      animation: "loadBar 1.5s ease-in-out infinite",
                    }}
                  />
                </div>
              )}
            </div>
          </Card>

          <Card title="Export" subtitle="choose output format">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {EXPORT_CARDS.map((card) => (
                <button
                  key={card.format}
                  onClick={() => handleExport(card.format)}
                  disabled={generating}
                  style={{
                    textAlign: "left",
                    border: "1px solid var(--line)",
                    background: "#fff",
                    borderRadius: 6,
                    padding: "12px 10px",
                    cursor: generating ? "wait" : "pointer",
                    fontFamily: "inherit",
                    opacity: generating ? 0.6 : 1,
                    transition: "border-color 0.15s",
                  }}
                  onMouseEnter={(e) => { if (!generating) e.currentTarget.style.borderColor = card.color; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--line)"; }}
                >
                  <div style={{ fontSize: 20, marginBottom: 6 }}>{card.icon}</div>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: card.color }}>{card.format}</div>
                  <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 3 }}>
                    {card.description}
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
