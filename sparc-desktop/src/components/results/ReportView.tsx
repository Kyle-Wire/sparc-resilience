import { useState, useEffect } from "react";
import { getReportData, generatePdfReport, stagePlotUrl } from "@/lib/api";
import type { ReportPayload } from "@/lib/types";

export default function ReportView() {
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [format, setFormat] = useState<"executive" | "technical">("executive");

  useEffect(() => {
    getReportData()
      .then(setReport)
      .catch(() => setError("Could not load report data. Run the pipeline first."))
      .finally(() => setLoading(false));
  }, []);

  const handlePrint = () => window.print();

  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [pdfMessage, setPdfMessage] = useState<string | null>(null);

  const handleGeneratePdf = async () => {
    setPdfGenerating(true);
    setPdfMessage(null);
    try {
      const result = await generatePdfReport();
      if (result.blob) {
        // Download the PDF
        const url = URL.createObjectURL(result.blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "sparc_report.pdf";
        a.click();
        URL.revokeObjectURL(url);
        setPdfMessage("PDF saved");
      } else if (result.htmlFallback) {
        setPdfMessage(`HTML fallback saved: ${result.htmlFallback}`);
      }
    } catch (err) {
      setPdfMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setPdfGenerating(false);
    }
  };

  const handleCopyMarkdown = () => {
    if (!report) return;
    const md = buildMarkdown(report);
    navigator.clipboard.writeText(md);
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Compiling report…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error ?? "No report data available."}</p>
      </div>
    );
  }

  const project = report.project ?? {};
  const dataSummary = report.data_summary ?? {};

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-sparc-gray-200 px-4 py-3 print:hidden">
        <h1 className="text-lg font-bold">Report</h1>
        <div className="flex items-center gap-2">
          {/* Format toggle */}
          <div className="flex rounded border border-sparc-gray-300 overflow-hidden mr-2">
            <button
              onClick={() => setFormat("executive")}
              className={`px-3 py-1.5 text-xs font-medium ${
                format === "executive"
                  ? "bg-sparc-purple text-white"
                  : "bg-white text-sparc-gray-600 hover:bg-sparc-gray-50"
              }`}
            >
              Executive
            </button>
            <button
              onClick={() => setFormat("technical")}
              className={`px-3 py-1.5 text-xs font-medium border-l border-sparc-gray-300 ${
                format === "technical"
                  ? "bg-sparc-purple text-white"
                  : "bg-white text-sparc-gray-600 hover:bg-sparc-gray-50"
              }`}
            >
              Technical
            </button>
          </div>
          <button
            onClick={handleCopyMarkdown}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100"
          >
            Copy Markdown
          </button>
          <button
            onClick={handlePrint}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100"
          >
            Print
          </button>
          <button
            onClick={handleGeneratePdf}
            disabled={pdfGenerating}
            className="rounded bg-sparc-purple px-3 py-1.5 text-xs font-medium text-white hover:bg-sparc-purple/90 disabled:opacity-50"
          >
            {pdfGenerating ? "Generating…" : "Download PDF"}
          </button>
          {pdfMessage && (
            <span className="text-[10px] text-sparc-gray-500">{pdfMessage}</span>
          )}
        </div>
      </div>

      {/* Report content */}
      <div className="flex-1 overflow-auto p-6 print:p-0">
        <div className="mx-auto max-w-4xl space-y-8 print:max-w-none">
          {/* Project Header */}
          <section>
            <h2 className="text-xl font-bold">
              {String(project.name ?? project.title ?? "SPARC Analysis Report")}
            </h2>
            {project.description != null && (
              <p className="mt-1 text-sm text-sparc-gray-600">{String(project.description)}</p>
            )}
            <div className="mt-3 flex flex-wrap gap-4 text-xs text-sparc-gray-500">
              {project.location != null && <span>Location: {String(project.location)}</span>}
              {project.crs != null && <span>CRS: {String(project.crs)}</span>}
              {project.target != null && <span>Target: {String(project.target)}</span>}
            </div>
          </section>

          {/* Data Overview */}
          <section>
            <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">Data Overview</h3>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {Object.entries(dataSummary)
                .filter(([, v]) => typeof v === "number" || typeof v === "string")
                .slice(0, 8)
                .map(([k, v]) => (
                  <div key={k} className="rounded border border-sparc-gray-200 p-3">
                    <p className="text-[10px] text-sparc-gray-500">{k}</p>
                    <p className="text-sm font-bold tabular-nums">{String(v)}</p>
                  </div>
                ))}
            </div>
            {report.predictors && report.predictors.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-medium text-sparc-gray-600">
                  Predictors ({report.predictors.length}):
                </p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {report.predictors.map((p) => (
                    <span key={p} className="rounded bg-sparc-gray-100 px-2 py-0.5 text-[10px] font-mono">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Correlogram summary — technical only */}
          {format === "technical" && report.correlogram && Object.keys(report.correlogram).length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Stage 0 — Correlogram
              </h3>
              <div className="overflow-auto rounded border border-sparc-gray-200">
                <table className="w-full text-left text-xs">
                  <thead className="bg-sparc-gray-50">
                    <tr>
                      <th className="px-3 py-2 font-medium">Variable</th>
                      {Object.keys(Object.values(report.correlogram)[0] ?? {}).map((k) => (
                        <th key={k} className="px-3 py-2 font-medium">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.correlogram).map(([varName, vals]) => (
                      <tr key={varName} className="border-t border-sparc-gray-100">
                        <td className="px-3 py-1.5 font-mono">{varName}</td>
                        {Object.values(vals).map((v, i) => (
                          <td key={i} className="px-3 py-1.5 tabular-nums">
                            {typeof v === "number" ? v.toFixed(4) : String(v ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* GWEN summary — technical only */}
          {format === "technical" && report.gwen && report.gwen.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Stage 1 — GWEN Variable Selection
              </h3>
              <div className="overflow-auto rounded border border-sparc-gray-200">
                <table className="w-full text-left text-xs">
                  <thead className="bg-sparc-gray-50">
                    <tr>
                      {Object.keys(report.gwen[0]).map((k) => (
                        <th key={k} className="px-3 py-2 font-medium">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {report.gwen.slice(0, 30).map((row, i) => (
                      <tr key={i} className="border-t border-sparc-gray-100">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-3 py-1.5 tabular-nums">
                            {typeof v === "number" ? v.toFixed(4) : String(v ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Spatial CV models */}
          {report.spatial_cv_models && report.spatial_cv_models.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Stage 2 — Spatial CV
              </h3>
              <p className="text-xs text-sparc-gray-600">
                Models evaluated: {report.spatial_cv_models.join(", ")}
              </p>
            </section>
          )}

          {/* Causal coefficients */}
          {report.causal_results?.direct_effects && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Stage 3 — Causal Analysis
              </h3>
              <div className="overflow-auto rounded border border-sparc-gray-200">
                <table className="w-full text-left text-xs">
                  <thead className="bg-sparc-gray-50">
                    <tr>
                      <th className="px-3 py-2 font-medium">Treatment</th>
                      <th className="px-3 py-2 font-medium">Coefficient</th>
                      <th className="px-3 py-2 font-medium">Elasticity</th>
                      <th className="px-3 py-2 font-medium">E-value</th>
                      <th className="px-3 py-2 font-medium">CI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.causal_results.direct_effects).map(([t, e]) => (
                      <tr key={t} className="border-t border-sparc-gray-100">
                        <td className="px-3 py-1.5 font-mono">{t}</td>
                        <td className="px-3 py-1.5 tabular-nums">{e.structural_coeff?.toFixed(4) ?? "—"}</td>
                        <td className="px-3 py-1.5 tabular-nums">{e.elasticity?.toFixed(4) ?? "—"}</td>
                        <td className="px-3 py-1.5 tabular-nums">{e.e_value?.toFixed(2) ?? "—"}</td>
                        <td className="px-3 py-1.5 tabular-nums text-sparc-gray-400">
                          {e.bootstrap_ci_lower != null && e.bootstrap_ci_upper != null
                            ? `[${e.bootstrap_ci_lower.toFixed(4)}, ${e.bootstrap_ci_upper.toFixed(4)}]`
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Scenario summary */}
          {report.scenario_summary && report.scenario_summary.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Stage 4 — Scenarios
              </h3>
              <div className="overflow-auto rounded border border-sparc-gray-200">
                <table className="w-full text-left text-xs">
                  <thead className="bg-sparc-gray-50">
                    <tr>
                      {Object.keys(report.scenario_summary[0]).map((k) => (
                        <th key={k} className="px-3 py-2 font-medium">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {report.scenario_summary.map((row, i) => (
                      <tr key={i} className="border-t border-sparc-gray-100">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-3 py-1.5 tabular-nums">
                            {typeof v === "number" ? v.toFixed(4) : String(v ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Stage plots (inline images) — technical only */}
          {format === "technical" && report.plots && Object.keys(report.plots).length > 0 && (
            <section className="print:break-before-page">
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Diagnostic Plots
              </h3>
              <div className="space-y-6">
                {Object.entries(report.plots).map(([stageKey, plotList]) => (
                  <div key={stageKey}>
                    <h4 className="mb-2 text-sm font-semibold">Stage {stageKey}</h4>
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                      {plotList.slice(0, 8).map((p) => (
                        <div key={p.path} className="rounded border border-sparc-gray-200 p-2">
                          <img
                            src={stagePlotUrl(p.stage, p.path)}
                            alt={p.name}
                            className="w-full object-contain"
                          />
                          <p className="mt-1 text-center text-[10px] text-sparc-gray-500">{p.name}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Config details — technical only */}
          {format === "technical" && (report.causal || report.physics) && (
            <section>
              <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
                Configuration
              </h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {report.causal && Object.keys(report.causal).length > 0 && (
                  <div>
                    <h4 className="mb-1 text-xs font-semibold">Causal</h4>
                    <pre className="whitespace-pre-wrap rounded bg-sparc-gray-50 p-3 text-[10px] font-mono text-sparc-gray-700">
                      {JSON.stringify(report.causal, null, 2)}
                    </pre>
                  </div>
                )}
                {report.physics && Object.keys(report.physics).length > 0 && (
                  <div>
                    <h4 className="mb-1 text-xs font-semibold">Physics Priors</h4>
                    <pre className="whitespace-pre-wrap rounded bg-sparc-gray-50 p-3 text-[10px] font-mono text-sparc-gray-700">
                      {JSON.stringify(report.physics, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Limitations — always shown, non-removable */}
          <section className="print:break-before-page">
            <h3 className="mb-3 text-lg font-semibold border-b border-sparc-gray-200 pb-1">
              Limitations and Caveats
            </h3>
            <div className="space-y-3 text-xs text-sparc-gray-700">
              <div className="rounded border border-amber-200 bg-amber-50 p-3">
                <p className="font-semibold text-amber-800 mb-1">Observational Data</p>
                <p className="text-amber-700">
                  All findings are derived from observational data. While causal inference methods
                  (DAG-based structural estimation, backdoor adjustment, and sensitivity analysis)
                  are employed, unmeasured confounding cannot be entirely ruled out. E-values are
                  provided to quantify the strength of unmeasured confounding required to explain
                  away observed effects.
                </p>
              </div>
              <div className="rounded border border-amber-200 bg-amber-50 p-3">
                <p className="font-semibold text-amber-800 mb-1">Spatial Scale</p>
                <p className="text-amber-700">
                  Results reflect relationships at the spatial resolution of the analysis grid.
                  Ecological fallacy may apply when interpreting aggregate spatial patterns at
                  individual or sub-grid scales. Coefficients represent average effects within
                  grid cells and may not transfer to different resolutions.
                </p>
              </div>
              <div className="rounded border border-amber-200 bg-amber-50 p-3">
                <p className="font-semibold text-amber-800 mb-1">Temporal Scope</p>
                <p className="text-amber-700">
                  This analysis represents a cross-sectional snapshot. Seasonal variation,
                  long-term trends, and climate change projections are not captured unless
                  explicitly modeled. Results should not be extrapolated to substantially
                  different time periods without additional validation.
                </p>
              </div>
              <div className="rounded border border-amber-200 bg-amber-50 p-3">
                <p className="font-semibold text-amber-800 mb-1">Transferability</p>
                <p className="text-amber-700">
                  Models and effect estimates are calibrated to the specific study area.
                  Transferring results to other geographic contexts requires domain expert review
                  and ideally local recalibration. The extrapolation guard marks scenario
                  predictions that exceed the observed data envelope.
                </p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

// ── Markdown export helper ──────────────────────────────────────

function buildMarkdown(r: ReportPayload): string {
  const lines: string[] = [];
  const project = r.project ?? {};

  lines.push(`# ${project.name ?? project.title ?? "SPARC Analysis Report"}`);
  if (project.description) lines.push(`\n${project.description}`);
  lines.push("");

  // Data overview
  if (r.data_summary && Object.keys(r.data_summary).length > 0) {
    lines.push("## Data Overview\n");
    for (const [k, v] of Object.entries(r.data_summary)) {
      lines.push(`- **${k}**: ${v}`);
    }
    lines.push("");
  }

  if (r.predictors && r.predictors.length > 0) {
    lines.push(`**Predictors (${r.predictors.length}):** ${r.predictors.join(", ")}\n`);
  }

  // Causal
  if (r.causal_results?.direct_effects) {
    lines.push("## Causal Analysis\n");
    lines.push("| Treatment | Coefficient | Elasticity | E-value |");
    lines.push("|---|---|---|---|");
    for (const [t, e] of Object.entries(r.causal_results.direct_effects)) {
      lines.push(
        `| ${t} | ${e.structural_coeff?.toFixed(4) ?? "—"} | ${e.elasticity?.toFixed(4) ?? "—"} | ${e.e_value?.toFixed(2) ?? "—"} |`,
      );
    }
    lines.push("");
  }

  // Scenarios
  if (r.scenario_summary && r.scenario_summary.length > 0) {
    lines.push("## Scenario Summary\n");
    const keys = Object.keys(r.scenario_summary[0]);
    lines.push(`| ${keys.join(" | ")} |`);
    lines.push(`| ${keys.map(() => "---").join(" | ")} |`);
    for (const row of r.scenario_summary) {
      lines.push(
        `| ${keys.map((k) => { const v = row[k]; return typeof v === "number" ? v.toFixed(4) : String(v ?? ""); }).join(" | ")} |`,
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}
