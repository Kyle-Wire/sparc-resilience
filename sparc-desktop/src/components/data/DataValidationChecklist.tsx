import { useEffect, useState } from "react";
import { validateData } from "@/lib/api";
import type { ValidationReport, ValidationItem } from "@/lib/types";

const SEVERITY_ICONS: Record<string, string> = {
  critical: "✕",
  warning: "⚠",
  info: "✓",
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "text-red-600 bg-red-50 border-red-200",
  warning: "text-amber-600 bg-amber-50 border-amber-200",
  info: "text-green-600 bg-green-50 border-green-200",
};

const SEVERITY_BADGE: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  warning: "bg-amber-100 text-amber-700",
  info: "bg-green-100 text-green-700",
};

const CATEGORY_LABELS: Record<string, string> = {
  missing: "Missing Values",
  crs: "Coordinate System",
  duplicate: "Duplicates",
  distribution: "Data Distribution",
  type: "Column Types",
  column: "Required Columns",
};

interface Props {
  /** Called with true when all critical issues are resolved or acknowledged. */
  onReady?: (ready: boolean) => void;
  /** If true, run validation immediately on mount. */
  autoRun?: boolean;
}

export default function DataValidationChecklist({ onReady, autoRun = false }: Props) {
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const runValidation = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await validateData();
      setReport(result);
    } catch (e: any) {
      setError(e.message || "Validation failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (autoRun) runValidation();
  }, [autoRun]);

  // Determine if user can proceed
  useEffect(() => {
    if (!report || !onReady) return;
    const unacknowledgedCritical = report.items.filter(
      (item) => item.severity === "critical" && !acknowledged.has(item.id)
    );
    onReady(unacknowledgedCritical.length === 0);
  }, [report, acknowledged, onReady]);

  const toggleAcknowledge = (id: string) => {
    setAcknowledged((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleCategory = (cat: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  // Group items by category
  const grouped: Record<string, ValidationItem[]> = {};
  if (report) {
    for (const item of report.items) {
      const cat = item.category;
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(item);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Pre-Flight Data Validation</h3>
          <p className="text-xs text-sparc-gray-500">
            Check data quality before processing to prevent pipeline failures.
          </p>
        </div>
        <button
          onClick={runValidation}
          disabled={loading}
          className="rounded bg-black px-3 py-1.5 text-xs font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50"
        >
          {loading ? "Validating…" : report ? "Re-validate" : "Run Validation"}
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </div>
      )}

      {report && (
        <>
          {/* Summary bar */}
          <div
            className={`flex items-center gap-3 rounded-lg border px-4 py-3 ${
              report.passed
                ? "border-green-200 bg-green-50"
                : "border-red-200 bg-red-50"
            }`}
          >
            <span className={`text-lg font-bold ${report.passed ? "text-green-600" : "text-red-600"}`}>
              {report.passed ? "✓" : "✕"}
            </span>
            <div className="flex-1">
              <span className="text-sm font-medium">
                {report.passed ? "All checks passed" : "Issues found — review before processing"}
              </span>
              <div className="mt-0.5 flex gap-3 text-xs">
                {report.n_critical > 0 && (
                  <span className="text-red-600">{report.n_critical} critical</span>
                )}
                {report.n_warning > 0 && (
                  <span className="text-amber-600">{report.n_warning} warnings</span>
                )}
                {report.n_info > 0 && (
                  <span className="text-green-600">{report.n_info} passed</span>
                )}
              </div>
            </div>
          </div>

          {/* Items grouped by category */}
          <div className="space-y-2">
            {Object.entries(grouped).map(([category, items]) => {
              const hasCritical = items.some((i) => i.severity === "critical");
              const isCollapsed = collapsed.has(category);

              return (
                <div key={category} className="rounded border border-sparc-gray-200 overflow-hidden">
                  <button
                    onClick={() => toggleCategory(category)}
                    className="flex w-full items-center justify-between bg-sparc-gray-50 px-3 py-2 text-left text-xs font-medium hover:bg-sparc-gray-100"
                  >
                    <span className="flex items-center gap-2">
                      <span className={isCollapsed ? "" : "rotate-90 inline-block"}>▶</span>
                      {CATEGORY_LABELS[category] || category}
                      <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                        hasCritical ? "bg-red-100 text-red-700" : "bg-sparc-gray-200 text-sparc-gray-600"
                      }`}>
                        {items.length}
                      </span>
                    </span>
                  </button>

                  {!isCollapsed && (
                    <div className="divide-y divide-sparc-gray-100">
                      {items.map((item) => (
                        <div
                          key={item.id}
                          className={`flex items-start gap-3 px-3 py-2 ${
                            acknowledged.has(item.id) ? "opacity-50" : ""
                          }`}
                        >
                          <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                            SEVERITY_BADGE[item.severity]
                          }`}>
                            {SEVERITY_ICONS[item.severity]}
                          </span>

                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-sparc-gray-800">
                              {item.message}
                            </p>
                            {item.detail && (
                              <p className="mt-0.5 text-[11px] text-sparc-gray-500">{item.detail}</p>
                            )}
                          </div>

                          {item.severity === "critical" && (
                            <button
                              onClick={() => toggleAcknowledge(item.id)}
                              className={`shrink-0 rounded border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                                acknowledged.has(item.id)
                                  ? "border-green-300 bg-green-50 text-green-700"
                                  : "border-sparc-gray-300 text-sparc-gray-600 hover:border-amber-300 hover:bg-amber-50"
                              }`}
                            >
                              {acknowledged.has(item.id) ? "Acknowledged" : "Acknowledge"}
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
