/**
 * useCommandPaletteItems — derives the full command palette item list from
 * the current data context. Extracted from App.tsx to keep the root thin.
 */
import { useMemo } from "react";
import { PAGES } from "@/components/layout/Sidebar";
import type { PaletteItem } from "@/components/common/CommandPalette";
import type { PromptDataContext } from "@/lib/prompts";
import type { ExplainContextValue } from "@/hooks/ExplainContext.tsx";

export function useCommandPaletteItems(
  dataCtx: PromptDataContext | null,
  explainHostValue: ExplainContextValue,
  navigate: (page: string) => void,
  setChatOpen: (fn: (o: boolean) => boolean) => void,
): PaletteItem[] {
  return useMemo<PaletteItem[]>(() => {
    const out: PaletteItem[] = PAGES.map((p) => ({
      id: `page:${p}`, kind: "page", label: p, page: p, hint: "→ navigate",
    }));
    out.push({ id: "page:Settings", kind: "page", label: "Settings", page: "Settings", hint: "→ navigate" });
    out.push({
      id: "action:toggle-chat", kind: "action", label: "Toggle chat panel",
      hint: "Cmd+J", run: () => setChatOpen((o) => !o),
    });
    out.push({
      id: "action:hypothesis", kind: "action", label: "Generate hypotheses",
      hint: "narrator",
      run: () => explainHostValue.requestExplain(
        "Propose 3-5 ranked, testable causal hypotheses for the loaded dataset.",
        "hypothesis",
      ),
    });
    if (dataCtx?.columns) {
      for (const col of dataCtx.columns.slice(0, 80)) {
        out.push({ id: `pred:${col}`, kind: "predictor", label: col, page: "Configure" });
      }
    }
    if (dataCtx?.scenarios) {
      for (const s of dataCtx.scenarios) {
        out.push({ id: `scn:${s.name}`, kind: "scenario", label: s.name, hint: s.variable, page: "Configure" });
      }
    }
    if (dataCtx?.causalResults) {
      for (const treatment of Object.keys(dataCtx.causalResults)) {
        out.push({
          id: `tr:${treatment}`, kind: "treatment", label: treatment,
          hint: "explain effect",
          run: () => explainHostValue.requestExplain(
            `Explain the causal effect of ${treatment} on the outcome, citing magnitudes, mechanism, and confidence.`,
            "narrator",
          ),
        });
      }
    }
    return out;
  }, [dataCtx, explainHostValue, navigate, setChatOpen]);
}
