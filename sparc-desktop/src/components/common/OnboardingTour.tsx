/**
 * OnboardingTour (Phase 20) — first-run guided walkthrough.
 *
 * - Steps point at sidebar pages and explain SPARC's causal-first flow.
 * - Skippable + resumable: state stored in localStorage under
 *   ``sparc-onboarding`` (``{ step: number, dismissed: boolean }``).
 * - Designed to be mounted unconditionally inside App; it self-renders
 *   only when the user has neither finished nor dismissed the tour.
 */

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "sparc-onboarding";
const TOTAL_VERSION = 2; // bump to force re-show

interface OnboardingState {
  step: number;
  dismissed: boolean;
  version: number;
}

export interface TourStep {
  page?: string; // sidebar PageName or "Settings"
  title: string;
  body: string;
}

const DEFAULT_STEPS: TourStep[] = [
  {
    title: "Welcome to SPARC",
    body: "SPARC is a causal-first desktop for spatial physics-informed models. This 60-second tour shows the workflow against the bundled Providence UHI example.",
  },
  {
    page: "Project",
    title: "1 · Project",
    body: "Set the project name, domain, and target column here. The YAML preview on the right is the canonical config — every other page is a structured editor for it.",
  },
  {
    page: "Data Collection",
    title: "2 · Data Collection",
    body: "Define a study boundary, fetch geospatial variables from free public APIs, confirm data quality on a 30m choropleth map, and export a GeoParquet file for the pipeline.",
  },
  {
    page: "DAG",
    title: "3 · Causal DAG",
    body: "Declare what causes what. Use j/k to walk nodes (vim-style), e to draw an edge, and d to delete. MC³ later returns posterior edge probabilities to validate your prior structure.",
  },
  {
    page: "Physics",
    title: "4 · Physics priors",
    body: "Add monotone constraints and dose-response shapes. These become hard physics constraints the neural network must respect.",
  },
  {
    page: "Run",
    title: "5 · Run the pipeline",
    body: "Execute Stage 0 (correlogram) → 1 (GWEN weighting) → 2 (spatial CV) → 3 (causal validation, MC³) → 4 (scenarios). Live progress streams from the FastAPI backend. When the run completes you'll see a summary card with a direct link to Results.",
  },
  {
    page: "Results",
    title: "6 · Results",
    body: "Walk through each stage using the pipeline navigator at the top. Stage 0 shows spatial geometry, Stage 2 model predictions on a map, Stage 3 causal dose-response, and Stage 4 ranked interventions. Each section has a Technical Disclosure for deeper diagnostics.",
  },
  {
    page: "Report",
    title: "7 · Reporting",
    body: "Generate audience-tuned reports (planner, public, technical, council, equity, scientist, auditor) as PDF or Word. Decision Support on the same panel lets you set a budget cap and run the equity optimiser.",
  },
  {
    title: "You're ready",
    body: "Press Cmd+K (Ctrl+K on Windows) any time to jump to a page, predictor, scenario, or treatment. Your project autosaves. Reopen this tour from Settings → Help.",
  },
];

function loadState(): OnboardingState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<OnboardingState>;
      if (parsed.version === TOTAL_VERSION) {
        return {
          step: parsed.step ?? 0,
          dismissed: !!parsed.dismissed,
          version: TOTAL_VERSION,
        };
      }
    }
  } catch {
    /* ignore */
  }
  return { step: 0, dismissed: false, version: TOTAL_VERSION };
}

function saveState(state: OnboardingState) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
  catch { /* ignore */ }
}

interface OnboardingTourProps {
  steps?: TourStep[];
  onNavigate?: (page: string) => void;
  /** Force the tour open from outside (e.g. from Settings → "Restart tour") */
  open?: boolean;
  onClose?: () => void;
}

export default function OnboardingTour({ steps, onNavigate, open, onClose }: OnboardingTourProps) {
  const tourSteps = steps ?? DEFAULT_STEPS;
  const [state, setState] = useState<OnboardingState>(() => loadState());

  // Sync to external open prop (forces reopen even if dismissed)
  useEffect(() => {
    if (open) setState((s) => ({ ...s, dismissed: false, step: s.step >= tourSteps.length ? 0 : s.step }));
  }, [open, tourSteps.length]);

  const visible = !state.dismissed && state.step < tourSteps.length;
  const current = tourSteps[Math.min(state.step, tourSteps.length - 1)];

  // Trigger navigation when entering a step that has a page hint
  useEffect(() => {
    if (!visible) return;
    if (current?.page && onNavigate) onNavigate(current.page);
  }, [visible, current, onNavigate]);

  const next = useCallback(() => {
    setState((s) => {
      const nextStep = s.step + 1;
      const ns: OnboardingState = nextStep >= tourSteps.length
        ? { ...s, step: tourSteps.length, dismissed: true }
        : { ...s, step: nextStep };
      saveState(ns);
      if (ns.dismissed && onClose) onClose();
      return ns;
    });
  }, [tourSteps.length, onClose]);

  const prev = useCallback(() => {
    setState((s) => {
      const ns = { ...s, step: Math.max(0, s.step - 1) };
      saveState(ns);
      return ns;
    });
  }, []);

  const skip = useCallback(() => {
    setState((s) => {
      const ns = { ...s, dismissed: true };
      saveState(ns);
      return ns;
    });
    if (onClose) onClose();
  }, [onClose]);

  if (!visible) return null;

  return (
    <div
      role="dialog" aria-label="SPARC onboarding tour"
      style={{
        position: "fixed", left: 0, right: 0, bottom: 0,
        zIndex: 9999, display: "flex", justifyContent: "center", padding: 16, pointerEvents: "none",
      }}
    >
      <div
        style={{
          pointerEvents: "auto",
          maxWidth: 520, width: "100%",
          background: "#fff", border: "1px solid var(--line)",
          borderRadius: 8, boxShadow: "0 12px 32px rgba(0,0,0,0.12)",
          padding: 18, fontFamily: "inherit",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <span className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            Tour · {state.step + 1} / {tourSteps.length}
          </span>
          <button
            onClick={skip}
            style={{ border: 0, background: "none", color: "var(--muted)", fontSize: 11, cursor: "pointer", padding: 0 }}
          >
            Skip tour ✕
          </button>
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "var(--ink)", marginBottom: 6 }}>{current.title}</div>
        <div style={{ fontSize: 13, lineHeight: 1.55, color: "var(--ink-2)", marginBottom: 14 }}>{current.body}</div>
        <div style={{
          height: 3, background: "var(--line)", borderRadius: 2, overflow: "hidden", marginBottom: 12,
        }}>
          <div style={{
            width: `${((state.step + 1) / tourSteps.length) * 100}%`,
            height: "100%", background: "var(--crimson)", transition: "width .25s ease",
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <button
            onClick={prev} disabled={state.step === 0}
            style={{
              border: "1px solid var(--line)", background: "#fff", padding: "6px 14px",
              borderRadius: 4, fontFamily: "inherit", fontSize: 12,
              cursor: state.step === 0 ? "default" : "pointer",
              color: state.step === 0 ? "var(--muted)" : "var(--ink-2)",
            }}
          >Back</button>
          <button
            onClick={next}
            style={{
              border: "1px solid var(--ink)", background: "var(--ink)", color: "#fff",
              padding: "6px 16px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", fontWeight: 600,
            }}
          >{state.step === tourSteps.length - 1 ? "Got it" : "Next →"}</button>
        </div>
      </div>
    </div>
  );
}

/** Helper for Settings → "Restart tour" button. */
export function resetOnboarding() {
  try { localStorage.removeItem(STORAGE_KEY); }
  catch { /* ignore */ }
}
