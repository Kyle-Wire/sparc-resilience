/**
 * StageResultsPage — top-level container for the per-stage Results experience.
 *
 * Renders PipelineNavigator plus the active stage page. Stage switching is
 * managed by local state; the "Results" sidebar entry always opens here.
 *
 * All 5 stages are available for navigation. The active stage tracks which
 * page the user is currently viewing; visited stages become "complete" so the
 * navigator shows progress without requiring actual pipeline run status.
 *
 * TODO: Wire `status` values to the run manifest so locked/complete states
 * reflect actual pipeline completion rather than visit history.
 */
import { useState } from "react";
import PipelineNavigator, { type StageNode } from "./PipelineNavigator";
import Stage0Page from "./stages/Stage0Page";
import Stage1Page from "./stages/Stage1Page";
import Stage2Page from "./stages/Stage2Page";
import Stage3Page from "./stages/Stage3Page";
import Stage4Page from "./stages/Stage4Page";

const STAGE_LABELS: [string, string][] = [
  ["Geometry",   "var(--s0, #602468)"],
  ["Selection",  "var(--s1, #e94d9b)"],
  ["Prediction", "var(--s2, #e73c25)"],
  ["Causation",  "var(--s3, #e79024)"],
  ["Decision",   "var(--s4, #f0b632)"],
];

const STAGE_PAGES = [Stage0Page, Stage1Page, Stage2Page, Stage3Page, Stage4Page];

export default function StageResultsPage() {
  const [activeStage, setActiveStage] = useState(0);

  function handleStageClick(id: number) {
    setActiveStage(id);
  }

  // All stages are navigable when viewing Results (the pipeline has already run).
  // TODO: Wire to run manifest to mark only completed stages as non-locked.
  const stages: StageNode[] = STAGE_LABELS.map(([label, color], i) => ({
    id: i,
    label,
    color,
    status: i === activeStage ? "active" : "complete",
  }));

  const ActivePage = STAGE_PAGES[activeStage];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <PipelineNavigator stages={stages} onStageClick={handleStageClick} />

      <main
        style={{
          flex: 1,
          overflowY: "auto",
          background: "var(--paper, #f7f4ee)",
        }}
      >
        <ActivePage />
      </main>
    </div>
  );
}
