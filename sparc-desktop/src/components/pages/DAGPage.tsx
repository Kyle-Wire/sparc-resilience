import DagEditor from "@/components/dag/DagEditor";

interface DagPageProps {
  onNavigate?: (page: string) => void;
}

/**
 * DAG page — thin wrapper around the self-contained DagEditor/DAGView.
 * DagEditor manages its own state (fetch, MC3, undo/redo, approval flow)
 * via usePipeline() context.
 */
export default function DAGPage({ onNavigate: _onNavigate }: DagPageProps) {
  return <DagEditor />;
}
