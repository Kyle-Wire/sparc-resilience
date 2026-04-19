import DAGView from "@/components/dag/DagEditor";

interface DagPageProps {
  onNavigate?: (page: string) => void;
}

export default function DAGPage({ onNavigate: _onNavigate }: DagPageProps) {
  return (
    <div style={{ height: "calc(100vh - 56px)", display: "flex", flexDirection: "column" }}>
      <DAGView />
    </div>
  );
}
