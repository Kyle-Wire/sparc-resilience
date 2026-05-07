/**
 * PanelGate — uniform empty-state + status-pill wrapper for insights panels.
 *
 * Subscribes to `useAvailability()` and renders one of three things:
 *
 *   1. The panel body (children) when status is ``ready``.
 *   2. A `<Panel>` containing `<PanelEmpty>` + `<StatusPill>` for any
 *      non-ready status (``awaiting``, ``partial``, ``running``, ``failed``).
 *   3. The panel body anyway (with a small status pill in the header) when
 *      ``mode="badge"`` — useful for panels that have meaningful partial
 *      content even before everything is loaded.
 *
 * The goal is that individual panels stop hand-rolling "no data yet"
 * messages and instead delegate to the central `/panels/availability`
 * snapshot for consistent messaging across the workspace.
 */
import type { ReactNode } from "react";
import {
  Panel,
  PanelEmpty,
  StatusPill,
  type PanelStatusValue,
} from "@/components/ui/DesignSystem";
import { useAvailability } from "@/hooks/useAvailability";

interface PanelGateProps {
  panelId: string;
  /** Title rendered when we have to show the empty state. */
  title: string;
  /** Subtitle for the empty-state Panel header. */
  subtitle?: string;
  /** Panel body — only rendered when the gate resolves to ``ready``. */
  children: ReactNode;
  /** Optional override hint when the snapshot has no opinion. */
  fallbackHint?: ReactNode;
}

const STATUS_LABELS: Record<PanelStatusValue, (stageLabel: string) => string> = {
  ready:    () => "READY",
  awaiting: (s) => `${s} · NOT STARTED`,
  partial:  (s) => `${s} · PARTIAL`,
  running:  (s) => `${s} · RUNNING`,
  failed:   (s) => `${s} · FAILED`,
};

const STATUS_REASONS: Record<PanelStatusValue, string> = {
  ready:    "Ready.",
  awaiting: "This panel is waiting on upstream stage output.",
  partial:  "Some inputs are present; a few are still missing.",
  running:  "Stage is currently running — this panel will refresh on completion.",
  failed:   "The most recent stage run reported an error.",
};

export function PanelGate({
  panelId,
  title,
  subtitle,
  children,
  fallbackHint,
}: PanelGateProps) {
  const { panel } = useAvailability();
  const entry = panel(panelId);

  // No snapshot yet (server unreachable / first paint) — render children.
  if (!entry) return <>{children}</>;

  if (entry.status === "ready") return <>{children}</>;

  const status = entry.status;
  const stageLabel = entry.stage_label || `STAGE ${entry.stage}`;
  const pillLabel = STATUS_LABELS[status](stageLabel);

  return (
    <Panel
      title={title}
      subtitle={subtitle}
      actions={<StatusPill status={status} label={pillLabel} />}
    >
      <PanelEmpty
        status={<StatusPill status={status} label={pillLabel} />}
        reason={STATUS_REASONS[status]}
        hint={entry.hint || fallbackHint}
      />
    </Panel>
  );
}
