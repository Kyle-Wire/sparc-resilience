// The run's span tree as a Gantt (SPEC §5.10): run › stage › task › subtask (depth 4, collapsible,
// virtualised), bars growing live, warning ticks and checkpoint flags on the time axis, and
// hatched "system sleep?" bands where heartbeats are more than 45 s apart. Selecting a stage on
// the rail zooms the axis to that stage.
import { useMemo } from "react";
import { getEvents } from "../../../api/tracking";
import { useResource } from "../../../api/resource";
import { Gantt } from "../../../charts";
import { spansList, type TrackerState } from "../../../stores/tracker";
import { fmtDuration } from "../../../theme/format";
import { ganttRows } from "../model";

export function GanttPanel({ jid, state, nowS, zoom, onStage }: { jid: string; state: TrackerState; nowS: number; zoom: string | null; onStage?: (stage: string) => void }) {
  const spanMap = state.spans;
  const spans = useMemo(() => spansList({ ...state, spans: spanMap }), [spanMap]);
  const warnCount = Object.values(state.warnings).reduce((a, w) => a + w.count, 0);
  // Warning ticks need each warning's time: the projection keeps counts, the event log keeps times.
  const warnEvents = useResource(
    warnCount ? `job:${jid}:warning-events:${warnCount}` : null,
    (sig) => getEvents(jid, { types: ["warning"], limit: 1000 }, sig),
    { tags: [`job:${jid}`], keepPrevious: true },
  );
  const zoomWin = useMemo(() => {
    if (!zoom) return null;
    const st = state.stages?.[zoom];
    if (!st || st.started_ts === null) return null;
    return { from: st.started_ts, to: st.ended_ts ?? nowS };
  }, [zoom, state.stages, nowS]);
  const rows = useMemo(() => ganttRows(spans, zoomWin), [spans, zoomWin]);
  if (!rows.length) return <p className="mc-note">The timeline starts with the first stage.</p>;
  const running = rows.some((r) => r.end === null);
  const markers = [
    ...state.checkpoints.filter((c) => c.ts !== null).map((c) => ({ t: c.ts as number, kind: "checkpoint" as const, label: `${c.action}: ${c.done.join(", ") || "—"}` })),
    ...(warnEvents.data?.events ?? []).map((e) => ({ t: e.ts, kind: "warning" as const, label: `${(e as { code?: string }).code ?? "warning"}: ${(e as { message?: string }).message ?? ""}` })),
  ].filter((m) => !zoomWin || (m.t >= zoomWin.from && m.t <= zoomWin.to));
  const bands = state.heartbeat_gaps
    .filter((g) => !zoomWin || (g.to_ts >= zoomWin.from && g.from_ts <= zoomWin.to))
    .map((g) => ({ from: g.from_ts, to: g.to_ts, label: "system sleep?" }));
  const caption = [
    `${rows.length} spans${zoomWin ? ` in ${zoom}` : ""}`,
    bands.length ? `${bands.length} gap${bands.length === 1 ? "" : "s"} without heartbeats (${fmtDuration(bands.reduce((a, b) => a + b.to - b.from, 0))}): system sleep?` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const stageOfRow = new Map(spans.filter((s) => s.kind === "stage").map((s) => [s.span_id, s.name]));
  return (
    <Gantt
      title="Timeline"
      caption={caption}
      rows={rows}
      t0={zoomWin?.from}
      now={zoomWin ? zoomWin.to : running ? Math.max(nowS, ...rows.map((r) => r.end ?? r.start)) : undefined}
      markers={markers}
      bands={bands}
      openDepth={2}
      viewportRows={14}
      pin={false}
      onRowClick={onStage ? (id) => stageOfRow.has(id) && onStage(stageOfRow.get(id)!) : undefined}
    />
  );
}
