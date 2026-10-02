// The run's span tree as a Gantt (SPEC §5.10): run › stage › task › subtask (depth 4, collapsible,
// virtualised), bars growing live, warning ticks and checkpoint flags on the time axis, and
// hatched "system sleep?" bands where heartbeats are more than 45 s apart. Selecting a stage on
// the rail zooms the axis to that stage.
import { useMemo } from "react";
import { getEvents, type LogLine } from "../../../api/tracking";
import { useResource } from "../../../api/resource";
import { Gantt } from "../../../charts";
import { spansList, type TrackerState } from "../../../stores/tracker";
import { fmtDuration } from "../../../theme/format";
import { ganttRows } from "../model";

/** Warning events read from the log for the time axis (the newest live ones are added on top). */
export const WARNING_TICKS_MAX = 1000;

export function GanttPanel({
  jid,
  state,
  nowS,
  zoom,
  onStage,
  logs = [],
}: {
  jid: string;
  state: TrackerState;
  nowS: number;
  zoom: string | null;
  onStage?: (stage: string) => void;
  /** Log and warning lines streamed since the page opened (the tracker entry's `logs`). */
  logs?: LogLine[];
}) {
  const spanMap = state.spans;
  const spans = useMemo(() => spansList({ ...state, spans: spanMap }), [spanMap]);
  const hasWarnings = Object.keys(state.warnings).length > 0;
  // Warning ticks need each warning's time: the projection keeps counts, the event log keeps
  // times. The log is read once (when the first warning is known); warnings streamed after the
  // cursor it reached come from the live lines, so a new warning never re-reads the whole file.
  const warnEvents = useResource(
    hasWarnings ? `job:${jid}:warning-events` : null,
    (sig) => getEvents(jid, { types: ["warning"], limit: WARNING_TICKS_MAX }, sig),
    { tags: [`job:${jid}`], keepPrevious: true },
  );
  const warnTicks = useMemo(() => {
    const page = warnEvents.data;
    if (!page) return [];
    const read = page.events.map((e) => ({ t: e.ts, label: `${(e as { code?: string }).code ?? "warning"}: ${(e as { message?: string }).message ?? ""}` }));
    const edge = Math.max(page.next_cursor, page.events.length ? page.events[page.events.length - 1].cursor : -1);
    const streamed = logs.filter((l) => l.cursor > edge && l.logger.startsWith("warning:")).map((l) => ({ t: l.ts, label: `${l.logger.slice("warning:".length)}: ${l.msg}` }));
    return [...read, ...streamed];
  }, [warnEvents.data, logs]);
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
    ...warnTicks.map((w) => ({ t: w.t, kind: "warning" as const, label: w.label })),
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
