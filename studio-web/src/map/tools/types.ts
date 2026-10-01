// Map tools (SPEC §12.5): pointer state machines in raster coordinates that produce a
// SelectionSpec (rect, circle, polygon, pick) or Float32 edit arrays (brush).
import type { SelectionSpec } from "../../api/types";

export type ToolId = "pan" | "brush" | "rect" | "circle" | "polygon" | "pick";

export type RasterPoint = { px: number; py: number };

/** What a tool draws while active, in raster units. */
export type ToolPreview =
  | { kind: "rect"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "circle"; cx: number; cy: number; r: number }
  | { kind: "polygon"; points: RasterPoint[]; closed: boolean }
  | { kind: "brush"; cx: number; cy: number; r: number; erase: boolean };

export type ToolResult =
  | { kind: "selection"; spec: SelectionSpec; mask: Uint8Array; label: string; portable: boolean }
  | { kind: "edit"; rows: number[]; label: string };

export interface MapTool {
  readonly id: ToolId;
  /** Whether a drag on the map pans (pan tool) instead of driving the tool. */
  readonly pans: boolean;
  down(p: RasterPoint): ToolResult | null;
  move(p: RasterPoint, buttons: number): ToolResult | null;
  up(p: RasterPoint): ToolResult | null;
  /** Finish a multi-click shape (polygon: double-click or Enter). */
  finish(): ToolResult | null;
  cancel(): void;
  preview(hover: RasterPoint | null): ToolPreview | null;
}
