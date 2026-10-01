// Small multiples (SPEC §6.4): a grid of bare panels sharing one frame (title, Table view,
// exports, Pin). Panels render any kit chart with `bare`; pass a shared domain so the panels
// compare. Exports tile every panel's SVG into one image (ChartFrame composes them).
import type { ReactNode } from "react";
import { ChartFrame, MeasureWidth, type ChartTable, type FrameOptions } from "./ChartFrame";

export type SmallMultiplesProps<T> = FrameOptions & {
  items: T[];
  panelTitle: (item: T) => string;
  /** Render one panel; use a kit chart with `bare` (the frame supplies exports). */
  renderPanel: (item: T, index: number) => ReactNode;
  table: ChartTable;
  minPanelWidth?: number;
};

export function SmallMultiples<T>({ items, panelTitle, renderPanel, table, minPanelWidth = 220, ...frame }: SmallMultiplesProps<T>) {
  return (
    <ChartFrame {...frame} table={table}>
      <div className="small-multiples" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(min(${minPanelWidth}px, 100%), 1fr))` }}>
        {items.map((it, i) => (
          <div className="panel" key={i}>
            <h4>{panelTitle(it)}</h4>
            <MeasureWidth>{renderPanel(it, i)}</MeasureWidth>
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}
