// The SVG chart kit (SPEC §12.6). Every chart renders inside ChartFrame (title, caption,
// units, Table view, SVG/PNG/CSV export, Pin to Findings, focusable marks). `KIT` lists the
// charts so tests can assert those guarantees for each one.
import { Bars } from "./Bars";
import { BoxStrip } from "./BoxStrip";
import { DotRange } from "./DotRange";
import { Forest } from "./Forest";
import { Gantt } from "./Gantt";
import { Gauge } from "./Gauge";
import { Heatmap } from "./Heatmap";
import { HexbinScatter } from "./HexbinScatter";
import { Histogram } from "./Histogram";
import { IntervalStack } from "./IntervalStack";
import { LineBand } from "./LineBand";
import { Pareto } from "./Pareto";
import { RingProfile } from "./RingProfile";
import { Rose } from "./Rose";
import { SmallMultiples } from "./SmallMultiples";
import { Sparkline } from "./Sparkline";

export { Axis, BandAxis, axisTitle } from "./Axis";
export { ChartFrame, MeasureWidth, markProps, useChart, useChartWidth, serializeChartSvg, svgToPng, composeSvgs, type ChartHandle, type ChartTable, type FrameOptions } from "./ChartFrame";
export * from "./scales";
export { Bars, BoxStrip, DotRange, Forest, Gantt, Gauge, Heatmap, HexbinScatter, Histogram, IntervalStack, LineBand, Pareto, RingProfile, Rose, SmallMultiples, Sparkline };
export type { BarSeries, BarsProps } from "./Bars";
export type { BoxGroup, BoxStripProps } from "./BoxStrip";
export type { DotRangeRow, DotRangeProps } from "./DotRange";
export type { ForestRow, ForestProps } from "./Forest";
export type { GanttRow, GanttProps } from "./Gantt";
export type { GaugeProps } from "./Gauge";
export type { HeatmapProps } from "./Heatmap";
export { bin2d, type Bins2D, type HexbinScatterProps } from "./HexbinScatter";
export { histogram, snapToEdges, type Bins, type HistogramProps } from "./Histogram";
export type { IntervalLayer, IntervalRow, IntervalStackProps } from "./IntervalStack";
export { seriesColors, type LineSeries, type LineBandProps, type RefLine } from "./LineBand";
export type { ParetoPoint, ParetoProps } from "./Pareto";
export type { Ring, RingProfileProps } from "./RingProfile";
export type { RosePetal, RoseProps } from "./Rose";
export type { SmallMultiplesProps } from "./SmallMultiples";
export type { SparklineProps } from "./Sparkline";

export const KIT = {
  LineBand,
  Bars,
  DotRange,
  Forest,
  Heatmap,
  Histogram,
  HexbinScatter,
  BoxStrip,
  Gantt,
  Sparkline,
  Pareto,
  Rose,
  Gauge,
  IntervalStack,
  RingProfile,
  SmallMultiples,
} as const;

export type KitName = keyof typeof KIT;
