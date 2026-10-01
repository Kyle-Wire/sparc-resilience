// Map kit (SPEC §6.5, §12.5): canvas raster renderer, map view, legend, inspector, overlays,
// swipe compare, the shared view transform and the selection/edit tools.
export { GridCanvas, type GridCanvasHandle, type GridCanvasProps } from "./GridCanvas";
export { MapView, differenceValues, type CompareMode, type MapSelectionEvent, type MapViewProps } from "./MapView";
export { LayerPicker, findLayer } from "./LayerPicker";
export { Legend, categorySwatches, divergingEnds } from "./Legend";
export { Inspector, type CellInfo, type CellCurve } from "./Inspector";
export { Overlay, MapChrome, type OverlayFeature } from "./Overlay";
export { SwipeCompare, type SwipeLayer } from "./SwipeCompare";
export { useMapView, fitView, zoomView, MAX_SCALE, type MapViewApi, type ViewState } from "./useMapView";
export { buildCellToPt, buildRowToPix, colorize, mapPalette, maskRaster, outlinePath, type Palette } from "./colour";
export { computeDomain, valueToT, tToValue, quantiles, layerSummary, categoryCounts, type Domain } from "./domain";
export * from "./grid";
export { runLayerLoader, fetchRunGrid, useRunGrid, useRunLayers, uploadMaskSelection, maskFromRange, forgetRunLayers, type LayerValues } from "./data";
export * from "./tools";
