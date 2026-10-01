// Pan: the default tool. Dragging pans the view (GridCanvas handles it); a click picks the
// cell under the pointer for the inspector.
import type { MapTool } from "./types";

export function panTool(): MapTool {
  return {
    id: "pan",
    pans: true,
    down: () => null,
    move: () => null,
    up: () => null,
    finish: () => null,
    cancel: () => {},
    preview: () => null,
  };
}
