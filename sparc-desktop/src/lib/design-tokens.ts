/**
 * SPARC Labs brand color ramp — maps directly to the brand guidelines.
 * Used for spatial map color scales and UI accents.
 */

export const SPARC_COLORS = {
  purple: "#602468",
  magenta: "#9e337d",
  pink: "#e94d9b",
  red: "#e94461",
  crimson: "#e73c25",
  orange: "#e76c25",
  amber: "#e79024",
  gold: "#f0b632",
  yellow: "#fbdd46",
} as const;

/**
 * Sequential ramp as RGB tuples for deck.gl color accessors.
 */
export const SPARC_RAMP_RGB: [number, number, number][] = [
  [96, 36, 104],   // purple
  [158, 51, 125],  // magenta
  [233, 77, 155],  // pink
  [233, 68, 97],   // red
  [231, 60, 37],   // crimson
  [231, 108, 37],  // orange
  [231, 144, 36],  // amber
  [240, 182, 50],  // gold
  [251, 221, 70],  // yellow
];

/**
 * Hex ramp for D3 / Recharts scales.
 */
export const SPARC_RAMP_HEX = [
  "#602468", "#9e337d", "#e94d9b", "#e94461",
  "#e73c25", "#e76c25", "#e79024", "#f0b632", "#fbdd46",
];

/**
 * Default height for spatial map containers across the app.
 * Earlier versions used 380px which felt cramped (especially in 2-col
 * grids). This value gives every map a tall, full-width canvas.
 */
export const MAP_HEIGHT_DEFAULT = 560;
