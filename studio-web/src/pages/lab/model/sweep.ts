// Sweep curve helpers (SPEC §7.9): the fitted saturation overlay and the pipeline's own
// response curve, drawn on the same signed scale as the swept ΔT points.
import type { Sweep } from "../../../api/lab";

/** Majority sign of the finite values (−1 when the points are mostly cooling). */
export function dominantSign(ys: (number | null)[]): 1 | -1 {
  let neg = 0;
  let pos = 0;
  for (const y of ys) {
    if (y === null || !Number.isFinite(y)) continue;
    if (y < 0) neg++;
    else if (y > 0) pos++;
  }
  return neg > pos ? -1 : 1;
}

/**
 * The fitted curve at `xs` on the points' sign: saturating B = A·(1 − e^(−d/d_s)); linear (no
 * A/d_s reported) is the least-squares line through the origin of the points; other shapes
 * have no closed form here and return null.
 */
export function fitOverlay(fit: Sweep["fit"], xs: number[], points: { x: number; y: number | null }[]): (number | null)[] | null {
  if (!fit) return null;
  const sign = dominantSign(points.map((p) => p.y));
  if (fit.model === "saturating" && fit.A !== null && fit.ds !== null && fit.ds > 0) {
    const A = Math.abs(fit.A);
    return xs.map((d) => sign * A * (1 - Math.exp(-d / fit.ds!)));
  }
  if (fit.model === "linear") {
    let sxy = 0;
    let sxx = 0;
    for (const p of points) {
      if (p.y === null || !Number.isFinite(p.y)) continue;
      sxy += p.x * p.y;
      sxx += p.x * p.x;
    }
    if (!(sxx > 0)) return null;
    const slope = sxy / sxx;
    return xs.map((d) => slope * d);
  }
  return null;
}

/** Evenly spaced x values from 0 to the largest dose (for a smooth overlay). */
export function overlayXs(doses: number[], n = 40): number[] {
  const top = Math.max(0, ...doses.filter(Number.isFinite));
  if (!(top > 0)) return [];
  return Array.from({ length: n + 1 }, (_, i) => (top * i) / n);
}

/** The pipeline's response curve if it has a recognisable shape: {dose|x: number[], benefit|delta|y: number[]}. */
export function pipelineCurve(obj: Sweep["pipeline_curve"], sign: 1 | -1): { x: number[]; y: number[] } | null {
  if (!obj) return null;
  const xs = (obj.dose ?? obj.x ?? obj.doses) as unknown;
  const ys = (obj.benefit ?? obj.delta ?? obj.y ?? obj.mean) as unknown;
  if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length !== ys.length || !xs.length) return null;
  const x = xs.map(Number);
  const y = ys.map(Number);
  if (x.some((v) => !Number.isFinite(v))) return null;
  // Benefits are reported positive; put them on the points' sign.
  const s = dominantSign(y) === sign ? 1 : -1;
  return { x, y: y.map((v) => s * v) };
}
