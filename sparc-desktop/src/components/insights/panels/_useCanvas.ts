/**
 * useCanvas — minimal wrapper that prepares a HiDPI 2D canvas context
 * sized to the element's client box and re-runs the draw on the
 * provided dependency array.
 *
 * Each panel passes a `draw(ctx, w, h)` callback that does the actual
 * painting. This kills the ~30-line boilerplate present in every
 * canvas-drawing useEffect in the legacy ResultsPage.
 */
import { useEffect, useRef } from "react";

export type DrawFn = (ctx: CanvasRenderingContext2D, w: number, h: number) => void;

export function useCanvas(draw: DrawFn, deps: React.DependencyList) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 400;
    const h = canvas.clientHeight || 240;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}
