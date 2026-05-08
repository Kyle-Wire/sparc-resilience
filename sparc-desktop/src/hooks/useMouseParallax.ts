import { useEffect, useRef, useCallback } from "react";

interface ParallaxOptions {
  /** Maximum rotation in degrees (default 12) */
  maxDeg?: number;
  /** Maximum translation in px for layered elements (default 20) */
  maxPx?: number;
  /** Whether the effect is enabled */
  enabled?: boolean;
}

interface ParallaxState {
  /** Normalized -1..1 horizontal position */
  nx: number;
  /** Normalized -1..1 vertical position */
  ny: number;
}

type ParallaxCallback = (state: ParallaxState) => void;

/**
 * Tracks mouse position normalized to [-1, 1] relative to the window.
 * Calls the provided callback with smooth lerped values on each animation frame.
 */
export function useMouseParallax(
  callback: ParallaxCallback,
  { maxDeg = 12, maxPx = 20, enabled = true }: ParallaxOptions = {},
) {
  const rawRef = useRef({ x: 0, y: 0 });
  const smoothRef = useRef({ x: 0, y: 0 });
  const rafRef = useRef<number | null>(null);
  const stableCallback = useRef(callback);
  stableCallback.current = callback;

  const handleMouseMove = useCallback((e: MouseEvent) => {
    rawRef.current = {
      x: (e.clientX / window.innerWidth) * 2 - 1,
      y: (e.clientY / window.innerHeight) * 2 - 1,
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      // Reset to neutral when disabled
      stableCallback.current({ nx: 0, ny: 0 });
      return;
    }

    window.addEventListener("mousemove", handleMouseMove, { passive: true });

    const tick = () => {
      const lerp = 0.06;
      smoothRef.current.x += (rawRef.current.x - smoothRef.current.x) * lerp;
      smoothRef.current.y += (rawRef.current.y - smoothRef.current.y) * lerp;
      stableCallback.current({ nx: smoothRef.current.x, ny: smoothRef.current.y });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [enabled, handleMouseMove]);

  return { maxDeg, maxPx };
}
