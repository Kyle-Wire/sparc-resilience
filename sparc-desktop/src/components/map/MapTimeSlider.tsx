/**
 * MapTimeSlider — animated time/step selector for time-varying map data.
 *
 * Plays through a list of time steps (years, dates, scenario names) and
 * notifies the parent when the index changes so it can swap the GeoJSON or
 * the active column. Designed to be dropped into a map page; the parent
 * controls the data layer.
 */
import { useCallback, useEffect, useRef, useState } from "react";

interface MapTimeSliderProps {
  steps: ReadonlyArray<string | number>;
  /** Index that the slider currently shows. */
  index: number;
  onIndexChange: (i: number) => void;
  /** Milliseconds per step when playing (default 600). */
  speedMs?: number;
  /** Optional label shown to the left of the slider. */
  label?: string;
}

export default function MapTimeSlider({
  steps,
  index,
  onIndexChange,
  speedMs = 600,
  label,
}: MapTimeSliderProps) {
  const [playing, setPlaying] = useState(false);
  const idxRef = useRef(index);
  idxRef.current = index;
  const stepsRef = useRef(steps);
  stepsRef.current = steps;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const next = (idxRef.current + 1) % Math.max(1, stepsRef.current.length);
      onIndexChange(next);
    }, speedMs);
    return () => clearInterval(id);
  }, [playing, speedMs, onIndexChange]);

  const togglePlay = useCallback(() => setPlaying((p) => !p), []);
  const current = steps[Math.max(0, Math.min(steps.length - 1, index))];

  if (steps.length === 0) return null;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "6px 10px",
        background: "rgba(255,255,255,0.95)",
        border: "1px solid rgba(0,0,0,0.06)",
        borderRadius: 6,
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
        backdropFilter: "blur(4px)",
        fontFamily: "system-ui, sans-serif",
        fontSize: 11,
      }}
    >
      <button
        onClick={togglePlay}
        title={playing ? "Pause" : "Play"}
        style={{
          width: 24,
          height: 24,
          borderRadius: 4,
          border: "1px solid #ccc",
          background: "#fff",
          cursor: "pointer",
          fontWeight: 700,
          color: "#444",
        }}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      {label && (
        <span style={{ fontSize: 10, color: "#666", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {label}
        </span>
      )}
      <input
        type="range"
        min={0}
        max={steps.length - 1}
        value={Math.max(0, Math.min(steps.length - 1, index))}
        onChange={(e) => onIndexChange(Number(e.target.value))}
        style={{ flex: 1, minWidth: 140 }}
      />
      <span
        className="mono"
        style={{
          minWidth: 48,
          textAlign: "right",
          fontFamily: "ui-monospace, monospace",
          color: "#222",
        }}
      >
        {String(current)}
      </span>
    </div>
  );
}
