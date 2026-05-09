/**
 * ResizableMapWrapper — wraps any fixed-height map block and lets the
 * user drag-resize it vertically. Height is clamped between minHeight
 * and the current viewport height so it never leaves the screen.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";

interface Props {
  children: ReactNode;
  defaultHeight?: number;
  minHeight?: number;
}

export default function ResizableMapWrapper({
  children,
  defaultHeight = MAP_HEIGHT_DEFAULT,
  minHeight = 220,
}: Props) {
  const [height, setHeight] = useState(defaultHeight);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startH = useRef(0);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current) return;
    const delta = e.clientY - startY.current;
    const next = Math.max(minHeight, Math.min(window.innerHeight - 80, startH.current + delta));
    setHeight(next);
  }, [minHeight]);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const onHandleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startY.current = e.clientY;
    startH.current = height;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ height, position: "relative" }}>
        {children}
      </div>
      {/* Drag handle */}
      <div
        onMouseDown={onHandleMouseDown}
        title="Drag to resize map"
        style={{
          height: 8,
          cursor: "row-resize",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "transparent",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: 36,
            height: 3,
            borderRadius: 2,
            background: "var(--line)",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = "var(--muted)")}
          onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = "var(--line)")}
        />
      </div>
    </div>
  );
}
