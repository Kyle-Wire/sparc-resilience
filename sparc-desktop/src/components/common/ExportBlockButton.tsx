import { useState, type RefObject } from "react";
import { toPng } from "html-to-image";
import { exportBlock } from "@/lib/api";

/**
 * ExportBlockButton — capture the DOM node referenced by `targetRef`
 * (or, when `canvasOnly`, the first <canvas> inside it) as a PNG and
 * persist it via POST /results/export so it appears in the registry
 * under stage "export".
 *
 * Drop into <Card actions={<ExportBlockButton ... />}>.
 */
export interface ExportBlockButtonProps {
  /** DOM node to snapshot. */
  targetRef: RefObject<HTMLElement | null>;
  /** Stable id for this artifact (e.g. "cate_map", "dose_response"). */
  artifactId: string;
  /** Registry stage to file under. Defaults to "export". */
  stage?: string;
  /** Human-readable filename suffix; falls back to artifactId. */
  label?: string;
  /** When true, snapshot only the first <canvas> (cheap for charts). */
  canvasOnly?: boolean;
  /** Compact icon-only style. */
  compact?: boolean;
}

export function ExportBlockButton({
  targetRef,
  artifactId,
  stage = "export",
  label,
  canvasOnly,
  compact,
}: ExportBlockButtonProps) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function handleClick() {
    const node = targetRef.current;
    if (!node) {
      setMsg("nothing to export");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      let dataUrl: string;
      if (canvasOnly) {
        const canvas = node.querySelector("canvas") as HTMLCanvasElement | null;
        if (!canvas) throw new Error("no <canvas> inside block");
        dataUrl = canvas.toDataURL("image/png");
      } else {
        dataUrl = await toPng(node, {
          cacheBust: true,
          pixelRatio: window.devicePixelRatio || 2,
          backgroundColor: "#ffffff",
          // Skip cross-origin images (basemap tiles) rather than failing.
          skipFonts: false,
          filter: (n) => {
            // Hide the export button itself from the snapshot.
            if (n instanceof HTMLElement && n.dataset?.exportSkip === "1") {
              return false;
            }
            return true;
          },
        });
      }
      const res = await exportBlock({
        artifact_id: artifactId,
        stage,
        label: label ?? artifactId,
        png_b64: dataUrl,
      });
      setMsg(res.registered ? "saved + registered" : "saved (no registry)");
    } catch (exc) {
      setMsg(`failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    } finally {
      setBusy(false);
      setTimeout(() => setMsg(null), 4000);
    }
  }

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }} data-export-skip="1">
      <button
        onClick={handleClick}
        disabled={busy}
        title={`Export ${label ?? artifactId} as PNG`}
        style={{
          fontSize: compact ? 10 : 11,
          padding: compact ? "2px 6px" : "4px 8px",
          border: "1px solid var(--line, #ddd)",
          borderRadius: 3,
          background: busy ? "var(--muted, #888)" : "rgba(255,255,255,0.95)",
          color: busy ? "#fff" : "var(--ink, #333)",
          cursor: busy ? "wait" : "pointer",
          fontFamily: "inherit",
        }}
      >
        {busy ? "exporting…" : compact ? "↓ PNG" : "Export PNG"}
      </button>
      {msg && (
        <span
          className="mono"
          style={{
            fontSize: 9,
            color: msg.startsWith("failed") ? "var(--crimson, #e73c25)" : "var(--muted, #777)",
          }}
        >
          {msg}
        </span>
      )}
    </span>
  );
}

export default ExportBlockButton;
