/**
 * DownloadMenu — uniform per-block download dropdown.
 *
 * Three actions:
 *  - **Data**: fetch the source endpoint (CSV/JSON/GeoJSON) and trigger a
 *    browser download.
 *  - **Figure (PNG)**: GET /artifacts/{stage}/{id}.png — the registry-backed
 *    rendered figure (uses server matplotlib + content-hash cache).
 *  - **Capture (PNG)**: client-side DOM screenshot via ExportBlockButton.
 *
 * Also exposes a top-level "Download all" action that hits /results/bundle
 * and saves the ZIP.
 */
import { useState, type RefObject } from "react";
import { ExportBlockButton } from "./ExportBlockButton";
import { downloadResultsBundle } from "@/lib/api";

const BASE = "http://127.0.0.1:8008";

export interface DownloadMenuProps {
  /** Stable artifact id. */
  artifactId: string;
  /** Registry stage id. */
  stage: string;
  /** Human label shown in the menu header. */
  label?: string;
  /** API path for the underlying data export (CSV/JSON/GeoJSON). */
  dataEndpoint?: string;
  /** Suggested filename for data download (without extension). */
  dataFilename?: string;
  /** DOM node to snapshot for client capture. */
  targetRef?: RefObject<HTMLElement | null>;
  /** When true, shows the "Download all artifacts" item. */
  includeBundle?: boolean;
  compact?: boolean;
}

function inferExt(contentType: string | null, fallback = "bin"): string {
  if (!contentType) return fallback;
  if (contentType.includes("json")) return contentType.includes("geo") ? "geojson" : "json";
  if (contentType.includes("csv")) return "csv";
  if (contentType.includes("zip")) return "zip";
  if (contentType.includes("png")) return "png";
  return fallback;
}

async function downloadUrl(url: string, suggestedName: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  const blob = await res.blob();
  const ext = inferExt(res.headers.get("content-type"), suggestedName.split(".").pop() ?? "bin");
  const filename = suggestedName.includes(".") ? suggestedName : `${suggestedName}.${ext}`;
  triggerBlob(blob, filename);
}

function triggerBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export default function DownloadMenu({
  artifactId,
  stage,
  label,
  dataEndpoint,
  dataFilename,
  targetRef,
  includeBundle,
  compact,
}: DownloadMenuProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const close = () => setOpen(false);
  const onError = (e: unknown) => {
    setErr(e instanceof Error ? e.message : String(e));
    setTimeout(() => setErr(null), 4000);
  };

  const handleData = async () => {
    if (!dataEndpoint) return;
    setBusy("data");
    try {
      await downloadUrl(`${BASE}${dataEndpoint}`, dataFilename ?? artifactId);
      close();
    } catch (e) { onError(e); } finally { setBusy(null); }
  };

  const handlePng = async () => {
    setBusy("png");
    try {
      await downloadUrl(`${BASE}/artifacts/${stage}/${artifactId}.png`, `${artifactId}.png`);
      close();
    } catch (e) { onError(e); } finally { setBusy(null); }
  };

  const handleBundle = async () => {
    setBusy("bundle");
    try {
      const blob = await downloadResultsBundle();
      triggerBlob(blob, "sparc_results.zip");
      close();
    } catch (e) { onError(e); } finally { setBusy(null); }
  };

  return (
    <span
      data-export-skip="1"
      style={{ position: "relative", display: "inline-flex", alignItems: "center", gap: 6 }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={`Download options for ${label ?? artifactId}`}
        style={{
          fontSize: compact ? 10 : 11,
          padding: compact ? "2px 6px" : "4px 8px",
          border: "1px solid var(--line, #ddd)",
          borderRadius: 3,
          background: open ? "rgba(122, 58, 58, 0.1)" : "rgba(255,255,255,0.95)",
          color: "var(--ink, #333)",
          cursor: "pointer",
          fontFamily: "inherit",
        }}
      >
        ↓ Download
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            right: 0,
            minWidth: 220,
            background: "#fff",
            border: "1px solid var(--line, #ddd)",
            borderRadius: 4,
            boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
            zIndex: 50,
            padding: 4,
          }}
        >
          <div style={{ fontSize: 10, color: "#888", padding: "4px 8px", fontWeight: 600 }}>
            {label ?? artifactId}
          </div>
          <MenuItem
            label="Data (CSV/JSON)"
            disabled={!dataEndpoint || busy !== null}
            busy={busy === "data"}
            onClick={handleData}
          />
          <MenuItem
            label="Figure (PNG)"
            disabled={busy !== null}
            busy={busy === "png"}
            onClick={handlePng}
          />
          {targetRef && (
            <div style={{ padding: "4px 8px" }}>
              <ExportBlockButton
                targetRef={targetRef}
                artifactId={`${artifactId}_capture`}
                label={`${label ?? artifactId} capture`}
                compact
              />
            </div>
          )}
          {includeBundle && (
            <>
              <div style={{ height: 1, background: "var(--line, #eee)", margin: "4px 0" }} />
              <MenuItem
                label="Download all artifacts (ZIP)"
                disabled={busy !== null}
                busy={busy === "bundle"}
                onClick={handleBundle}
              />
            </>
          )}
          {err && (
            <div className="mono" style={{ padding: "4px 8px", fontSize: 9, color: "var(--crimson, #e73c25)" }}>
              {err}
            </div>
          )}
        </div>
      )}
    </span>
  );
}

function MenuItem({
  label,
  disabled,
  busy,
  onClick,
}: {
  label: string;
  disabled?: boolean;
  busy?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "6px 8px",
        background: "transparent",
        border: "none",
        fontSize: 11,
        color: disabled ? "#aaa" : "#222",
        cursor: disabled ? "not-allowed" : "pointer",
        fontFamily: "inherit",
        borderRadius: 3,
      }}
      onMouseEnter={(e) => {
        if (!disabled) (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,0,0,0.05)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = "transparent";
      }}
    >
      {busy ? "downloading…" : label}
    </button>
  );
}
