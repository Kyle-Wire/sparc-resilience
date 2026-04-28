/**
 * DownloadMenu — uniform per-block download dropdown.
 *
 * Actions:
 *  - **Source artifact (DB)**: GET /artifacts/{stage}/{id}/download?fmt=…
 *    pulls the canonical bytes straight out of artifacts.db, with a
 *    format selector when multiple are sensible (csv/parquet for tables,
 *    joblib/pkl for blobs).  This is the preferred path now that the
 *    pipeline writes db-only by default.
 *  - **Data (legacy endpoint)**: optional; fetch a /results/* endpoint and
 *    save its CSV/JSON/GeoJSON.  Kept for views that compose multiple
 *    artifacts server-side.
 *  - **Figure (PNG)**: GET /artifacts/{stage}/{id}.png — registry-backed
 *    rendered figure (server matplotlib + content-hash cache).
 *  - **Capture (PNG)**: client-side DOM screenshot via ExportBlockButton.
 *  - **Stage as ZIP**: POST /artifacts/{stage}/export — bundle every
 *    artifact for the stage.
 *  - **Download all (run bundle)**: /results/bundle ZIP.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { ExportBlockButton } from "./ExportBlockButton";
import {
  downloadResultsBundle,
  downloadDbArtifact,
  downloadStageZip,
  listStageArtifacts,
  type DbArtifactInfo,
} from "@/lib/api";

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

  // DB-resident artifact metadata (populated when the menu opens).
  const [dbInfo, setDbInfo] = useState<DbArtifactInfo | null>(null);
  const [dbFmt, setDbFmt] = useState<string>("");
  const probedRef = useRef(false);

  // Resolve which formats are sensible for the artifact's storage_kind.
  const dbFormats = (() => {
    if (!dbInfo) return [] as string[];
    switch (dbInfo.storage_kind) {
      case "table":
        return ["csv", "parquet"];
      case "struct":
        return ["json"];
      case "blob_inline":
      case "blob_external":
        return ["joblib", "pkl"];
      default:
        return []; // legacy_path — server picks native format
    }
  })();

  // Probe artifacts.db for this artifact the first time the menu opens.
  useEffect(() => {
    if (!open || probedRef.current) return;
    probedRef.current = true;
    listStageArtifacts(stage)
      .then((info) => {
        const entry = info.artifacts.find((a) => a.artifact_id === artifactId);
        if (entry) {
          setDbInfo(entry);
          // Default to first sensible fmt.
          const opts =
            entry.storage_kind === "table"
              ? ["csv", "parquet"]
              : entry.storage_kind === "struct"
              ? ["json"]
              : entry.storage_kind === "blob_inline" || entry.storage_kind === "blob_external"
              ? ["joblib", "pkl"]
              : [];
          setDbFmt(opts[0] ?? "");
        }
      })
      .catch(() => {
        /* artifact may not be db-resident; "Source artifact (DB)" stays disabled */
      });
  }, [open, stage, artifactId]);

  const close = () => setOpen(false);
  const onError = (e: unknown) => {
    setErr(e instanceof Error ? e.message : String(e));
    setTimeout(() => setErr(null), 4000);
  };

  const handleDb = () => {
    if (!dbInfo) return;
    setBusy("db");
    try {
      downloadDbArtifact(stage, artifactId, (dbFmt || undefined) as never);
      close();
    } catch (e) { onError(e); } finally { setBusy(null); }
  };

  const handleStageZip = async () => {
    setBusy("stagezip");
    try {
      await downloadStageZip(stage);
      close();
    } catch (e) { onError(e); } finally { setBusy(null); }
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

          {/* DB-resident source bytes (preferred). */}
          <div style={{ padding: "4px 8px", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 11, color: dbInfo ? "#222" : "#aaa", flex: 1 }}>
              Source artifact (DB)
            </span>
            {dbFormats.length > 1 && (
              <select
                value={dbFmt}
                onChange={(e) => setDbFmt(e.target.value)}
                disabled={busy !== null}
                style={{
                  fontSize: 10,
                  padding: "1px 4px",
                  border: "1px solid var(--line, #ddd)",
                  borderRadius: 3,
                  background: "#fff",
                  fontFamily: "inherit",
                }}
              >
                {dbFormats.map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            )}
            <button
              type="button"
              onClick={handleDb}
              disabled={!dbInfo || busy !== null}
              style={{
                fontSize: 10,
                padding: "2px 6px",
                border: "1px solid var(--line, #ddd)",
                borderRadius: 3,
                background: dbInfo ? "rgba(255,255,255,0.95)" : "#f4f4f4",
                color: dbInfo ? "var(--ink, #333)" : "#aaa",
                cursor: dbInfo ? "pointer" : "not-allowed",
                fontFamily: "inherit",
              }}
            >
              {busy === "db" ? "…" : "Get"}
            </button>
          </div>

          <MenuItem
            label="Data (legacy endpoint)"
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
                label={`Stage ${stage} as ZIP`}
                disabled={busy !== null}
                busy={busy === "stagezip"}
                onClick={handleStageZip}
              />
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
