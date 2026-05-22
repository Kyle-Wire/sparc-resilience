/**
 * BoundarySelector — lets the researcher define the study boundary via:
 *   1. Place-name text search (Nominatim)
 *   2. Local file upload (shapefile/GeoJSON/GeoPackage)
 *   3. Bounding-box draw (future: free-hand polygon draw)
 */
import { useState, useRef } from "react";
import { Btn } from "@/components/ui/DesignSystem";
import { collectBoundary } from "@/lib/api";
import type { BoundaryResponse } from "@/lib/api";

interface Props {
  onBoundaryChange: (result: BoundaryResponse) => void;
}

type Tab = "search" | "file" | "draw";

export default function BoundarySelector({ onBoundaryChange }: Props) {
  const [tab, setTab] = useState<Tab>("search");
  const [placeName, setPlaceName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSearch() {
    if (!placeName.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await collectBoundary({ place_name: placeName.trim() });
      onBoundaryChange(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to resolve boundary");
    } finally {
      setLoading(false);
    }
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Use the absolute file path (Tauri provides it via the file object path)
    const filePath = (file as { path?: string }).path ?? file.name;
    setLoading(true);
    setError(null);
    try {
      const result = await collectBoundary({ file_path: filePath });
      onBoundaryChange(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to read boundary file");
    } finally {
      setLoading(false);
    }
  }

  const tabStyle = (t: Tab): React.CSSProperties => ({
    padding: "6px 14px",
    border: "none",
    borderBottom: tab === t ? "2px solid var(--sparc-purple)" : "2px solid transparent",
    background: "transparent",
    cursor: "pointer",
    fontWeight: tab === t ? 700 : 400,
    color: tab === t ? "var(--sparc-purple)" : "var(--ink)",
    fontSize: 13,
  });

  return (
    <div>
      {/* Tab bar */}
      <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--border)" }}>
        <button style={tabStyle("search")} onClick={() => setTab("search")}>Place Search</button>
        <button style={tabStyle("file")} onClick={() => setTab("file")}>File Upload</button>
        <button style={tabStyle("draw")} onClick={() => setTab("draw")}>Draw Bounds</button>
      </div>

      <div style={{ paddingTop: 16 }}>
        {tab === "search" && (
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input
              type="text"
              value={placeName}
              onChange={(e) => setPlaceName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. Phoenix, AZ"
              style={{
                flex: 1,
                padding: "8px 12px",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontSize: 14,
                background: "var(--surface)",
                color: "var(--ink)",
              }}
            />
            <Btn primary onClick={handleSearch} disabled={loading || !placeName.trim()}>
              {loading ? "Searching…" : "Search"}
            </Btn>
          </div>
        )}

        {tab === "file" && (
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".geojson,.json,.gpkg,.shp,.zip"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <Btn onClick={() => fileInputRef.current?.click()} disabled={loading}>
              {loading ? "Loading…" : "Choose boundary file…"}
            </Btn>
            <p style={{ marginTop: 8, fontSize: 12, color: "var(--ink-muted)" }}>
              Accepts GeoJSON, GeoPackage, Shapefile (ZIP), or .shp
            </p>
          </div>
        )}

        {tab === "draw" && (
          <p style={{ fontSize: 13, color: "var(--ink-muted)" }}>
            Draw mode coming soon — use the map controls to draw a bounding box,
            then click <strong>Use Selection</strong>.
          </p>
        )}

        {error && (
          <p style={{ marginTop: 10, color: "var(--error, #c0392b)", fontSize: 13 }}>
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
