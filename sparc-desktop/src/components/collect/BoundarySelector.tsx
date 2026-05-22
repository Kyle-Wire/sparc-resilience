/**
 * BoundarySelector — lets the researcher define the study boundary via:
 *   1. Place-name text search (Nominatim)
 *   2. Local file upload (shapefile/GeoJSON/GeoPackage)
 *   3. Bounding-box draw (manual lat/lon coordinate entry)
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

  // Draw-mode bbox state
  const [minLon, setMinLon] = useState("");
  const [minLat, setMinLat] = useState("");
  const [maxLon, setMaxLon] = useState("");
  const [maxLat, setMaxLat] = useState("");

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

  async function handleBboxApply() {
    const vals = [minLon, minLat, maxLon, maxLat].map(Number);
    if (vals.some(isNaN)) {
      setError("All four coordinate fields are required and must be numeric.");
      return;
    }
    const [mLon, mLat, xLon, xLat] = vals;
    if (mLon >= xLon || mLat >= xLat) {
      setError("Min values must be less than max values.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await collectBoundary({ bbox: [mLon, mLat, xLon, xLat] });
      onBoundaryChange(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set boundary");
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
          <div>
            <p style={{ marginBottom: 10, fontSize: 12, color: "var(--ink-muted)" }}>
              Enter bounding-box coordinates in decimal degrees (WGS-84).
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
              {(
                [
                  ["Min Longitude (W)", minLon, setMinLon, "-180", "180"],
                  ["Min Latitude (S)", minLat, setMinLat, "-90", "90"],
                  ["Max Longitude (E)", maxLon, setMaxLon, "-180", "180"],
                  ["Max Latitude (N)", maxLat, setMaxLat, "-90", "90"],
                ] as [string, string, (v: string) => void, string, string][]
              ).map(([label, value, setter, min, max]) => (
                <label key={label} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "var(--ink)" }}>{label}</span>
                  <input
                    type="number"
                    value={value}
                    min={min}
                    max={max}
                    step="any"
                    onChange={(e) => setter(e.target.value)}
                    placeholder="e.g. -112.07"
                    style={{
                      padding: "6px 10px",
                      border: "1px solid var(--border)",
                      borderRadius: 6,
                      fontSize: 13,
                      background: "var(--surface)",
                      color: "var(--ink)",
                    }}
                  />
                </label>
              ))}
            </div>
            <Btn primary onClick={handleBboxApply} disabled={loading || !minLon || !minLat || !maxLon || !maxLat}>
              {loading ? "Applying…" : "Apply bounding box"}
            </Btn>
          </div>
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
