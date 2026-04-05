/**
 * Native file/folder dialog wrappers using Tauri's dialog plugin.
 * Falls back to browser prompts when running outside Tauri (dev mode).
 */

let dialogModule: typeof import("@tauri-apps/plugin-dialog") | null = null;

async function getDialog() {
  if (dialogModule) return dialogModule;
  try {
    dialogModule = await import("@tauri-apps/plugin-dialog");
    return dialogModule;
  } catch {
    return null;
  }
}

/**
 * Open a native folder picker. Returns the selected folder path or null.
 */
export async function pickFolder(): Promise<string | null> {
  const dialog = await getDialog();
  if (dialog) {
    const result = await dialog.open({ directory: true, multiple: false });
    return result as string | null;
  }
  // Dev fallback: prompt
  return prompt("Enter folder path:");
}

/**
 * Open a native file picker filtered to YAML files. Returns selected path or null.
 */
export async function pickYaml(): Promise<string | null> {
  const dialog = await getDialog();
  if (dialog) {
    const result = await dialog.open({
      directory: false,
      multiple: false,
      filters: [{ name: "YAML", extensions: ["yml", "yaml"] }],
    });
    return result as string | null;
  }
  // Dev fallback
  return prompt("Enter path to project.yml:");
}

/**
 * Open a native file picker filtered to CSV files. Returns selected path or null.
 */
export async function pickCsv(): Promise<string | null> {
  const dialog = await getDialog();
  if (dialog) {
    const result = await dialog.open({
      directory: false,
      multiple: false,
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    return result as string | null;
  }
  // Dev fallback
  return prompt("Enter path to CSV file:");
}

/**
 * Open a native file picker for raster files (GeoTIFF). Returns selected paths or null.
 */
export async function pickRasters(): Promise<string[] | null> {
  const dialog = await getDialog();
  if (dialog) {
    const result = await dialog.open({
      directory: false,
      multiple: true,
      filters: [{ name: "GeoTIFF", extensions: ["tif", "tiff"] }],
    });
    if (!result) return null;
    return Array.isArray(result) ? (result as string[]) : [result as string];
  }
  const input = prompt("Enter raster paths (comma-separated):");
  return input ? input.split(",").map((s) => s.trim()) : null;
}

/**
 * Open a native file picker for boundary files (Shapefile, GPKG, GeoJSON).
 */
export async function pickBoundary(): Promise<string | null> {
  const dialog = await getDialog();
  if (dialog) {
    const result = await dialog.open({
      directory: false,
      multiple: false,
      filters: [
        { name: "Spatial", extensions: ["shp", "gpkg", "geojson"] },
      ],
    });
    return result as string | null;
  }
  return prompt("Enter path to boundary file (.shp, .gpkg, .geojson):");
}
