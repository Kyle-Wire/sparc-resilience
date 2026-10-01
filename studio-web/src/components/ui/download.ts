// Client-side downloads and clipboard (chart/map export, CSV copy). Kept in one module so
// tests can observe what would be saved.

export type SavedFile = { name: string; blob: Blob };

let lastSaved: SavedFile | null = null;

/** The most recent download (tests and diagnostics). */
export function lastDownload(): SavedFile | null {
  return lastSaved;
}

export function downloadBlob(blob: Blob, name: string): void {
  lastSaved = { name, blob };
  if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.rel = "noopener";
  a.style.display = "none";
  document.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }
}

export function downloadText(text: string, name: string, type = "text/plain;charset=utf-8"): void {
  downloadBlob(new Blob([text], { type }), name);
}

/** Copy text; falls back to a hidden textarea when the async clipboard is unavailable. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = typeof document.execCommand === "function" ? document.execCommand("copy") : false;
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** File-name-safe slug: "Skill vs block size (°F)" → "skill-vs-block-size-f". */
export function fileSlug(s: string): string {
  return (
    s
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80) || "export"
  );
}
