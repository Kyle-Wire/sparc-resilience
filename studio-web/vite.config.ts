/// <reference types="vitest/config" />
// Vite config for SPARC Studio (SPEC §12.1, §13.2, §13.3).
//
// - `npm run build` writes hashed assets to ../sparc/studio/static (emptied first) plus
//   BUILD_INFO.json = {src_sha256, vite, react}. There is no timestamp, so rebuilding
//   unchanged sources gives byte-identical output (the CI rebuild-diff relies on it).
// - `npm run dev` serves on 5173 and proxies /api, /auth and /openapi.json to the Studio
//   server (default http://127.0.0.1:8765, override with SPARC_STUDIO_URL). SSE responses
//   pass straight through (no buffering, no compression).
// - Vitest runs every src/**/*.test.{ts,tsx} in jsdom with src/test/setup.ts.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, version as viteVersion, type Plugin } from "vite";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const API_TARGET = process.env.SPARC_STUDIO_URL ?? "http://127.0.0.1:8765";

function walk(dir: string, out: string[]): void {
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue; // editor and OS junk never counts
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
}

/**
 * Source hash recorded in BUILD_INFO.json and re-computed by scripts/check_studio_assets.py.
 * Inputs: every file under src/ (dotfiles skipped), package-lock.json, vite.config.ts and
 * index.html. Paths are POSIX, relative to studio-web/, sorted by code point. The digest is
 * sha256 over the concatenation of `relpath + "\0" + bytes + "\0"` for each file.
 */
export function sourceHash(root: string = ROOT): string {
  const files: string[] = [];
  walk(join(root, "src"), files);
  for (const f of ["package-lock.json", "vite.config.ts", "index.html"]) files.push(join(root, f));
  const rel = files.map((f) => relative(root, f).split(sep).join("/")).sort();
  const h = createHash("sha256");
  for (const r of rel) {
    h.update(r, "utf8");
    h.update("\0");
    h.update(readFileSync(join(root, r)));
    h.update("\0");
  }
  return h.digest("hex");
}

function buildInfo(): Plugin {
  return {
    name: "sparc-build-info",
    apply: "build",
    generateBundle() {
      const reactPkg = JSON.parse(readFileSync(join(ROOT, "node_modules/react/package.json"), "utf8")) as { version: string };
      const info = { src_sha256: sourceHash(), vite: viteVersion, react: reactPkg.version };
      this.emitFile({ type: "asset", fileName: "BUILD_INFO.json", source: JSON.stringify(info, null, 2) + "\n" });
    },
  };
}

const proxyEntry = {
  target: API_TARGET,
  // Keep Host/Origin as the browser sent them; the server allows them via --dev-origin.
  changeOrigin: false,
  configure(proxy: { on(event: "proxyRes", cb: (res: { headers: Record<string, string | string[] | undefined> }) => void): void }) {
    proxy.on("proxyRes", (res) => {
      const ct = String(res.headers["content-type"] ?? "");
      if (ct.startsWith("text/event-stream")) {
        // Never let an intermediary buffer or transform an event stream.
        res.headers["cache-control"] = "no-store, no-transform";
        res.headers["x-accel-buffering"] = "no";
      }
    });
  },
};

export default defineConfig({
  root: ROOT,
  base: "/",
  plugins: [react(), buildInfo()],
  build: {
    outDir: "../sparc/studio/static",
    emptyOutDir: true,
    assetsDir: "assets",
    target: "es2022",
    sourcemap: false,
    // Fonts are always emitted as files (never inlined as data: URIs) so the CSS stays small.
    assetsInlineLimit: (file) => (/\.(woff2?|ttf)$/.test(file) ? false : undefined),
    chunkSizeWarningLimit: 600,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": proxyEntry, "/auth": proxyEntry, "/openapi.json": proxyEntry },
  },
  preview: {
    port: 4173,
    proxy: { "/api": proxyEntry, "/auth": proxyEntry, "/openapi.json": proxyEntry },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setup.ts"],
    pool: "threads",
    maxWorkers: 2,
    testTimeout: 20000,
  },
});
