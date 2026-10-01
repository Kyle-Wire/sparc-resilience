// Feature folders never import each other (SPEC §17.1): the projects pages and their api
// modules import only foundation modules (api client/sse/binary/resource/types, components,
// charts, map, layouts, router, stores jobs/ui/selection, theme), their own api modules
// (api/projects, api/inputs) and their own folder.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "../../..");
const OWN = resolve(HERE, "..");

const ALLOWED: RegExp[] = [
  /^api\/(client|sse|binary|resource|types|projects|inputs)$/,
  /^components\/ui\/[A-Za-z]+$/,
  /^charts(\/[A-Za-z]+)?$/,
  /^map(\/[A-Za-z]+)*$/,
  /^layouts(\/[A-Za-z]+)?$/,
  /^router$/,
  /^stores\/(jobs|ui|selection)$/,
  /^theme\/[A-Za-z]+(\.css)?$/,
  /^test\/render$/,
  /^pages\/core\/routes$/,
];
const PACKAGES = new Set(["react", "react-dom", "react-dom/client", "zustand", "vitest", "node:fs", "node:path", "node:url"]);

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(ts|tsx)$/.test(name)) out.push(p);
  }
  return out;
}

/** Every module specifier of a source file (static imports, re-exports, dynamic imports). */
export function specifiers(source: string): string[] {
  const out: string[] = [];
  const re = /(?:^|\n)\s*(?:import|export)\s[^;]*?from\s+["']([^"']+)["']|(?:^|\n)\s*import\s+["']([^"']+)["']|import\(\s*["']([^"']+)["']\s*\)/g;
  for (let m = re.exec(source); m; m = re.exec(source)) out.push(m[1] ?? m[2] ?? m[3]);
  return out;
}

function violations(file: string): string[] {
  const bad: string[] = [];
  for (const spec of specifiers(readFileSync(file, "utf8"))) {
    if (!spec.startsWith(".")) {
      if (!PACKAGES.has(spec)) bad.push(spec);
      continue;
    }
    const target = resolve(dirname(file), spec);
    if (target === OWN || target.startsWith(OWN + sep)) continue; // own folder
    const rel = relative(SRC, target).split(sep).join("/");
    if (!ALLOWED.some((r) => r.test(rel))) bad.push(rel);
  }
  return bad;
}

describe("import boundaries", () => {
  const files = [...walk(OWN), join(SRC, "api/projects.ts"), join(SRC, "api/inputs.ts")];

  it("scans every file of the item", () => {
    expect(files.length).toBeGreaterThan(20);
    expect(specifiers('import { a } from "./x";\nimport type { B } from "../y";\nexport { c } from "../../z";\nconst L = lazy(() => import("./Home"));\nimport "./projects.css";')).toEqual([
      "./x",
      "../y",
      "../../z",
      "./Home",
      "./projects.css",
    ]);
  });

  it("imports only foundation modules, its own api modules and its own folder", () => {
    const found: Record<string, string[]> = {};
    for (const f of files) {
      const v = violations(f);
      if (v.length) found[relative(SRC, f)] = v;
    }
    expect(found).toEqual({});
  });

  it("would flag another feature folder or another item's api module", () => {
    const tmp = join(OWN, "Home.tsx");
    const fake = 'import { x } from "../tracking/hooks";\nimport { y } from "../../api/runs";\nimport { z } from "../../stores/tracker";';
    const bad = specifiers(fake).map((s) => relative(SRC, resolve(dirname(tmp), s)).split(sep).join("/"));
    expect(bad).toEqual(["pages/tracking/hooks", "api/runs", "stores/tracker"]);
    expect(bad.every((b) => !ALLOWED.some((r) => r.test(b)))).toBe(true);
  });
});
