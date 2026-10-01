// Map dotted config paths ("actionable.Pct_Canopy.doses.2") to YAML line numbers, so the
// config editor can put `/config/validate` issues in the CodeArea gutter. A small line
// scanner, not a YAML parser: block mappings, block sequences (indented or PyYAML's compact
// "- " at the key's indent), "- key: value" items, flow lists/maps on one line, quoted keys,
// comments and block scalars (|, >). Paths are relative to the `core:` block when the file
// has one (config.yml always does).

type Frame = { indent: number; path: string[]; kind: "key" | "item" };

const KEY_RE = /^("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^\s#'"{[\-?:][^#]*?|-[^\s#][^#]*?)\s*:(?:\s+(.*))?$/;

function unquote(k: string): string {
  if (k.startsWith('"') && k.endsWith('"')) return k.slice(1, -1).replace(/\\"/g, '"');
  if (k.startsWith("'") && k.endsWith("'")) return k.slice(1, -1).replace(/''/g, "'");
  return k.trim();
}

/** The value part without a trailing comment (a " #" outside quotes). */
function stripComment(v: string): string {
  let q: string | null = null;
  for (let i = 0; i < v.length; i++) {
    const c = v[i];
    if (q) {
      if (c === q) q = null;
    } else if (c === '"' || c === "'") q = c;
    else if (c === "#" && (i === 0 || /\s/.test(v[i - 1]))) return v.slice(0, i).trimEnd();
  }
  return v.trimEnd();
}

/** Split the inside of a one-line flow collection at top-level commas. */
function splitFlow(inner: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let q: string | null = null;
  let cur = "";
  for (const c of inner) {
    if (q) {
      if (c === q) q = null;
      cur += c;
      continue;
    }
    if (c === '"' || c === "'") q = c;
    else if (c === "[" || c === "{") depth++;
    else if (c === "]" || c === "}") depth--;
    if (c === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** Record paths inside a one-line flow value (`[0, 5, 10]`, `{a: 1, b: [2]}`). */
function indexFlow(value: string, path: string[], line: number, out: Map<string, number>): void {
  const v = value.trim();
  if (v.startsWith("[") && v.endsWith("]")) {
    splitFlow(v.slice(1, -1)).forEach((el, i) => {
      const p = [...path, String(i)];
      if (!out.has(p.join("."))) out.set(p.join("."), line);
      indexFlow(el, p, line, out);
    });
  } else if (v.startsWith("{") && v.endsWith("}")) {
    for (const el of splitFlow(v.slice(1, -1))) {
      const m = /^("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^:]+?)\s*:\s*(.*)$/.exec(el);
      if (!m) continue;
      const p = [...path, unquote(m[1])];
      if (!out.has(p.join("."))) out.set(p.join("."), line);
      indexFlow(m[2], p, line, out);
    }
  }
}

/** Every path → its 1-based line (as written in the file, `core.` prefix included). */
export function yamlPathIndex(text: string): Map<string, number> {
  const out = new Map<string, number>();
  const lines = text.split("\n");
  const stack: Frame[] = [];
  const counters = new Map<string, number>();
  let blockIndent: number | null = null; // inside a block scalar: skip lines indented deeper
  const record = (path: string[], line: number) => {
    const k = path.join(".");
    if (!out.has(k)) out.set(k, line);
  };
  const keyLine = (content: string, indent: number, parent: string[], line: number) => {
    const m = KEY_RE.exec(content);
    if (!m) return;
    const path = [...parent, unquote(m[1])];
    record(path, line);
    counters.delete(path.join("."));
    stack.push({ indent, path, kind: "key" });
    const value = stripComment(m[2] ?? "");
    if (/^[|>][0-9+-]*$/.test(value)) blockIndent = indent;
    else indexFlow(value, path, line, out);
  };
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].replace(/\t/g, "  ").replace(/\r$/, "");
    const content = raw.trim();
    const indent = raw.length - raw.trimStart().length;
    if (blockIndent !== null) {
      if (content === "" || indent > blockIndent) continue;
      blockIndent = null;
    }
    if (content === "" || content.startsWith("#") || content === "---" || content === "...") continue;
    const line = i + 1;
    if (content === "-" || content.startsWith("- ")) {
      while (stack.length && (stack[stack.length - 1].indent > indent || (stack[stack.length - 1].indent === indent && stack[stack.length - 1].kind === "item"))) stack.pop();
      const parent = stack.length ? stack[stack.length - 1].path : [];
      const pk = parent.join(".");
      const idx = counters.get(pk) ?? 0;
      counters.set(pk, idx + 1);
      const itemPath = [...parent, String(idx)];
      record(itemPath, line);
      stack.push({ indent, path: itemPath, kind: "item" });
      const rest = content.slice(1).trimStart();
      if (!rest) continue;
      const restIndent = indent + (content.length - rest.length);
      if (KEY_RE.test(rest) && !rest.startsWith("[") && !rest.startsWith("{")) keyLine(rest, restIndent, itemPath, line);
      else indexFlow(stripComment(rest), itemPath, line, out);
      continue;
    }
    while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
    keyLine(content, indent, stack.length ? stack[stack.length - 1].path : [], line);
  }
  return out;
}

/**
 * Line of a dotted path (relative to the `core:` block when present). Falls back to the
 * nearest indexed ancestor, then to null.
 */
export function locatePath(index: Map<string, number>, path: string): number | null {
  const hasCore = index.has("core");
  const parts = path ? path.split(".") : [];
  for (let n = parts.length; n >= 0; n--) {
    const p = parts.slice(0, n).join(".");
    const full = hasCore ? (p ? `core.${p}` : "core") : p;
    const hit = index.get(full) ?? (hasCore ? index.get(p) : undefined);
    if (hit !== undefined) return hit;
  }
  return null;
}
