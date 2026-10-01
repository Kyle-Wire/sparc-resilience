// History-API router and route registry (SPEC §3.3, §12.3).
//
// Each feature folder declares its routes in `src/pages/<group>/routes.ts`:
//
//   export const routes: RouteDef[] = [
//     { path: "/r/:rid/accuracy", component: lazy(() => import("./Accuracy")), title: "Accuracy",
//       runTab: { id: "accuracy", label: "Accuracy", group: "Model", order: 30 } },
//   ];
//
// The registry collects them with `import.meta.glob("./pages/*/routes.ts", {eager: true})`;
// folders that do not exist yet contribute nothing. Route modules should import only types
// from this file at module top level (values are fine inside components and functions).
//
// The URL holds all navigation state: path params plus query parameters read and written
// with `useUrlState(key, codec)`. Every view is deep-linkable and reload-safe.
import {
  createContext,
  createElement,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
  type AnchorHTMLAttributes,
  type ComponentType,
  type LazyExoticComponent,
  type MouseEvent,
  type ReactNode,
} from "react";
import { RUN_TAB_IDS, type Project, type RunTabId } from "./api/types";

export type { RunTabId } from "./api/types";

// ---------------------------------------------------------------- route definitions

export type RunTabGroup = "Model" | "Effects" | "Decisions" | "Trust" | "Run";
export const RUN_TAB_GROUPS: readonly RunTabGroup[] = ["Model", "Effects", "Decisions", "Trust", "Run"];

export type RunTabDef = { id: RunTabId; label: string; group: RunTabGroup; order: number };

export type ProjectNavDef = {
  id: string;
  label: string;
  order: number;
  /** Target URL for this project, or null to hide the entry. */
  to: (pid: string, project: Project) => string | null;
  /** Non-null greys the entry out with this text as its tooltip. */
  disabledReason?: (project: Project) => string | null;
};

export type RouteDef = {
  path: string;
  // Pages take no props (they read params with useRoute), so any component type is accepted.
  component: LazyExoticComponent<ComponentType<any>>;
  title: string;
  runTab?: RunTabDef;
  projectNav?: ProjectNavDef;
  /** Use the full content width (Mission Control, Map, Lab). */
  fullWidth?: boolean;
};

export type RouteModule = { routes?: RouteDef[] };

// ---------------------------------------------------------------- matching

type Seg = { kind: "static"; value: string } | { kind: "param"; name: string; optional: boolean } | { kind: "wild" };

export type CompiledRoute = { def: RouteDef; segs: Seg[] };

export type RouteMatch = { route: RouteDef; params: Record<string, string> };

function splitPath(p: string): string[] {
  return p.split("/").filter((s) => s.length > 0);
}

export function compilePath(path: string): Seg[] {
  if (path === "*") return [{ kind: "wild" }];
  return splitPath(path).map((s): Seg => {
    if (s === "*") return { kind: "wild" };
    if (s.startsWith(":")) {
      const optional = s.endsWith("?");
      return { kind: "param", name: s.slice(1, optional ? -1 : undefined), optional };
    }
    return { kind: "static", value: s };
  });
}

const RANK = { static: 3, param: 2, optional: 1, wild: 0 } as const;

function segRank(s: Seg | undefined): number {
  if (!s) return -1;
  if (s.kind === "param") return s.optional ? RANK.optional : RANK.param;
  return RANK[s.kind];
}

/** More specific first: compare segment by segment (static > param > optional > *), then length. */
function compareSpecificity(a: Seg[], b: Seg[]): number {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = segRank(b[i]) - segRank(a[i]);
    if (d !== 0) return d;
  }
  return 0;
}

function decode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

function matchSegs(segs: Seg[], parts: string[]): Record<string, string> | null {
  const params: Record<string, string> = {};
  let i = 0;
  for (let k = 0; k < segs.length; k++) {
    const s = segs[k];
    if (s.kind === "wild") {
      params["*"] = parts.slice(i).map(decode).join("/");
      return params;
    }
    const part = parts[i];
    if (part === undefined) {
      if (s.kind === "param" && s.optional) continue;
      return null;
    }
    if (s.kind === "static") {
      if (part !== s.value) return null;
    } else {
      params[s.name] = decode(part);
    }
    i++;
  }
  return i === parts.length ? params : null;
}

const hasWild = (segs: Seg[]) => segs.some((s) => s.kind === "wild");

/** Routes in match order: wildcard routes last, then by segment specificity. */
export function compileRoutes(routes: RouteDef[]): CompiledRoute[] {
  return routes
    .map((def) => ({ def, segs: compilePath(def.path) }))
    .sort((a, b) => Number(hasWild(a.segs)) - Number(hasWild(b.segs)) || compareSpecificity(a.segs, b.segs));
}

/** Best match for a pathname (the most specific pattern wins), or null. */
export function matchRoute(compiled: CompiledRoute[], pathname: string): RouteMatch | null {
  const parts = splitPath(pathname);
  for (const c of compiled) {
    const params = matchSegs(c.segs, parts);
    if (params) return { route: c.def, params };
  }
  return null;
}

/** Fill a pattern: `/r/:rid/docs/:doc?` + {rid: "x"} → `/r/x/docs`. Unknown required params throw. */
export function fillPath(pattern: string, params: Record<string, string | number | null | undefined>): string {
  const out: string[] = [];
  for (const s of compilePath(pattern)) {
    if (s.kind === "static") out.push(s.value);
    else if (s.kind === "wild") {
      const w = params["*"];
      if (w !== undefined && w !== null && w !== "") out.push(String(w));
    } else {
      const v = params[s.name];
      if (v === undefined || v === null || v === "") {
        if (s.optional) continue;
        throw new Error(`fillPath: missing :${s.name} for ${pattern}`);
      }
      out.push(encodeURIComponent(String(v)));
    }
  }
  return "/" + out.join("/");
}

// ---------------------------------------------------------------- registry

export type RunTabEntry = RunTabDef & { route: RouteDef };
export type ProjectNavEntry = ProjectNavDef & { route: RouteDef };

export type Registry = {
  routes: RouteDef[];
  compiled: CompiledRoute[];
  /** Run tabs in display order: by group (Model, Effects, Decisions, Trust, Run), then `order`. */
  runTabs: RunTabEntry[];
  projectNav: ProjectNavEntry[];
  problems: string[];
};

/** Build the registry from `import.meta.glob(..., {eager: true})` output. */
export function buildRegistry(modules: Record<string, RouteModule>): Registry {
  const routes: RouteDef[] = [];
  const problems: string[] = [];
  const seenPath = new Set<string>();
  const seenTab = new Map<string, string>();
  const seenNav = new Map<string, string>();
  const runTabs: RunTabEntry[] = [];
  const projectNav: ProjectNavEntry[] = [];
  for (const file of Object.keys(modules).sort()) {
    const list = modules[file]?.routes;
    if (!Array.isArray(list)) continue;
    for (const r of list) {
      if (seenPath.has(r.path)) {
        problems.push(`${file}: duplicate route ${r.path}`);
        continue;
      }
      seenPath.add(r.path);
      routes.push(r);
      if (r.runTab) {
        if (!RUN_TAB_IDS.includes(r.runTab.id)) problems.push(`${file}: unknown run tab id "${r.runTab.id}"`);
        else if (!RUN_TAB_GROUPS.includes(r.runTab.group)) problems.push(`${file}: unknown run tab group "${r.runTab.group}"`);
        else if (seenTab.has(r.runTab.id)) problems.push(`${file}: run tab "${r.runTab.id}" already declared by ${seenTab.get(r.runTab.id)}`);
        else {
          seenTab.set(r.runTab.id, file);
          runTabs.push({ ...r.runTab, route: r });
        }
      }
      if (r.projectNav) {
        if (seenNav.has(r.projectNav.id)) problems.push(`${file}: nav entry "${r.projectNav.id}" already declared by ${seenNav.get(r.projectNav.id)}`);
        else {
          seenNav.set(r.projectNav.id, file);
          projectNav.push({ ...r.projectNav, route: r });
        }
      }
    }
  }
  runTabs.sort((a, b) => RUN_TAB_GROUPS.indexOf(a.group) - RUN_TAB_GROUPS.indexOf(b.group) || a.order - b.order || a.label.localeCompare(b.label));
  projectNav.sort((a, b) => a.order - b.order || a.label.localeCompare(b.label));
  if (problems.length && typeof console !== "undefined") for (const p of problems) console.warn(`[routes] ${p}`);
  return { routes, compiled: compileRoutes(routes), runTabs, projectNav, problems };
}

/** The app registry: every `src/pages/<group>/routes.ts`. */
export const registry: Registry = buildRegistry(import.meta.glob<RouteModule>("./pages/*/routes.ts", { eager: true }));

const RegistryContext = createContext<Registry>(registry);

/** Override the registry for a subtree (tests and storybook-like harnesses). */
export function RegistryProvider(props: { registry: Registry; children?: ReactNode }) {
  return createElement(RegistryContext.Provider, { value: props.registry }, props.children);
}

export function useRegistry(): Registry {
  return useContext(RegistryContext);
}

/** `/r/<rid>/<tab>` for a run tab (optional params and wildcards dropped). */
export function runTabHref(tab: RunTabEntry, rid: string): string {
  return fillPath(tab.route.path, { rid });
}

/**
 * The run tab a route belongs to: its own `runTab`, else the tab whose path is the longest
 * prefix of the route's pattern (e.g. `/r/:rid/lab/library` → `lab`).
 */
export function runTabForRoute(reg: Registry, route: RouteDef | null): RunTabEntry | null {
  if (!route) return null;
  const own = reg.runTabs.find((t) => t.route === route);
  if (own) return own;
  const parts = compilePath(route.path);
  let best: RunTabEntry | null = null;
  let bestLen = -1;
  for (const t of reg.runTabs) {
    const tp = compilePath(t.route.path).filter((s) => !(s.kind === "param" && s.optional) && s.kind !== "wild");
    if (tp.length > parts.length) continue;
    const ok = tp.every((s, i) => {
      const p = parts[i];
      if (s.kind === "static") return p.kind === "static" && p.value === s.value;
      return p.kind === "param";
    });
    if (ok && tp.length > bestLen) {
      best = t;
      bestLen = tp.length;
    }
  }
  return best && bestLen > 2 ? best : null;
}

// ---------------------------------------------------------------- location store

export type Location = { pathname: string; search: string; hash: string; key: string };

type HistState = { __key?: string; [k: string]: unknown } | null;

let keySeq = 0;
const newKey = () => `${Date.now().toString(36)}-${(keySeq++).toString(36)}`;

function readLocation(): Location {
  if (typeof window === "undefined") return { pathname: "/", search: "", hash: "", key: "ssr" };
  const st = window.history.state as HistState;
  let key = st?.__key;
  if (!key) {
    key = newKey();
    try {
      window.history.replaceState({ ...(st ?? {}), __key: key }, "");
    } catch {
      /* sandboxed */
    }
  }
  return { pathname: window.location.pathname, search: window.location.search, hash: window.location.hash, key };
}

let current: Location = readLocation();
const listeners = new Set<() => void>();
const scrollPositions = new Map<string, number>();

function emit(): void {
  current = readLocation();
  for (const l of [...listeners]) l();
}

if (typeof window !== "undefined") {
  try {
    window.history.scrollRestoration = "manual";
  } catch {
    /* not supported */
  }
  window.addEventListener("popstate", () => {
    emit();
    const y = scrollPositions.get(current.key) ?? 0;
    const restore = () => window.scrollTo?.(0, y);
    if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(restore);
    else restore();
  });
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getLocation(): Location {
  return current;
}

export type NavigateOptions = { replace?: boolean; state?: Record<string, unknown>; scroll?: boolean };

/** Navigate within the SPA (pushState by default). Absolute http(s) URLs fall back to a page load. */
export function navigate(to: string, opts: NavigateOptions = {}): void {
  if (typeof window === "undefined") return;
  if (/^[a-z]+:\/\//i.test(to)) {
    window.location.assign(to);
    return;
  }
  const url = new URL(to, window.location.href);
  const target = url.pathname + url.search + url.hash;
  const here = window.location.pathname + window.location.search + window.location.hash;
  try {
    scrollPositions.set(current.key, window.scrollY ?? 0);
  } catch {
    /* ignore */
  }
  const state = { ...(opts.state ?? {}), __key: newKey() };
  if (opts.replace || target === here) window.history.replaceState(state, "", target);
  else window.history.pushState(state, "", target);
  emit();
  if (opts.scroll !== false && !opts.replace) {
    if (url.hash) document.getElementById(decode(url.hash.slice(1)))?.scrollIntoView?.();
    else window.scrollTo?.(0, 0);
  }
}

export function useLocation(): Location {
  return useSyncExternalStore(subscribe, getLocation, getLocation);
}

export type RouteState = {
  pathname: string;
  query: URLSearchParams;
  hash: string;
  params: Record<string, string>;
  route: RouteDef | null;
};

/** Current route, params and query (re-renders on navigation). */
export function useRoute(): RouteState {
  const loc = useLocation();
  const reg = useRegistry();
  return useMemo(() => {
    const m = matchRoute(reg.compiled, loc.pathname);
    return { pathname: loc.pathname, query: new URLSearchParams(loc.search), hash: loc.hash, params: m?.params ?? {}, route: m?.route ?? null };
  }, [loc.pathname, loc.search, loc.hash, reg]);
}

/** Merge query changes into the current URL (null removes a key). */
export function setQuery(patch: Record<string, string | null | undefined>, opts: { push?: boolean } = {}): void {
  if (typeof window === "undefined") return;
  const q = new URLSearchParams(window.location.search);
  for (const [k, v] of Object.entries(patch)) {
    if (v === null || v === undefined || v === "") q.delete(k);
    else q.set(k, v);
  }
  const s = q.toString();
  navigate(window.location.pathname + (s ? "?" + s : "") + window.location.hash, { replace: !opts.push, scroll: false });
}

// ---------------------------------------------------------------- query codecs

export type Codec<T> = { parse: (raw: string | null) => T; format: (value: T) => string | null };

function b64urlEncode(s: string): string {
  const bytes = new TextEncoder().encode(s);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): string {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/") + pad);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

export const codecs = {
  string(def = ""): Codec<string> {
    return { parse: (r) => (r === null ? def : r), format: (v) => (v === def ? null : v) };
  },
  optString(): Codec<string | null> {
    return { parse: (r) => (r === null || r === "" ? null : r), format: (v) => (v === null || v === "" ? null : v) };
  },
  int(def: number): Codec<number> {
    return {
      parse: (r) => {
        if (r === null || !/^-?\d+$/.test(r.trim())) return def;
        return parseInt(r, 10);
      },
      format: (v) => (v === def ? null : String(Math.trunc(v))),
    };
  },
  float(def: number | null): Codec<number | null> {
    return {
      parse: (r) => {
        if (r === null || r.trim() === "") return def;
        const v = Number(r);
        return Number.isFinite(v) ? v : def;
      },
      format: (v) => (v === def || v === null ? null : String(v)),
    };
  },
  bool(def = false): Codec<boolean> {
    return {
      parse: (r) => (r === null ? def : r === "1" || r === "true" || r === "yes"),
      format: (v) => (v === def ? null : v ? "1" : "0"),
    };
  },
  enum<T extends string>(values: readonly T[], def: T): Codec<T> {
    return { parse: (r) => (r !== null && (values as readonly string[]).includes(r) ? (r as T) : def), format: (v) => (v === def ? null : v) };
  },
  /** Comma-separated list (items may not contain commas). */
  list(): Codec<string[]> {
    return {
      parse: (r) => (r === null || r === "" ? [] : r.split(",").map(decode).filter((s) => s.length > 0)),
      format: (v) => (v.length ? v.map((s) => s.replace(/,/g, "%2C")).join(",") : null),
    };
  },
  /** Any JSON value, URL-safe base64 encoded. */
  json<T>(def: T): Codec<T> {
    return {
      parse: (r) => {
        if (r === null || r === "") return def;
        try {
          return JSON.parse(b64urlDecode(r)) as T;
        } catch {
          return def;
        }
      },
      format: (v) => (v === def || v === null || v === undefined ? null : b64urlEncode(JSON.stringify(v))),
    };
  },
};

export const base64url = { encode: b64urlEncode, decode: b64urlDecode };

/**
 * A value stored in the query string. Writes replace the history entry unless `push` is set,
 * so filter tweaks do not flood the back button.
 */
export function useUrlState<T>(key: string, codec: Codec<T>, opts: { push?: boolean } = {}): [T, (next: T | ((prev: T) => T)) => void] {
  const loc = useLocation();
  const raw = new URLSearchParams(loc.search).get(key);
  // Parsing is cheap and codecs return fresh objects for json; memoise on the raw string.
  const value = useMemo(() => codec.parse(raw), [raw]);
  const push = !!opts.push;
  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      const curRaw = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get(key);
      const prev = codec.parse(curRaw);
      const v = typeof next === "function" ? (next as (p: T) => T)(prev) : next;
      setQuery({ [key]: codec.format(v) }, { push });
    },
    [key, push],
  );
  return [value, set];
}

// ---------------------------------------------------------------- <Link>

export type LinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & {
  to: string;
  replace?: boolean;
  /** Set aria-current="page" when the current path equals (or, with "prefix", starts with) `to`. */
  activeMatch?: "exact" | "prefix";
  children?: ReactNode;
};

function isModified(e: MouseEvent): boolean {
  return e.metaKey || e.altKey || e.ctrlKey || e.shiftKey;
}

export function Link({ to, replace, activeMatch, onClick, target, children, ...rest }: LinkProps) {
  const loc = useLocation();
  const path = to.split(/[?#]/)[0];
  const active =
    activeMatch === "exact" ? loc.pathname === path : activeMatch === "prefix" ? loc.pathname === path || loc.pathname.startsWith(path.endsWith("/") ? path : path + "/") : false;
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || isModified(e) || (target && target !== "_self")) return;
    if (/^[a-z]+:\/\//i.test(to) || to.startsWith("/api/")) return; // external or download
    e.preventDefault();
    navigate(to, { replace });
  };
  return createElement("a", { ...rest, href: to, target, onClick: handle, "aria-current": rest["aria-current"] ?? (active ? "page" : undefined) }, children);
}
