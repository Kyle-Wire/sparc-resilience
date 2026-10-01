// Minimal React testing helpers (no testing-library dependency): render into a container
// under act(), fire DOM events, type into inputs, and a fetch mock keyed by "METHOD path".
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";

export type Rendered = { container: HTMLElement; root: Root; rerender: (ui: ReactNode) => void; unmount: () => void };

const mounted = new Set<Root>();

export function render(ui: ReactNode): Rendered {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mounted.add(root);
  act(() => root.render(ui));
  return {
    container,
    root,
    rerender: (next) => act(() => root.render(next)),
    unmount: () => {
      if (!mounted.delete(root)) return;
      act(() => root.unmount());
    },
  };
}

/**
 * Unmount every tree rendered by the test (setup.ts runs it after each test), so effects,
 * subscriptions and portals into document.body do not leak into the next test.
 */
export function cleanup(): void {
  for (const root of [...mounted]) {
    mounted.delete(root);
    act(() => root.unmount());
  }
}

/** Let pending promises and effects settle. */
export async function flush(times = 3): Promise<void> {
  for (let i = 0; i < times; i++) await act(async () => await new Promise((r) => setTimeout(r, 0)));
}

/**
 * Flush until `check` returns a truthy value (or throws no more), then return it. For work
 * whose duration is not a fixed number of ticks, such as a lazy route chunk that vitest
 * transforms on first import.
 */
export async function waitFor<T>(check: () => T, timeoutMs = 5000, what = "condition"): Promise<NonNullable<T>> {
  const end = Date.now() + timeoutMs;
  let last: unknown = null;
  for (;;) {
    try {
      const v = check();
      if (v) return v as NonNullable<T>;
    } catch (e) {
      last = e;
    }
    if (Date.now() > end) throw new Error(`waitFor: ${what} not met within ${timeoutMs} ms${last ? ` (${String(last)})` : ""}`);
    await act(async () => await new Promise((r) => setTimeout(r, 10)));
  }
}

export function click(el: Element | null): void {
  if (!el) throw new Error("click: element not found");
  act(() => {
    (el as HTMLElement).dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 }));
  });
}

export function key(el: Element | null, k: string, init: KeyboardEventInit = {}): void {
  if (!el) throw new Error("key: element not found");
  act(() => {
    el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init }));
  });
}

/** Set an input's value the way React notices (native setter + input event). */
export function typeInto(el: Element | null, value: string): void {
  if (!el) throw new Error("typeInto: element not found");
  const input = el as HTMLInputElement;
  const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : input instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  act(() => {
    setter?.call(input, value);
    input.dispatchEvent(new Event(input instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

export function byText(root: ParentNode, selector: string, text: string | RegExp): HTMLElement | null {
  for (const el of root.querySelectorAll<HTMLElement>(selector)) {
    const t = el.textContent ?? "";
    if (typeof text === "string" ? t.includes(text) : text.test(t)) return el;
  }
  return null;
}

export type MockRoute = { status?: number; body?: unknown; headers?: Record<string, string>; raw?: ArrayBuffer | string };
export type MockHandler = MockRoute | ((url: URL, init: RequestInit) => MockRoute | Promise<MockRoute>);

/**
 * Replace global fetch with a table of "METHOD /path" → response (query strings ignored for
 * matching unless the key contains "?"). Returns the call log.
 */
export function mockFetch(routes: Record<string, MockHandler>): { calls: { method: string; url: string; body: unknown }[]; restore: () => void } {
  const calls: { method: string; url: string; body: unknown }[] = [];
  const orig = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, "http://localhost");
    const method = (init.method ?? "GET").toUpperCase();
    let body: unknown = init.body;
    if (typeof body === "string") {
      try {
        body = JSON.parse(body);
      } catch {
        /* raw text */
      }
    }
    calls.push({ method, url: url.pathname + url.search, body });
    const h = routes[`${method} ${url.pathname}${url.search}`] ?? routes[`${method} ${url.pathname}`];
    if (!h) return new Response(JSON.stringify({ error: { code: "not_found", message: `no mock for ${method} ${url.pathname}` } }), { status: 404, headers: { "content-type": "application/json" } });
    const r = typeof h === "function" ? await h(url, init) : h;
    const headers = new Headers(r.headers ?? {});
    if (r.raw !== undefined) return new Response(r.raw, { status: r.status ?? 200, headers });
    if (!headers.has("content-type")) headers.set("content-type", "application/json");
    return new Response(r.body === undefined ? null : JSON.stringify(r.body), { status: r.status ?? 200, headers });
  }) as typeof fetch;
  return { calls, restore: () => (globalThis.fetch = orig) };
}
