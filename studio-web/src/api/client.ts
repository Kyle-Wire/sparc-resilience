// Typed fetch client for the Studio API (api.md §0).
//
// - JSON in and out; every non-2xx response becomes an `ApiError` built from the error
//   envelope {error: {code, message, detail?, action?}}.
// - Auth is the HttpOnly `sparc_studio` cookie set by `/auth?t=…` (same-origin fetches send
//   it). In `npm run dev` the page reads VITE_STUDIO_TOKEN and calls `/auth?t=<token>` once,
//   and again on the first 401.
// - Binary arrays: `getBin` returns a typed view chosen by X-SPARC-Dtype (or the requested
//   dtype) plus the packed-array offsets (X-SPARC-Offsets) when present.
// - Raw uploads: `putRaw` streams a Blob/ArrayBuffer body; with `onProgress` it uses XHR so
//   upload progress is observable.
import { parseOffsets, viewOf, type Dtype, type OffsetEntry, type TypedArray } from "./binary";
import type { Action, ApiErrorBody, ValidationErrorItem } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown> | null;
  readonly action: Action | null;

  constructor(status: number, code: string, message: string, detail?: Record<string, unknown> | null, action?: Action | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail ?? null;
    this.action = action ?? null;
  }

  /** Validation errors (`422 validation`) carry `detail.errors`. */
  get validationErrors(): ValidationErrorItem[] {
    const errs = this.detail?.errors;
    return Array.isArray(errs) ? (errs as ValidationErrorItem[]) : [];
  }

  get isNotFound(): boolean {
    return this.status === 404;
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

/** Map a response status and (maybe) parsed body to an ApiError (api.md §0.3). */
export function errorFromResponse(status: number, body: unknown, statusText = ""): ApiError {
  const env = (body && typeof body === "object" ? (body as { error?: unknown }).error : undefined) as Partial<ApiErrorBody> | undefined;
  if (env && typeof env === "object" && typeof env.code === "string") {
    const message = typeof env.message === "string" && env.message ? env.message : statusText || `HTTP ${status}`;
    const detail = env.detail && typeof env.detail === "object" ? (env.detail as Record<string, unknown>) : null;
    const action = env.action && typeof env.action === "object" && typeof (env.action as Action).kind === "string" ? (env.action as Action) : null;
    return new ApiError(status, env.code, message, detail, action);
  }
  // FastAPI's own errors ({detail: …}) before the envelope handler, proxies, or HTML pages.
  const fallbackCode =
    status === 401 ? "unauthorized" : status === 404 ? "not_found" : status === 422 ? "validation" : status >= 500 ? "internal" : `http_${status}`;
  let message = statusText || `HTTP ${status}`;
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") message = d;
  } else if (typeof body === "string" && body.trim() && body.length < 300 && !body.trimStart().startsWith("<")) {
    message = body.trim();
  }
  return new ApiError(status, fallbackCode, message, null, null);
}

export type Query = Record<string, string | number | boolean | null | undefined | (string | number)[]>;

/** Build `/api/...?a=1&b=x,y` (arrays become comma lists; null/undefined are dropped). */
export function apiUrl(path: string, query?: Query): string {
  if (!query) return path;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null) continue;
    qs.set(k, Array.isArray(v) ? v.join(",") : String(v));
  }
  const s = qs.toString();
  if (!s) return path;
  return path + (path.includes("?") ? "&" : "?") + s;
}

// ---------------------------------------------------------------- dev auth bootstrap

let devAuth: Promise<boolean> | null = null;

function devToken(): string | null {
  try {
    const env = import.meta.env as Record<string, unknown>;
    const t = env.DEV ? env.VITE_STUDIO_TOKEN : undefined;
    return typeof t === "string" && t ? t : null;
  } catch {
    return null;
  }
}

/**
 * In dev, exchange VITE_STUDIO_TOKEN for the session cookie via `/auth?t=…` (SPEC §13.3).
 * Memoised; `force` repeats it once (after a 401, e.g. when the server restarted).
 */
export function ensureDevAuth(force = false): Promise<boolean> {
  const token = devToken();
  if (!token) return Promise.resolve(false);
  if (devAuth && !force) return devAuth;
  devAuth = fetch(`/auth?t=${encodeURIComponent(token)}&next=${encodeURIComponent("/api/health")}`, { credentials: "same-origin" })
    .then((r) => r.ok)
    .catch(() => false);
  return devAuth;
}

// ---------------------------------------------------------------- core request

export type RequestOptions = {
  query?: Query;
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Raw body (Blob/ArrayBuffer/typed array/string) sent as-is instead of JSON. */
  raw?: BodyInit;
};

async function readBody(res: Response): Promise<unknown> {
  const ct = res.headers.get("content-type") ?? "";
  const text = await res.text();
  if (!text) return null;
  if (ct.includes("json")) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}

async function send(method: string, path: string, opts: RequestOptions, retried: boolean): Promise<Response> {
  const headers: Record<string, string> = { Accept: "application/json", ...(opts.headers ?? {}) };
  let body: BodyInit | undefined;
  if (opts.raw !== undefined) body = opts.raw;
  else if (opts.body !== undefined) {
    body = JSON.stringify(opts.body);
    headers["Content-Type"] = "application/json";
  }
  let res: Response;
  try {
    res = await fetch(apiUrl(path, opts.query), { method, headers, body, credentials: "same-origin", signal: opts.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw new ApiError(0, "aborted", "Request cancelled");
    throw new ApiError(0, "network", "Studio server is not reachable");
  }
  if (res.status === 401 && !retried && devToken()) {
    if (await ensureDevAuth(true)) return send(method, path, opts, true);
  }
  return res;
}

/** Perform a request and parse JSON; throws ApiError on any non-2xx status. */
export async function request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await send(method, path, opts, false);
  const body = await readBody(res);
  if (!res.ok) throw errorFromResponse(res.status, body, res.statusText);
  return body as T;
}

export const api = {
  get: <T>(path: string, query?: Query, signal?: AbortSignal) => request<T>("GET", path, { query, signal }),
  post: <T>(path: string, body?: unknown, opts: Omit<RequestOptions, "body"> = {}) => request<T>("POST", path, { ...opts, body: body ?? {} }),
  put: <T>(path: string, body?: unknown, opts: Omit<RequestOptions, "body"> = {}) => request<T>("PUT", path, { ...opts, body }),
  patch: <T>(path: string, body?: unknown, opts: Omit<RequestOptions, "body"> = {}) => request<T>("PATCH", path, { ...opts, body }),
  del: <T>(path: string, query?: Query, opts: Omit<RequestOptions, "query"> = {}) => request<T>("DELETE", path, { ...opts, query }),
};

// ---------------------------------------------------------------- binary

export type BinResult<T extends TypedArray = TypedArray> = {
  data: T;
  dtype: Dtype;
  length: number;
  etag: string | null;
  offsets: OffsetEntry[] | null;
  buffer: ArrayBuffer;
  headers: Headers;
};

/**
 * GET a binary array (api.md §0.4). The dtype comes from X-SPARC-Dtype, else `dtype`, else
 * float32. Packed bodies (grid.bin, preview) expose `offsets`; use `unpack` from binary.ts.
 */
export async function getBin<T extends TypedArray = TypedArray>(path: string, dtype?: Dtype, opts: { query?: Query; signal?: AbortSignal } = {}): Promise<BinResult<T>> {
  const res = await send("GET", path, { query: opts.query, signal: opts.signal, headers: { Accept: "application/octet-stream" } }, false);
  if (!res.ok) throw errorFromResponse(res.status, await readBody(res), res.statusText);
  const buffer = await res.arrayBuffer();
  const hdrDtype = res.headers.get("X-SPARC-Dtype") as Dtype | null;
  const dt: Dtype = hdrDtype ?? dtype ?? "float32";
  const offH = res.headers.get("X-SPARC-Offsets");
  const offsets = offH ? parseOffsets(offH) : null;
  const lenH = res.headers.get("X-SPARC-Length");
  const data = (offsets ? new Uint8Array(buffer) : viewOf(buffer, dt)) as T;
  return {
    data,
    dtype: offsets ? "uint8" : dt,
    length: lenH ? Number(lenH) : offsets ? 0 : data.length,
    etag: res.headers.get("ETag"),
    offsets,
    buffer,
    headers: res.headers,
  };
}

export type PutRawOptions = {
  contentType?: string;
  headers?: Record<string, string>;
  query?: Query;
  method?: "PUT" | "POST";
  signal?: AbortSignal;
  /** Called with (sentBytes, totalBytes) while the body uploads (uses XHR). */
  onProgress?: (sent: number, total: number) => void;
};

/** Stream a raw body (uploads, blobs, finding images) and parse the JSON reply. */
export async function putRaw<T>(path: string, body: Blob | ArrayBuffer | ArrayBufferView, opts: PutRawOptions = {}): Promise<T> {
  const method = opts.method ?? "PUT";
  const headers: Record<string, string> = { Accept: "application/json", "Content-Type": opts.contentType ?? "application/octet-stream", ...(opts.headers ?? {}) };
  const payload: Blob | ArrayBuffer =
    body instanceof Blob || body instanceof ArrayBuffer
      ? body
      : (body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength) as ArrayBuffer);
  if (!opts.onProgress) return request<T>(method, path, { query: opts.query, headers, raw: payload, signal: opts.signal });
  const url = apiUrl(path, opts.query);
  const onProgress = opts.onProgress;
  const attempt = () =>
    new Promise<{ status: number; statusText: string; body: unknown }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(method, url);
      xhr.withCredentials = true;
      for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
      xhr.upload.onprogress = (ev) => onProgress(ev.loaded, ev.lengthComputable ? ev.total : 0);
      xhr.onload = () => {
        const ct = xhr.getResponseHeader("content-type") ?? "";
        let parsed: unknown = xhr.responseText;
        if (ct.includes("json") && xhr.responseText) {
          try {
            parsed = JSON.parse(xhr.responseText);
          } catch {
            /* keep text */
          }
        }
        resolve({ status: xhr.status, statusText: xhr.statusText, body: parsed });
      };
      xhr.onerror = () => reject(new ApiError(0, "network", "Upload failed: the Studio server is not reachable"));
      xhr.onabort = () => reject(new ApiError(0, "aborted", "Upload cancelled"));
      opts.signal?.addEventListener("abort", () => xhr.abort(), { once: true });
      xhr.send(payload);
    });
  let r = await attempt();
  if (r.status === 401 && devToken() && (await ensureDevAuth(true))) r = await attempt();
  if (r.status < 200 || r.status >= 300) throw errorFromResponse(r.status, r.body, r.statusText);
  return r.body as T;
}

/** Turn any thrown value into a message suitable for a toast or an empty state. */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
