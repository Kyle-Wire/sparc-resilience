// ChartFrame (SPEC §6.4, §12.6): title, computed caption, units, a "Table view" <details>,
// SVG/PNG/CSV export and "Pin to Findings" for every chart in the kit, plus keyboard-focusable
// marks with an aria-live readout and a shared tooltip.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type FocusEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
  type Ref,
} from "react";
import { api, errorMessage, putRaw } from "../api/client";
import type { Finding, FindingCreate } from "../api/types";
import { Icon } from "../components/ui/Icon";
import { Table, type Column } from "../components/ui/Table";
import { toCsv, type CsvCell } from "../components/ui/csv";
import { copyText, downloadBlob, downloadText, fileSlug } from "../components/ui/download";
import { useRoute } from "../router";
import { toast, useUi } from "../stores/ui";
import { resolveCssVars, themeColors, type ThemeColors } from "../theme/palette";

export type ChartTable = {
  columns: { key: string; label: string; unit?: string }[];
  rows: CsvCell[][];
};

export type ChartHandle = {
  /** Self-contained SVG markup (CSS variables resolved, fonts named, canvas layers embedded). */
  svg: () => string;
  /** PNG rendered from the SVG at 2×. */
  png: () => Promise<Blob>;
  csv: () => string;
  exportSvg: () => void;
  exportPng: () => Promise<void>;
  downloadCsv: () => void;
  copyCsv: () => Promise<boolean>;
  pin: () => Promise<Finding | null>;
  table: ChartTable;
};

/** Props every kit chart accepts for its frame. */
export type FrameOptions = {
  title: string;
  caption?: ReactNode;
  /** Units shown under the title (also in the table header). */
  units?: string;
  legend?: ReactNode;
  /** Render only the plot (inside SmallMultiples or table cells). */
  bare?: boolean;
  /** Override the export file name (defaults to the title). */
  exportName?: string;
  /** Extra snapshot data for Findings, or false to hide the Pin button. */
  pin?: false | { view?: string; snapshot?: Record<string, unknown> };
  handleRef?: Ref<ChartHandle>;
  actions?: ReactNode;
  className?: string;
};

type ChartCtx = {
  dark: boolean;
  colors: ThemeColors;
  announce: (text: string) => void;
  tip: (text: string, x: number, y: number) => void;
  untip: () => void;
};

const Ctx = createContext<ChartCtx | null>(null);

const WidthCtx = createContext<number | null>(null);

/**
 * The measured width of the enclosing chart body (or small-multiples panel), so charts draw
 * their viewBox at 1:1 and text stays at its CSS size. Falls back before layout (and in jsdom).
 */
export function useChartWidth(fallback = 640): number {
  const w = useContext(WidthCtx);
  return w && w > 0 ? Math.max(240, Math.round(w)) : fallback;
}

/** A block that measures its own width and provides it to the charts inside. */
export function MeasureWidth({ children, className }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const width = el.getBoundingClientRect().width;
      if (width > 0) setW((prev) => (prev !== null && Math.abs(prev - width) < 1 ? prev : width));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className={className} style={{ minWidth: 0 }}>
      <WidthCtx.Provider value={w}>{children}</WidthCtx.Provider>
    </div>
  );
}

const fallbackCtx = (dark: boolean): ChartCtx => ({ dark, colors: themeColors(dark), announce: () => {}, tip: () => {}, untip: () => {} });

/** Theme colours, the readout and the tooltip of the enclosing frame. */
export function useChart(): ChartCtx {
  const dark = useUi((s) => s.dark);
  const c = useContext(Ctx);
  return c ?? fallbackCtx(dark);
}

/**
 * Props that make an SVG mark focusable and described: aria-label, focus → readout,
 * hover → tooltip, Enter/Space/click → `onActivate`.
 */
export function markProps(ctx: ChartCtx, label: string, onActivate?: () => void) {
  return {
    tabIndex: 0,
    role: "img" as const,
    "aria-label": label,
    onFocus: (e: FocusEvent<SVGElement>) => {
      ctx.announce(label);
      const r = (e.currentTarget as Element).getBoundingClientRect?.();
      if (r) ctx.tip(label, r.left + r.width / 2, r.top);
    },
    onBlur: () => ctx.untip(),
    onMouseEnter: (e: ReactMouseEvent<SVGElement>) => ctx.tip(label, e.clientX, e.clientY),
    onMouseMove: (e: ReactMouseEvent<SVGElement>) => ctx.tip(label, e.clientX, e.clientY),
    onMouseLeave: () => ctx.untip(),
    onClick: onActivate,
    onKeyDown: onActivate
      ? (e: ReactKeyboardEvent<SVGElement>) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onActivate();
          }
        }
      : undefined,
    cursor: onActivate ? "pointer" : undefined,
  };
}

const EXPORT_CSS = `
text{font-family:var(--font-body);font-variant-numeric:tabular-nums}
.t-axis{font-size:11px;fill:var(--muted)}
.t-label{font-size:11.5px;fill:var(--ink-2)}
.t-strong{font-size:12px;fill:var(--ink);font-weight:600}
.gridline{stroke:var(--grid);stroke-width:1}
.axisline{stroke:var(--axis);stroke-width:1}
.zero{stroke:var(--ink-2);stroke-width:1;stroke-dasharray:3 3}
.brush{fill:rgba(42,120,214,0.18);stroke:var(--accent)}
`;

/** Serialise a chart <svg> into a standalone document (exported for tests). */
export function serializeChartSvg(svg: SVGSVGElement, dark: boolean, canvases: HTMLCanvasElement[] = []): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const vb = svg.getAttribute("viewBox")?.split(/\s+/).map(Number);
  if (vb && vb.length === 4) {
    clone.setAttribute("width", String(vb[2]));
    clone.setAttribute("height", String(vb[3]));
  }
  clone.removeAttribute("class");
  for (const el of clone.querySelectorAll("[tabindex]")) el.removeAttribute("tabindex");
  // Canvas layers (hexbin rasters) become embedded images at their plot position.
  for (const c of canvases) {
    if (!c.getContext("2d")) continue; // never drawn (no canvas support)
    let href = "";
    try {
      href = c.toDataURL("image/png");
    } catch {
      href = "";
    }
    if (!href || href === "data:,") continue;
    const img = document.createElementNS("http://www.w3.org/2000/svg", "image");
    for (const k of ["x", "y", "width", "height"]) img.setAttribute(k, c.dataset[k === "width" ? "w" : k === "height" ? "h" : k] ?? "0");
    img.setAttribute("href", href);
    img.setAttribute("preserveAspectRatio", "none");
    clone.insertBefore(img, clone.firstChild);
  }
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", vb ? String(vb[0]) : "0");
  bg.setAttribute("y", vb ? String(vb[1]) : "0");
  bg.setAttribute("width", "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("style", "fill:var(--surface)");
  clone.insertBefore(bg, clone.firstChild);
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = EXPORT_CSS;
  clone.insertBefore(style, clone.firstChild);
  const text = new XMLSerializer().serializeToString(clone);
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + resolveCssVars(text, dark);
}

/**
 * Tile several chart SVGs (small multiples) into one SVG, three per row, each under its
 * panel title.
 */
export function composeSvgs(svgs: SVGSVGElement[], perRow = 3): SVGSVGElement {
  const NS = "http://www.w3.org/2000/svg";
  const boxes = svgs.map((s) => s.getAttribute("viewBox")?.split(/\s+/).map(Number) ?? [0, 0, 300, 200]);
  const cw = Math.max(...boxes.map((b) => b[2]));
  const ch = Math.max(...boxes.map((b) => b[3])) + 22;
  const cols = Math.min(perRow, svgs.length);
  const rows = Math.ceil(svgs.length / cols);
  const out = document.createElementNS(NS, "svg");
  out.setAttribute("viewBox", `0 0 ${cols * cw} ${rows * ch}`);
  svgs.forEach((s, i) => {
    const x = (i % cols) * cw;
    const y = Math.floor(i / cols) * ch;
    const title = s.closest(".panel")?.querySelector("h4")?.textContent ?? "";
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", String(x + 4));
    t.setAttribute("y", String(y + 14));
    t.setAttribute("class", "t-strong");
    t.textContent = title;
    out.appendChild(t);
    const inner = s.cloneNode(true) as SVGSVGElement;
    inner.setAttribute("x", String(x));
    inner.setAttribute("y", String(y + 20));
    inner.setAttribute("width", String(boxes[i][2]));
    inner.setAttribute("height", String(boxes[i][3]));
    inner.removeAttribute("style");
    out.appendChild(inner);
  });
  return out;
}

/** Rasterise SVG markup to PNG (2×) through an <img> and a canvas. */
export async function svgToPng(svgText: string, scale = 2, timeoutMs = 8000): Promise<Blob> {
  const m = /<svg[^>]*\swidth="([\d.]+)"[^>]*\sheight="([\d.]+)"/.exec(svgText) ?? /<svg[^>]*\sheight="([\d.]+)"[^>]*\swidth="([\d.]+)"/.exec(svgText);
  const w = m ? Number(m[1]) : 640;
  const h = m ? Number(m[2]) : 400;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("PNG export needs canvas support in this browser");
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("PNG export timed out")), timeoutMs);
      img.onload = () => {
        clearTimeout(t);
        resolve();
      };
      img.onerror = () => {
        clearTimeout(t);
        reject(new Error("The chart could not be rasterised"));
      };
      img.src = url;
    });
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  } finally {
    URL.revokeObjectURL(url);
  }
  return new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"));
}

function tableToCsv(t: ChartTable): string {
  return toCsv(
    t.columns.map((c) => (c.unit ? `${c.label} (${c.unit})` : c.label)),
    t.rows,
  );
}

function captionText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(captionText).join("");
  return "";
}

export type ChartFrameProps = FrameOptions & {
  table: ChartTable;
  children: ReactNode;
};

/** The frame every kit chart renders inside. */
export function ChartFrame({ title, caption, units, legend, bare, exportName, pin, handleRef, actions, className, table, children }: ChartFrameProps) {
  const dark = useUi((s) => s.dark);
  const context = useUi((s) => s.context);
  const route = useRoute();
  const body = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState("");
  const [tipState, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const [pinning, setPinning] = useState(false);
  const titleId = useId();
  const name = fileSlug(exportName ?? title);

  const ctx = useMemo<ChartCtx>(
    () => ({
      dark,
      colors: themeColors(dark),
      announce: (t) => setLive((prev) => (prev === t ? t + "​" : t)),
      tip: (text, x, y) => setTip({ text, x, y }),
      untip: () => setTip(null),
    }),
    [dark],
  );

  const svgText = useCallback(() => {
    const all = [...(body.current?.querySelectorAll<SVGSVGElement>("svg.chart") ?? [])];
    if (!all.length) throw new Error("This chart has no SVG to export");
    const svg = all.length === 1 ? all[0] : composeSvgs(all);
    const canvases = [...(body.current?.querySelectorAll<HTMLCanvasElement>("canvas[data-chart-layer]") ?? [])];
    return serializeChartSvg(svg, dark, canvases);
  }, [dark]);

  const csv = useCallback(() => tableToCsv(table), [table]);

  const doPin = useCallback(async (): Promise<Finding | null> => {
    const pid = context.projectId;
    if (!pid) {
      toast("warning", "Open a project to pin findings");
      return null;
    }
    setPinning(true);
    try {
      const body: FindingCreate = {
        project_id: pid,
        run_id: context.runId,
        view: (pin && pin.view) || route.route?.path || route.pathname,
        url_state: route.pathname + (route.query.toString() ? "?" + route.query.toString() : ""),
        title,
        note_md: "",
        snapshot: { kind: "chart", title, caption: captionText(caption), units: units ?? null, table, ...(pin && pin.snapshot ? pin.snapshot : {}) },
      };
      const f = await api.post<Finding>("/api/findings", body);
      try {
        const svg = svgText();
        await putRaw(`/api/findings/${encodeURIComponent(f.id)}/image`, new Blob([svg], { type: "image/svg+xml" }), { contentType: "image/svg+xml" });
      } catch {
        /* the finding stands without an image */
      }
      toast("success", "Pinned to Findings", { href: `/p/${pid}/findings`, linkLabel: "Open Findings" });
      return f;
    } catch (e) {
      toast("error", "Could not pin to Findings", { body: errorMessage(e) });
      return null;
    } finally {
      setPinning(false);
    }
  }, [context.projectId, context.runId, pin, route, title, caption, units, table, svgText]);

  const handle = useMemo<ChartHandle>(
    () => ({
      svg: svgText,
      png: () => svgToPng(svgText()),
      csv,
      table,
      exportSvg: () => downloadBlob(new Blob([svgText()], { type: "image/svg+xml" }), `${name}.svg`),
      exportPng: async () => {
        try {
          downloadBlob(await svgToPng(svgText()), `${name}.png`);
        } catch (e) {
          toast("error", "PNG export failed", { body: errorMessage(e) });
        }
      },
      downloadCsv: () => downloadText(csv(), `${name}.csv`, "text/csv;charset=utf-8"),
      copyCsv: async () => {
        const ok = await copyText(csv());
        toast(ok ? "success" : "warning", ok ? "Chart data copied as CSV" : "Copy failed: use Download CSV");
        return ok;
      },
      pin: doPin,
    }),
    [svgText, csv, table, name, doPin],
  );
  useImperativeHandle(handleRef, () => handle, [handle]);

  const columns = useMemo<Column<Record<string, CsvCell>>[]>(
    () =>
      table.columns.map((c, i) => ({
        key: `c${i}`,
        label: c.label,
        unit: c.unit,
        align: table.rows.some((r) => typeof r[i] === "number") ? "right" : "left",
        value: (r) => r[`c${i}`],
        render: (r) => {
          const v = r[`c${i}`];
          if (typeof v === "number") return Number.isFinite(v) ? String(Number(v.toPrecision(6))).replace(/^-/, "−") : "—";
          return v === null || v === undefined || v === "" ? "—" : String(v);
        },
      })),
    [table],
  );
  const rows = useMemo(() => table.rows.map((r) => Object.fromEntries(r.map((v, i) => [`c${i}`, v]))), [table]);

  if (bare) return <Ctx.Provider value={ctx}>{children}</Ctx.Provider>;

  return (
    <Ctx.Provider value={ctx}>
      <figure className={["chart-frame", className ?? ""].filter(Boolean).join(" ")} aria-labelledby={titleId} style={{ margin: 0 }} data-chart={title}>
        <header>
          <div>
            <h3 id={titleId}>{title}</h3>
            {units ? <p className="cap">Units: {units}</p> : null}
          </div>
          <div className="chart-tools" role="toolbar" aria-label={`${title} export`}>
            {actions}
            <button type="button" className="btn small ghost" onClick={handle.exportSvg} aria-label={`Export ${title} as SVG`}>
              <Icon name="download" /> SVG
            </button>
            <button type="button" className="btn small ghost" onClick={() => void handle.exportPng()} aria-label={`Export ${title} as PNG`}>
              <Icon name="image" /> PNG
            </button>
            <button type="button" className="btn small ghost" onClick={() => void handle.copyCsv()} aria-label={`Copy ${title} data as CSV`}>
              <Icon name="copy" /> CSV
            </button>
            {pin !== false ? (
              <button type="button" className="btn small ghost" onClick={() => void doPin()} disabled={pinning} aria-label={`Pin ${title} to Findings`}>
                {pinning ? <span className="spinner" aria-hidden="true" /> : <Icon name="pin" />} Pin
              </button>
            ) : null}
          </div>
        </header>
        <div className="chart-body" ref={body}>
          <MeasureWidth>{children}</MeasureWidth>
        </div>
        {legend ? <div className="legend-row">{legend}</div> : null}
        {caption ? <figcaption className="cap">{caption}</figcaption> : null}
        <details className="table-view">
          <summary>Table view</summary>
          <Table columns={columns} rows={rows} caption={`${title} data`} csvName={name} />
        </details>
        <div className="sr-only" aria-live="polite" role="status">
          {live}
        </div>
        {tipState ? (
          <div className="tip" role="presentation" style={{ left: Math.min(tipState.x + 12, (typeof window !== "undefined" ? window.innerWidth : 1200) - 290), top: tipState.y + 12 }}>
            {tipState.text}
          </div>
        ) : null}
      </figure>
    </Ctx.Provider>
  );
}
