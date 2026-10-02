// Report builder and export model (SPEC §6.7, api.md §8 `export.*` params, §10): the section
// picker's defaults, the preview and export bodies built from the selection, the preview
// document made inert (no scripts, no handlers) before it goes into the sandboxed iframe, and
// the bundle's size arithmetic.
import { REPORT_SECTIONS, type BundleParams, type Export, type ReportParams, type ReportPreviewBody, type ReportSection } from "../../../api/exports";
import type { OutputEntry } from "../../../api/types";

/** Sections ticked when the builder opens (plans and findings follow their selections). */
export const DEFAULT_SECTIONS: readonly ReportSection[] = ["summary", "accuracy", "validation", "scenarios", "climate", "equity", "caveats", "limitations", "provenance"];

export type ReportSelection = {
  runId: string;
  sections: ReportSection[];
  resultIds: string[];
  planIds: string[];
  findingIds: string[];
};

/** Sections in the server's canonical order (the report renders them in this order). */
export function orderedSections(sections: readonly ReportSection[]): ReportSection[] {
  return REPORT_SECTIONS.map((s) => s.id).filter((id) => sections.includes(id));
}

/** `POST /api/projects/{pid}/report/preview` body. */
export function previewBody(sel: ReportSelection): ReportPreviewBody {
  return { run_id: sel.runId, sections: orderedSections(sel.sections), result_ids: [...sel.resultIds], plan_ids: [...sel.planIds], finding_ids: [...sel.findingIds] };
}

/** `export.report` params (`POST /api/exports` with kind "report"): the preview body plus the format. */
export function reportParams(sel: ReportSelection, format: "html" | "md"): ReportParams {
  return { ...previewBody(sel), format };
}

/** Tick or untick a section; picking ids of plans or findings ticks their section. */
export function withSelection(sel: ReportSelection, patch: Partial<Omit<ReportSelection, "runId">>): ReportSelection {
  const next = { ...sel, ...patch };
  const sections = new Set(next.sections);
  if (patch.planIds && patch.planIds.length && !sel.planIds.length) sections.add("plans");
  if (patch.findingIds && patch.findingIds.length && !sel.findingIds.length) sections.add("findings");
  return { ...next, sections: orderedSections([...sections]) };
}

const URL_ATTRS = /^(href|src|xlink:href|action|formaction|srcdoc)$/i;

/**
 * The preview HTML made inert for the sandboxed iframe: script, iframe, object and embed
 * elements, `<base>`, `<meta http-equiv>` (a refresh would navigate the frame away), `on*`
 * handlers and `javascript:` URLs are removed, and a CSP meta forbids scripts. The iframe's
 * empty `sandbox` already blocks scripts; this keeps the document itself clean.
 */
export function inertPreview(html: string): string {
  if (typeof DOMParser === "undefined") {
    return html.replace(/<script\b[\s\S]*?<\/script\s*>/gi, "").replace(/\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  }
  const doc = new DOMParser().parseFromString(html, "text/html");
  for (const el of [...doc.querySelectorAll("script, iframe, object, embed, frame, frameset, base, meta[http-equiv]")]) el.remove();
  for (const el of [...doc.querySelectorAll("*")]) {
    for (const a of [...el.attributes]) {
      if (/^on/i.test(a.name) || (URL_ATTRS.test(a.name) && /^\s*(javascript|vbscript):/i.test(a.value))) el.removeAttribute(a.name);
    }
  }
  const meta = doc.createElement("meta");
  meta.setAttribute("http-equiv", "Content-Security-Policy");
  meta.setAttribute("content", "script-src 'none'; object-src 'none'");
  doc.head.prepend(meta);
  return "<!doctype html>\n" + doc.documentElement.outerHTML;
}

// ---------------------------------------------------------------- bundle

/** Outputs a bundle can include: present or stale catalog outputs, without the checkpoint. */
export function bundleOutputs(outputs: readonly OutputEntry[]): OutputEntry[] {
  return outputs.filter((o) => (o.state === "present" || o.state === "stale" || o.state === "partial") && !o.id.startsWith("checkpoint"));
}

export function outputBytes(o: Pick<OutputEntry, "files">): number {
  return o.files.reduce((a, f) => a + (Number.isFinite(f.bytes) ? f.bytes : 0), 0);
}

/** Checkpoints above this size get a warning when ticked. */
export const CHECKPOINT_WARN_BYTES = 100 * 1024 * 1024;

/** `export.bundle` params: the ticked outputs and the checkpoint toggle. */
export function bundleParams(runId: string, outputs: readonly string[], includeCheckpoint: boolean): BundleParams {
  return { run_id: runId, outputs: [...outputs], include_checkpoint: includeCheckpoint };
}

// ---------------------------------------------------------------- history

/** History order: newest first. */
export function sortedExports(list: readonly Export[]): Export[] {
  return [...list].sort((a, b) => b.created_utc.localeCompare(a.created_utc) || b.id.localeCompare(a.id));
}

/** Put a just-created export at the top of the history (replacing an older copy of it). */
export function withExport(list: readonly Export[] | undefined, e: Export): Export[] {
  return sortedExports([e, ...(list ?? []).filter((x) => x.id !== e.id)]);
}
