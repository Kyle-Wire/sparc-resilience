import { Marked } from "marked";
import { useMemo, type MouseEvent } from "react";
import { navigate } from "../../router";

const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

/** Resolve HTML character references (`&colon;`, `&#x09;`, …) the way the browser will. */
export function decodeEntities(s: string): string {
  if (!s.includes("&")) return s;
  if (typeof document !== "undefined") {
    // A textarea's content is RCDATA: references are decoded, markup is never parsed.
    const ta = document.createElement("textarea");
    ta.innerHTML = s;
    return ta.value;
  }
  return s
    .replace(/&#x([0-9a-f]+);?/gi, (_, h: string) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);?/g, (_, d: string) => String.fromCodePoint(Number(d)));
}

// Characters URL parsers ignore or strip inside a scheme ("java\tscript:" is javascript:).
const URL_INVISIBLE = /[\u0000-\u0020\u007f-\u009f\u00ad\u200b-\u200f\u2028\u2029\ufeff]/g;

/** The URL scheme a browser would see, lower-cased, or null for a relative URL. */
function schemeOf(url: string): string | null {
  const m = /^([a-z][a-z0-9+.-]*):/i.exec(url.replace(URL_INVISIBLE, ""));
  return m ? m[1].toLowerCase() : null;
}

/**
 * A link target safe to emit: http(s), mailto, in-page anchors and same-site paths.
 * Character references are resolved first, so `javascript&colon;…` cannot slip through as
 * an attribute the browser decodes after the check. Null means "render the text only".
 */
export function safeLinkHref(raw: string): string | null {
  const href = decodeEntities(raw).trim();
  if (!href) return null;
  const scheme = schemeOf(href);
  if (scheme !== null) return scheme === "http" || scheme === "https" || scheme === "mailto" ? href : null;
  if (/^[\\/]{2}/.test(href.replace(URL_INVISIBLE, ""))) return null; // protocol-relative: another host
  return href;
}

/** Image sources that need no network request: relative paths and inline raster data. */
export function safeImageSrc(raw: string): string | null {
  const src = decodeEntities(raw).trim();
  if (!src) return null;
  const scheme = schemeOf(src);
  if (scheme === "data") return /^data:image\/(png|gif|jpeg|webp)[;,]/i.test(src.replace(URL_INVISIBLE, "")) ? src : null;
  if (scheme !== null) return null; // remote (http, https, …) or anything executable
  if (/^[\\/]{2}/.test(src.replace(URL_INVISIBLE, ""))) return null;
  return src;
}

const attr = (s: string) => escapeHtml(s);

const md = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    // Raw HTML in documents is shown as text, never interpreted (SPEC §6.4 Docs).
    html({ text }) {
      return escapeHtml(text);
    },
    // Links and images are emitted here (not by marked's default renderer) so the URL that
    // was checked is exactly the attribute value the browser receives.
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens);
      const safe = safeLinkHref(href);
      if (safe === null) return text;
      return `<a href="${attr(safe)}"${title ? ` title="${attr(decodeEntities(title))}"` : ""}>${text}</a>`;
    },
    // Remote images would be network requests; only local and inline images render.
    image({ href, title, text }) {
      const alt = decodeEntities(text || "");
      const safe = safeImageSrc(href);
      if (safe === null) return `<span class="muted">[image: ${escapeHtml(alt || decodeEntities(href))}]</span>`;
      return `<img src="${attr(safe)}" alt="${attr(alt)}"${title ? ` title="${attr(decodeEntities(title))}"` : ""}>`;
    },
  },
});

/** Markdown → HTML with raw HTML escaped, unsafe links neutralised, remote images dropped. */
export function renderMarkdown(source: string): string {
  return md.parse(source ?? "", { async: false });
}

/**
 * Rendered markdown (`marked`). In-app links (starting with "/") navigate without a reload.
 */
export function Markdown({ source, className }: { source: string; className?: string }) {
  const html = useMemo(() => renderMarkdown(source), [source]);
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const a = (e.target as HTMLElement).closest("a");
    if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
    const href = a.getAttribute("href") ?? "";
    if (href.startsWith("/") && !href.startsWith("//") && !href.startsWith("/api/")) {
      e.preventDefault();
      navigate(href);
    } else if (/^https?:/i.test(href)) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
  };
  return <div className={["md", className ?? ""].filter(Boolean).join(" ")} onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />;
}
