import { Marked } from "marked";
import { useMemo, type MouseEvent } from "react";
import { navigate } from "../../router";

const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

const SAFE_LINK = /^(https?:|mailto:|#|\/(?!\/)|\.{0,2}\/?[\w.-])/i;
const LOCAL_IMG = /^(data:image\/(png|gif|jpeg|webp);|\/(?!\/)|\.{0,2}\/?[\w.-])/i;

const md = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    // Raw HTML in documents is shown as text, never interpreted (SPEC §6.4 Docs).
    html({ text }) {
      return escapeHtml(text);
    },
    link({ href, tokens }) {
      if (SAFE_LINK.test(href) && !/^\s*(javascript|vbscript|data):/i.test(href)) return false;
      return this.parser.parseInline(tokens);
    },
    // Remote images would be network requests; only local and inline images render.
    image({ href, text }) {
      if (LOCAL_IMG.test(href) && !/^https?:/i.test(href)) return false;
      return `<span class="muted">[image: ${escapeHtml(text || href)}]</span>`;
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
