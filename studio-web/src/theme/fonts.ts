// Bundled fonts (SPEC §12.1, §12.7): Inter for display and body text, JetBrains Mono for
// kickers, labels, numbers in tables and logs. Only the latin woff2 files of the weights in
// use are bundled (from the @fontsource packages), so Studio works offline and makes no
// external requests.
import inter400 from "@fontsource/inter/files/inter-latin-400-normal.woff2?url";
import inter500 from "@fontsource/inter/files/inter-latin-500-normal.woff2?url";
import inter600 from "@fontsource/inter/files/inter-latin-600-normal.woff2?url";
import inter700 from "@fontsource/inter/files/inter-latin-700-normal.woff2?url";
import inter800 from "@fontsource/inter/files/inter-latin-800-normal.woff2?url";
import mono400 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2?url";
import mono500 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2?url";
import mono600 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-600-normal.woff2?url";

export const FONT_FACES: { family: string; weight: number; url: string }[] = [
  { family: "Inter", weight: 400, url: inter400 },
  { family: "Inter", weight: 500, url: inter500 },
  { family: "Inter", weight: 600, url: inter600 },
  { family: "Inter", weight: 700, url: inter700 },
  { family: "Inter", weight: 800, url: inter800 },
  { family: "JetBrains Mono", weight: 400, url: mono400 },
  { family: "JetBrains Mono", weight: 500, url: mono500 },
  { family: "JetBrains Mono", weight: 600, url: mono600 },
];

export function fontFaceCss(): string {
  return FONT_FACES.map(
    (f) => `@font-face{font-family:"${f.family}";font-style:normal;font-weight:${f.weight};font-display:swap;src:url("${f.url}") format("woff2");}`,
  ).join("\n");
}

/** Inject the @font-face rules once (idempotent). */
export function installFonts(doc: Document = document): void {
  if (doc.getElementById("sparc-fonts")) return;
  const style = doc.createElement("style");
  style.id = "sparc-fonts";
  style.textContent = fontFaceCss();
  doc.head.appendChild(style);
}
