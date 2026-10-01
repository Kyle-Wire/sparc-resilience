// Bundled fonts (SPEC §12.1, §12.7): Archivo for display, Public Sans for body text and
// IBM Plex Mono for eyebrows, numbers and logs. Only the latin woff2 files of the weights in
// use are bundled (from the @fontsource packages), so Studio works offline and makes no
// external requests. The static Archivo package has no width axis: `font-stretch: 87%` in
// base.css falls back to the normal width.
import archivo600 from "@fontsource/archivo/files/archivo-latin-600-normal.woff2?url";
import archivo700 from "@fontsource/archivo/files/archivo-latin-700-normal.woff2?url";
import plexMono400 from "@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2?url";
import plexMono500 from "@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2?url";
import publicSans400 from "@fontsource/public-sans/files/public-sans-latin-400-normal.woff2?url";
import publicSans500 from "@fontsource/public-sans/files/public-sans-latin-500-normal.woff2?url";
import publicSans600 from "@fontsource/public-sans/files/public-sans-latin-600-normal.woff2?url";

export const FONT_FACES: { family: string; weight: number; url: string }[] = [
  { family: "Archivo", weight: 600, url: archivo600 },
  { family: "Archivo", weight: 700, url: archivo700 },
  { family: "Public Sans", weight: 400, url: publicSans400 },
  { family: "Public Sans", weight: 500, url: publicSans500 },
  { family: "Public Sans", weight: 600, url: publicSans600 },
  { family: "IBM Plex Mono", weight: 400, url: plexMono400 },
  { family: "IBM Plex Mono", weight: 500, url: plexMono500 },
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
