// Vitest setup (jsdom): React act() environment, and the browser APIs jsdom lacks. Canvas is
// not available in jsdom (no `canvas` package): getContext returns null, which every renderer
// handles (the colour buffers are still computed and tested directly).
import { afterEach } from "vitest";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

if (typeof window !== "undefined") {
  if (!window.matchMedia) {
    window.matchMedia = (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList;
  }
  if (!("ResizeObserver" in window)) {
    class RO {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    (window as unknown as { ResizeObserver: unknown }).ResizeObserver = RO;
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = RO;
  }
  // jsdom prints "Not implemented: HTMLCanvasElement.prototype.getContext"; return null quietly.
  HTMLCanvasElement.prototype.getContext = function getContext() {
    return null;
  } as typeof HTMLCanvasElement.prototype.getContext;
  // Downloads: jsdom cannot follow an <a download> click ("navigation to another Document").
  if (typeof URL.createObjectURL !== "function") URL.createObjectURL = () => "blob:test";
  if (typeof URL.revokeObjectURL !== "function") URL.revokeObjectURL = () => {};
  const anchorClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function click(this: HTMLAnchorElement) {
    if (this.hasAttribute("download")) return;
    anchorClick.call(this);
  };
  window.scrollTo = (() => {}) as typeof window.scrollTo; // jsdom: "not implemented"
  Element.prototype.scrollIntoView = Element.prototype.scrollIntoView ?? function () {};
  if (!(Element.prototype as { setPointerCapture?: unknown }).setPointerCapture) {
    (Element.prototype as unknown as { setPointerCapture: () => void }).setPointerCapture = () => {};
  }
}

afterEach(() => {
  document.body.innerHTML = "";
});
