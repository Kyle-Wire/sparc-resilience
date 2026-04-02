import { useEffect } from "react";

/**
 * Global keyboard shortcut handler for SPARC desktop app.
 *
 * Shortcuts (all use Cmd/Ctrl modifier):
 *  - Cmd+K          Toggle chat panel
 *  - Cmd+1..9,0     Navigate to page by index (1=Project … 0=Results)
 *  - Cmd+,          Open settings
 *  - Cmd+Shift+R    Reload current page (refresh key)
 */
export function useKeyboardShortcuts(handlers: {
  toggleChat: () => void;
  navigateByIndex: (index: number) => void;
  openSettings: () => void;
  refresh: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (!meta) return;

      // Cmd+K — toggle chat
      if (e.key === "k" && !e.shiftKey) {
        e.preventDefault();
        handlers.toggleChat();
        return;
      }

      // Cmd+, — settings
      if (e.key === ",") {
        e.preventDefault();
        handlers.openSettings();
        return;
      }

      // Cmd+Shift+R — refresh current page
      if (e.key === "r" && e.shiftKey) {
        e.preventDefault();
        handlers.refresh();
        return;
      }

      // Cmd+1-9,0 — navigate by page index
      const digit = parseInt(e.key);
      if (!isNaN(digit) && digit >= 0 && digit <= 9) {
        e.preventDefault();
        // 1-9 map to pages 0-8, 0 maps to page 9
        const idx = digit === 0 ? 9 : digit - 1;
        handlers.navigateByIndex(idx);
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handlers]);
}
