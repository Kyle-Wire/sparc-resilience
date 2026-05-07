import { useState, useEffect } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

type State = "idle" | "checking" | "available" | "downloading" | "error";

export default function UpdateBanner() {
  const [state, setState] = useState<State>("idle");
  const [update, setUpdate] = useState<Update | null>(null);
  const [newVersion, setNewVersion] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!import.meta.env.PROD) return;

    let cancelled = false;

    const runCheck = async () => {
      // Don't re-check if we already found an update or are mid-download
      if (cancelled) return;
      try {
        const u = await check();
        if (cancelled) return;
        if (u?.available) {
          setUpdate(u);
          setNewVersion(u.version);
          setState((prev) =>
            prev === "downloading" || prev === "available" ? prev : "available",
          );
        }
      } catch {
        // Silently fail — don't disturb the user if update check fails
      }
    };

    // Initial check ~3s after launch.
    const initialTimer = setTimeout(runCheck, 3000);

    // Periodic background check every 4 hours.
    const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;
    const interval = setInterval(runCheck, FOUR_HOURS_MS);

    return () => {
      cancelled = true;
      clearTimeout(initialTimer);
      clearInterval(interval);
    };
  }, []);

  const handleInstall = async () => {
    if (!update) return;
    setState("downloading");
    setProgress(0);
    try {
      let downloaded = 0;
      let total = 0;
      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case "Started":
            total = event.data.contentLength ?? 0;
            break;
          case "Progress":
            downloaded += event.data.chunkLength;
            if (total > 0) setProgress(Math.round((downloaded / total) * 100));
            break;
          case "Finished":
            setProgress(100);
            break;
        }
      });
      await relaunch();
    } catch {
      setState("error");
    }
  };

  if (dismissed || state === "idle" || state === "checking") return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: "1.25rem",
        right: "1.25rem",
        zIndex: 9999,
        minWidth: "260px",
        maxWidth: "320px",
      }}
      className="rounded border border-sparc-gray-200 bg-white shadow-lg"
    >
      <div className="flex items-start gap-3 p-4">
        <div className="flex-1 space-y-1">
          {state === "available" && (
            <>
              <p className="text-sm font-semibold">Update available</p>
              <p className="text-xs text-sparc-gray-600">
                Version {newVersion} is ready to install.
              </p>
            </>
          )}
          {state === "downloading" && (
            <>
              <p className="text-sm font-semibold">Downloading update…</p>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-sparc-gray-200">
                <div
                  className="h-full bg-sparc-purple transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-sparc-gray-600">{progress}%</p>
            </>
          )}
          {state === "error" && (
            <p className="text-sm text-sparc-crimson">
              Update failed. Try again later.
            </p>
          )}
        </div>
        {state !== "downloading" && (
          <button
            onClick={() => setDismissed(true)}
            className="text-xs leading-none text-sparc-gray-600 hover:text-sparc-gray-800"
            aria-label="Dismiss"
          >
            ×
          </button>
        )}
      </div>
      {state === "available" && (
        <div className="border-t border-sparc-gray-200 px-4 py-3">
          <button
            onClick={handleInstall}
            className="w-full rounded bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sparc-gray-800"
          >
            Restart to Update
          </button>
        </div>
      )}
    </div>
  );
}
