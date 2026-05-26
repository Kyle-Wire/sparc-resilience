/**
 * useStartup — owns the 4-step splash state machine:
 *   sidecar → token (acquire auth token) → session (auth rehydration) → ready
 *
 * Returns a discriminated state that App.tsx uses to decide what to render,
 * plus parallax preference (persisted in localStorage).
 */
import { useState, useEffect, useRef } from "react";
import type { SplashStep } from "@/components/layout/Splash";
import { setToken } from "@/stores/tokenStore";
import { loadThemeKey, applyTheme } from "@/lib/theme";
import { useAuthStore } from "@/stores/authStore";

interface StartupState {
  splashStep: SplashStep;
  splashDone: boolean;
  parallaxEnabled: boolean;
  setParallaxEnabled: (v: boolean) => void;
  handleRetry?: () => Promise<void>;
}

export function useStartup(
  serverReady: boolean,
  startupFailed: boolean,
  restartServer: () => Promise<void>,
): StartupState {
  const { init: initAuth } = useAuthStore();
  const [splashStep, setSplashStep] = useState<SplashStep>("sidecar");
  const [splashDone, setSplashDone] = useState(false);
  const splashStartRef = useRef<number>(Date.now());

  const [parallaxEnabled, setParallaxEnabledState] = useState(
    () => localStorage.getItem("sparc-parallax") !== "false",
  );

  const setParallaxEnabled = (v: boolean) => {
    setParallaxEnabledState(v);
    localStorage.setItem("sparc-parallax", v ? "true" : "false");
  };

  // Apply persisted theme on mount
  useEffect(() => {
    applyTheme(loadThemeKey());
  }, []);

  // Startup failure → show error step
  useEffect(() => {
    if (startupFailed && splashStep === "sidecar") setSplashStep("failed");
  }, [startupFailed, splashStep]);

  // Step 1 → 2: sidecar ready → acquire token
  useEffect(() => {
    if (serverReady && splashStep === "sidecar") setSplashStep("token");
  }, [serverReady, splashStep]);

  // Step 2 → 3: acquire token → then advance to session
  // Token is stored in memory only — never written to disk or localStorage.
  // On Tauri unavailability (browser/dev), we skip and advance anyway.
  useEffect(() => {
    if (splashStep !== "token") return;
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke<string>("get_sidecar_token"))
      .then((token) => { setToken(token); setSplashStep("session"); })
      .catch(() => { setSplashStep("session"); /* browser/dev — token stays empty */ });
  }, [splashStep]);

  // Step 3 → 4: restore auth session
  useEffect(() => {
    if (splashStep !== "session") return;
    const advance = () => {
      setSplashStep("ready");
      const elapsed = Date.now() - splashStartRef.current;
      const remaining = Math.max(400, 5_000 - elapsed);
      console.log(`[DEBUG-sparc-blank] initAuth done, splashDone in ${remaining}ms`);
      setTimeout(() => setSplashDone(true), remaining);
    };
    initAuth().then(advance).catch((err) => {
      console.warn("[DEBUG-sparc-blank] initAuth failed, advancing anyway:", err);
      advance();
    });
    // initAuth is stable; only re-run when splashStep changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splashStep]);

  const handleRetry = splashStep === "failed"
    ? async () => {
        setSplashStep("sidecar");
        await restartServer();
      }
    : undefined;

  return { splashStep, splashDone, parallaxEnabled, setParallaxEnabled, handleRetry };
}
