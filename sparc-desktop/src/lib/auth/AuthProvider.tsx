/**
 * Top-level auth context.
 *
 * Owns:
 *   - the cached license + state machine result
 *   - the Supabase session (when present)
 *   - sign-in / sign-up / sign-out / redeem-key actions
 *
 * Consumed by:
 *   - `App.tsx` for the splash/login/blocking-modal gates
 *   - `useEntitlements()` for feature gating throughout the UI
 *   - `SettingsPage` for the License & Account cards
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { open as openExternal } from "@tauri-apps/plugin-shell";

import {
  isSupabaseConfigured,
  supabase,
  persistAccessTokenMirror,
} from "@/lib/auth/supabaseClient";
import {
  setLicenseKey,
  signOutAndClear,
  verifyLicense,
  type VerifyResult,
} from "@/lib/auth/license";
import type { AuthProvider as Provider, LicenseCache, LicenseState } from "@/lib/auth/types";

// ─── Context value ──────────────────────────────────────────────────

export interface AuthContextValue {
  /** Current state-machine state. */
  state: LicenseState;
  /** Last cached license, if any. */
  license: LicenseCache | null;
  /** Days remaining in the offline grace window. */
  daysRemaining: number;
  /** Logged-in user email, if known. */
  email: string | null;
  /** Provider that issued the current session, if known. */
  provider: Provider | null;
  /** True while verifyLicense() is running. */
  busy: boolean;

  /** Re-run verifyLicense (force=true triggers a network call). */
  refresh(opts?: { force?: boolean }): Promise<void>;

  /** Start an OAuth flow via the loopback redirect. Returns when the
   *  browser exchange completes and a session is stored. */
  signInWithProvider(provider: "google" | "github" | "azure"): Promise<void>;

  /** Email + password sign-in. */
  signInWithEmail(email: string, password: string): Promise<void>;

  /** Email + password sign-up. Triggers a confirmation email. */
  signUpWithEmail(email: string, password: string): Promise<void>;

  /** Send password reset email. */
  resetPassword(email: string): Promise<void>;

  /** Persist a license key + verify it. */
  redeemKey(key: string): Promise<void>;

  /** Wipe session + cache, return to LoginScreen. */
  signOut(): Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const v = useContext(AuthContext);
  if (!v) throw new Error("useAuth() must be used inside <AuthProvider>");
  return v;
}

// ─── Provider implementation ────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LicenseState>("verifying");
  const [license, setLicense] = useState<LicenseCache | null>(null);
  const [daysRemaining, setDaysRemaining] = useState(0);
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);

  const apply = useCallback((r: VerifyResult) => {
    if (!mounted.current) return;
    setState(r.state);
    setLicense(r.cache);
    setDaysRemaining(r.daysRemaining);
  }, []);

  const refresh = useCallback(
    async (opts: { force?: boolean } = {}) => {
      setBusy(true);
      try {
        const r = await verifyLicense(opts);
        apply(r);
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [apply],
  );

  // Initial verify on mount + re-verify when Supabase auth state changes.
  useEffect(() => {
    mounted.current = true;
    void refresh();

    if (isSupabaseConfigured()) {
      const { data: sub } = supabase().auth.onAuthStateChange((_event, session) => {
        void persistAccessTokenMirror(session?.access_token ?? null);
        // Re-verify on sign-in / sign-out / token-refresh.
        void refresh({ force: true });
      });
      return () => {
        mounted.current = false;
        sub.subscription.unsubscribe();
      };
    }
    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  // ─── Provider OAuth via loopback ──────────────────────────────────
  const signInWithProvider = useCallback(
    async (provider: "google" | "github" | "azure") => {
      if (!isSupabaseConfigured()) throw new Error("Supabase not configured");

      // 1. Start loopback listener and grab the redirect URI + CSRF state.
      const { port, state: csrf } = await invoke<{ port: number; state: string }>(
        "oauth_start",
      );
      const redirectTo = `http://127.0.0.1:${port}`;

      try {
        // 2. Build the Supabase OAuth URL with our state token.
        const { data, error } = await supabase().auth.signInWithOAuth({
          provider,
          options: {
            redirectTo,
            queryParams: { state: csrf, prompt: "select_account" },
            skipBrowserRedirect: true,
          },
        });
        if (error || !data?.url) {
          await invoke("oauth_cancel", { stateToken: csrf }).catch(() => {});
          throw new Error(error?.message ?? "OAuth start failed");
        }

        // 3. Open the system browser for the user to sign in.
        await openExternal(data.url);

        // 4. Await the loopback callback → exchange code → session.
        const code = await invoke<string>("oauth_await_code", { stateToken: csrf });
        const { error: exErr } = await supabase().auth.exchangeCodeForSession(code);
        if (exErr) throw new Error(exErr.message);
      } catch (e) {
        await invoke("oauth_cancel", { stateToken: csrf }).catch(() => {});
        throw e;
      }

      // verifyLicense will re-fire via onAuthStateChange.
    },
    [],
  );

  const signInWithEmail = useCallback(async (email: string, password: string) => {
    const { error } = await supabase().auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
  }, []);

  const signUpWithEmail = useCallback(async (email: string, password: string) => {
    const { error } = await supabase().auth.signUp({
      email,
      password,
      options: { emailRedirectTo: undefined },
    });
    if (error) throw new Error(error.message);
  }, []);

  const resetPassword = useCallback(async (email: string) => {
    const { error } = await supabase().auth.resetPasswordForEmail(email);
    if (error) throw new Error(error.message);
  }, []);

  const redeemKey = useCallback(
    async (key: string) => {
      await setLicenseKey(key);
      await refresh({ force: true });
    },
    [refresh],
  );

  const signOut = useCallback(async () => {
    await signOutAndClear();
    setState("no_license");
    setLicense(null);
    setDaysRemaining(0);
  }, []);

  // Keep a small derived state for convenience.
  const email = license?.email ?? null;
  const provider = license?.provider ?? null;

  const value = useMemo<AuthContextValue>(
    () => ({
      state,
      license,
      daysRemaining,
      email,
      provider,
      busy,
      refresh,
      signInWithProvider,
      signInWithEmail,
      signUpWithEmail,
      resetPassword,
      redeemKey,
      signOut,
    }),
    [
      state,
      license,
      daysRemaining,
      email,
      provider,
      busy,
      refresh,
      signInWithProvider,
      signInWithEmail,
      signUpWithEmail,
      resetPassword,
      redeemKey,
      signOut,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
