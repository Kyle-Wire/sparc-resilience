/**
 * The five-state license verification machine.
 *
 *   1. ACTIVE         — cache valid AND age < 15d                 → allow
 *   2. NEEDS_RECHECK  — cache valid AND age ≥ 15d AND network ok  → revalidate, allow
 *   3. GRACE_WARNING  — cache valid AND age < 15d AND no network  → allow + banner
 *   4. GRACE_EXPIRED  — cache valid AND age ≥ 15d AND no network  → block, Settings only
 *   5. NO_LICENSE     — no cache (or invalid cache)               → Login
 *
 * Plus one transient state used during boot: `email_unverified`
 * (logged in, but Supabase says `email_confirmed_at IS NULL`).
 */
import {
  buildCache,
  graceDaysRemaining,
  isWithinGrace,
  readLicenseCache,
  writeLicenseCache,
  clearLicenseCache,
} from "./cache";
import { isSupabaseConfigured, supabase, persistAccessTokenMirror } from "./supabaseClient";
import { secretGet, secretSet, SECRET_KEYS } from "./secrets";
import type { LicenseCache, LicenseState, VerifyLicenseResponse } from "./types";

// ─── Creator / master-key bypass ──────────────────────────
// Two ways to unlock a perpetual `team` license without a backend:
//   (1) build with `VITE_SPARC_CREATOR_MODE=1`  → auto-active on launch
//   (2) enter a license key starting with "SPARC-CREATOR-"
// Both produce a synthetic LicenseCache that never expires and skips
// every Supabase / Lemon Squeezy round-trip.
const CREATOR_KEY_PREFIX = "SPARC-CREATOR-";
const CREATOR_MODE_ENV =
  (import.meta as any).env?.VITE_SPARC_CREATOR_MODE === "1" ||
  (import.meta as any).env?.VITE_SPARC_CREATOR_MODE === "true";

export function isCreatorKey(key: string | null | undefined): boolean {
  return !!key && key.toUpperCase().startsWith(CREATOR_KEY_PREFIX);
}

function buildCreatorCache(licenseKey: string | null): LicenseCache {
  return buildCache({
    valid: true,
    plan: "team",
    status: "active",
    expiry: null,
    email: "creator@sparclabs.local",
    provider: "license_key",
    licenseKey: licenseKey ?? `${CREATOR_KEY_PREFIX}LOCAL`,
  });
}

async function detectCreatorBypass(): Promise<LicenseCache | null> {
  if (CREATOR_MODE_ENV) {
    return buildCreatorCache(null);
  }
  const stored = await secretGet(SECRET_KEYS.licenseKey);
  if (isCreatorKey(stored)) {
    return buildCreatorCache(stored ?? null);
  }
  return null;
}

export interface VerifyResult {
  state: LicenseState;
  cache: LicenseCache | null;
  daysRemaining: number; // 0 when no cache; negative when expired
}

/** Centralized verification entry point. Called on splash and from
 *  the Settings "Refresh License" button. */
export async function verifyLicense(opts: { force?: boolean } = {}): Promise<VerifyResult> {
  // Branch 0 — creator bypass: env flag or magic key wins immediately.
  const creatorCache = await detectCreatorBypass();
  if (creatorCache) {
    await writeLicenseCache(creatorCache);
    return { state: "active", cache: creatorCache, daysRemaining: 9999 };
  }

  const cache = await readLicenseCache();
  const now = Date.now();

  // Branch 1 — cache valid and fresh: short-circuit.
  if (!opts.force && cache && cache.valid && isWithinGrace(cache, now)) {
    const ageMs = now - new Date(cache.last_verified).getTime();
    if (ageMs < 15 * 24 * 60 * 60 * 1000) {
      return {
        state: "active",
        cache,
        daysRemaining: graceDaysRemaining(cache, now),
      };
    }
  }

  // Branch 2 — try a network revalidation.
  if (!isSupabaseConfigured()) {
    // Unconfigured app == no license backend. Fall back to whatever
    // the cache says.
    if (cache && cache.valid && isWithinGrace(cache, now)) {
      return { state: "grace_warning", cache, daysRemaining: graceDaysRemaining(cache, now) };
    }
    return { state: "no_license", cache: null, daysRemaining: 0 };
  }

  try {
    const fresh = await callVerifyEdgeFn();
    if (!fresh.email_confirmed) {
      return { state: "email_unverified", cache, daysRemaining: 0 };
    }
    if (!fresh.valid) {
      // Server says license invalid — clear cache, send to Login.
      await clearLicenseCache();
      return { state: "no_license", cache: null, daysRemaining: 0 };
    }
    // Refresh cache and return active.
    const updated = await refreshCacheFromResponse(fresh);
    return {
      state: "active",
      cache: updated,
      daysRemaining: graceDaysRemaining(updated, now),
    };
  } catch (e) {
    // Network failed (or auth missing). Fall back to cache.
    if (cache && cache.valid) {
      if (isWithinGrace(cache, now)) {
        return {
          state: "grace_warning",
          cache,
          daysRemaining: graceDaysRemaining(cache, now),
        };
      }
      return {
        state: "grace_expired",
        cache,
        daysRemaining: graceDaysRemaining(cache, now),
      };
    }
    return { state: "no_license", cache: null, daysRemaining: 0 };
  }
}

/** Wipe everything — keyring tokens AND license cache. Used by
 *  Settings → Sign Out. */
export async function signOutAndClear(): Promise<void> {
  if (isSupabaseConfigured()) {
    try {
      await supabase().auth.signOut();
    } catch {
      // ignore
    }
  }
  await clearLicenseCache();
  await persistAccessTokenMirror(null);
}

// ─── internals ───────────────────────────────────────────────────────

async function callVerifyEdgeFn(): Promise<VerifyLicenseResponse> {
  // Prefer JWT (logged-in user). Fall back to license key payload for
  // key-only users who never created a Supabase account.
  const sb = supabase();
  const { data: sessionData } = await sb.auth.getSession();
  const session = sessionData?.session;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  let body: Record<string, unknown> = {};

  if (session?.access_token) {
    headers["Authorization"] = `Bearer ${session.access_token}`;
    await persistAccessTokenMirror(session.access_token);
  } else {
    const key = await secretGet(SECRET_KEYS.licenseKey);
    if (!key) throw new Error("no_session");
    body = { key };
  }

  const { data, error } = await sb.functions.invoke<VerifyLicenseResponse>(
    "verify-license",
    { body, headers },
  );
  if (error || !data) {
    throw new Error(error?.message ?? "verify-license failed");
  }
  return data;
}

async function refreshCacheFromResponse(r: VerifyLicenseResponse): Promise<LicenseCache> {
  const sb = supabase();
  const { data: sessionData } = await sb.auth.getSession();
  const session = sessionData?.session;
  const userEmail = session?.user.email ?? null;
  const provider = session ? "google" : "license_key"; // refined by caller for non-Google providers
  const licenseKey = await secretGet(SECRET_KEYS.licenseKey);

  const cache = buildCache({
    valid: r.valid,
    plan: r.plan,
    status: r.status,
    expiry: r.expiry,
    email: userEmail,
    provider: provider,
    licenseKey: licenseKey,
  });
  await writeLicenseCache(cache);
  return cache;
}

/** Persist the user's license key (used by the LoginPage fallback path). */
export async function setLicenseKey(key: string): Promise<void> {
  await secretSet(SECRET_KEYS.licenseKey, key);
  // If it's a creator key, eagerly mint the team-tier cache so the
  // next render flips to `active` without a network round-trip.
  if (isCreatorKey(key)) {
    await writeLicenseCache(buildCreatorCache(key));
  }
}
