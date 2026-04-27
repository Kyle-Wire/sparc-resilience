/**
 * Shared types for the auth/licensing layer.
 *
 * Kept deliberately small and serializable — these objects move
 * between Rust (keyring/store) and TypeScript via JSON.
 */

/** Plans, ordered roughly free → enterprise. */
export type Plan = "free" | "pro" | "team";

/** Possible auth providers presented on the LoginPage. */
export type AuthProvider = "google" | "github" | "azure" | "email" | "license_key";

/**
 * The five license states that drive UI gating, plus a small set of
 * transient states the AuthProvider exposes during boot.
 */
export type LicenseState =
  | "verifying" // splash, network call in flight
  | "active" // cache valid OR fresh server response
  | "grace_warning" // cache valid (<15d) + network failed → show banner
  | "needs_recheck" // cache stale (≥15d) + network ok → silent revalidate
  | "grace_expired" // cache stale + network failed → block, Settings only
  | "no_license" // no cache + unauthenticated → Login screen
  | "email_unverified"; // logged in but `email_confirmed_at` IS NULL

/** Cache shape persisted via `tauri-plugin-store`. */
export interface LicenseCache {
  valid: boolean;
  plan: Plan;
  status: string; // "active" | "trialing" | "cancelled" | …
  expiry: string | null; // ISO; null for license-key (no recurring)
  last_verified: string; // ISO
  email: string | null;
  provider: AuthProvider | null;
  /** Last 4 chars of license key, displayed as "****ABCD". */
  license_key_last4: string | null;
}

/** Canonical response shape returned by the `verify-license` Edge Fn. */
export interface VerifyLicenseResponse {
  valid: boolean;
  plan: Plan;
  status: string;
  expiry: string | null;
  email_confirmed: boolean;
  // Optional UI hints from server.
  message?: string;
}

export interface AuthSession {
  access_token: string;
  refresh_token: string;
  user_id: string;
  email: string | null;
  email_confirmed_at: string | null;
  provider: AuthProvider;
  expires_at: number; // unix seconds
}

/** Bag exposed by `useEntitlements()`. */
export interface EntitlementSet {
  plan: Plan;
  /** Hard caps. -1 means unlimited. */
  maxProjects: number;
  maxRunsPerMonth: number;
  /** Boolean feature flags. */
  templates: "blank_only" | "all";
  scenarioRunner: boolean;
  causalDiscoveryLearn: boolean;
  aiChat: boolean;
  reportExport: boolean;
  multiSeat: boolean;
}
