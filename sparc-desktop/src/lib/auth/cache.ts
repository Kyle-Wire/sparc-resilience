/**
 * Persistent (non-secret) license cache.
 *
 * Lives in `tauri-plugin-store`'s JSON file under the app config dir.
 * The store keeps everything needed to make a *cached* license
 * decision when the network is down:
 *
 *   - the canonical license fields (plan, status, expiry, …)
 *   - `last_verified` so we can compute the 15-day grace window
 *   - the free-tier monthly run counter (for `useDailyRunCounter`)
 *
 * Anything genuinely secret (JWTs, license keys) lives in the OS
 * keyring — see `secrets.ts`.
 */
import { LazyStore } from "@tauri-apps/plugin-store";
import type { LicenseCache, Plan } from "./types";

const STORE_FILE = "license.json";

let _store: LazyStore | null = null;
function store(): LazyStore {
  if (!_store) _store = new LazyStore(STORE_FILE);
  return _store;
}

const KEY_LICENSE = "license";
const KEY_RUN_COUNTER = "run_counter";

export async function readLicenseCache(): Promise<LicenseCache | null> {
  try {
    const v = await store().get<LicenseCache>(KEY_LICENSE);
    return v ?? null;
  } catch {
    return null;
  }
}

export async function writeLicenseCache(cache: LicenseCache): Promise<void> {
  await store().set(KEY_LICENSE, cache);
  await store().save();
}

export async function clearLicenseCache(): Promise<void> {
  await store().delete(KEY_LICENSE);
  await store().save();
}

// ─── Free-tier run counter ──────────────────────────────────────────

export interface RunCounter {
  /** Calendar month bucket "YYYY-MM". */
  month: string;
  count: number;
}

function currentMonthKey(now = new Date()): string {
  const y = now.getUTCFullYear();
  const m = String(now.getUTCMonth() + 1).padStart(2, "0");
  return `${y}-${m}`;
}

export async function readRunCounter(): Promise<RunCounter> {
  const cur = currentMonthKey();
  const v = await store().get<RunCounter>(KEY_RUN_COUNTER);
  if (!v || v.month !== cur) {
    // New month → fresh bucket.
    const fresh: RunCounter = { month: cur, count: 0 };
    return fresh;
  }
  return v;
}

export async function incrementRunCounter(): Promise<RunCounter> {
  const cur = await readRunCounter();
  const next: RunCounter = { month: cur.month, count: cur.count + 1 };
  await store().set(KEY_RUN_COUNTER, next);
  await store().save();
  return next;
}

// ─── Helpers for the verifyLicense() state machine ──────────────────

const FIFTEEN_DAYS_MS = 15 * 24 * 60 * 60 * 1000;

/** Days remaining until the cached license falls out of the 15-day
 *  offline grace window. Negative when expired. */
export function graceDaysRemaining(cache: LicenseCache, now = Date.now()): number {
  const ageMs = now - new Date(cache.last_verified).getTime();
  return Math.ceil((FIFTEEN_DAYS_MS - ageMs) / (24 * 60 * 60 * 1000));
}

export function isWithinGrace(cache: LicenseCache, now = Date.now()): boolean {
  return graceDaysRemaining(cache, now) > 0;
}

export function buildCache(args: {
  plan: Plan;
  status: string;
  valid: boolean;
  expiry: string | null;
  email: string | null;
  provider: LicenseCache["provider"];
  licenseKey?: string | null;
}): LicenseCache {
  const last4 = args.licenseKey ? args.licenseKey.slice(-4) : null;
  return {
    valid: args.valid,
    plan: args.plan,
    status: args.status,
    expiry: args.expiry,
    last_verified: new Date().toISOString(),
    email: args.email,
    provider: args.provider,
    license_key_last4: last4,
  };
}
