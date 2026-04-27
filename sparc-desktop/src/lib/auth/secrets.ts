/**
 * Tauri-aware secret storage.
 *
 * Wraps the `secret_get/set/delete` Rust commands so the rest of the
 * code can call them as plain async functions and still build under
 * `pnpm dev` (browser) — where the Tauri shell is unavailable, we
 * fall back to a session-scoped in-memory map.
 */
import { invoke } from "@tauri-apps/api/core";

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

const _memFallback = new Map<string, string>();

export async function secretSet(key: string, value: string): Promise<void> {
  if (!isTauri()) {
    _memFallback.set(key, value);
    return;
  }
  await invoke("secret_set", { key, value });
}

export async function secretGet(key: string): Promise<string | null> {
  if (!isTauri()) {
    return _memFallback.get(key) ?? null;
  }
  const v = await invoke<string | null>("secret_get", { key });
  return v ?? null;
}

export async function secretDelete(key: string): Promise<void> {
  if (!isTauri()) {
    _memFallback.delete(key);
    return;
  }
  await invoke("secret_delete", { key });
}

// Canonical key names used by the auth layer. Centralized here so we
// never typo a keyring entry.
export const SECRET_KEYS = {
  supabaseAccessToken: "supabase_access_token",
  supabaseRefreshToken: "supabase_refresh_token",
  licenseKey: "license_key",
  anthropicApiKey: "anthropic_api_key",
} as const;
