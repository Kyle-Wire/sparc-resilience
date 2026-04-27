/**
 * Supabase JS client with a custom storage adapter that stashes the
 * session in the OS keyring instead of `localStorage`.
 *
 * The Supabase URL + anon key are read from Vite env vars
 * (`VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`) at build time. When
 * unset, the client is constructed lazily and the auth lib treats
 * itself as "not configured" (verifyLicense returns `no_license`).
 */
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { secretGet, secretSet, secretDelete, SECRET_KEYS } from "./secrets";

const URL = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export function isSupabaseConfigured(): boolean {
  return Boolean(URL && ANON_KEY);
}

/**
 * Storage adapter — Supabase calls these synchronously through a
 * Promise-returning interface. We map every key to the keyring so
 * tokens never touch `localStorage`.
 */
const keyringStorage = {
  async getItem(key: string): Promise<string | null> {
    return await secretGet(`sb_${key}`);
  },
  async setItem(key: string, value: string): Promise<void> {
    await secretSet(`sb_${key}`, value);
  },
  async removeItem(key: string): Promise<void> {
    await secretDelete(`sb_${key}`);
  },
};

let _client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (_client) return _client;
  if (!isSupabaseConfigured()) {
    throw new Error(
      "Supabase is not configured. Set VITE_SUPABASE_URL and " +
        "VITE_SUPABASE_ANON_KEY in your .env file.",
    );
  }
  _client = createClient(URL!, ANON_KEY!, {
    auth: {
      storage: keyringStorage,
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: false, // we use loopback OAuth, not URL fragments
    },
    realtime: {
      // We don't use realtime; disabling avoids the WebSocket and
      // keeps our CSP `connect-src` tight.
      params: { eventsPerSecond: 0 },
    },
  });
  return _client;
}

/** Persist a small mirror of the access token in the keyring under a
 *  stable name so other tools (debugging scripts, the sidecar token
 *  exchange) can find it without depending on Supabase's storage key
 *  format. */
export async function persistAccessTokenMirror(token: string | null): Promise<void> {
  if (token) {
    await secretSet(SECRET_KEYS.supabaseAccessToken, token);
  } else {
    await secretDelete(SECRET_KEYS.supabaseAccessToken);
  }
}
