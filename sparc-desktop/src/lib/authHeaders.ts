/**
 * sparc-desktop/src/lib/authHeaders.ts
 *
 * Centralises the construction of request auth headers so that `api.ts`
 * does not need to import Supabase types directly, and so the logic is
 * testable in isolation.
 */

import { getToken } from "@/stores/tokenStore";
import { useAuthStore } from "@/stores/authStore";

/**
 * Return the auth headers for an outgoing sidecar request.
 *
 * - `x-sparc-token` — local sidecar token (from `tokenStore`)
 * - `Authorization: Bearer …` — Supabase JWT (from `authStore`)
 *
 * Only keys with non-empty values are included.
 */
export function getAuthHeaders(): Record<string, string> {
  const token = getToken();
  const jwt = useAuthStore.getState().session?.access_token;
  const headers: Record<string, string> = {};
  if (token) headers["x-sparc-token"] = token;
  if (jwt) headers["Authorization"] = `Bearer ${jwt}`;
  return headers;
}
