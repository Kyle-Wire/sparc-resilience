// supabase/functions/verify-license/index.ts
//
// Single source of truth for license state. Two callers:
//   (1) Logged-in user — Authorization: Bearer <user JWT> (no body)
//   (2) Key-only user  — body { "key": "..." } (no auth header)
//
// Returns canonical shape consumed by the desktop:
//   { valid, plan, status, expiry, email_confirmed }
//
// The desktop's `verifyLicense()` state machine handles caching,
// 15-day grace, and offline transitions — this function is purely
// "what does the server currently think?" and never refers to the
// client's cache.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";
import { corsHeaders, jsonResponse } from "../_shared/cors.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

interface VerifyResponse {
  valid: boolean;
  plan: "free" | "pro" | "team";
  status: string;
  expiry: string | null;
  email_confirmed: boolean;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (req.method !== "POST") {
    return jsonResponse({ error: "method_not_allowed" }, { status: 405 });
  }

  // Service-role client — bypasses RLS so we can read both
  // `licenses` and `license_keys` regardless of caller identity.
  const admin = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  // Try JWT path first.
  const authHeader = req.headers.get("Authorization");
  if (authHeader?.startsWith("Bearer ")) {
    const jwt = authHeader.slice("Bearer ".length).trim();
    const { data: userData, error: userErr } = await admin.auth.getUser(jwt);
    if (!userErr && userData?.user) {
      const u = userData.user;
      const emailConfirmed = !!u.email_confirmed_at;
      const { data: lic } = await admin
        .from("licenses")
        .select("plan, status, current_period_end")
        .eq("user_id", u.id)
        .maybeSingle();

      const payload: VerifyResponse = {
        valid: !!lic && (lic.status === "active" || lic.status === "trialing"),
        plan: (lic?.plan ?? "free") as "free" | "pro" | "team",
        status: lic?.status ?? "active",
        expiry: lic?.current_period_end ?? null,
        email_confirmed: emailConfirmed,
      };
      return jsonResponse(payload);
    }
  }

  // Fall back to license-key path.
  let body: { key?: string } = {};
  try {
    body = await req.json();
  } catch {
    /* ignore */
  }
  if (!body.key) {
    return jsonResponse({ error: "missing_credentials" }, { status: 401 });
  }

  const { data: row } = await admin
    .from("license_keys")
    .select("plan, status, expiry")
    .eq("key", body.key)
    .maybeSingle();

  if (!row) {
    return jsonResponse(
      { valid: false, plan: "free", status: "expired", expiry: null, email_confirmed: true },
      { status: 200 },
    );
  }

  // Touch last_seen_at fire-and-forget.
  void admin
    .from("license_keys")
    .update({ last_seen_at: new Date().toISOString() })
    .eq("key", body.key);

  const payload: VerifyResponse = {
    valid: row.status === "active",
    plan: row.plan as "free" | "pro" | "team",
    status: row.status,
    expiry: row.expiry,
    email_confirmed: true, // key-only users have no email to confirm
  };
  return jsonResponse(payload);
});
