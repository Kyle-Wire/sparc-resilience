// supabase/functions/redeem-key/index.ts
//
// Redeems a Lemon Squeezy license key. Two responsibilities:
//   1. Validate the key with Lemon Squeezy's `/v1/licenses/activate`
//      (proves the key is real and not exhausted).
//   2. Upsert the result into `license_keys` so future `verify-license`
//      calls hit our DB instead of LS.
//
// Input (POST JSON):
//   { "key": "string", "machine_id": "string" }
// Output:
//   { ok: true,  plan: 'pro'|'team', status: 'active', expiry: null }
//   { ok: false, error: 'string' }
//
// Set environment variables in the Supabase project:
//   - LEMON_SQUEEZY_API_KEY (server-side LS API key)
//   - LEMON_SQUEEZY_STORE_ID (numeric, used for sanity-checking)

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";
import { corsHeaders, jsonResponse } from "../_shared/cors.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const LS_API_KEY = Deno.env.get("LEMON_SQUEEZY_API_KEY") ?? "";
const LS_STORE_ID = Deno.env.get("LEMON_SQUEEZY_STORE_ID") ?? "";

// Map Lemon Squeezy variant IDs → SPARC plan tiers. Add new entries
// here as the storefront grows.
const VARIANT_PLAN_MAP: Record<string, "pro" | "team"> = {
  "358410": "pro",  // SPARC Pro (license-key product)
  // "<team_variant_id>": "team",
};

function variantToPlan(variantId: string | undefined): "pro" | "team" {
  if (variantId && VARIANT_PLAN_MAP[variantId]) return VARIANT_PLAN_MAP[variantId];
  return "pro"; // default for unknown variants
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (req.method !== "POST") {
    return jsonResponse({ error: "method_not_allowed" }, { status: 405 });
  }

  let body: { key?: string; machine_id?: string } = {};
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ ok: false, error: "invalid_json" }, { status: 400 });
  }
  if (!body.key) {
    return jsonResponse({ ok: false, error: "missing_key" }, { status: 400 });
  }
  if (!LS_API_KEY) {
    return jsonResponse(
      { ok: false, error: "license_backend_not_configured" },
      { status: 500 },
    );
  }

  // Call Lemon Squeezy.
  const lsResp = await fetch("https://api.lemonsqueezy.com/v1/licenses/activate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      Authorization: `Bearer ${LS_API_KEY}`,
    },
    body: JSON.stringify({
      license_key: body.key,
      instance_name: body.machine_id ?? "sparc-desktop",
    }),
  });

  const lsData = await lsResp.json().catch(() => ({}));
  if (!lsResp.ok || !lsData?.activated) {
    return jsonResponse(
      { ok: false, error: lsData?.error ?? "activation_failed" },
      { status: 400 },
    );
  }

  const meta = lsData.meta ?? {};
  if (LS_STORE_ID && String(meta.store_id) !== LS_STORE_ID) {
    return jsonResponse(
      { ok: false, error: "wrong_store" },
      { status: 400 },
    );
  }

  const plan = variantToPlan(String(meta.variant_id ?? ""));
  const lsKey = lsData.license_key ?? {};
  const expiry: string | null = lsKey.expires_at ?? null;
  const lsLicenseId: string | undefined = lsKey.id ? String(lsKey.id) : undefined;
  const orderId: string | undefined = meta.order_id ? String(meta.order_id) : undefined;

  // Upsert into our DB.
  const admin = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { error: upsertErr } = await admin.from("license_keys").upsert(
    {
      key: body.key,
      plan,
      status: "active",
      ls_order_id: orderId,
      ls_license_key_id: lsLicenseId,
      activated_at: new Date().toISOString(),
      last_seen_at: new Date().toISOString(),
      expiry,
    },
    { onConflict: "key" },
  );

  if (upsertErr) {
    return jsonResponse(
      { ok: false, error: `db_upsert_failed: ${upsertErr.message}` },
      { status: 500 },
    );
  }

  return jsonResponse({
    ok: true,
    plan,
    status: "active",
    expiry,
  });
});
