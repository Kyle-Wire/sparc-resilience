import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

// OAuth + magic-link callback. Exchanges ?code= for a session cookie, then either
// redirects to ?next= or — if ?desktop=1 — bounces to the Tauri deep link.
export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const code = url.searchParams.get("code");
  const next = url.searchParams.get("next") ?? "/account";
  const desktop = url.searchParams.get("desktop") === "1";
  const state = url.searchParams.get("state") ?? "";

  if (!code) {
    return NextResponse.redirect(new URL("/login?error=missing_code", req.url));
  }

  const supabase = await createSupabaseServerClient();
  const { data, error } = await supabase.auth.exchangeCodeForSession(code);
  if (error || !data.session) {
    return NextResponse.redirect(new URL(`/login?error=${encodeURIComponent(error?.message ?? "exchange_failed")}`, req.url));
  }

  if (desktop) {
    // Hand the desktop app a one-shot code via custom protocol.
    const target = `sparc://auth/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
    return new NextResponse(
      `<!doctype html><html><body style="font-family:system-ui;background:#1a1416;color:#f5efe4;display:grid;place-items:center;height:100vh;margin:0">` +
      `<div style="text-align:center"><h1 style="font-weight:800;letter-spacing:-0.02em">Returning you to SPARC…</h1>` +
      `<p style="opacity:.7">If the app doesn't open automatically, <a style="color:#e73c25" href="${target}">click here</a>.</p></div>` +
      `<script>setTimeout(function(){window.location.href=${JSON.stringify(target)}},150)</script>` +
      `</body></html>`,
      { status: 200, headers: { "content-type": "text/html; charset=utf-8" } },
    );
  }

  return NextResponse.redirect(new URL(next, req.url));
}
