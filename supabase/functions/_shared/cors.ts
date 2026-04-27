// supabase/functions/_shared/cors.ts
// Shared CORS headers for all SPARC Edge Functions. The desktop app
// runs from the Tauri webview (origin `tauri://localhost` on prod
// builds, `http://localhost:1420` in dev), so we can't use a strict
// allow-list — we accept any origin but require the Authorization
// header to actually authorize the request.
export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

export function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
}
