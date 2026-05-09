// Deno Edge Function: report-crash
// Receives crash payloads from the SPARC Desktop app and forwards them
// to sparcurbanlabs@gmail.com via the Resend API.
//
// Required environment variables (set in the Supabase dashboard):
//   RESEND_API_KEY  — Resend API key for sending email

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";

const RESEND_API_URL = "https://api.resend.com/emails";
const TO_ADDRESS = "sparcurbanlabs@gmail.com";
const FROM_ADDRESS = "SPARC Crash Reporter <noreply@sparclabs.io>";

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return new Response("Bad Request: invalid JSON", { status: 400 });
  }

  const resendKey = Deno.env.get("RESEND_API_KEY");
  if (!resendKey) {
    console.error("[report-crash] RESEND_API_KEY not set");
    return new Response("Internal Server Error", { status: 500 });
  }

  const { page, error, sessionLog, userAgent, timestamp } = body as {
    page?: string;
    error?: { message?: string; stack?: string };
    sessionLog?: unknown[];
    userAgent?: string;
    timestamp?: string;
  };

  const logLines = Array.isArray(sessionLog)
    ? sessionLog
        .slice(-20)
        .map((e) => JSON.stringify(e))
        .join("\n")
    : "(no log)";

  const html = `
<h2>SPARC Desktop Crash Report</h2>
<table>
  <tr><th>Page</th><td>${htmlEscape(page ?? "unknown")}</td></tr>
  <tr><th>Time</th><td>${htmlEscape(timestamp ?? "unknown")}</td></tr>
  <tr><th>User Agent</th><td>${htmlEscape(userAgent ?? "unknown")}</td></tr>
</table>

<h3>Error</h3>
<pre>${htmlEscape(error?.message ?? "(no message)")}</pre>
<pre>${htmlEscape(error?.stack ?? "(no stack)")}</pre>

<h3>Session Log (last 20 entries)</h3>
<pre>${htmlEscape(logLines)}</pre>
`;

  const res = await fetch(RESEND_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${resendKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: FROM_ADDRESS,
      to: [TO_ADDRESS],
      subject: `[SPARC Crash] ${page ?? "unknown"} — ${timestamp?.slice(0, 10) ?? ""}`,
      html,
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    console.error("[report-crash] Resend error:", detail);
    return new Response("Failed to send email", { status: 502 });
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
});

function htmlEscape(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
