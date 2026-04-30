// Server-side Supabase client (App Router). Reads cookies in server components / route handlers.
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

export async function createSupabaseServerClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) {
    throw new Error("Supabase env vars are not configured. See sparc-website/.env.example.");
  }
  const cookieStore = await cookies();
  return createServerClient(url, anon, {
    cookies: {
      getAll: () => cookieStore.getAll(),
      setAll: (toSet) => {
        try {
          toSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options));
        } catch {
          // Called from a server component during render — cookies are read-only there.
          // Middleware already keeps the session refreshed.
        }
      },
    },
  });
}

// Service-role client — server-only, bypasses RLS. Use only in webhooks / admin routes.
import { createClient } from "@supabase/supabase-js";
export function createSupabaseAdminClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const service = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !service) {
    throw new Error("Supabase admin env vars are not configured.");
  }
  return createClient(url, service, { auth: { autoRefreshToken: false, persistSession: false } });
}
