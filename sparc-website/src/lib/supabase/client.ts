// Browser-side Supabase client (App Router). Uses cookies via @supabase/ssr.
import { createBrowserClient } from "@supabase/ssr";

export function createSupabaseBrowserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) {
    throw new Error("Supabase env vars are not configured. See sparc-website/.env.example.");
  }
  return createBrowserClient(url, anon);
}
