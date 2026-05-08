import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("[SPARC] Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY — auth will be unavailable.");
}

export const supabase = createClient(supabaseUrl ?? "https://placeholder.supabase.co", supabaseAnonKey ?? "placeholder", {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    storageKey: "sparc-auth",
  },
});
