import { create } from "zustand";
import type { User, Session } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

interface AuthState {
  user: User | null;
  session: Session | null;
  /** true while the initial session restore is in-flight */
  loading: boolean;
  error: string | null;

  /** Call once on app boot to restore any persisted session */
  init: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
  /** Unsubscribe the auth-state listener; call on app teardown. */
  destroy: () => void;
}

// Module-level ref so `destroy()` can cancel the Supabase subscription.
let _unsubscribe: (() => void) | null = null;

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  session: null,
  loading: true,
  error: null,

  init: async () => {
    set({ loading: true, error: null });
    try {
      const { data, error } = await supabase.auth.getSession();
      if (error) throw error;
      set({
        session: data.session,
        user: data.session?.user ?? null,
        loading: false,
      });
      // Keep store in sync with Supabase token refreshes.
      // Capture the returned subscription so destroy() can clean up.
      const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
        set({ session, user: session?.user ?? null });
      });
      _unsubscribe = () => sub.subscription.unsubscribe();
    } catch (err) {
      set({
        loading: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },

  signIn: async (email: string, password: string) => {
    set({ loading: true, error: null });
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      set({ loading: false, error: error.message });
      return;
    }
    set({ session: data.session, user: data.user, loading: false });
  },

  signOut: async () => {
    await supabase.auth.signOut();
    set({ session: null, user: null });
  },

  clearError: () => set({ error: null }),

  destroy: () => {
    _unsubscribe?.();
    _unsubscribe = null;
  },
}));
