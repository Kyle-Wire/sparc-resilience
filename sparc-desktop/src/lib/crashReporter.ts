import { supabase } from "@/lib/supabase";
import { getRunLog } from "@/lib/api";

interface CrashPayload {
  page: string;
  error: Error;
}

/**
 * Collect local context (session log tail) and forward a crash report to the
 * `report-crash` Supabase Edge Function, which emails `sparcurbanlabs@gmail.com`
 * via the Resend API.
 */
export async function reportCrash({ page, error }: CrashPayload): Promise<void> {
  // Attempt to pull the last 50 log entries from the sidecar.
  let sessionLog: unknown[] = [];
  try {
    const { entries } = await getRunLog();
    sessionLog = entries.slice(-50);
  } catch {
    // If the sidecar is down or no project is loaded, omit the log.
  }

  await supabase.functions.invoke("report-crash", {
    body: {
      page,
      error: {
        message: error.message,
        stack: error.stack ?? null,
      },
      sessionLog,
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString(),
    },
  });
}
